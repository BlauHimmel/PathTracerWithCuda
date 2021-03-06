#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust\random.h>
#include <thrust\device_ptr.h>

#include "curand.h"
#include "curand_kernel.h"

#include "Math\basic_math.hpp"
#include "Math\cuda_math.hpp"

#include "Core\path_tracer_kernel.h"
#include "Core\sphere.h"
#include "Core\triangle.h"
#include "Core\image.h"
#include "Core\ray.h"
#include "Core\camera.h"
#include "Core\fresnel.h"
#include "Core\material.h"
#include "Core\parallel_function.h"
#include "Core\cube_map.h"
#include "Core\triangle_mesh.h"
#include "Core\configuration.h"
#include "Core\texture.h"
#include "Bvh\bvh_node.h"

enum class object_type
{
	none,
	sphere,
	triangle
};

__device__ int hash(int a)
{
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

__device__ float3 reflection(
	const float3& normal,
	const float3& in_direction
)
{
	return in_direction - 2.0f * dot(normal, in_direction) * normal;
}

__device__ float3 refraction(
	const float3& normal,
	const float3& in_direction,
	float in_refraction_index,
	float out_refraction_index
)
{
	//[ ��r (N �� I) �C ��(1 �C ��r^2 (1 �C(N �� I)^2)) ] N �C ��r I
	//��r = ��i / ��o
	float3 i = in_direction * -1.0f;
	float n_dot_i = dot(normal, i);

	float refraction_ratio = in_refraction_index / out_refraction_index;
	float a = refraction_ratio * n_dot_i;
	float b = 1.0f - refraction_ratio * refraction_ratio * (1.0f - n_dot_i * n_dot_i);

	if (b < 0.0f)
	{
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	if (n_dot_i > 0)
	{
		return normal * (a - sqrt(b)) - refraction_ratio * i;
	}
	else
	{
		return normal * (a + sqrt(b)) - refraction_ratio * i;
	}
}

__device__ bool intersect_triangle_mesh_bvh(
	triangle* triangles,			//in	
	bvh_node_device** bvh_nodes,	//in
	int mesh_index,					//in
	const ray& ray,					//in
	configuration* config,			//in
	float& hit_t,					//out	
	float& hit_t1,					//out	
	float& hit_t2,					//out	
	int& hit_triangle_index			//out
)
{
	float min_t = INFINITY;
	float min_t1 = INFINITY;
	float min_t2 = INFINITY;
	int min_triangle_index;

	float current_t = INFINITY;
	float current_t1 = INFINITY;
	float current_t2 = INFINITY;

	bool is_hit = false;
	float bounding_box_hit_t = INFINITY;

	bvh_node_device* bvh_node = bvh_nodes[mesh_index];

	int node_num = bvh_node[0].next_node_index;
	int traversal_position = 0;
	while (traversal_position != node_num)
	{
		bvh_node_device current_node = bvh_node[traversal_position];

		//if hit the box them check if it is leaf node, otherwise check stop traversing on this branch
		if (current_node.box.intersect_bounding_box(ray, bounding_box_hit_t) && bounding_box_hit_t <= min_t)
		{
			if (current_node.is_leaf)
			{
				//intersect with each triangles in the leaf node and update the relevant minimal parameters
				for (int i = 0; i < config->bvh_leaf_node_triangle_num; i++)
				{
					int triangle_index = current_node.triangle_indices[i];
					if (triangle_index != -1)
					{
						if (triangles[triangle_index].intersect(ray, current_t, current_t1, current_t2) && current_t > 0.0f && current_t < min_t)
						{
							min_t = current_t;
							min_t1 = current_t1;
							min_t2 = current_t2;
							min_triangle_index = triangle_index;
							is_hit = true;
						}
					}
					else
					{
						break;
					}
				}
			}

			traversal_position++;
		}
		else
		{
			traversal_position = current_node.next_node_index;
		}
	}

	if (is_hit)
	{
		hit_t = min_t;
		hit_t1 = min_t1;
		hit_t2 = min_t2;
		hit_triangle_index = min_triangle_index;
	}

	return is_hit;
}

__device__ float3 sample_on_hemisphere_cosine_weight(
	const float3& normal,	//in
	float rand1,			//in
	float rand2				//in
)
{
	//spherical coordinate
	float cos_theta = sqrt(rand1);
	float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
	float phi = rand2 * TWO_PI;

	//find orthogonal base
	float3 any_direction;
	if (abs(normal.x) < SQRT_ONE_THIRD)
	{
		any_direction = make_float3(1.0f, 0.0f, 0.0f);
	}
	else if (abs(normal.y) < SQRT_ONE_THIRD)
	{
		any_direction = make_float3(0.0f, 1.0f, 0.0f);
	}
	else if (abs(normal.z) < SQRT_ONE_THIRD)
	{
		any_direction = make_float3(0.0f, 0.0f, 1.0f);
	}
	float3 vec_i = normalize(cross(normal, any_direction));
	float3 vec_j = cross(normal, vec_i);

	return cos_theta * normal + __cosf(phi) * sin_theta * vec_i + __sinf(phi) * sin_theta * vec_j;
}

__device__ float3 sample_on_hemisphere_ggx_weight(
	const float3& normal,	//in
	float roughness,		//in
	float rand1,			//in
	float rand2				//in
)
{
	float theta = atanf(roughness * sqrtf(rand1) / sqrtf(1.0f - rand1));
	float phi = rand2 * TWO_PI;
	float cos_theta = __cosf(theta);
	float sin_theta = __sinf(theta);

	float3 any_direction;
	if (abs(normal.x) < SQRT_ONE_THIRD)
	{
		any_direction = make_float3(1.0f, 0.0f, 0.0f);
	}
	else if (abs(normal.y) < SQRT_ONE_THIRD)
	{
		any_direction = make_float3(0.0f, 1.0f, 0.0f);
	}
	else if (abs(normal.z) < SQRT_ONE_THIRD)
	{
		any_direction = make_float3(0.0f, 0.0f, 1.0f);
	}
	float3 vec_i = normalize(cross(normal, any_direction));
	float3 vec_j = cross(normal, vec_i);

	return cos_theta * normal + __cosf(phi) * sin_theta * vec_i + __sinf(phi) * sin_theta * vec_j;
}

__device__ float3 sample_on_sphere(
	float rand1,			//in
	float rand2				//in
)
{
	//spherical coordinate
	float cos_theta = rand1 * 2.0f - 1.0f;
	float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
	float phi = rand2 * TWO_PI;

	return make_float3(cos_theta, __cosf(phi) * sin_theta, __sinf(phi) * sin_theta);
}

__device__ color compute_absorption_through_medium(
	const color& absorption_coefficient,	//in 
	float scaterring_distance				//in
)
{
	//E = E_0 * e^(-��x)
	return make_float3(
		__powf(E, -1.0f * absorption_coefficient.x * scaterring_distance),
		__powf(E, -1.0f * absorption_coefficient.y * scaterring_distance),
		__powf(E, -1.0f * absorption_coefficient.z * scaterring_distance)
	);
}

__device__ float compute_ggx_shadowing_masking(
	float roughness,				//in
	const float3& macro_normal,		//in
	const float3& micro_normal,		//in
	const float3& ray_direction		//in
)
{
	float3 v = -1.0f * ray_direction;
	float v_dot_n = dot(v, macro_normal);
	float v_dot_m = dot(v, micro_normal);

	float positive_value = (v_dot_m / v_dot_n) > 0.0f ? 1.0f : 0.0f;
	if (positive_value == 0.0f)
	{
		return 0.0f;
	}

	float roughness_square = roughness * roughness;

	float cos_v_square = v_dot_n * v_dot_n;
	float tan_v_square = (1.0f - cos_v_square) / cos_v_square;
	return 2.0f / (1.0f + sqrtf(1.0f + roughness_square * tan_v_square));
}

__global__ void init_data_kernel(
	int pixel_count,					//in
	int* energy_exist_pixels,			//in out
	color* not_absorbed_colors,			//in out
	color* accumulated_colors,			//in out
	scattering* scatterings,			//in out
	configuration* config				//in
)
{
	int block_x = blockIdx.x;
	int thread_x = threadIdx.x;

	int pixel_index = config->block_size * block_x + thread_x;
	bool is_index_valid = pixel_index < pixel_count;

	if (is_index_valid)
	{
		energy_exist_pixels[pixel_index] = pixel_index;
		not_absorbed_colors[pixel_index] = make_float3(1.0f, 1.0f, 1.0f);
		accumulated_colors[pixel_index] = make_float3(0.0f, 0.0f, 0.0f);
		scatterings[pixel_index] = scattering::get_default_scattering(config);
	}
}

__global__ void generate_ray_kernel(
	float3 eye,							//in
	float3 view,						//in
	float3 up,							//in
	float2 resolution,					//in
	float2 fov,							//in
	float aperture_radius,				//in
	float focal_distance,				//in
	int pixel_count,					//in
	ray* rays,							//in out
	int seed,							//in
	configuration* config				//in
)
{
	int block_x = blockIdx.x;
	int thread_x = threadIdx.x;

	int pixel_index = config->block_size * block_x + thread_x;
	bool is_index_valid = pixel_index < pixel_count;

	if (is_index_valid)
	{
		int image_y = (int)(pixel_index / resolution.x);
		int image_x = pixel_index - (image_y * resolution.x);

		thrust::default_random_engine random_engine(hash(seed) * hash(seed) * hash(pixel_index));
		thrust::uniform_real_distribution<float> uniform_distribution(-0.5f, 0.5f);

		//for anti-aliasing
		float jitter_x = 0.0f;
		float jitter_y = 0.0f;

		if (config->use_anti_alias)
		{
			jitter_x = uniform_distribution(random_engine);
			jitter_y = uniform_distribution(random_engine);
		}

		float distance = length(view);

		//vector base
		float3 horizontal = normalize(cross(view, up));
		float3 vertical = normalize(cross(horizontal, view));

		//edge of canvas
		float3 x_axis = horizontal * (distance * __tanf(fov.x * 0.5f * (PI / 180.0f)));
		float3 y_axis = vertical * (distance * __tanf(-fov.y * 0.5f * (PI / 180.0f)));
		
		float normalized_image_x = (((float)image_x + jitter_x) / (resolution.x - 1.0f)) * 2.0f - 1.0f;
		float normalized_image_y = (((float)image_y + jitter_y) / (resolution.y - 1.0f)) * 2.0f - 1.0f;

		//for all the ray (cast from one point) refracted by the convex will be cast on one point finally(focal point)
		float3 point_on_canvas_plane = eye + view + normalized_image_x * x_axis + normalized_image_y * y_axis;
		float3 point_on_image_plane = eye + normalize(point_on_canvas_plane - eye) * focal_distance;

		float3 point_on_aperture;
		if (aperture_radius > 0.00001f)
		{
			//sample on convex, note that convex is not a plane
			float rand1 = uniform_distribution(random_engine) + 0.5f;
			float rand2 = uniform_distribution(random_engine) + 0.5f;

			float angle = rand1 * TWO_PI;
			float distance = aperture_radius * sqrt(rand2);

			float aperture_x = __cosf(angle) * distance;
			float aperture_y = __sinf(angle) * distance;

			point_on_aperture = eye + aperture_x * horizontal + aperture_y * vertical;
		}
		else
		{
			point_on_aperture = eye;
		}

		float3 direction = normalize(point_on_image_plane - point_on_aperture);

		rays[pixel_index].origin = point_on_aperture;
		rays[pixel_index].direction = direction;
	}
}

__global__ void trace_ray_kernel(
	int mesh_num,							//in
	bvh_node_device** bvh_nodes,			//in
	triangle* triangles,					//in
	int sphere_num,							//in
	sphere* spheres,						//in
	int pixel_count,						//in
	int depth,								//in
	int energy_exist_pixels_count,			//in
	int* energy_exist_pixels,				//in out
	ray* rays,								//in out
	scattering* scatterings,				//in out
	float3* not_absorbed_colors,			//in out
	float3* accumulated_colors,				//in out
	cube_map* sky_cube_map,					//in
	texture_wrapper* mesh_textures,			//in
	int seed,								//in
	configuration* config					//in
)
{
	int block_x = blockIdx.x;
	int thread_x = threadIdx.x;

	int energy_exist_pixel_index = config->block_size * block_x + thread_x;
	bool is_index_valid = energy_exist_pixel_index < energy_exist_pixels_count;

	if (!is_index_valid)
	{
		return;
	}

	int pixel_index = energy_exist_pixels[energy_exist_pixel_index];
	ray tracing_ray = rays[pixel_index];

	thrust::default_random_engine random_engine(hash(seed) * hash(pixel_index) * hash(depth));
	thrust::uniform_real_distribution<float> uniform_distribution(0.0f, 1.0f);

	float hit_t, hit_t1, hit_t2;
	float3 hit_point, hit_normal;
	int hit_triangle_index;

	float min_t = INFINITY;
	float min_t1 = INFINITY;
	float min_t2 = INFINITY;
	float3 min_point = make_float3(0.0f, 0.0f, 0.0f);
	float3 min_normal = make_float3(0.0f, 0.0f, 0.0f);
	object_type min_type = object_type::none;
	int min_sphere_index = -1;
	int min_triangle_index = -1;

	for (int i = 0; i < sphere_num; i++)
	{
		if (spheres[i].intersect(tracing_ray, hit_point, hit_normal, hit_t) && hit_t < min_t && hit_t > 0.0f)
		{
			min_t = hit_t;
			min_point = hit_point;
			min_normal = hit_normal;
			min_sphere_index = i;
			min_type = object_type::sphere;
		}
	}
	
	for (int mesh_index = 0; mesh_index < mesh_num; mesh_index++)
	{
		if (intersect_triangle_mesh_bvh(triangles, bvh_nodes, mesh_index, tracing_ray, config, hit_t, hit_t1, hit_t2, hit_triangle_index) && hit_t < min_t && hit_t > 0.0f)
		{
			min_t = hit_t;
			min_t1 = hit_t1;
			min_t2 = hit_t2;
			min_point = tracing_ray.point_on_ray(hit_t);
			min_triangle_index = hit_triangle_index;
			min_type = object_type::triangle;
		}
	}

	//absorption and scattering of medium
	//TODO:WHEN USE MULTIPLE MATERIAL, HOW TO CHOOSE MEDIUM?
	scattering current_scattering = scatterings[pixel_index];

	if (current_scattering.reduced_scattering_coefficient.x > 0.0f ||
		length(current_scattering.absorption_coefficient) > config->sss_threshold)
	{
		float rand = uniform_distribution(random_engine);
		float scaterring_distance = -__logf(rand) / current_scattering.reduced_scattering_coefficient.x;
		//TODO:ANISOTROPIC SCATTERING
		if (scaterring_distance < min_t)
		{
			//absorption and scattering
			float rand1 = uniform_distribution(random_engine);
			float rand2 = uniform_distribution(random_engine);

			ray next_ray;
			next_ray.origin = tracing_ray.point_on_ray(scaterring_distance);
			next_ray.direction = sample_on_sphere(rand1, rand2);
			rays[pixel_index] = next_ray;

			not_absorbed_colors[pixel_index] *= compute_absorption_through_medium(current_scattering.absorption_coefficient, scaterring_distance);

			//kill the low energy ray
			if (length(not_absorbed_colors[pixel_index]) <= config->energy_exist_threshold)
			{
				energy_exist_pixels[energy_exist_pixel_index] = -1;
			}

			return;
		}
		else
		{
			//absorption
			not_absorbed_colors[pixel_index] *= compute_absorption_through_medium(current_scattering.absorption_coefficient, min_t);
		}
	}

	if (min_type != object_type::none)
	{
		material min_mat;
		if (min_type == object_type::sphere)
		{
			min_mat = spheres[min_sphere_index].mat;
		}
		else if (min_type == object_type::triangle)
		{
			min_mat = *(triangles[min_triangle_index].mat);
			triangle hit_triangle = triangles[min_triangle_index];
			min_normal = hit_triangle.normal0 * (1.0f - min_t1 - min_t2) +
				hit_triangle.normal1 * min_t1 + 
				hit_triangle.normal2 * min_t2;

			float2 uv = make_float2(0.0f, 0.0f);

			if (min_mat.diffuse_texture_id != -1 || min_mat.specular_texture_id != -1)
			{
				uv = hit_triangle.uv0 * (1.0f - min_t1 - min_t2) +
					hit_triangle.uv1 * min_t1 +
					hit_triangle.uv2 * min_t2;
			}

			if (min_mat.diffuse_texture_id != -1)
			{
				min_mat.diffuse_color = min_mat.diffuse_color * mesh_textures[min_mat.diffuse_texture_id].sample_texture(uv, config->use_bilinear);
			}

			if (min_mat.specular_texture_id != -1)
			{
				min_mat.specular_color = min_mat.specular_color * mesh_textures[min_mat.specular_texture_id].sample_texture(uv, config->use_bilinear);
			}
		}

		float3 in_direction = tracing_ray.direction;

		medium in_medium;
		in_medium = medium::get_default_medium(config);
		medium out_medium = min_mat.medium;

		bool is_hit_on_back = dot(in_direction, min_normal) > 0;

		if (is_hit_on_back)
		{
			min_normal *= -1.0f;

			if (min_mat.is_transparent)
			{
				medium temp = in_medium;
				in_medium = out_medium;
				out_medium = temp;
			}
		}

		float3 reflection_direction = reflection(min_normal, in_direction);
		float3 refraction_direction = refraction(min_normal, in_direction, in_medium.refraction_index, out_medium.refraction_index);
		float3 bias_vector = config->vector_bias_length * min_normal;

		fresnel fresnel;
		if (min_mat.medium.extinction_coefficient == 0 || min_mat.is_transparent)
		{
			fresnel = fresnel::get_fresnel_dielectrics(min_normal, in_direction, in_medium.refraction_index, out_medium.refraction_index, reflection_direction, refraction_direction);
		}
		else
		{
			fresnel = fresnel::get_fresnel_conductors(min_normal, in_direction, out_medium.refraction_index, out_medium.extinction_coefficient);
		}
		
		float rand = uniform_distribution(random_engine);

		if (rand < fresnel.reflection_index)
		{
			//reflection
			float rand1 = uniform_distribution(random_engine);
			float rand2 = uniform_distribution(random_engine);

			float remap_roughness = __powf(min_mat.roughness, 1.85f) * 0.238f;
			float3 micro_normal = sample_on_hemisphere_ggx_weight(min_normal, remap_roughness, rand1, rand2);
			
			float3 micro_reflection_direction = reflection(micro_normal, in_direction);
			float self_shadowing = compute_ggx_shadowing_masking(remap_roughness, min_normal, micro_normal, tracing_ray.direction) * 
				compute_ggx_shadowing_masking(remap_roughness, min_normal, micro_normal, micro_reflection_direction);

			ray next_ray;
			next_ray.origin = min_point + bias_vector;
			next_ray.direction = micro_reflection_direction;
			rays[pixel_index] = next_ray;

			not_absorbed_colors[pixel_index] *= (min_mat.specular_color * self_shadowing);
		}
		else if (min_mat.is_transparent)
		{
			//refraction
			ray next_ray;
			next_ray.origin = min_point - bias_vector;
			next_ray.direction = refraction_direction;
			rays[pixel_index] = next_ray;

			//transmitted into a new medium
			scatterings[pixel_index] = out_medium.scattering;
			not_absorbed_colors[pixel_index] *= __powf((out_medium.refraction_index / in_medium.refraction_index), 2.0f);
		}
		else
		{
			accumulated_colors[pixel_index] += not_absorbed_colors[pixel_index] * min_mat.emission_color;
			not_absorbed_colors[pixel_index] *= min_mat.diffuse_color;

			//diffuse
			float rand1 = uniform_distribution(random_engine);
			float rand2 = uniform_distribution(random_engine);

			ray next_ray;
			next_ray.origin = min_point + bias_vector;
			next_ray.direction = sample_on_hemisphere_cosine_weight(min_normal, rand1, rand2);
			rays[pixel_index] = next_ray;
		}

		//kill the low energy ray
		if (length(not_absorbed_colors[pixel_index]) <= config->energy_exist_threshold)
		{
			energy_exist_pixels[energy_exist_pixel_index] = -1;
		}
	}
	else
	{
		float3 background_color = sky_cube_map->get_background_color(tracing_ray.direction, config->use_sky_box, config->use_sky, config->use_bilinear);
		accumulated_colors[pixel_index] += not_absorbed_colors[pixel_index] * background_color;
		//kill the low ray because it has left the scene
		energy_exist_pixels[energy_exist_pixel_index] = -1;
	}
}

__global__ void pixel_256_transform_gamma_corrected_kernel(
	color* accumulated_colors,			//in 
	color* image_pixels,				//in
	color256* image_pixels_256,			//in out
	int pixel_count,					//in
	int pass_counter,					//in
	configuration* config				//in
)
{
	int block_x = blockIdx.x;
	int thread_x = threadIdx.x;

	int pixel_index = config->block_size * block_x + thread_x;
	bool is_index_valid = pixel_index < pixel_count;

	if (is_index_valid)
	{
		if (pass_counter != 1)
		{
			image_pixels[pixel_index] += clamp(accumulated_colors[pixel_index], 0.0f, static_cast<float>(config->max_tracer_depth) * 2.0f);
		}
		else
		{
			image_pixels[pixel_index] = clamp(accumulated_colors[pixel_index], 0.0f, static_cast<float>(config->max_tracer_depth) * 2.0f);
		}

		color pixel = image_pixels[pixel_index] / (float)pass_counter;
		float x, y, z;

		if (config->gamma_correction)
		{
			float inverse_gamma = 0.45454545f;
			color corrected_pixel;
			corrected_pixel.x = __expf(inverse_gamma * __logf(pixel.x));
			corrected_pixel.y = __expf(inverse_gamma * __logf(pixel.y));
			corrected_pixel.z = __expf(inverse_gamma * __logf(pixel.z));

			x = clamp(corrected_pixel.x * 255.0f, 0.0f, 255.0f);
			y = clamp(corrected_pixel.y * 255.0f, 0.0f, 255.0f);
			z = clamp(corrected_pixel.z * 255.0f, 0.0f, 255.0f);
		}
		else
		{
			x = clamp(pixel.x * 255.0f, 0.0f, 255.0f);
			y = clamp(pixel.y * 255.0f, 0.0f, 255.0f);
			z = clamp(pixel.z * 255.0f, 0.0f, 255.0f);
		}

		color256 color_256;
		color_256.x = (uchar)x;
		color_256.y = (uchar)y;
		color_256.z = (uchar)z;

		image_pixels_256[pixel_index] = color_256;
	}
}

//===============================================================================================================
extern "C" void path_tracer_kernel(
	int mesh_num,												//in
	bvh_node_device** bvh_nodes_device,							//in
	triangle* triangles_device,									//in
	int sphere_num,												//in
	sphere* spheres_device, 									//in
	int pixel_count, 											//in
	color* image_pixels,										//in out
	color256* image_pixels_256,									//in out
	int pass_counter, 											//in
	render_camera* render_camera_device,						//in
	cube_map* sky_cube_map_device,								//in
	color* not_absorbed_colors_device,							//in 
	color* accumulated_colors_device,							//in 
	ray* rays_device,											//in 
	int* energy_exist_pixels_device,							//in 
	scattering* scatterings_device,								//in 
	texture_wrapper* mesh_textures_device,						//in
	configuration* config_device								//in 
)
{
	configuration config = *config_device;

	int threads_num_per_block = config.block_size;
	int total_blocks_num_per_gird = (pixel_count + threads_num_per_block - 1) / threads_num_per_block;

	int energy_exist_pixels_count = pixel_count;
	int seed = pass_counter;
	int max_depth = config.max_tracer_depth;

	init_data_kernel <<<total_blocks_num_per_gird, threads_num_per_block >>> (
		pixel_count, 
		energy_exist_pixels_device,
		not_absorbed_colors_device,
		accumulated_colors_device,
		scatterings_device,
		config_device
		);

	generate_ray_kernel <<<total_blocks_num_per_gird, threads_num_per_block>>> (
		render_camera_device->eye,
		render_camera_device->view,
		render_camera_device->up,
		render_camera_device->resolution,
		render_camera_device->fov,
		render_camera_device->aperture_radius,
		render_camera_device->focal_distance,
		pixel_count, 
		rays_device,
		seed,
		config_device
		);

	for (int depth = 0; depth < max_depth; depth++)
	{
		if (energy_exist_pixels_count == 0)
		{
			break;
		}

		int used_blocks_num_per_gird = (energy_exist_pixels_count + threads_num_per_block - 1) / threads_num_per_block;

		trace_ray_kernel <<<used_blocks_num_per_gird, threads_num_per_block>>> (
			mesh_num,
			bvh_nodes_device,
			triangles_device,
			sphere_num, 
			spheres_device,
			pixel_count, 
			depth,
			energy_exist_pixels_count,
			energy_exist_pixels_device,
			rays_device,
			scatterings_device,
			not_absorbed_colors_device,
			accumulated_colors_device,
			sky_cube_map_device,
			mesh_textures_device,
			seed,
			config_device
			);
		
		energy_exist_pixels_count = thread_shrink(energy_exist_pixels_device, energy_exist_pixels_count);
	}

	pixel_256_transform_gamma_corrected_kernel <<<total_blocks_num_per_gird, threads_num_per_block>>> (
		accumulated_colors_device,
		image_pixels,
		image_pixels_256,
		pixel_count,
		pass_counter,
		config_device
		);

	CUDA_CALL(cudaDeviceSynchronize());
}

extern "C" void path_tracer_kernel_memory_allocate(
	color** not_absorbed_colors_device,							//in out
	color** accumulated_colors_device,							//in out
	ray** rays_device,											//in out
	int** energy_exist_pixels_device,							//in 
	scattering** scatterings_device,							//in out
	int pixel_count												//in
)
{
	CUDA_CALL(cudaMallocManaged((void**)not_absorbed_colors_device, pixel_count * sizeof(color)));
	CUDA_CALL(cudaMallocManaged((void**)accumulated_colors_device, pixel_count * sizeof(color)));
	CUDA_CALL(cudaMallocManaged((void**)rays_device, pixel_count * sizeof(ray)));
	CUDA_CALL(cudaMallocManaged((void**)energy_exist_pixels_device, pixel_count * sizeof(int)));
	CUDA_CALL(cudaMallocManaged((void**)scatterings_device, pixel_count * sizeof(scattering)));
}

extern "C" void path_tracer_kernel_memory_free(
	color* not_absorbed_colors_device,	//in out
	color* accumulated_colors_device,	//in out
	ray* rays_device,					//in out
	int* energy_exist_pixels_device,	//in out
	scattering* scatterings_device		//in out
)
{
	CUDA_CALL(cudaFree(not_absorbed_colors_device));
	CUDA_CALL(cudaFree(accumulated_colors_device));
	CUDA_CALL(cudaFree(rays_device));
	CUDA_CALL(cudaFree(energy_exist_pixels_device));
	CUDA_CALL(cudaFree(scatterings_device));
}