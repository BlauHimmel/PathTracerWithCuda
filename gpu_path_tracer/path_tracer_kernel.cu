#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust\random.h>
#include <thrust\functional.h>
#include <thrust\iterator\counting_iterator.h>
#include <thrust\device_ptr.h>
#include <thrust\remove.h>

#include "curand.h"
#include "curand_kernel.h"

#include "basic_math.h"
#include "cuda_math.hpp"

#include "sphere.hpp"
#include "image.hpp"
#include "ray.hpp"
#include "camera.hpp"
#include "fresnel.h"
#include "material.hpp"
#include "cube_map.hpp"

#define BLOCK_SIZE 1024
#define MAX_TRACER_DEPTH 20
#define VECTOR_BIAS_LENGTH 0.0001f
#define ENERGY_EXIST_THRESHOLD 0.000001f
#define SSS_THRESHOLD 0.000001f

//TODO:LET MAX_TRACER_DEPTH BE A PARAMETER, WHICH CAN BE CONTROL IN THE APPLICATION

enum class object_type
{
	none,
	sphere,
	ground
};

struct is_negative_predicate
{
	__host__ __device__ bool operator()(int value)
	{
		return value < 0;
	}
};

//using a function instead of macro will cause a weird memory access violation bug, BOOOOOM!
#define GET_DEFAULT_SCATERRING_DEVICE(air_scattering)\
{\
	air_scattering.absorption_coefficient = make_float3(0.0f, 0.0f, 0.0f);\
	air_scattering.reduced_scattering_coefficient = make_float3(0.0f, 0.0f, 0.0f);\
}\

#define GET_DEFAULT_MEDIUM_DEVICE(air_medium)\
{\
	GET_DEFAULT_SCATERRING_DEVICE(air_medium.scattering);\
	air_medium.refraction_index = AIR_REFRACTION_INDEX;\
}\

#define GET_DEFAULT_MATERIAL_DEVICE(material)\
{\
	material.diffuse_color = make_float3(0.0f, 0.0f, 0.0f);\
	material.emission_color = make_float3(0.0f, 0.0f, 0.0f);\
	material.specular_color = make_float3(0.0f, 0.0f, 0.0f);\
	material.is_transparent = false;\
	GET_DEFAULT_MEDIUM_DEVICE(material.medium);\
}\

__host__ __device__  bool intersect_ground(
	float altitude,			//in 
	const ray& ray,			//in
	float3& hit_point,		//out
	float3& hit_normal,		//out
	float& hit_t			//out
)
{
	if (ray.direction.y != 0)
	{
		hit_t = (altitude - ray.origin.y) / ray.direction.y;
		hit_point = ray.origin + ray.direction * hit_t;
		hit_normal = make_float3(0.0f, 1.0f, 0.0f);
		return true;
	}
	else
	{
		return false;
	}
}

__host__ __device__  bool intersect_sphere(
	const sphere& sphere,	//in	
	const ray& ray,			//in
	float3& hit_point,		//out
	float3& hit_normal,		//out
	float& hit_t			//out
)
{
	float3 op = sphere.center - ray.origin;
	float b = dot(op, ray.direction);
	float delta = b * b - dot(op, op) + sphere.radius * sphere.radius;

	if (delta < 0)
	{
		return false;
	}

	float delta_root = sqrt(delta);
	float t1 = b - delta_root;
	float t2 = b + delta_root;

	if (t1 < 0 && t2 < 0)
	{
		return false;
	}

	if (t1 > 0 && t2 > 0)
	{
		hit_t = min(t1, t2);
	}
	else
	{
		hit_t = max(t1, t2);
	}

	hit_point = ray.origin + ray.direction * hit_t;
	hit_normal = normalize(hit_point - sphere.center);
	return true;
}

__host__ __device__ float3 sample_on_hemisphere(
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

	return cos_theta * normal + cos(phi) * sin_theta * vec_i + sin(phi) * sin_theta * vec_j;
}

__host__ __device__ float3 sample_on_sphere(
	float rand1,			//in
	float rand2				//in
)
{
	//spherical coordinate
	float cos_theta = rand1 * 2.0f - 1.0f;
	float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
	float phi = rand2 * TWO_PI;

	return make_float3(cos_theta, cos(phi) * sin_theta, sin(phi) * sin_theta);
}

__host__ __device__ color compute_absorption_through_medium(
	const color& absorption_coefficient,	//in 
	float scaterring_distance				//in
)
{
	//E = E_0 * e^(-汐x)
	return make_float3(
		pow(E, -1.0f * absorption_coefficient.x * scaterring_distance),
		pow(E, -1.0f * absorption_coefficient.y * scaterring_distance),
		pow(E, -1.0f * absorption_coefficient.z * scaterring_distance)
	);
}

__host__ __device__ int rand_hash(
	int a			//in
)
{
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

__host__ __device__ float3 get_reflection_direction(
	const float3& normal,		//in
	const float3& in_direction	//in
)
{
	return in_direction - 2.0f * dot(normal, in_direction) * normal;
}

__host__ __device__ float3 get_refraction_direction(
	const float3& normal,		//in
	const float3& in_direction,	//in
	float in_refraction_index,	//in
	float out_refraction_index	//in
)
{
	//[ 灰r (N ﹞ I) 每 ﹟(1 每 灰r^2 (1 每(N ﹞ I)^2)) ] N 每 灰r I
	//灰r = 灰i / 灰o
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

__host__ __device__ fresnel get_fresnel(
	const float3& normal,					//in
	const float3& in_direction,				//in
	float in_refraction_index,				//in
	float out_refraction_index,				//in
	const float3& reflection_direction,		//in
	const float3& refraction_direction		//in
)
{
	//using real fresnel equation
	fresnel fresnel;

	if (length(refraction_direction) <= 0.12345f || dot(normal, refraction_direction) > 0)
	{
		fresnel.reflection_index = 1.0f;
		fresnel.refractive_index = 0.0f;
		return fresnel;
	}

	float cos_theta_in = dot(normal, in_direction * -1.0f);
	float cos_theta_out = dot(-1.0f * normal, refraction_direction);

	float reflection_coef_s_polarized = pow((in_refraction_index * cos_theta_in - out_refraction_index * cos_theta_out) / (in_refraction_index * cos_theta_in + out_refraction_index * cos_theta_out), 2.0f);
	float reflection_coef_p_polarized = pow((in_refraction_index * cos_theta_out - out_refraction_index * cos_theta_in) / (in_refraction_index * cos_theta_out + out_refraction_index * cos_theta_in), 2.0f);

	float reflection_coef_unpolarized = (reflection_coef_s_polarized + reflection_coef_p_polarized) / 2.0f;

	fresnel.reflection_index = reflection_coef_unpolarized;
	fresnel.refractive_index = 1 - fresnel.reflection_index;
	return fresnel;
}

__host__ __device__ float3 get_background_color(
	const float3& direction,		//in
	cube_map* sky_cube_map			//in
)
{
	//TODO:COULD SAMPLE CUBE MAP HERE
	float u, v;
	int index;
	convert_xyz_to_cube_uv(direction.x, direction.y, direction.z, index, u, v);
	int x_image = u * sky_cube_map->length;
	int y_image = (1.0f - v) * sky_cube_map->length;

	uchar* pixels;
	if (index == 0) pixels = sky_cube_map->m_x_positive_map;
	else if (index == 1) pixels = sky_cube_map->m_x_negative_map;
	else if (index == 2) pixels = sky_cube_map->m_y_positive_map;
	else if (index == 3) pixels = sky_cube_map->m_y_negative_map;
	else if (index == 4) pixels = sky_cube_map->m_z_positive_map;
	else if (index == 5) pixels = sky_cube_map->m_z_negative_map;

	return make_float3(
		pixels[(y_image * sky_cube_map->length + x_image) * 4 + 0] / 255.0f,
		pixels[(y_image * sky_cube_map->length + x_image) * 4 + 1] / 255.0f,
		pixels[(y_image * sky_cube_map->length + x_image) * 4 + 2] / 255.0f
	);

	//float t = (dot(direction, make_float3(-0.41f, 0.41f, -0.82f)) + 1.0f) / 2.0f;
	//float3 a = make_float3(0.15f, 0.3f, 0.5f);
	//float3 b = make_float3(1.0f, 1.0f, 1.0f);
	//return ((1.0f - t) * a + t * b) * 1.0f;
}

__host__ __device__ float3 point_on_ray(
	const ray& ray,	//in
	float t			//in
)
{
	return ray.origin + ray.direction * t;
}

__global__ void init_data_kernel(
	int pixel_count,						//in
	int* energy_exist_pixels,				//in out
	color* not_absorbed_colors,				//in out
	color* accumulated_colors,				//in out
	scattering* scatterings					//in out
)
{
	int block_x = blockIdx.x;
	int thread_x = threadIdx.x;

	int pixel_index = BLOCK_SIZE * block_x + thread_x;
	bool is_index_valid = pixel_index < pixel_count;

	if (is_index_valid)
	{
		energy_exist_pixels[pixel_index] = pixel_index;
		not_absorbed_colors[pixel_index] = make_float3(1.0f, 1.0f, 1.0f);
		accumulated_colors[pixel_index] = make_float3(0.0f, 0.0f, 0.0f);
		GET_DEFAULT_SCATERRING_DEVICE(scatterings[pixel_index]);
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
	int seed							//in
)
{
	int block_x = blockIdx.x;
	int thread_x = threadIdx.x;

	int pixel_index = BLOCK_SIZE * block_x + thread_x;
	bool is_index_valid = pixel_index < pixel_count;

	if (is_index_valid)
	{
		int image_y = (int)(pixel_index / resolution.y);
		int image_x = pixel_index - (image_y * resolution.y);

		thrust::default_random_engine random_engine(rand_hash(seed) * rand_hash(seed) * rand_hash(pixel_index));
		thrust::uniform_real_distribution<float> uniform_distribution(-0.5f, 0.5f);

		//for anti-aliasing
		float jitter_x = uniform_distribution(random_engine);
		float jitter_y = uniform_distribution(random_engine);

		float distance = length(view);

		//vector base
		float3 horizontal = normalize(cross(view, up));
		float3 vertical = normalize(cross(horizontal, view));

		//edge of canvas
		float3 x_axis = horizontal * (distance * tan(fov.x * 0.5f * (PI / 180.0f)));
		float3 y_axis = vertical * (distance * tan(-fov.y * 0.5f * (PI / 180.0f)));
		
		float normalized_image_x = ((image_x + jitter_x) / (resolution.x - 1.0f)) * 2.0f - 1.0f;
		float normalized_image_y = ((image_y + jitter_y) / (resolution.y - 1.0f)) * 2.0f - 1.0f;

		//for all the ray (cast from one point) refracted by the convex will be cast on one point finally(focal point)
		float3 point_on_canvas_plane = eye + view + normalized_image_x * x_axis + normalized_image_y * y_axis;
		float3 point_on_image_plane = eye + (point_on_canvas_plane - eye) * focal_distance;

		float3 point_on_aperture;
		if (aperture_radius > 0.00001f)
		{
			//sample on convex, note that convex is not a plane
			float rand1 = uniform_distribution(random_engine) + 0.5f;
			float rand2 = uniform_distribution(random_engine) + 0.5f;

			float angle = rand1 * TWO_PI;
			float distance = aperture_radius * sqrt(rand2);

			float aperture_x = cos(angle) * distance;
			float aperture_y = sin(angle) * distance;

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
	int seed								//in
)
{
	int block_x = blockIdx.x;
	int thread_x = threadIdx.x;

	int energy_exist_pixel_index = BLOCK_SIZE * block_x + thread_x;
	bool is_index_valid = energy_exist_pixel_index < energy_exist_pixels_count;

	if (!is_index_valid)
	{
		return;
	}

	int pixel_index = energy_exist_pixels[energy_exist_pixel_index];
	ray tracing_ray = rays[pixel_index];

	thrust::default_random_engine random_engine(rand_hash(seed) * rand_hash(pixel_index) * rand_hash(depth));
	thrust::uniform_real_distribution<float> uniform_distribution(0.0f, 1.0f);

	float hit_t;
	float3 hit_point, hit_normal;

	float min_t = INFINITY;
	float3 min_point = make_float3(0.0f, 0.0f, 0.0f);
	float3 min_normal = make_float3(0.0f, 0.0f, 0.0f);
	object_type min_type = object_type::none;
	int min_sphere_index = -1;

	//intersect will primitives in scene
	//if (intersect_ground(-0.8f, tracing_ray, hit_point, hit_normal, hit_t) && hit_t < min_t && hit_t > 0.0f)
	//{
	//	//TODO:HARDCODE HERE
	//	min_t = hit_t;
	//	min_point = hit_point;
	//	min_normal = hit_normal;
	//	min_type = object_type::ground;
	//}

	for (int i = 0; i < sphere_num; i++)
	{
		if (intersect_sphere(spheres[i], tracing_ray, hit_point, hit_normal, hit_t) && hit_t < min_t && hit_t > 0.0f)
		{
			min_t = hit_t;
			min_point = hit_point;
			min_normal = hit_normal;
			min_sphere_index = i;
			min_type = object_type::sphere;
		}
	}

	//absorption and scattering of medium
	scattering current_scattering = scatterings[pixel_index];

	if (current_scattering.reduced_scattering_coefficient.x > 0.0f ||
		length(current_scattering.absorption_coefficient) > SSS_THRESHOLD)
	{
		float rand = uniform_distribution(random_engine);
		float scaterring_distance = -log(rand) / current_scattering.reduced_scattering_coefficient.x;
		//TODO:ISOTROPIC SCATTERING
		if (scaterring_distance < min_t)
		{
			//absorption and scattering
			float rand1 = uniform_distribution(random_engine);
			float rand2 = uniform_distribution(random_engine);

			ray next_ray;
			next_ray.origin = point_on_ray(tracing_ray, scaterring_distance);
			next_ray.direction = sample_on_sphere(rand1, rand2);
			rays[pixel_index] = next_ray;

			not_absorbed_colors[pixel_index] *= compute_absorption_through_medium(current_scattering.absorption_coefficient, scaterring_distance);

			//kill the low energy ray
			if (length(not_absorbed_colors[pixel_index]) <= ENERGY_EXIST_THRESHOLD)
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
		if (min_type == object_type::ground)
		{
			//TODO:HARDCODE HERE
			GET_DEFAULT_MATERIAL_DEVICE(min_mat);
			min_mat.diffuse_color = make_float3(0.455f, 0.43f, 0.39f);
		}
		else if (min_type == object_type::sphere)
		{
			min_mat = spheres[min_sphere_index].mat;
		}

		float3 in_direction = tracing_ray.direction;

		medium in_medium;
		GET_DEFAULT_MEDIUM_DEVICE(in_medium);
		medium out_medium = min_mat.medium;

		bool is_hit_on_back = dot(in_direction, min_normal) > 0;

		if (is_hit_on_back)
		{
			min_normal *= -1.0f;

			medium temp = in_medium;
			in_medium = out_medium;
			out_medium = temp;
		}

		float3 reflection_direction = get_reflection_direction(min_normal, in_direction);
		float3 refraction_direction = get_refraction_direction(min_normal, in_direction, in_medium.refraction_index, out_medium.refraction_index);//may be nan
		float3 bias_vector = VECTOR_BIAS_LENGTH * min_normal;

		fresnel fresnel = get_fresnel(min_normal, in_direction, in_medium.refraction_index, out_medium.refraction_index, reflection_direction, refraction_direction);

		float rand = uniform_distribution(random_engine);

		if (min_mat.medium.refraction_index > 1.0f && rand < fresnel.reflection_index)
		{
			//reflection
			not_absorbed_colors[pixel_index] *= min_mat.specular_color;

			ray next_ray;
			next_ray.origin = min_point + bias_vector;
			next_ray.direction = reflection_direction;
			rays[pixel_index] = next_ray;
		}
		else if (min_mat.is_transparent)
		{
			//refraction
			//transmitted into a new medium
			scatterings[pixel_index] = out_medium.scattering;

			ray next_ray;
			next_ray.origin = min_point - bias_vector;
			next_ray.direction = refraction_direction;
			rays[pixel_index] = next_ray;
		}
		else
		{
			//diffuse
			accumulated_colors[pixel_index] += not_absorbed_colors[pixel_index] * min_mat.emission_color;
			not_absorbed_colors[pixel_index] *= min_mat.diffuse_color;

			float rand1 = uniform_distribution(random_engine);
			float rand2 = uniform_distribution(random_engine);

			ray next_ray;
			next_ray.origin = min_point + bias_vector;
			next_ray.direction = sample_on_hemisphere(min_normal, rand1, rand2);
			rays[pixel_index] = next_ray;
		}

		//kill the low energy ray
		if (length(not_absorbed_colors[pixel_index]) <= ENERGY_EXIST_THRESHOLD)
		{
			energy_exist_pixels[energy_exist_pixel_index] = -1;
		}
	}
	else
	{
		float3 background_color = get_background_color(tracing_ray.direction, sky_cube_map);
		accumulated_colors[pixel_index] += not_absorbed_colors[pixel_index] * background_color;
		//kill the low ray because it has left the scene
		energy_exist_pixels[energy_exist_pixel_index] = -1;
	}
}

extern "C" void path_tracer_kernel(
	int sphere_num,						//in
	sphere* spheres, 					//in
	int pixel_count, 					//in
	color* pixels,						//in out
	int pass_counter, 					//in out
	render_camera* render_cam,			//in
	cube_map* sky_cube_map				//in
)
{
	int threads_num_per_block = BLOCK_SIZE;
	int total_blocks_num_per_gird = (pixel_count + threads_num_per_block - 1) / threads_num_per_block;

	color* not_absorbed_colors = nullptr;
	color* accumulated_colors = nullptr;
	ray* rays = nullptr;
	int* energy_exist_pixels = nullptr;
	int energy_exist_pixels_count = pixel_count;
	int seed = pass_counter;
	scattering* scatterings = nullptr;

	CUDA_CALL(cudaMalloc((void**)&not_absorbed_colors, pixel_count * sizeof(color)));
	CUDA_CALL(cudaMalloc((void**)&accumulated_colors, pixel_count * sizeof(color)));
	CUDA_CALL(cudaMalloc((void**)&rays, pixel_count * sizeof(ray)));
	CUDA_CALL(cudaMalloc((void**)&energy_exist_pixels, pixel_count * sizeof(int)));
	CUDA_CALL(cudaMalloc((void**)&scatterings, pixel_count * sizeof(scattering)));

	init_data_kernel <<<total_blocks_num_per_gird, threads_num_per_block >>> (
		pixel_count, 
		energy_exist_pixels, 
		not_absorbed_colors, 
		accumulated_colors,
		scatterings
		);

	generate_ray_kernel <<<total_blocks_num_per_gird, threads_num_per_block>>> (
		render_cam->eye,
		render_cam->view,
		render_cam->up,
		render_cam->resolution,
		render_cam->fov,
		render_cam->aperture_radius,
		render_cam->focal_distance,
		pixel_count, 
		rays,
		seed
		);

	for (int depth = 0; depth < MAX_TRACER_DEPTH; depth++)
	{
		if (energy_exist_pixels_count == 0)
		{
			break;
		}

		int used_blocks_num_per_gird = (energy_exist_pixels_count + threads_num_per_block - 1) / threads_num_per_block;

		trace_ray_kernel <<<used_blocks_num_per_gird, threads_num_per_block>>> (
			sphere_num, 
			spheres, 
			pixel_count, 
			depth,
			energy_exist_pixels_count,
			energy_exist_pixels, 
			rays, 
			scatterings,
			not_absorbed_colors, 
			accumulated_colors, 
			sky_cube_map,
			seed
			);
		
		thrust::device_ptr<int> energy_exist_pixels_start_on_device = thrust::device_pointer_cast(energy_exist_pixels);
		thrust::device_ptr<int> energy_exist_pixels_end_on_device = thrust::remove_if(
			energy_exist_pixels_start_on_device,
			energy_exist_pixels_start_on_device + energy_exist_pixels_count,
			is_negative_predicate()
		);
		
		energy_exist_pixels_count = (int)(thrust::raw_pointer_cast(energy_exist_pixels_end_on_device) - energy_exist_pixels);
	}

	CUDA_CALL(cudaMemcpy(pixels, accumulated_colors, pixel_count * sizeof(color), cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(not_absorbed_colors));
	CUDA_CALL(cudaFree(accumulated_colors));
	CUDA_CALL(cudaFree(rays));
	CUDA_CALL(cudaFree(energy_exist_pixels));
	CUDA_CALL(cudaFree(scatterings));
}