#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust\random.h>
#include <thrust\functional.h>
#include <thrust\iterator\counting_iterator.h>
#include <thrust\device_ptr.h>
#include <thrust\remove.h>

#include "curand.h"
#include "curand_kernel.h"

#include "basic_math.hpp"
#include "cuda_math.hpp"

#include "sphere.hpp"
#include "triangle.hpp"
#include "image.hpp"
#include "ray.hpp"
#include "camera.hpp"
#include "fresnel.hpp"
#include "material.hpp"
#include "cube_map.hpp"
#include "triangle_mesh.hpp"
#include "configuration.hpp"
#include "bvh_node.h"

enum class object_type
{
	none,
	sphere,
	ground,
	triangle
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
	air_scattering.absorption_coefficient = AIR_ABSORPTION_COEFFICIENT;\
	air_scattering.reduced_scattering_coefficient = AIR_REDUCED_SCATTERING_COEFFICIENT;\
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

__host__ __device__ float3 point_on_ray(
	const ray& ray,	//in
	float t			//in
)
{
	return ray.origin + ray.direction * t;
}

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

__host__ __device__ bool intersect_triangle(
	const triangle& triangle,		//in
	const ray& ray,					//in
	float& hit_t,					//out	
	float& hit_t1,					//out
	float& hit_t2					//out
)
{
	float3 edge1 = triangle.vertex1 - triangle.vertex0;
	float3 edge2 = triangle.vertex2 - triangle.vertex0;

	float3 p_vec = cross(ray.direction, edge2);
	float det = dot(edge1, p_vec);

	if (det == 0.0f)
	{
		return false;
	}

	float inverse_det = 1.0f / det;
	float3 t_vec = ray.origin - triangle.vertex0;
	float3 q_vec = cross(t_vec, edge1);

	float t1 = dot(t_vec, p_vec) * inverse_det;
	float t2 = dot(ray.direction, q_vec) * inverse_det;
	float t= dot(edge2, q_vec) * inverse_det;

	if (t1 >= 0.0f && t2 >= 0.0f && t1 + t2 <= 1.0f)
	{
		hit_t = t;
		hit_t1 = t1;
		hit_t2 = t2;
		return true;
	}

	return false;
}

__host__ __device__ bool intersect_bounding_box(
	const bounding_box& box,		//in
	const ray& ray					//in
)
{
	float3 inverse_direction = 1.0f / ray.direction;

	float t_x1 = (box.left_bottom.x - ray.origin.x) * inverse_direction.x;
	float t_x2 = (box.right_top.x - ray.origin.x) * inverse_direction.x;

	float t_y1 = (box.left_bottom.y - ray.origin.y) * inverse_direction.y;
	float t_y2 = (box.right_top.y - ray.origin.y) * inverse_direction.y;

	float t_z1 = (box.left_bottom.z - ray.origin.z) * inverse_direction.z;
	float t_z2 = (box.right_top.z - ray.origin.z) * inverse_direction.z;

	float t_min = fmaxf(fmaxf(fminf(t_x1, t_x2), fminf(t_y1, t_y2)), fminf(t_z1, t_z2));
	float t_max = fminf(fminf(fmaxf(t_x1, t_x2), fmaxf(t_y1, t_y2)), fmaxf(t_z1, t_z2));

	bool is_hit = t_max >= t_min;
	return is_hit;
}

__host__ __device__ bool intersect_triangle_mesh_bvh(
	triangle* triangles,			//in	
	bvh_node_device** bvh_nodes,	//in
	int mesh_index,					//in
	const ray& ray,					//in
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

	bvh_node_device* bvh_node = bvh_nodes[mesh_index];

	int node_num = bvh_node[0].next_node_index;
	int traversal_position = 0;
	while (traversal_position != node_num)
	{
		bvh_node_device current_node = bvh_node[traversal_position];

		//if hit the box them check if it is leaf node, otherwise check stop traversing on this branch
		if (intersect_bounding_box(current_node.box, ray))
		{
			if (current_node.is_leaf)
			{
				//intersect with each triangles in the leaf node and update the relevant minimal parameters
				for (int i = 0; i < BVH_LEAF_NODE_TRIANGLE_NUM; i++)
				{
					int triangle_index = current_node.triangle_indices[i];
					if (triangle_index != -1)
					{
						if (intersect_triangle(triangles[triangle_index], ray, current_t, current_t1, current_t2) && current_t > 0.0f && current_t < min_t)
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

__host__ __device__ float3 sample_on_hemisphere_cosine_weight(
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

__host__ __device__ float3 sample_on_hemisphere_ggx_weight(
	const float3& normal,	//in
	float roughness,		//in
	float rand1,			//in
	float rand2				//in
)
{
	float theta = atanf(roughness * sqrtf(rand1) / sqrtf(1.0f - rand1));
	float phi = rand2 * TWO_PI;
	float cos_theta = cosf(theta);
	float sin_theta = sinf(theta);

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

__host__ __device__ float compute_ggx_shadowing_masking(
	float roughness,				//in
	const float3& macro_normal,		//in
	const float3& micro_normal,		//in
	const float3& ray_direction		//in
)
{
	float3 v = -1.0f * ray_direction;
	float v_dot_n = dot(v, macro_normal);
	float v_dot_m = dot(v, micro_normal);
	float roughness_square = roughness * roughness;
	
	float cos_v_square = v_dot_n * v_dot_n;
	float tan_v_square = (1 - cos_v_square) / cos_v_square;
	
	float positive_value = (v_dot_m / v_dot_n) > 0.0f ? 1.0f : 0.0f;

	if (positive_value == 0.0f)
	{
		return 0.0f;
	}
	else
	{
		return 2.0f / (1.0f + sqrtf(1.0f + roughness_square * tan_v_square));
	}
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

__host__ __device__ fresnel get_fresnel_dielectrics(
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

	float cos_theta_in = dot(normal, in_direction * -1.0f);
	float cos_theta_out = dot(-1.0f * normal, refraction_direction);

	//total internal reflection 
	if (in_refraction_index > out_refraction_index && acosf(cos_theta_in) >= asinf(out_refraction_index / in_refraction_index))
	{
		fresnel.reflection_index = 1.0f;
		fresnel.refractive_index = 0.0f;
		return fresnel;
	}

	if (length(refraction_direction) <= 0.000005f || cos_theta_out < 0)
	{
		fresnel.reflection_index = 1.0f;
		fresnel.refractive_index = 0.0f;
		return fresnel;
	}

	float reflection_coef_s_polarized = pow((in_refraction_index * cos_theta_in - out_refraction_index * cos_theta_out) / (in_refraction_index * cos_theta_in + out_refraction_index * cos_theta_out), 2.0f);
	float reflection_coef_p_polarized = pow((in_refraction_index * cos_theta_out - out_refraction_index * cos_theta_in) / (in_refraction_index * cos_theta_out + out_refraction_index * cos_theta_in), 2.0f);

	float reflection_coef_unpolarized = (reflection_coef_s_polarized + reflection_coef_p_polarized) / 2.0f;

	fresnel.reflection_index = reflection_coef_unpolarized;
	fresnel.refractive_index = 1.0f - fresnel.reflection_index;
	return fresnel;
}

__host__ __device__ fresnel get_fresnel_conductors(
	const float3& normal,					//in
	const float3& in_direction,				//in
	float refraction_index,					//in
	float extinction_coefficient,			//in
	const float3& reflection_direction		//in
)
{
	//using real fresnel equation
	fresnel fresnel;

	float cos_theta_in = dot(normal, in_direction * -1.0f);
	float refraction_index_square = refraction_index * refraction_index;
	float extinction_coefficient_square = extinction_coefficient * extinction_coefficient;
	float refraction_extinction_square_add = refraction_index_square + extinction_coefficient_square;
	float cos_theta_in_square = cos_theta_in * cos_theta_in;
	float two_refraction_cos_theta_in = 2 * refraction_index * cos_theta_in;

	float reflection_coef_s_polarized = (refraction_extinction_square_add * cos_theta_in_square - two_refraction_cos_theta_in + 1.0f) / (refraction_extinction_square_add * cos_theta_in_square + two_refraction_cos_theta_in + 1.0f);
	float reflection_coef_p_polarized = (refraction_extinction_square_add - two_refraction_cos_theta_in + cos_theta_in_square) / (refraction_extinction_square_add + two_refraction_cos_theta_in + cos_theta_in_square);

	float reflection_coef_unpolarized = (reflection_coef_s_polarized + reflection_coef_p_polarized) / 2.0f;

	fresnel.reflection_index = reflection_coef_unpolarized;
	fresnel.refractive_index = 1.0f - fresnel.reflection_index;
	return fresnel;
}

__host__ __device__ float3 get_background_color(
	const float3& direction,		//in
	cube_map* sky_cube_map,			//in
	configuration* config			//in
)
{
	if (config->use_sky_box)
	{
		if (config->use_bilinear)
		{
			float u, v;
			int index;
			convert_xyz_to_cube_uv(direction.x, direction.y, direction.z, index, u, v);
			float x_image_real = u * (float)sky_cube_map->length;
			float y_image_real = (1.0f - v) * (float)sky_cube_map->length;

			int floor_x_image = (int)clamp(floorf(x_image_real), 0.0f, (float)(sky_cube_map->length - 1));
			int ceil_x_image = (int)clamp(ceilf(x_image_real), 0.0f, (float)(sky_cube_map->length - 1));
			int floor_y_image = (int)clamp(floorf(y_image_real), 0.0f, (float)(sky_cube_map->length - 1));
			int ceil_y_image = (int)clamp(ceilf(y_image_real), 0.0f, (float)(sky_cube_map->length - 1));

			//0:left bottm	1:right bottom	2:left top	3:right top
			int x_images[4] = { floor_x_image, ceil_x_image, floor_x_image, ceil_x_image };
			int y_images[4] = { floor_y_image, floor_y_image, ceil_y_image, ceil_y_image };

			float left_right_t = x_image_real - floorf(x_image_real);
			float bottom_top_t = y_image_real - floorf(y_image_real);

			uchar* pixels;
			if (index == 0) pixels = sky_cube_map->m_x_positive_map;
			else if (index == 1) pixels = sky_cube_map->m_x_negative_map;
			else if (index == 2) pixels = sky_cube_map->m_y_positive_map;
			else if (index == 3) pixels = sky_cube_map->m_y_negative_map;
			else if (index == 4) pixels = sky_cube_map->m_z_positive_map;
			else if (index == 5) pixels = sky_cube_map->m_z_negative_map;

			float3 sample_colors[4] = {
				make_float3(
					pixels[(y_images[0] * sky_cube_map->length + x_images[0]) * 4 + 0] / 255.0f,
					pixels[(y_images[0] * sky_cube_map->length + x_images[0]) * 4 + 1] / 255.0f,
					pixels[(y_images[0] * sky_cube_map->length + x_images[0]) * 4 + 2] / 255.0f
				),
				make_float3(
					pixels[(y_images[1] * sky_cube_map->length + x_images[1]) * 4 + 0] / 255.0f,
					pixels[(y_images[1] * sky_cube_map->length + x_images[1]) * 4 + 1] / 255.0f,
					pixels[(y_images[1] * sky_cube_map->length + x_images[1]) * 4 + 2] / 255.0f
				),
				make_float3(
					pixels[(y_images[2] * sky_cube_map->length + x_images[2]) * 4 + 0] / 255.0f,
					pixels[(y_images[2] * sky_cube_map->length + x_images[2]) * 4 + 1] / 255.0f,
					pixels[(y_images[2] * sky_cube_map->length + x_images[2]) * 4 + 2] / 255.0f
				),
				make_float3(
					pixels[(y_images[3] * sky_cube_map->length + x_images[3]) * 4 + 0] / 255.0f,
					pixels[(y_images[3] * sky_cube_map->length + x_images[3]) * 4 + 1] / 255.0f,
					pixels[(y_images[3] * sky_cube_map->length + x_images[3]) * 4 + 2] / 255.0f
				),
			};

			return lerp(
				lerp(sample_colors[0], sample_colors[1], left_right_t),
				lerp(sample_colors[2], sample_colors[3], left_right_t),
				bottom_top_t
			);
		}
		else
		{
			float u, v;
			int index;
			convert_xyz_to_cube_uv(direction.x, direction.y, direction.z, index, u, v);
			int x_image = (int)clamp((u * sky_cube_map->length), 0.0f, (float)(sky_cube_map->length - 1));
			int y_image = (int)clamp(((1.0f - v) * sky_cube_map->length), 0.0f, (float)(sky_cube_map->length - 1));

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
		}
	}

	if (config->use_ground)
	{
		float t = (dot(direction, make_float3(-0.41f, 0.41f, -0.82f)) + 1.0f) / 2.0f;
		float3 a = make_float3(0.15f, 0.3f, 0.5f);
		float3 b = make_float3(1.0f, 1.0f, 1.0f);
		return ((1.0f - t) * a + t * b) * 1.0f;
	}

	return make_float3(0.0f, 0.0f, 0.0f);
}

__global__ void init_data_kernel(
	int pixel_count,						//in
	int* energy_exist_pixels,				//in out
	color* not_absorbed_colors,				//in out
	color* accumulated_colors,				//in out
	scattering* scatterings,				//in out
	configuration* config					//in
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

	thrust::default_random_engine random_engine(rand_hash(seed) * rand_hash(pixel_index) * rand_hash(depth));
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

	//intersect with primitives in scene
	if (config->use_ground && intersect_ground(-0.8f, tracing_ray, hit_point, hit_normal, hit_t) && hit_t < min_t && hit_t > 0.0f)
	{
		min_t = hit_t;
		min_point = hit_point;
		min_normal = hit_normal;
		min_type = object_type::ground;
	}

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
	
	for (int mesh_index = 0; mesh_index < mesh_num; mesh_index++)
	{
		if (intersect_triangle_mesh_bvh(triangles, bvh_nodes, mesh_index, tracing_ray, hit_t, hit_t1, hit_t2, hit_triangle_index) && hit_t < min_t && hit_t > 0.0f)
		{
			min_t = hit_t;
			min_t1 = hit_t1;
			min_t2 = hit_t2;
			min_point = point_on_ray(tracing_ray, hit_t);
			min_triangle_index = hit_triangle_index;
			min_type = object_type::triangle;
		}
	}

	//absorption and scattering of medium
	scattering current_scattering = scatterings[pixel_index];

	if (current_scattering.reduced_scattering_coefficient.x > 0.0f ||
		length(current_scattering.absorption_coefficient) > config->sss_threshold)
	{
		float rand = uniform_distribution(random_engine);
		float scaterring_distance = -log(rand) / current_scattering.reduced_scattering_coefficient.x;
		//TODO:ANISOTROPIC SCATTERING
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
		if (min_type == object_type::ground)
		{
			GET_DEFAULT_MATERIAL_DEVICE(min_mat);
			min_mat.diffuse_color = make_float3(0.455f, 0.43f, 0.39f);
		}
		else if (min_type == object_type::sphere)
		{
			min_mat = spheres[min_sphere_index].mat;
		}
		else if (min_type == object_type::triangle)
		{
			min_mat = *(triangles[min_triangle_index].mat);
			triangle hit_triangle = triangles[min_triangle_index];
			min_normal = hit_triangle.normal0 * (1.0f - min_t1 - min_t2) + hit_triangle.normal1 * min_t1 + hit_triangle.normal2 * min_t2;
		}

		float3 in_direction = tracing_ray.direction;

		medium in_medium;
		GET_DEFAULT_MEDIUM_DEVICE(in_medium);
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

		float3 reflection_direction = get_reflection_direction(min_normal, in_direction);
		float3 refraction_direction = get_refraction_direction(min_normal, in_direction, in_medium.refraction_index, out_medium.refraction_index);
		float3 bias_vector = config->vector_bias_length * min_normal;

		fresnel fresnel;
		if (min_mat.medium.extinction_coefficient == 0 || min_mat.is_transparent)
		{
			fresnel = get_fresnel_dielectrics(min_normal, in_direction, in_medium.refraction_index, out_medium.refraction_index, reflection_direction, refraction_direction);
		}
		else
		{
			fresnel = get_fresnel_conductors(min_normal, in_direction, out_medium.refraction_index, out_medium.extinction_coefficient, reflection_direction);
		}
		
		float rand = uniform_distribution(random_engine);

		if (rand < fresnel.reflection_index)
		{
			//reflection
			not_absorbed_colors[pixel_index] *= min_mat.specular_color;

			float rand1 = uniform_distribution(random_engine);
			float rand2 = uniform_distribution(random_engine);

			float remap_roughness = powf(min_mat.roughness, 2.0f) / 2.0f;
			float3 micro_normal = sample_on_hemisphere_ggx_weight(min_normal, remap_roughness, rand1, rand2);
			float self_shadowing = compute_ggx_shadowing_masking(remap_roughness, min_normal, micro_normal, tracing_ray.direction);

			float rand = uniform_distribution(random_engine);
			if (rand < self_shadowing)
			{
				ray next_ray;
				next_ray.origin = min_point + bias_vector;
				next_ray.direction = get_reflection_direction(micro_normal, in_direction);
				rays[pixel_index] = next_ray;
			}
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
		float3 background_color = get_background_color(tracing_ray.direction, sky_cube_map, config);
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
			image_pixels[pixel_index] += accumulated_colors[pixel_index];
		}
		else
		{
			image_pixels[pixel_index] = accumulated_colors[pixel_index];
		}

		color pixel = image_pixels[pixel_index] / (float)pass_counter;

		float inverse_gamma = 0.45454545f;
		color corrected_pixel;
		corrected_pixel.x = std::expf(inverse_gamma * std::logf(pixel.x));
		corrected_pixel.y = std::expf(inverse_gamma * std::logf(pixel.y));
		corrected_pixel.z = std::expf(inverse_gamma * std::logf(pixel.z));

		float x = clamp(corrected_pixel.x * 255.0f, 0.0f, 255.0f);
		float y = clamp(corrected_pixel.y * 255.0f, 0.0f, 255.0f);
		float z = clamp(corrected_pixel.z * 255.0f, 0.0f, 255.0f);

		color256 color_256;
		color_256.x = (uchar)x;
		color_256.y = (uchar)y;
		color_256.z = (uchar)z;

		image_pixels_256[pixel_index] = color_256;
	}
}

//===============================================================================================================
//TODO:BUILD SPATIAL STRUCTURE ON GPU

extern "C" void path_tracer_kernel(
	int mesh_num,							//in
	bvh_node_device** bvh_nodes_device,		//in
	triangle* triangles_device,				//in
	int sphere_num,							//in
	sphere* spheres_device, 				//in
	int pixel_count, 						//in
	color* image_pixels,					//in out
	color256* image_pixels_256,				//in out
	int pass_counter, 						//in
	render_camera* render_camera_device,	//in
	cube_map* sky_cube_map_device,			//in
	color* not_absorbed_colors_device,		//in 
	color* accumulated_colors_device,		//in 
	ray* rays_device,						//in 
	int* energy_exist_pixels_device,		//in 
	scattering* scatterings_device,			//in 
	configuration* config_device			//in 
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
			seed,
			config_device
			);
		
		thrust::device_ptr<int> energy_exist_pixels_start_on_device = thrust::device_pointer_cast(energy_exist_pixels_device);
		thrust::device_ptr<int> energy_exist_pixels_end_on_device = thrust::remove_if(
			energy_exist_pixels_start_on_device,
			energy_exist_pixels_start_on_device + energy_exist_pixels_count,
			is_negative_predicate()
		);
		
		energy_exist_pixels_count = (int)(thrust::raw_pointer_cast(energy_exist_pixels_end_on_device) - energy_exist_pixels_device);
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
