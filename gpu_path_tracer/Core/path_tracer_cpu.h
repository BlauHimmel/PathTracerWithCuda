#pragma once

#ifndef __PATH_TRACER_CPU__
#define __PATH_TRACER_CPU__

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

enum class object_type_cpu
{
	none,
	sphere,
	triangle
};

int hash_cpu(int a);

float3 reflection_cpu(
	const float3& normal,
	const float3& in_direction
);

float3 refraction_cpu(
	const float3& normal,
	const float3& in_direction,
	float in_refraction_index,
	float out_refraction_index
);

bool intersect_triangle_mesh_bvh_cpu(
	triangle* triangles,			//in	
	bvh_node_device** bvh_nodes,	//in
	int mesh_index,					//in
	const ray& ray,					//in
	configuration* config,			//in
	float& hit_t,					//out	
	float& hit_t1,					//out	
	float& hit_t2,					//out	
	int& hit_triangle_index			//out
);

float3 sample_on_hemisphere_cosine_weight_cpu(
	const float3& normal,	//in
	float rand1,			//in
	float rand2				//in
);

float3 sample_on_hemisphere_ggx_weight_cpu(
	const float3& normal,	//in
	float roughness,		//in
	float rand1,			//in
	float rand2				//in
);

float3 sample_on_sphere_cpu(
	float rand1,			//in
	float rand2				//in
);

float compute_ggx_shadowing_masking_cpu(
	float roughness,				//in
	const float3& macro_normal,		//in
	const float3& micro_normal,		//in
	const float3& ray_direction		//in
);

void init_data_cpu(
	int pixel_count,					//in
	int* energy_exist_pixels,			//in out
	color* not_absorbed_colors,			//in out
	color* accumulated_colors,			//in out
	scattering* scatterings,			//in out
	configuration* config				//in
);

void generate_ray_cpu(
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
);

void trace_ray_cpu(
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
);

void pixel_256_transform_gamma_corrected_cpu(
	color* accumulated_colors,			//in 
	color* image_pixels,				//in
	color256* image_pixels_256,			//in out
	int pixel_count,					//in
	int pass_counter,					//in
	configuration* config				//in
);

//===============================================================================================================
void path_tracer_cpu(
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
);


#endif // !__PATH_TRACER_CPU__
