#pragma once

#ifndef __PATH_TRACER_KERNEL__
#define __PATH_TRACER_KERNEL__

#include "sphere.hpp"
#include "image.hpp"
#include "ray.hpp"
#include "camera.hpp"
#include "cube_map.hpp"
#include "triangle_mesh.hpp"
#include "bvh.hpp"
#include "configuration.hpp"

extern "C" void path_tracer_kernel(
	int triangle_num,						//in
	bvh_node_device* bvh_nodes_device,		//in
	triangle* triangles_device,				//in
	int sphere_num,							//in
	sphere* spheres_device, 				//in
	int pixel_count, 						//in
	color* pixels,							//in out
	int depth, 								//in
	render_camera* render_camera_device,	//in
	cube_map* sky_cube_map_device,			//in
	color* not_absorbed_colors_device,		//in 
	color* accumulated_colors_device,		//in 
	ray* rays_device,						//in 
	int* energy_exist_pixels_device,		//in 
	scattering* scatterings_device,			//in 
	configuration* config_device			//in 
);

extern "C" void path_tracer_kernel_memory_allocate(
	color** not_absorbed_colors_device,		//in out
	color** accumulated_colors_device,		//in out
	ray** rays_device,						//in out
	int** energy_exist_pixels_device,		//in out
	scattering** scatterings_device,		//in out
	int pixel_count							//in
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

#endif // !__PATH_TRACER_KERNEL__
