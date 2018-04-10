#pragma once

#ifndef __PATH_TRACER_KERNEL__
#define __PATH_TRACER_KERNEL__

#include "Core\sphere.h"
#include "Core\image.h"
#include "Core\ray.h"
#include "Core\camera.h"
#include "Core\cube_map.h"
#include "Core\triangle.h"
#include "Core\configuration.h"
#include "Bvh\bvh_node.h"

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
);

extern "C" void path_tracer_kernel_memory_allocate(
	color** not_absorbed_colors_device,		//in out
	color** accumulated_colors_device,		//in out
	ray** rays_device,						//in out
	int** energy_exist_pixels_device,		//in out
	scattering** scatterings_device,		//in out
	int pixel_count							//in
);

extern "C" void path_tracer_kernel_memory_free(
	color* not_absorbed_colors_device,	//in out
	color* accumulated_colors_device,	//in out
	ray* rays_device,					//in out
	int* energy_exist_pixels_device,	//in out
	scattering* scatterings_device		//in out
);

#endif // !__PATH_TRACER_KERNEL__
