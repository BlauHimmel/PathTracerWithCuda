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
	configuration* config					//in
);

#endif // !__PATH_TRACER_KERNEL__
