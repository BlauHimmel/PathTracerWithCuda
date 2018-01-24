#pragma once

#ifndef __PATH_TRACER_KERNEL__
#define __PATH_TRACER_KERNEL__

#include "sphere.hpp"
#include "image.hpp"
#include "ray.hpp"
#include "camera.hpp"
#include "cube_map.hpp"
#include "triangle_mesh.hpp"

extern "C" void path_tracer_kernel(
	int triangle_num,						//in
	triangle* triangles,					//in
	int sphere_num,							//in
	sphere* spheres, 						//in
	int pixel_count, 						//in
	color* pixels,							//in out
	int depth, 								//in
	render_camera* render_camera,			//in
	cube_map* sky_cube_map					//in
);

#endif // !__PATH_TRACER_KERNEL__
