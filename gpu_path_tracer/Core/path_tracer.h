#pragma once

#ifndef __PATH_TRACER__
#define __PATH_TRACER__

#include <cuda_runtime.h>
#include <thrust\device_vector.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "Core\sphere.h"
#include "Core\image.h"
#include "Core\ray.h"
#include "Core\camera.h"
#include "Core\material.h"
#include "Math\cuda_math.hpp"
#include "Others\utilities.hpp"
#include "Math\basic_math.hpp"
#include "Core\cube_map.h"
#include "Core\material.h"
#include "Core\triangle_mesh.h"
#include "Core\path_tracer_kernel.h"
#include "Bvh\bvh.h"
#include "Core\scene_parser.h"
#include "Core\config_parser.h"
#include "Core\path_tracer_cpu.h"

#include "lib\glm\glm.hpp"
#include "lib\imgui\imgui.h"
#include "lib\imgui\imgui_impl_glfw_gl3.h"

class path_tracer
{
private:
	scene_parser m_scene;

	config_parser* m_config = nullptr;
	render_camera* m_render_camera = nullptr;
	image* m_image = nullptr;

	bool m_is_initiated = false;
	bool m_is_scene_choose = false;

	color* m_not_absorbed_colors_device = nullptr;
	color* m_accumulated_colors_device = nullptr;
	ray* m_rays_device = nullptr;
	int* m_energy_exist_pixels_device = nullptr;
	scattering* m_scatterings_device = nullptr;

public:
	~path_tracer();

	std::vector<std::string> init(render_camera* render_camera, config_parser* config, const std::string& scene_file_directory);
	image* render();
	void clear();

	void render_ui();

	bool init_scene_device_data(int index);
	void release_scene_device_data();
};

#endif // !__PATH_TRACER__