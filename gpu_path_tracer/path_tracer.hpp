#pragma once

#ifndef __PATH_TRACER__
#define __PATH_TRACER__

#include <cuda_runtime.h>

#include <string.h>
#include <stdio.h>
#include <time.h>

#include "sphere.hpp"
#include "image.hpp"
#include "ray.hpp"
#include "camera.hpp"
#include "material.hpp"
#include "cuda_math.hpp"
#include "utilities.hpp"
#include "basic_math.h"
#include "path_tracer_kernel.h"
#include "material.hpp"
#include "cube_map.hpp"
#include "triangle_mesh.hpp"
#include "bvh.hpp"
#include "scene_parser.hpp"
#include "config_parser.hpp"

#include "lib\imgui\imgui.h"
#include "lib\imgui\imgui_impl_glfw_gl3.h"

class path_tracer
{
private:
	scene_parser m_scene;

	config_parser* m_config = nullptr;
	render_camera* m_render_camera = nullptr;
	image* m_image = nullptr;
	image* m_buffer = nullptr;

	bool m_is_initiated = false;

	color* m_not_absorbed_colors_device = nullptr;
	color* m_accumulated_colors_device = nullptr;
	ray* m_rays_device = nullptr;
	int* m_energy_exist_pixels_device = nullptr;
	scattering* m_scatterings_device = nullptr;

public:
	~path_tracer();

	void init(render_camera* render_camera, config_parser* config);
	image* render();
	void clear();

	void render_ui();

private:
	void init_scene_device_data();
};

inline path_tracer::~path_tracer()
{
	if (m_is_initiated)
	{
		release_image(m_image);
	}
	path_tracer_kernel_memory_free(
		m_not_absorbed_colors_device,
		m_accumulated_colors_device,
		m_rays_device,
		m_energy_exist_pixels_device,
		m_scatterings_device
	);
}

inline void path_tracer::init(render_camera* render_camera, config_parser* config)
{
	m_is_initiated = true;
	m_config = config;
	m_render_camera = render_camera;
	m_image = create_image(static_cast<int>(render_camera->resolution.x), static_cast<int>(render_camera->resolution.y));
	m_buffer = create_image(static_cast<int>(render_camera->resolution.x), static_cast<int>(render_camera->resolution.y));
	path_tracer_kernel_memory_allocate(
		&m_not_absorbed_colors_device,
		&m_accumulated_colors_device,
		&m_rays_device,
		&m_energy_exist_pixels_device,
		&m_scatterings_device,
		m_buffer->pixel_count
	);
	init_scene_device_data();
}

inline image* path_tracer::render()
{
	if (m_is_initiated)
	{
		path_tracer_kernel(
			m_scene.get_triangles_num(),
			m_scene.get_bvh_node_device_ptr(),
			m_scene.get_triangles_device_ptr(),
			m_scene.get_sphere_num(),
			m_scene.get_sphere_device_ptr(),
			m_buffer->pixel_count,
			m_buffer->pixels,
			m_image->pass_counter,
			m_render_camera,
			m_scene.get_cube_map_device_ptr(),
			m_not_absorbed_colors_device,
			m_accumulated_colors_device,
			m_rays_device,
			m_energy_exist_pixels_device,
			m_scatterings_device,
			m_config->get_config_device_ptr()

		);
		for (auto i = 0; i < m_image->pixel_count; i++)
		{
			m_image->pixels[i] += m_buffer->pixels[i];
		}
		m_image->pass_counter++;
		return m_image;
	}
	else
	{
		return nullptr;
	}
}

inline void path_tracer::clear()
{
	if (m_is_initiated)
	{
		release_image(m_image);
		m_image = create_image(static_cast<int>(m_render_camera->resolution.x), static_cast<int>(m_render_camera->resolution.y));
	}
}

inline void path_tracer::render_ui()
{
	char buffer[2048];

	bool is_sphere_modified = false;

	if (m_scene.get_sphere_num() > 0 && ImGui::TreeNode("Sphere"))
	{
		for (auto i = 0; i < m_scene.get_sphere_num(); i++)
		{
			sprintf(buffer, "Sphere-%d", i + 1);
			if (ImGui::TreeNode(buffer))
			{
				bool is_modified = false;

				ImGui::Separator();

				sphere sphere = m_scene.get_sphere(i);

				float position[3] = { sphere.center.x, sphere.center.y ,sphere.center.z };
				float radius = sphere.radius;
				
				ImGui::Text("Base:");
				is_modified = is_modified || ImGui::DragFloat3("Position", position, 0.001f);
				is_modified = is_modified || ImGui::DragFloat("Radius", &radius, 0.001f);

				ImGui::Separator();

				material mat = sphere.mat;

				float diffuse[3] = { mat.diffuse_color.x, mat.diffuse_color.y, mat.diffuse_color.z };
				float specular[3] = { mat.specular_color.x, mat.specular_color.y, mat.specular_color.z };
				float emission[3] = { mat.emission_color.x, mat.emission_color.y, mat.emission_color.z };
				bool is_transparent = mat.is_transparent;
				float roughness = mat.roughness;

				ImGui::Text("Material:");
				is_modified = is_modified || ImGui::ColorEdit3("Diffuse", diffuse);
				is_modified = is_modified || ImGui::ColorEdit3("Specular", specular);
				is_modified = is_modified || ImGui::ColorEdit3("Emission", emission);
				is_modified = is_modified || ImGui::Checkbox("Transparent", &is_transparent);
				is_modified = is_modified || ImGui::SliderFloat("Roughness", &roughness, 0.0f, 1.0f);

				float refraction_index = mat.medium.refraction_index;
				float extinction_coefficient = mat.medium.extinction_coefficient;

				is_modified = is_modified || ImGui::DragFloat("Refraction Index", &refraction_index, 0.001f);
				is_modified = is_modified || ImGui::DragFloat("Extinction Coefficient", &extinction_coefficient, 0.001f);

				float absorption_coefficient[] = { mat.medium.scattering.absorption_coefficient.x, mat.medium.scattering.absorption_coefficient.y, mat.medium.scattering.absorption_coefficient.z };
				float reduced_scattering_coefficient = mat.medium.scattering.reduced_scattering_coefficient.x;

				is_modified = is_modified || ImGui::DragFloat3("Absorption Coefficient", absorption_coefficient, 0.001f);
				is_modified = is_modified || ImGui::DragFloat("Reduced Scattering Coefficient Coefficient", &reduced_scattering_coefficient, 0.001f);

				ImGui::Separator();

				if (is_modified)
				{
					is_sphere_modified = true;

					sphere.center = make_float3(position[0], position[1], position[2]);
					sphere.radius = radius;

					mat.diffuse_color = make_float3(diffuse[0], diffuse[1], diffuse[2]);
					mat.specular_color = make_float3(specular[0], specular[1], specular[2]);
					mat.emission_color = make_float3(emission[0], emission[1], emission[2]);
					mat.is_transparent = is_transparent;
					mat.roughness = roughness;

					mat.medium.refraction_index = refraction_index;
					mat.medium.extinction_coefficient = extinction_coefficient;
					mat.medium.scattering.absorption_coefficient = make_float3(absorption_coefficient[0], absorption_coefficient[1], absorption_coefficient[2]);
					mat.medium.scattering.reduced_scattering_coefficient = make_float3(reduced_scattering_coefficient, reduced_scattering_coefficient, reduced_scattering_coefficient);

					sphere.mat = mat;

					m_scene.set_sphere_device(i, sphere);
				}

				ImGui::TreePop();
			}
		}

		ImGui::TreePop();
	}

	bool is_triangle_mesh_modified = false;
	
	if (m_scene.get_mesh_num() > 0 && ImGui::TreeNode("Mesh"))
	{
		for (auto i = 0; i < m_scene.get_mesh_num(); i++)
		{
			sprintf(buffer, "Mesh-%d", i + 1);
			if (ImGui::TreeNode(buffer))
			{
				bool is_modified = false;

				ImGui::Separator();

				float3 position = m_scene.get_mesh_position(i);

				ImGui::Text("Base:");
				sprintf(buffer, "Vertices: %d", m_scene.get_mesh_vertices_num(i));
				ImGui::Text(buffer);
				sprintf(buffer, "Facets: %d", m_scene.get_mesh_triangle_num(i));
				ImGui::Text(buffer);
				sprintf(buffer, "Position: (%.2f, %.2f, %.2f)", position.x, position.y, position.z);
				ImGui::Text(buffer);

				ImGui::Separator();

				material mat = m_scene.get_mesh_material(i);

				float diffuse[3] = { mat.diffuse_color.x, mat.diffuse_color.y, mat.diffuse_color.z };
				float specular[3] = { mat.specular_color.x, mat.specular_color.y, mat.specular_color.z };
				float emission[3] = { mat.emission_color.x, mat.emission_color.y, mat.emission_color.z };
				bool is_transparent = mat.is_transparent;
				float roughness = mat.roughness;

				ImGui::Text("Material:");
				is_modified = is_modified || ImGui::ColorEdit3("Diffuse", diffuse);
				is_modified = is_modified || ImGui::ColorEdit3("Specular", specular);
				is_modified = is_modified || ImGui::ColorEdit3("Emission", emission);
				is_modified = is_modified || ImGui::Checkbox("Transparent", &is_transparent);
				is_modified = is_modified || ImGui::SliderFloat("Roughness", &roughness, 0.0f, 1.0f);

				float refraction_index = mat.medium.refraction_index;
				float extinction_coefficient = mat.medium.extinction_coefficient;

				is_modified = is_modified || ImGui::DragFloat("Refraction Index", &refraction_index, 0.001f);
				is_modified = is_modified || ImGui::DragFloat("Extinction Coefficient", &extinction_coefficient, 0.001f);

				float absorption_coefficient[] = { mat.medium.scattering.absorption_coefficient.x, mat.medium.scattering.absorption_coefficient.y, mat.medium.scattering.absorption_coefficient.z };
				float reduced_scattering_coefficient = mat.medium.scattering.reduced_scattering_coefficient.x;

				is_modified = is_modified || ImGui::DragFloat3("Absorption Coefficient", absorption_coefficient, 0.001f);
				is_modified = is_modified || ImGui::DragFloat("Reduced Scattering Coefficient Coefficient", &reduced_scattering_coefficient, 0.001f);

				ImGui::Separator();

				if (is_modified)
				{
					is_triangle_mesh_modified = true;

					material new_mat = get_default_material();

					new_mat.diffuse_color = make_float3(diffuse[0], diffuse[1], diffuse[2]);
					new_mat.specular_color = make_float3(specular[0], specular[1], specular[2]);
					new_mat.emission_color = make_float3(emission[0], emission[1], emission[2]);
					new_mat.is_transparent = is_transparent;
					new_mat.roughness = roughness;

					new_mat.medium.refraction_index = refraction_index;
					new_mat.medium.extinction_coefficient = extinction_coefficient;
					new_mat.medium.scattering.absorption_coefficient = make_float3(absorption_coefficient[0], absorption_coefficient[1], absorption_coefficient[2]);
					new_mat.medium.scattering.reduced_scattering_coefficient = make_float3(reduced_scattering_coefficient, reduced_scattering_coefficient, reduced_scattering_coefficient);

					m_scene.set_mesh_material_device(i, new_mat);
				}

				ImGui::TreePop();
			}
		}

		ImGui::TreePop();
	}

	if (is_sphere_modified || is_triangle_mesh_modified)
	{
		clear();
	}
}

inline void path_tracer::init_scene_device_data()
{
	double time;
	printf("[Info]Load scene data...\n");
	TIME_COUNT_CALL_START();
	if (!m_scene.load_scene("res\\scene\\scene.json"))
	{
		m_is_initiated = false;
		std::cout << "[Error]Load scene failed!" << std::endl;
		return;
	}
	m_scene.create_scene_data_device();
	TIME_COUNT_CALL_END(time);
	printf("[Info]Load scene completed, total time consuming: %.4f ms\n", time);
}

#endif // !__PATH_TRACER__