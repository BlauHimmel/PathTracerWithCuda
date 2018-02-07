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

#include "lib\imgui\imgui.h"
#include "lib\imgui\imgui_impl_glfw_gl3.h"

class path_tracer
{
private:
	cube_map_loader m_cube_map_loader;
	cube_map* m_cube_map;
	image* m_image;

	triangle_mesh m_triangle_mesh;
	int m_triangle_num;
	triangle* m_triangles_device;
	bvh_node_device* m_bvh_nodes;

	int m_sphere_num;
	sphere* m_spheres_device;
	sphere* m_spheres;

	render_camera* m_render_camera;

	bool m_is_initiated;

public:
	path_tracer();
	~path_tracer();

	void init(render_camera* render_camera);
	image* render();
	void clear();

	void render_ui();

private:
	void create_scene_device_data();
	void release_scene_device_data();
};

path_tracer::path_tracer()
{
	m_is_initiated = false;
	m_sphere_num = 0;
	m_image = nullptr;
	m_spheres = nullptr;
	m_render_camera = nullptr;
	m_cube_map = nullptr;
	m_triangle_num = 0;
	m_triangles_device = nullptr;
	m_bvh_nodes = nullptr;
	m_spheres_device = nullptr;
}

inline path_tracer::~path_tracer()
{
	if (m_is_initiated)
	{
		release_scene_device_data();
		release_image(m_image);
	}
}

inline void path_tracer::init(render_camera* render_camera)
{
	m_is_initiated = true;
	m_render_camera = render_camera;
	m_image = create_image(static_cast<int>(render_camera->resolution.x), static_cast<int>(render_camera->resolution.y));
	create_scene_device_data();
}

inline image* path_tracer::render()
{
	if (m_is_initiated)
	{
		image* buffer = create_image(m_image->width, m_image->height);
		path_tracer_kernel(m_triangle_num, m_bvh_nodes, m_triangles_device,m_sphere_num, m_spheres_device, buffer->pixel_count, buffer->pixels, m_image->pass_counter, m_render_camera, m_cube_map);
		for (auto i = 0; i < m_image->pixel_count; i++)
		{
			m_image->pixels[i] += buffer->pixels[i];
		}
		m_image->pass_counter++;
		release_image(buffer);
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

	if (m_sphere_num > 0 && ImGui::TreeNode("Sphere"))
	{
		for (auto i = 0; i < m_sphere_num; i++)
		{
			sprintf(buffer, "Sphere-%d", i + 1);
			if (ImGui::TreeNode(buffer))
			{
				bool is_modified = false;

				ImGui::Separator();

				sphere sphere = m_spheres[i];

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

					m_spheres[i] = sphere;
				}

				ImGui::TreePop();
			}
		}

		ImGui::TreePop();
	}

	bool is_triangle_mesh_modified = false;

	if (m_triangle_num > 0 && ImGui::TreeNode("Triangle Mesh"))
	{
		bool is_modified = false;

		ImGui::Separator();

		float3 position = m_triangle_mesh.get_position();

		ImGui::Text("Base:");
		sprintf(buffer, "Vertices: %d", m_triangle_mesh.get_vertex_num());
		ImGui::Text(buffer);
		sprintf(buffer, "Facets: %d", m_triangle_mesh.get_triangle_num());
		ImGui::Text(buffer);
		sprintf(buffer, "Position: (%.2f, %.2f, %.2f)", position.x, position.y, position.z);
		ImGui::Text(buffer);

		ImGui::Separator();

		material mat = m_triangle_mesh.get_material();

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

			mat.diffuse_color = make_float3(diffuse[0], diffuse[1], diffuse[2]);
			mat.specular_color = make_float3(specular[0], specular[1], specular[2]);
			mat.emission_color = make_float3(emission[0], emission[1], emission[2]);
			mat.is_transparent = is_transparent;
			mat.roughness = roughness;

			mat.medium.refraction_index = refraction_index;
			mat.medium.extinction_coefficient = extinction_coefficient;
			mat.medium.scattering.absorption_coefficient = make_float3(absorption_coefficient[0], absorption_coefficient[1], absorption_coefficient[2]);
			mat.medium.scattering.reduced_scattering_coefficient = make_float3(reduced_scattering_coefficient, reduced_scattering_coefficient, reduced_scattering_coefficient);

			m_triangle_mesh.set_material(mat);
		}

		ImGui::TreePop();
	}

	if (is_sphere_modified)
	{
		CUDA_CALL(cudaFree(m_spheres_device));
		CUDA_CALL(cudaMalloc((void**)&m_spheres_device, m_sphere_num * sizeof(sphere)));
		CUDA_CALL(cudaMemcpy(m_spheres_device, m_spheres, m_sphere_num * sizeof(sphere), cudaMemcpyHostToDevice));
		clear();
	}

	if (is_triangle_mesh_modified)
	{
		CUDA_CALL(cudaFree(m_triangles_device));
		m_triangles_device = m_triangle_mesh.create_mesh_device_data();
		clear();
	}
}

inline void path_tracer::create_scene_device_data()
{
	//TODO:LOAD SCENE FROM FILE
	m_sphere_num = 2;

	m_spheres = new sphere[m_sphere_num];

	material red = get_default_material();
	red.diffuse_color = make_float3(0.87f, 0.15f, 0.15f);
	red.specular_color = make_float3(1.0f, 1.0f, 1.0f);
	red.medium.refraction_index = 1.491f;

	material green = get_default_material();
	green.diffuse_color = make_float3(0.15f, 0.87f, 0.15f);
	green.specular_color = make_float3(1.0f, 1.0f, 1.0f);
	green.medium.refraction_index = 1.491f;

	material orange = get_default_material();
	orange.diffuse_color = make_float3(0.93f, 0.33f, 0.04f);
	orange.specular_color = make_float3(1.0f, 1.0f, 1.0f);
	orange.medium.refraction_index = 1.491f;

	material purple = get_default_material();
	purple.diffuse_color = make_float3(0.5f, 0.1f, 0.9f);
	purple.specular_color = make_float3(1.0f, 1.0f, 1.0f);
	purple.medium.refraction_index = 1.491f;

	material blue = get_default_material();
	blue.diffuse_color = make_float3(0.4f, 0.6f, 0.8f);
	blue.specular_color = make_float3(1.0f, 1.0f, 1.0f);
	blue.medium.refraction_index = 1.2f;

	material glass = get_default_material();
	glass.specular_color = make_float3(1.0f, 1.0f, 1.0f);
	glass.is_transparent = true;
	glass.roughness = 0.75f;
	glass.medium.refraction_index = 2.42f;

	material green_glass = get_default_material();
	green_glass.specular_color = make_float3(1.0f, 1.0f, 1.0f);
	green_glass.is_transparent = true;
	green_glass.medium.refraction_index = 1.62f;
	green_glass.medium.scattering.absorption_coefficient = make_float3(1.0f, 0.01f, 1.0f);

	material marble = get_default_material();
	marble.specular_color = make_float3(1.0f, 1.0f, 1.0f);
	marble.is_transparent = true;
	marble.medium.refraction_index = 1.486f;
	marble.medium.scattering.absorption_coefficient = make_float3(0.6f, 0.6f, 0.6f);
	marble.medium.scattering.reduced_scattering_coefficient = make_float3(8.0f, 8.0f, 8.0f);

	material something = get_default_material();
	something.specular_color = make_float3(1.0f, 1.0f, 1.0f);
	something.is_transparent = true;
	something.medium.refraction_index = 1.333f;
	something.medium.scattering.absorption_coefficient = make_float3(0.9f, 0.3f, 0.02f);
	something.medium.scattering.reduced_scattering_coefficient = make_float3(2.0f, 2.0f, 2.0f);

	material ketchup = get_default_material();
	ketchup.specular_color = make_float3(1.0f, 1.0f, 1.0f);
	ketchup.is_transparent = true;
	ketchup.medium.refraction_index = 1.35f;
	ketchup.medium.scattering.absorption_coefficient = make_float3(0.02f, 5.1f, 5.7f);
	ketchup.medium.scattering.reduced_scattering_coefficient = make_float3(9.0f, 9.0f, 9.0f);



	material light = get_default_material();
	light.emission_color = make_float3(18.0f, 18.0f, 15.0f);

	m_spheres[0].center = make_float3(-0.9f, 2.0f, -0.9f);
	m_spheres[0].radius = 1.0f;
	m_spheres[0].mat = material_data::metal::gold();

	//m_spheres[1].center = make_float3(0.9f, -0.5f, 1.3f);
	//m_spheres[1].radius = 0.3f;
	//m_spheres[1].mat = green_glass;

	//m_spheres[2].center = make_float3(-0.5f, -0.4f, 1.0f);
	//m_spheres[2].radius = 0.4f;
	//m_spheres[2].mat = steel;

	//m_spheres[3].center = make_float3(-1.0f, -0.7f, 1.2f);
	//m_spheres[3].radius = 0.1f;
	//m_spheres[3].mat = blue;

	//m_spheres[4].center = make_float3(-0.5f, -0.7f, 1.7f);
	//m_spheres[4].radius = 0.1f;
	//m_spheres[4].mat = red;

	//m_spheres[5].center = make_float3(0.3f, -0.7f, 1.4f);
	//m_spheres[5].radius = 0.1f;
	//m_spheres[5].mat = blue;

	//m_spheres[6].center = make_float3(0.1f, -0.7f, 0.1f);
	//m_spheres[6].radius = 0.1f;
	//m_spheres[6].mat = blue;
	  
	//m_spheres[7].center = make_float3(0.2f, -0.55f, 0.7f);
	//m_spheres[7].radius = 0.25f;
	//m_spheres[7].mat = glass;
	  
	//m_spheres[8].center = make_float3(0.8f, 0.0f, -0.4f);
	//m_spheres[8].radius = 0.8f;
	//m_spheres[8].mat = green;
	  
	//m_spheres[9].center = make_float3(0.8f, 1.2f, -0.4f);
	//m_spheres[9].radius = 0.4f;
	//m_spheres[9].mat = purple;
	  
	//m_spheres[10].center = make_float3(0.8f, 1.8f, -0.4f);
	//m_spheres[10].radius = 0.2f;
	//m_spheres[10].mat = marble;
	  
	//m_spheres[11].center = make_float3(0.8f, 2.1f, -0.4f);
	//m_spheres[11].radius = 0.1f;
	//m_spheres[11].mat = red;
	  
	//m_spheres[12].center = make_float3(0.8f, 2.25f, -0.4f);
	//m_spheres[12].radius = 0.05f;
	//m_spheres[12].mat = green;
	  
	//m_spheres[13].center = make_float3(0.8f, 2.325f, -0.4f);
	//m_spheres[13].radius = 0.025f;
	//m_spheres[13].mat = green;
	 
	//m_spheres[14].center = make_float3(2.0f, -0.05f, 2.0f);
	//m_spheres[14].radius = 0.75f;
	//m_spheres[14].mat = something;

	m_spheres[1].center = make_float3(-8.0, 50.0, -5.0);
	m_spheres[1].radius = 10.0f;
	m_spheres[1].mat = light;

	CUDA_CALL(cudaMalloc((void**)&m_spheres_device, m_sphere_num * sizeof(sphere)));
	CUDA_CALL(cudaMemcpy(m_spheres_device, m_spheres, m_sphere_num * sizeof(sphere), cudaMemcpyHostToDevice));

	m_cube_map_loader.load_data(
		"res\\texture\\lancellotti_chapel\\xpos.bmp",
		"res\\texture\\lancellotti_chapel\\xneg.bmp",
		"res\\texture\\lancellotti_chapel\\ypos.bmp",
		"res\\texture\\lancellotti_chapel\\yneg.bmp",
		"res\\texture\\lancellotti_chapel\\zpos.bmp",
		"res\\texture\\lancellotti_chapel\\zneg.bmp"
	);

	m_cube_map = m_cube_map_loader.create_cube_device_data();

	m_triangle_mesh.load_obj("res\\obj\\diamond.obj");
	m_triangle_mesh.set_material(material_data::dielectric::diamond());
	m_triangle_mesh.set_position(make_float3(3.0f, 0.0f, 0.0f));
	m_triangles_device = m_triangle_mesh.create_mesh_device_data();
	m_triangle_num = m_triangle_mesh.get_triangle_num();

	clock_t start_time, end_time;

	printf("constructing bvh...\n");
	start_time = clock();
	bvh_node* root = build_bvh(m_triangle_mesh.get_triangles());
	end_time = clock();
	double build_bvh_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC * 1000.0;
	printf("bvh constructed on CPU, time consuming: %.4f ms\n", build_bvh_time);

	printf("copying bvh data to GPU...\n");
	start_time = clock();
	m_bvh_nodes = build_bvh_device_data(root);
	end_time = clock();
	double copy_bvh_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC * 1000.0;
	printf("bvh constructed on GPU, time consuming: %.4f ms\n", copy_bvh_time);
}

inline void path_tracer::release_scene_device_data()
{
	//TODO:Hardcode here
	CUDA_CALL(cudaFree(m_spheres_device));
	SAFE_DELETE_ARRAY(m_spheres);
	m_cube_map_loader.release_cube_device_data();
	m_cube_map_loader.unload_data();
	m_cube_map = nullptr;
}

#endif // !__PATH_TRACER__