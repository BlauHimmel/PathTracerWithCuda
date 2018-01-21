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
#include "cuda_math.hpp"
#include "utilities.hpp"
#include "basic_math.h"
#include "path_tracer_kernel.h"
#include "material.hpp"

class path_tracer
{
private:
	image* m_image;
	int m_sphere_num;
	sphere* m_spheres;
	render_camera* m_render_camera;
	bool m_is_initiated;

public:
	path_tracer();
	~path_tracer();

	void init(render_camera* render_camera);
	image* render();
	void clear();

private:
	void create_device_data();
	void release_device_data();
};

path_tracer::path_tracer()
{
	m_is_initiated = false;
	m_sphere_num = 0;
	m_image = nullptr;
	m_spheres = nullptr;
	m_render_camera = nullptr;
}

inline path_tracer::~path_tracer()
{
	if (m_is_initiated)
	{
		release_device_data();
		release_image(m_image);
	}
}

inline void path_tracer::init(render_camera* render_camera)
{
	m_is_initiated = true;
	m_render_camera = render_camera;
	m_image = create_image(static_cast<int>(render_camera->resolution.x), static_cast<int>(render_camera->resolution.y));
	create_device_data();
}

inline image* path_tracer::render()
{
	if (m_is_initiated)
	{
		image* buffer = create_image(m_image->width, m_image->height);
		path_tracer_kernel(m_sphere_num, m_spheres, buffer->pixel_count, buffer->pixels, m_image->pass_counter, m_render_camera);
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

inline void path_tracer::create_device_data()
{
	//TODO:LOAD SCENE FROM FILE
	m_sphere_num = 15;

	sphere* temp_spheres = new sphere[m_sphere_num];

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

	material glass = get_default_material();
	glass.specular_color = make_float3(1.0f, 1.0f, 1.0f);
	glass.is_transparent = true;
	glass.medium.refraction_index = 1.62f;

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
	marble.medium.scattering.reduced_scattering_coefficient = 8.0f;

	material something = get_default_material();
	something.specular_color = make_float3(1.0f, 1.0f, 1.0f);
	something.is_transparent = true;
	something.medium.refraction_index = 1.333f;
	something.medium.scattering.absorption_coefficient = make_float3(0.9f, 0.3f, 0.02f);
	something.medium.scattering.reduced_scattering_coefficient = 2.0f;

	material ketchup = get_default_material();
	ketchup.specular_color = make_float3(1.0f, 1.0f, 1.0f);
	ketchup.is_transparent = true;
	ketchup.medium.refraction_index = 1.35f;
	ketchup.medium.scattering.absorption_coefficient = make_float3(0.02f, 5.1f, 5.7f);
	ketchup.medium.scattering.reduced_scattering_coefficient = 9.0f;

	material white = get_default_material();
	white.diffuse_color = make_float3(0.9f, 0.9f, 0.9f);

	material gold = get_default_material();
	gold.specular_color = make_float3(0.869f, 0.621f, 0.027f);
	gold.medium.refraction_index = 1000.0f;

	material steel = get_default_material();
	steel.specular_color = make_float3(0.89f, 0.89f, 0.89f);
	steel.medium.refraction_index = 1000.0f;

	material light = get_default_material();
	light.emission_color = make_float3(13.0f, 13.0f, 11.0f);

	temp_spheres[0].center = make_float3(-0.9f, 0.0f, -0.9f);
	temp_spheres[0].radius = 0.8f;
	temp_spheres[0].mat = something;

	temp_spheres[1].center = make_float3(0.8f, 0.0f, -0.4f);
	temp_spheres[1].radius = 0.8f;
	temp_spheres[1].mat = ketchup;

	temp_spheres[2].center = make_float3(-0.5f, -0.4f, 1.0f);
	temp_spheres[2].radius = 0.4f;
	temp_spheres[2].mat = marble;

	temp_spheres[3].center = make_float3(1.5f, 1.6f, -2.3f);
	temp_spheres[3].radius = 0.4f;
	temp_spheres[3].mat = ketchup;

	temp_spheres[4].center = make_float3(-1.0f, -0.7f, 1.2f);
	temp_spheres[4].radius = 0.1f;
	temp_spheres[4].mat = marble;

	temp_spheres[5].center = make_float3(-0.5f, -0.7f, 1.7f);
	temp_spheres[5].radius = 0.1f;
	temp_spheres[5].mat = marble;

	temp_spheres[6].center = make_float3(0.3f, -0.7f, 1.4f);
	temp_spheres[6].radius = 0.1f;
	temp_spheres[6].mat = marble;

	temp_spheres[7].center = make_float3(-0.1f, -0.7f, 0.1f);
	temp_spheres[7].radius = 0.1f;
	temp_spheres[7].mat = marble;

	temp_spheres[8].center = make_float3(0.9f, -0.5f, 1.3f);
	temp_spheres[8].radius = 0.3f;
	temp_spheres[8].mat = green_glass;

	temp_spheres[9].center = make_float3(0.2f, -0.55f, 0.7f);
	temp_spheres[9].radius = 0.25f;
	temp_spheres[9].mat = marble;

	temp_spheres[10].center = make_float3(0.8f, 1.8f, -0.4f);
	temp_spheres[10].radius = 0.2f;
	temp_spheres[10].mat = ketchup;

	temp_spheres[11].center = make_float3(0.8f, 2.1f, -0.4f);
	temp_spheres[11].radius = 0.1f;
	temp_spheres[11].mat = ketchup;

	temp_spheres[12].center = make_float3(0.8f, 2.25f, -0.4f);
	temp_spheres[12].radius = 0.05f;
	temp_spheres[12].mat = ketchup;

	temp_spheres[13].center = make_float3(0.8f, 2.325f, -0.4f);
	temp_spheres[13].radius = 0.025f;
	temp_spheres[13].mat = ketchup;

	temp_spheres[14].center = make_float3(-4.0f, 15.0, 0.0f);
	temp_spheres[14].radius = 6.0f;
	temp_spheres[14].mat = light;

	CUDA_CALL(cudaMalloc((void**)&m_spheres, m_sphere_num * sizeof(sphere)));

	CUDA_CALL(cudaMemcpy(m_spheres, temp_spheres, m_sphere_num * sizeof(sphere), cudaMemcpyHostToDevice));

	SAFE_DELETE_ARRAY(temp_spheres);
}

inline void path_tracer::release_device_data()
{
	//TODO:Hardcode here
	CUDA_CALL(cudaFree(m_spheres));
}

#endif // !__PATH_TRACER__
