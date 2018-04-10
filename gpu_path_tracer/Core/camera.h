#pragma once

#ifndef __CAMERA__
#define __CAMERA__

#include <cuda_runtime.h>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include "Math\basic_math.hpp"

#define MAX_APERTURE_RADIUS 1.0f

//passed to cuda kernel for rendering, this structure is retrived from view_camera
struct render_camera
{
	float3 eye;
	float3 view;
	float3 up;
	float2 resolution;
	float2 fov;
	float aperture_radius;
	float focal_distance;
};

//used for glfw callback for interact
struct view_camera
{
private:
	glm::vec3 center;
	
	float yaw;
	float pitch;
	float radius;

	float aperture_radius;
	float focal_distance;

public:

	glm::vec2 resolution;
	glm::vec2 fov;

	view_camera();

	void modify_yaw(float delta);
	void modify_pitch(float delta);
	void modify_radius(float delta);
	void modify_pan(float x, float y);

	void modify_aperture_radius(float delta);
	void modify_focal_distance(float delta);
	
	void set_fov(float fov_x);
	void set_resolution(float width, float height);
	void set_aperture_radius(float value);
	void set_focal_distance(float value);

	void get_render_camera(render_camera* render_cam) const;
	float get_max_aperture_radius() const;
	float get_max_focal_distance() const;
	float get_aperture_radius();
	float get_focal_distance();

private:

	void clamp_yaw();
	void clamp_pitch();
	void clamp_radius();
	void clamp_aperture_radius();
	void clamp_focal_distance();
};

#endif // !__CAMERA__
