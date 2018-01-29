#pragma once

#ifndef __CAMERA__
#define __CAMERA__

#include <cuda_runtime.h>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>

//used by the path tracer
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

//used by openGL
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

	void set_yaw(float delta);
	void set_pitch(float delta);
	void set_radius(float delta);
	void set_pan(float x, float y);
	void set_aperture_radius(float delta);
	void set_focal_distance(float delta);
	
	void set_fov(float fov_x);
	void set_resolution(float width, float height);

	void get_render_camera(render_camera* render_cam);

private:

	void clamp_yaw();
	void clamp_pitch();
	void clamp_radius();
	void clamp_aperture_radius();
	void clamp_focal_distance();
};

inline view_camera::view_camera()
{
	center = glm::vec3(0.0f, 0.0f, 0.0f);
	yaw = 0.0f;
	pitch = 0.3f;
	radius = 4.0f;
	aperture_radius = 0.0f;
	focal_distance = radius;

	resolution = glm::vec2(640.0f, 640.0f);
	fov = glm::vec2(45.0f, 45.0f);
}

inline void view_camera::set_yaw(float delta)
{
	yaw += delta;
	clamp_yaw();
}

inline void view_camera::set_pitch(float delta)
{
	pitch += delta;
	clamp_pitch();
}

inline void view_camera::set_radius(float scale)
{
	radius += radius * scale;
	clamp_radius();
}

inline void view_camera::set_pan(float x, float y)
{
	center.x += x;
	center.y += y;
}

inline void view_camera::set_aperture_radius(float delta)
{
	aperture_radius += delta;
	clamp_aperture_radius();
}

inline void view_camera::set_focal_distance(float delta)
{
	focal_distance += delta;
	clamp_focal_distance();
}

inline void view_camera::set_fov(float fov_x)
{
	fov.x = fov_x;
	fov.y = math::radians_to_degrees(atan(tan(math::degrees_to_radians(fov_x) * 0.5f) * (resolution.y / resolution.x)) * 2.0f);
}

inline void view_camera::set_resolution(float width, float height)
{
	resolution.x = width;
	resolution.y = height;
}

inline void view_camera::get_render_camera(render_camera* render_cam)
{
	float x = sin(yaw) * cos(pitch);
	float y = sin(pitch);
	float z = cos(yaw) * cos(pitch);

	glm::vec3 to_camera = glm::vec3(x, y, z);

	glm::vec3 eye = center + to_camera * radius;
	glm::vec3 view = -1.0f * to_camera;

	render_cam->eye = make_float3(eye.x, eye.y, eye.z);
	render_cam->view = make_float3(view.x, view.y, view.z);
	render_cam->up = make_float3(0.0f, 1.0f, 0.0f);
	render_cam->resolution = make_float2(resolution.x, resolution.y);
	render_cam->fov = make_float2(fov.x, fov.y);
	render_cam->focal_distance = focal_distance;
	render_cam->aperture_radius = aperture_radius;
}

inline void view_camera::clamp_yaw()
{
	yaw = math::mod(yaw, TWO_PI);
}

inline void view_camera::clamp_pitch()
{
	pitch = math::clamp(pitch, -HALF_PI + 0.02f, HALF_PI - 0.02f);
}

inline void view_camera::clamp_radius()
{
	radius = math::clamp(radius, 0.02f, 400.0f);
	focal_distance = math::clamp(focal_distance, 0.0f, radius);
}

inline void view_camera::clamp_aperture_radius()
{
	aperture_radius = math::clamp(aperture_radius, 0.0f, 25.0f);
}

inline void view_camera::clamp_focal_distance()
{
	focal_distance = math::clamp(focal_distance, 0.0f, radius);
}

#endif // !__CAMERA__
