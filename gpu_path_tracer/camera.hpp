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
};

//used by openGL
struct view_camera
{
	glm::vec4 up;
	glm::vec4 target;
	glm::vec4 eye;
	glm::vec2 resolution;
	glm::vec2 fov;

	void orbit_left(float value);
	void orbit_right(float value);
	void orbit_up(float value);
	void orbit_down(float value);

	void zoom_in(float value);
	void zoom_out(float value);

	void move(float x, float y);

	void set_resolution(float width, float height);
	void set_fov_x(float fov);

	void get_render_camera(render_camera* render_cam);
};

inline void view_camera::orbit_left(float value)
{
	glm::mat4 rotate_mat = glm::rotate(glm::mat4(1.0f), -1.0f * value, glm::vec3(up));
	up = rotate_mat * up;
	eye = rotate_mat * eye;
}

inline void view_camera::orbit_right(float value)
{
	glm::mat4 rotate_mat = glm::rotate(glm::mat4(1.0f), 1.0f * value, glm::vec3(up));
	up = rotate_mat * up;
	eye = rotate_mat * eye;
}

inline void view_camera::orbit_up(float value)
{
	glm::mat4 rotate_mat = glm::rotate(glm::mat4(1.0f), -1.0f * value, glm::cross(glm::vec3(eye), glm::vec3(up)));
	up = rotate_mat * up;
	eye = rotate_mat * eye;
}

inline void view_camera::orbit_down(float value)
{
	glm::mat4 rotate_mat = glm::rotate(glm::mat4(1.0f), 1.0f * value, glm::cross(glm::vec3(eye), glm::vec3(up)));
	up = rotate_mat * up;
	eye = rotate_mat * eye;
}

inline void view_camera::zoom_in(float value)
{
	glm::vec3 view_dir = value * glm::vec3(glm::normalize(target - eye));
	eye = eye + glm::vec4(view_dir, 0.0f);
}

inline void view_camera::zoom_out(float value)
{
	glm::vec3 view_dir = -value * glm::vec3(glm::normalize(target - eye));
	eye = eye + glm::vec4(view_dir, 0.0f);
}

inline void view_camera::move(float x, float y)
{
	glm::vec3 direction = glm::vec3(glm::normalize(target - eye));
	glm::vec4 horizontal = glm::vec4(glm::cross(direction, glm::vec3(up)), 0.0f);
	eye = eye + horizontal * x + up * y;
	target = target + horizontal * x + up * y;
}

inline void view_camera::set_resolution(float width, float height)
{
	resolution = glm::vec2(width, height);
	set_fov_x(fov.x);
}

inline void view_camera::set_fov_x(float fov_x)
{
	fov.x = fov_x;
	fov.y = fov_x * (resolution.y / resolution.x);
}

inline void view_camera::get_render_camera(render_camera* render_cam)
{
	glm::vec3 direction = glm::vec3(glm::normalize(target - eye));
	render_cam->eye = make_float3(eye.x, eye.y, eye.z);
	render_cam->view = make_float3(direction.x, direction.y, direction.z);
	render_cam->up = make_float3(up.x, up.y, up.z);
	render_cam->resolution = make_float2(resolution.x, resolution.y);
	render_cam->fov = make_float2(fov.x, fov.y);
}

#endif // !__CAMERA__
