#include "Core\camera.h"

view_camera::view_camera()
{
	center = glm::vec3(0.0f, 0.0f, 0.0f);
	yaw = 0.0f;
	pitch = 0.3f;
	radius = 14.0f;
	aperture_radius = 0.0f;
	focal_distance = radius;

	resolution = glm::vec2(640.0f, 640.0f);
	fov = glm::vec2(45.0f, 45.0f);
}

void view_camera::modify_yaw(float delta)
{
	yaw += delta;
	clamp_yaw();
}

void view_camera::modify_pitch(float delta)
{
	pitch += delta;
	clamp_pitch();
}

void view_camera::modify_radius(float scale)
{
	radius += radius * scale;
	clamp_radius();
}

void view_camera::modify_pan(float x, float y)
{
	glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::vec3 view = glm::vec3(-sin(yaw) * cos(pitch), -sin(pitch), -cos(yaw) * cos(pitch));
	glm::vec3 horizontal = normalize(cross(view, up));
	up = normalize(cross(horizontal, view));

	center += (up * y + horizontal * x);
}

void view_camera::modify_aperture_radius(float delta)
{
	aperture_radius += delta;
	clamp_aperture_radius();
}

void view_camera::modify_focal_distance(float delta)
{
	focal_distance += delta;
	clamp_focal_distance();
}

void view_camera::set_fov(float fov_x)
{
	fov.x = fov_x;
	fov.y = math::radians_to_degrees(atan(tan(math::degrees_to_radians(fov_x) * 0.5f) * (resolution.y / resolution.x)) * 2.0f);
}

void view_camera::set_resolution(float width, float height)
{
	resolution.x = width;
	resolution.y = height;
}

void view_camera::set_aperture_radius(float value)
{
	aperture_radius = value;
	clamp_aperture_radius();
}

void view_camera::set_focal_distance(float value)
{
	focal_distance = value;
	clamp_focal_distance();
}

void view_camera::get_render_camera(render_camera* render_cam) const
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

float view_camera::get_max_aperture_radius() const
{
	return MAX_APERTURE_RADIUS;
}

float view_camera::get_max_focal_distance() const
{
	return 2.0f * radius;
}

float view_camera::get_aperture_radius()
{
	return aperture_radius;
}

float view_camera::get_focal_distance()
{
	return focal_distance;
}

void view_camera::clamp_yaw()
{
	yaw = math::mod(yaw, TWO_PI);
}

void view_camera::clamp_pitch()
{
	pitch = math::clamp(pitch, -HALF_PI + 0.02f, HALF_PI - 0.02f);
}

void view_camera::clamp_radius()
{
	radius = math::clamp(radius, 0.02f, 4000.0f);
	focal_distance = math::clamp(focal_distance, 0.0f, radius);
}

void view_camera::clamp_aperture_radius()
{
	aperture_radius = math::clamp(aperture_radius, 0.0f, MAX_APERTURE_RADIUS);
}

void view_camera::clamp_focal_distance()
{
	focal_distance = math::clamp(focal_distance, 0.0f, 2.0f * radius);
}
