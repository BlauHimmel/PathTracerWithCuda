#include "Bvh\bounding_box.h"

bounding_box::bounding_box()
{
	left_bottom.x = -1.0f;
	left_bottom.y = -1.0f;
	left_bottom.z = -1.0f;
	right_top.x = -1.0f;
	right_top.y = -1.0f;
	right_top.z = -1.0f;
	centroid.x = -1.0f;
	centroid.y = -1.0f;
	centroid.z = -1.0f;
}

bounding_box::bounding_box(const float3& left_bottom, const float3& right_top) :
	left_bottom(left_bottom), right_top(right_top), centroid(0.5f * (right_top + left_bottom))
{

}

void bounding_box::expand_to_fit_box(const float3& other_left_bottom, const float3& other_right_top)
{
	left_bottom = fminf(left_bottom, other_left_bottom);
	right_top = fmaxf(right_top, other_right_top);
	centroid = 0.5f * (right_top + left_bottom);
}

void bounding_box::expand_to_fit_triangle(const float3& vertex0, const float3& vertex1, const float3& vertex2)
{
	left_bottom = fminf(left_bottom, make_float3(min(vertex0.x, vertex1.x, vertex2.x), min(vertex0.y, vertex1.y, vertex2.y), min(vertex0.z, vertex1.z, vertex2.z)));
	right_top = fmaxf(right_top, make_float3(max(vertex0.x, vertex1.x, vertex2.x), max(vertex0.y, vertex1.y, vertex2.y), max(vertex0.z, vertex1.z, vertex2.z)));
	centroid = 0.5f * (right_top + left_bottom);
}

void bounding_box::get_bounding_box(const float3& other_left_bottom, const float3& other_right_top)
{
	left_bottom = other_left_bottom;
	right_top = other_right_top;
	centroid = 0.5f * (right_top + left_bottom);
}

void bounding_box::get_bounding_box(const float3& vertex0, const float3& vertex1, const float3& vertex2)
{
	left_bottom = make_float3(min(vertex0.x, vertex1.x, vertex2.x), min(vertex0.y, vertex1.y, vertex2.y), min(vertex0.z, vertex1.z, vertex2.z));
	right_top = make_float3(max(vertex0.x, vertex1.x, vertex2.x), max(vertex0.y, vertex1.y, vertex2.y), max(vertex0.z, vertex1.z, vertex2.z));

	centroid = 0.5f * (right_top + left_bottom);
}

float bounding_box::get_surface_area()
{
	return 2.0f * (right_top.x - left_bottom.x) * (right_top.y - left_bottom.y) * (right_top.z - left_bottom.z);
}

float bounding_box::get_axis_length(int axis)
{
	if (axis == 0)	return right_top.x - left_bottom.x;
	else if (axis == 1) return right_top.y - left_bottom.y;
	else return right_top.z - left_bottom.z;
}

bool bounding_box::is_thin_bounding_box()
{
	return
		right_top.x == left_bottom.x ||
		right_top.y == left_bottom.y ||
		right_top.z == left_bottom.z;
}