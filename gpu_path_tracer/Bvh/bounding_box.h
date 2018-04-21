#pragma once

#ifndef __BOUNDING_BOX__
#define __BOUNDING_BOX__

#include "Math\cuda_math.hpp"
#include "Core\ray.h"

struct bounding_box
{
	float3 left_bottom;
	float3 right_top;
	float3 centroid;

	__device__ __host__ bounding_box()
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
	
	__device__ __host__ bounding_box(const float3& left_bottom, const float3& right_top) :
		left_bottom(left_bottom), right_top(right_top), centroid(0.5f * (right_top + left_bottom))
	{

	}

	__device__ __host__ void expand_to_fit_box(const float3& other_left_bottom, const float3& other_right_top)
	{
		left_bottom = fminf(left_bottom, other_left_bottom);
		right_top = fmaxf(right_top, other_right_top);
		centroid = 0.5f * (right_top + left_bottom);
	}

	__device__ __host__ void expand_to_fit_triangle(const float3& vertex0, const float3& vertex1, const float3& vertex2)
	{
		left_bottom = fminf(left_bottom, make_float3(min(vertex0.x, vertex1.x, vertex2.x), min(vertex0.y, vertex1.y, vertex2.y), min(vertex0.z, vertex1.z, vertex2.z)));
		right_top = fmaxf(right_top, make_float3(max(vertex0.x, vertex1.x, vertex2.x), max(vertex0.y, vertex1.y, vertex2.y), max(vertex0.z, vertex1.z, vertex2.z)));
		centroid = 0.5f * (right_top + left_bottom);
	}

	__device__ __host__ void get_bounding_box(const float3& other_left_bottom, const float3& other_right_top)
	{
		left_bottom = other_left_bottom;
		right_top = other_right_top;
		centroid = 0.5f * (right_top + left_bottom);
	}

	__device__ __host__ void get_bounding_box(const float3& vertex0, const float3& vertex1, const float3& vertex2)
	{
		left_bottom = make_float3(min(vertex0.x, vertex1.x, vertex2.x), min(vertex0.y, vertex1.y, vertex2.y), min(vertex0.z, vertex1.z, vertex2.z));
		right_top = make_float3(max(vertex0.x, vertex1.x, vertex2.x), max(vertex0.y, vertex1.y, vertex2.y), max(vertex0.z, vertex1.z, vertex2.z));

		centroid = 0.5f * (right_top + left_bottom);
	}

	__device__ __host__ float get_surface_area()
	{
		return 2.0f * (right_top.x - left_bottom.x) * (right_top.y - left_bottom.y) * (right_top.z - left_bottom.z);
	}

	__device__ __host__ float get_axis_length(int axis) //return- 0:x 1:y 2:z
	{
		if (axis == 0)	return right_top.x - left_bottom.x;
		else if (axis == 1) return right_top.y - left_bottom.y;
		else return right_top.z - left_bottom.z;
	}

	__device__ __host__ bool is_thin_bounding_box()
	{
		return
			right_top.x == left_bottom.x ||
			right_top.y == left_bottom.y ||
			right_top.z == left_bottom.z;
	}

	__host__ __device__ bool intersect_bounding_box(const ray& ray)
	{
		float3 inverse_direction = 1.0f / ray.direction;

		float t_x1 = (left_bottom.x - ray.origin.x) * inverse_direction.x;
		float t_x2 = (right_top.x - ray.origin.x) * inverse_direction.x;

		float t_y1 = (left_bottom.y - ray.origin.y) * inverse_direction.y;
		float t_y2 = (right_top.y - ray.origin.y) * inverse_direction.y;

		float t_z1 = (left_bottom.z - ray.origin.z) * inverse_direction.z;
		float t_z2 = (right_top.z - ray.origin.z) * inverse_direction.z;

		float t_max = fmaxf(fmaxf(fminf(t_x1, t_x2), fminf(t_y1, t_y2)), fminf(t_z1, t_z2));
		float t_min = fminf(fminf(fmaxf(t_x1, t_x2), fmaxf(t_y1, t_y2)), fmaxf(t_z1, t_z2));

		bool is_hit = t_max <= t_min;
		return is_hit && t_min > 0.0f;
	}
};

#endif // !__BOUNDING_BOX__
