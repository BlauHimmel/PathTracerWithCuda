#pragma once

#ifndef __BOUNDING_BOX__
#define __BOUNDING_BOX__

#include "Math\cuda_math.hpp"

struct bounding_box
{
	float3 left_bottom;
	float3 right_top;
	float3 centroid;

	bounding_box();
	bounding_box(const float3& left_bottom, const float3& right_top);
	void expand_to_fit_box(const float3& other_left_bottom, const float3& other_right_top);
	void expand_to_fit_triangle(const float3& vertex0, const float3& vertex1, const float3& vertex2);
	void get_bounding_box(const float3& other_left_bottom, const float3& other_right_top);
	void get_bounding_box(const float3& vertex0, const float3& vertex1, const float3& vertex2);
	float get_surface_area();
	float get_axis_length(int axis); //return- 0:x 1:y 2:z
	bool is_thin_bounding_box();
};

#endif // !__BOUNDING_BOX__
