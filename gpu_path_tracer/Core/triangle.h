#pragma once

#ifndef __TRIANGLE__
#define __TRIANGLE__

#include <cuda_runtime.h>
#include "Math\cuda_math.hpp"
#include "Core\material.h"
#include "Core\ray.h"

struct triangle
{
	float3 vertex0;
	float3 vertex1;
	float3 vertex2;

	float3 normal0;
	float3 normal1;
	float3 normal2;

	float2 uv0;
	float2 uv1;
	float2 uv2;

	material* mat;

	__device__ bool intersect(
		const ray& ray,					
		float& hit_t,					
		float& hit_t1,					
		float& hit_t2					
	)
	{
		float3 edge1 = vertex1 - vertex0;
		float3 edge2 = vertex2 - vertex0;

		float3 p_vec = cross(ray.direction, edge2);
		float det = dot(edge1, p_vec);

		if (det == 0.0f)
		{
			return false;
		}

		float inverse_det = 1.0f / det;
		float3 t_vec = ray.origin - vertex0;
		float3 q_vec = cross(t_vec, edge1);

		float t1 = dot(t_vec, p_vec) * inverse_det;
		float t2 = dot(ray.direction, q_vec) * inverse_det;
		float t = dot(edge2, q_vec) * inverse_det;

		if (t1 >= 0.0f && t2 >= 0.0f && t1 + t2 <= 1.0f)
		{
			hit_t = t;
			hit_t1 = t1;
			hit_t2 = t2;
			return true;
		}

		return false;
	}
};

#endif // !__TRIANGLE__