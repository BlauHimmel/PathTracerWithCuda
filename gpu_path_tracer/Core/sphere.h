#pragma once

#ifndef __SPHERE__
#define __SPHERE__

#include <cuda_runtime.h>
#include "Core\material.h"
#include "Core\ray.h"
#include "Math\cuda_math.hpp"

struct sphere
{
	float3 center;
	float radius;

	material mat;

	__host__ __device__  bool intersect(
		const ray& ray,			//in
		float3& hit_point,		//out
		float3& hit_normal,		//out
		float& hit_t			//out
	)
	{
		float3 op = center - ray.origin;
		float b = dot(op, ray.direction);
		float delta = b * b - dot(op, op) + radius * radius;

		if (delta < 0)
		{
			return false;
		}

		float delta_root = sqrt(delta);
		float t1 = b - delta_root;
		float t2 = b + delta_root;

		if (t1 < 0 && t2 < 0)
		{
			return false;
		}

		if (t1 > 0 && t2 > 0)
		{
			hit_t = fminf(t1, t2);
		}
		else
		{
			hit_t = fmaxf(t1, t2);
		}

		hit_point = ray.origin + ray.direction * hit_t;
		hit_normal = normalize(hit_point - center);
		return true;
	}
};

#endif // !__SPHERE__
