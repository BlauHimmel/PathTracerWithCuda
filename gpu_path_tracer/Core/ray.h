#pragma once

#ifndef __RAY__
#define __RAY__

#include <cuda_runtime.h>
#include "Math\cuda_math.hpp"

struct ray
{
	float3 origin;
	float3 direction;

	__host__ __device__ float3 point_on_ray(float t)
	{
		return origin + direction * t;
	}
};

#endif // !__RAY__
