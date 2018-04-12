#pragma once

#ifndef __TRIANGLE__
#define __TRIANGLE__

#include <cuda_runtime.h>
#include "Math\cuda_math.hpp"
#include "Core\material.h"

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
};

#endif // !__TRIANGLE__