#pragma once

#ifndef __TRIANGLE__
#define __TRIANGLE__

#include <cuda_runtime.h>
#include "material.hpp"

struct triangle
{
	float3 vertex0;
	float3 vertex1;
	float3 vertex2;
	float3 normal;

	material* mat;
};

#endif // !__TRIANGLE__