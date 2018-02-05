#pragma once

#ifndef __SPHERE__
#define __SPHERE__

#include <cuda_runtime.h>
#include "basic_math.h"
#include "material.hpp"

struct sphere
{
	float3 center;
	float radius;

	material mat;
};

#endif // !__RAY__
