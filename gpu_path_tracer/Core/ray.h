#pragma once

#ifndef __RAY__
#define __RAY__

#include <cuda_runtime.h>

struct ray
{
	float3 origin;
	float3 direction;
};

#endif // !__RAY__
