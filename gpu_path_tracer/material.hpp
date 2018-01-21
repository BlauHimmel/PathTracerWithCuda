#pragma once

#ifndef __MATERIAL__
#define __MATERIAL__

#include <cuda_runtime.h>
#include "basic_math.h"

struct scattering
{
	color absorption_coefficient;
	float reduced_scattering_coefficient;
};

struct medium
{
	float refraction_index;
	scattering scattering;
};

struct material
{
	color diffuse_color;
	color emission_color;
	color specular_color;
	bool is_transparent;
	
	medium medium;
};

inline material get_default_material()
{
	material mat;
	mat.diffuse_color = make_float3(0.0f, 0.0f, 0.0f);
	mat.emission_color = make_float3(0.0f, 0.0f, 0.0f);
	mat.specular_color = make_float3(0.0f, 0.0f, 0.0f);
	mat.is_transparent = false;
	mat.medium.refraction_index = AIR_REFRACTION_INDEX;
	mat.medium.scattering.absorption_coefficient = make_float3(0.0f, 0.0f, 0.0f);
	mat.medium.scattering.reduced_scattering_coefficient = 0.0f;
	return mat;
}


#endif // !__MATERIAL__
