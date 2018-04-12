#pragma once

#ifndef __MATERIAL__
#define __MATERIAL__

#include <cuda_runtime.h>
#include "Math\basic_math.hpp"
#include "Core\texture.h"

#define AIR_REFRACTION_INDEX 1.000293f
#define AIR_ABSORPTION_COEFFICIENT make_float3(0.0f, 0.0f, 0.0f)
#define AIR_REDUCED_SCATTERING_COEFFICIENT make_float3(0.0f, 0.0f, 0.0f)

//for conductors(e.g. metals like aluminum or copper) extinction is set to be greater than zero, otherwise it is considered as dielectrics
//note : 
//1> metal do not have diffuse color, and its specular color can be white or others, but for dielectrics the specular color can only be 
//set to white(i.e. its r,g,b channels are equal).

struct scattering
{
	color absorption_coefficient;
	color reduced_scattering_coefficient;
};

struct medium
{
	float refraction_index;
	float extinction_coefficient;
	scattering scattering;
};

struct material
{
	color diffuse_color;
	color emission_color;
	color specular_color;
	bool is_transparent;
	float roughness;
	
	medium medium;

	int diffuse_texture_id;
};

material get_default_material();

material* new_default_material();

material* copy_material(const material& mat);

namespace material_data
{
	class metal
	{
	public:
		static material titanium();

		static material chromium();

		static material iron();

		static material nickel();

		static material platinum();

		static material copper();

		static material palladium();

		static material zinc();

		static material gold();

		static material aluminum();

		static material silver();
	};

	class dielectric
	{
	public:
		static material glass();

		static material green_glass();
	
		static material diamond();

		static material red();

		static material green();

		static material orange();

		static material purple();

		static material blue();

		static material wall_blue();

		static material marble();

		static material something_blue();

		static material something_red();

		static material light();
	};
}

#endif // !__MATERIAL__
