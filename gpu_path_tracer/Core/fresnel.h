#pragma once

#ifndef __FRESNEL__
#define __FRESNEL__

struct fresnel
{
	float reflection_index;
	float refractive_index;

	__host__ __device__ static fresnel get_fresnel_dielectrics(
		const float3& normal,					
		const float3& in_direction,				
		float in_refraction_index,				
		float out_refraction_index,				
		const float3& reflection_direction,		
		const float3& refraction_direction		
	)
	{
		//using real fresnel equation
		fresnel fresnel;

		float cos_theta_in = dot(normal, in_direction * -1.0f);
		float cos_theta_out = dot(-1.0f * normal, refraction_direction);

		//total internal reflection 
		if (in_refraction_index > out_refraction_index && acosf(cos_theta_in) >= asinf(out_refraction_index / in_refraction_index))
		{
			fresnel.reflection_index = 1.0f;
			fresnel.refractive_index = 0.0f;
			return fresnel;
		}

		if (length(refraction_direction) <= 0.000005f || cos_theta_out < 0)
		{
			fresnel.reflection_index = 1.0f;
			fresnel.refractive_index = 0.0f;
			return fresnel;
		}

		float reflection_coef_s_polarized = pow((in_refraction_index * cos_theta_in - out_refraction_index * cos_theta_out) / (in_refraction_index * cos_theta_in + out_refraction_index * cos_theta_out), 2.0f);
		float reflection_coef_p_polarized = pow((in_refraction_index * cos_theta_out - out_refraction_index * cos_theta_in) / (in_refraction_index * cos_theta_out + out_refraction_index * cos_theta_in), 2.0f);

		float reflection_coef_unpolarized = (reflection_coef_s_polarized + reflection_coef_p_polarized) / 2.0f;

		fresnel.reflection_index = reflection_coef_unpolarized;
		fresnel.refractive_index = 1.0f - fresnel.reflection_index;
		return fresnel;
	}

	__host__ __device__ static fresnel get_fresnel_conductors(
		const float3& normal,					
		const float3& in_direction,				
		float refraction_index,					
		float extinction_coefficient		
	)
	{
		//using real fresnel equation
		fresnel fresnel;

		float cos_theta_in = dot(normal, in_direction * -1.0f);
		float refraction_index_square = refraction_index * refraction_index;
		float extinction_coefficient_square = extinction_coefficient * extinction_coefficient;
		float refraction_extinction_square_add = refraction_index_square + extinction_coefficient_square;
		float cos_theta_in_square = cos_theta_in * cos_theta_in;
		float two_refraction_cos_theta_in = 2 * refraction_index * cos_theta_in;

		float reflection_coef_s_polarized = (refraction_extinction_square_add * cos_theta_in_square - two_refraction_cos_theta_in + 1.0f) / (refraction_extinction_square_add * cos_theta_in_square + two_refraction_cos_theta_in + 1.0f);
		float reflection_coef_p_polarized = (refraction_extinction_square_add - two_refraction_cos_theta_in + cos_theta_in_square) / (refraction_extinction_square_add + two_refraction_cos_theta_in + cos_theta_in_square);

		float reflection_coef_unpolarized = (reflection_coef_s_polarized + reflection_coef_p_polarized) / 2.0f;

		fresnel.reflection_index = reflection_coef_unpolarized;
		fresnel.refractive_index = 1.0f - fresnel.reflection_index;
		return fresnel;
	}
};

#endif // !__FRESNEL
