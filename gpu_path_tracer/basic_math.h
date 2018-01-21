#pragma once

#ifndef __MATH__
#define __MATH__

#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>

using color = float3;
using uchar = unsigned char;

#define PI						3.1415926535897f
#define INVERSE_PI				0.3183098861837f
#define TWO_PI					6.2831853071795f
#define FOUR_PI					12.566370614359f
#define HALF_PI					1.5707963267949f
#define QUARTER_PI				0.0795774715459f
#define E						2.7182818284590f
#define SQRT_ONE_THIRD			0.5773502691896f
#define AIR_REFRACTION_INDEX	1.000293f

namespace math
{
	//Math function
	extern unsigned int mod(int x, int y);

	extern float mod(float x, float y);

	extern float radians_to_degrees(float radians);

	extern float degrees_to_radians(float degrees);

	extern float average(float n1, float n2);

	extern float round(float n);

	extern float square(float n);

	extern float log2(float n);

	extern bool is_nan(float n);

	extern float clamp(float n, float min, float max);

	extern float repeat(float n, float modulus);

	extern int sign(float n);

	extern int positive_characteristic(float n);

	extern void swap(float& a, float& b);

	//Utility Function
	extern color gamma_correct(const color& primitive_color);

	extern uchar3 float_to_8bit(const color& primitive_color, bool is_corrected = false);
}

#endif // !__MATH__
