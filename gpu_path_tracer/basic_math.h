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
	inline unsigned int mod(int x, int y)
	{
		int result = x % y;
		if (result < 0)
		{
			result += y;
		}
		return result;
	}

	inline float mod(float x, float y)
	{
		return x - y * std::floor(x / y);
	}

	inline float radians_to_degrees(float radians)
	{
		return radians * 180.0f / PI;
	}

	inline float degrees_to_radians(float degrees)
	{
		return degrees / 180.0f * PI;
	}

	inline float average(float n1, float n2)
	{
		return (n1 + n2) * 0.5f;
	}

	inline float round(float n)
	{
		return std::floor(n + 0.5f);
	}

	inline float square(float n)
	{
		return n * n;
	}

	inline float log2(float n)
	{
		return std::log(n) / std::log(2.0f);
	}

	inline bool is_nan(float n)
	{
		return n != n;
	}

	inline float clamp(float n, float min, float max)
	{
		return n < min ? min : (n > max ? max : n);
	}

	inline float repeat(float n, float modulus)
	{
		return n - modulus * std::floor(n / modulus);
	}

	inline int sign(float n)
	{
		return n >= 0 ? 1 : -1;
	}

	inline int positive_characteristic(float n)
	{
		return n > 0 ? 1 : 0;
	}

	inline void swap(float& a, float& b)
	{
		float temp = a;
		a = b;
		b = temp;
	}

	//Utility Function
	inline color gamma_correct(const color& primitive_color)
	{
		float gamma = 2.2f;
		float inverse_gamma = 1.0f / gamma;
		color corrected_color; 
		corrected_color.x = std::expf(inverse_gamma * std::logf(primitive_color.x));
		corrected_color.y = std::expf(inverse_gamma * std::logf(primitive_color.y));
		corrected_color.z = std::expf(inverse_gamma * std::logf(primitive_color.z));
		return corrected_color;
	}

	inline uchar3 float_to_8bit(const color& primitive_color, bool is_corrected = false)
	{
		color color;
		if (!is_corrected)
		{
			color = gamma_correct(primitive_color);
		}
		else
		{
			color = primitive_color;
		}

		float x = clamp(color.x * 255.0f, 0.0f, 255.0f);
		float y = clamp(color.y * 255.0f, 0.0f, 255.0f);
		float z = clamp(color.z * 255.0f, 0.0f, 255.0f);
		uchar3 color_8bit;
		color_8bit.x = static_cast<uchar>(x);
		color_8bit.y = static_cast<uchar>(y);
		color_8bit.z = static_cast<uchar>(z);
		return color_8bit;
	}
}

#endif // !__MATH__
