#include "basic_math.h"

namespace math
{
	//Math function
	unsigned int mod(int x, int y)
	{
		int result = x % y;
		if (result < 0)
		{
			result += y;
		}
		return result;
	}

	float mod(float x, float y)
	{
		return x - y * std::floor(x / y);
	}

	float radians_to_degrees(float radians)
	{
		return radians * 180.0f / PI;
	}

	float degrees_to_radians(float degrees)
	{
		return degrees / 180.0f * PI;
	}

	float average(float n1, float n2)
	{
		return (n1 + n2) * 0.5f;
	}

	float round(float n)
	{
		return std::floor(n + 0.5f);
	}

	float square(float n)
	{
		return n * n;
	}

	float log2(float n)
	{
		return std::log(n) / std::log(2.0f);
	}

	bool is_nan(float n)
	{
		return n != n;
	}

	float clamp(float n, float min, float max)
	{
		return n < min ? min : (n > max ? max : n);
	}
	float repeat(float n, float modulus)
	{
		return n - modulus * std::floor(n / modulus);
	}

	int sign(float n)
	{
		return n >= 0 ? 1 : -1;
	}

	int positive_characteristic(float n)
	{
		return n > 0 ? 1 : 0;
	}

	void swap(float& a, float& b)
	{
		float temp = a;
		a = b;
		b = temp;
	}

	//Utility Function
	color gamma_correct(const color& primitive_color)
	{
		float gamma = 2.2f;
		float inverse_gamma = 1.0f / gamma;
		color corrected_color;
		corrected_color.x = std::powf(primitive_color.x, inverse_gamma);
		corrected_color.y = std::powf(primitive_color.y, inverse_gamma);
		corrected_color.z = std::powf(primitive_color.z, inverse_gamma);
		return corrected_color;
	}

	uchar3 float_to_8bit(const color& primitive_color, bool is_corrected)
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