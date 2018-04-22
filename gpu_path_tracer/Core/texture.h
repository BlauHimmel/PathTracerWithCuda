#pragma once

#ifndef __TEXTURE__
#define __TEXTURE__

#include "Math\basic_math.hpp"
#include "Math\cuda_math.hpp"

struct texture_wrapper
{
	int width;
	int height;
	uchar* pixels;

	__device__ float3 sample_texture(const float2& uv,	bool use_bilinear)
	{

		float2 actual_uv;
		actual_uv.x = uv.x - floorf(uv.x);
		actual_uv.y = uv.y - floorf(uv.y);

		if (use_bilinear)
		{
			float x_image_real = actual_uv.x * (float)(width - 1);
			float y_image_real = (1.0f - actual_uv.y) * (float)(height - 1);

			int floor_x_image = (int)clamp(floorf(x_image_real), 0.0f, (float)(width - 1));
			int ceil_x_image = (int)clamp(ceilf(x_image_real), 0.0f, (float)(width - 1));
			int floor_y_image = (int)clamp(floorf(y_image_real), 0.0f, (float)(height - 1));
			int ceil_y_image = (int)clamp(ceilf(y_image_real), 0.0f, (float)(height - 1));

			//0:left bottm	1:right bottom	2:left top	3:right top
			int x_images[4] = { floor_x_image, ceil_x_image, floor_x_image, ceil_x_image };
			int y_images[4] = { floor_y_image, floor_y_image, ceil_y_image, ceil_y_image };

			float left_right_t = x_image_real - floorf(x_image_real);
			float bottom_top_t = y_image_real - floorf(y_image_real);

			float3 sample_colors[4] = {
				make_float3(
					pixels[(y_images[0] * width + x_images[0]) * 4 + 0] / 255.0f,
					pixels[(y_images[0] * width + x_images[0]) * 4 + 1] / 255.0f,
					pixels[(y_images[0] * width + x_images[0]) * 4 + 2] / 255.0f
				),
				make_float3(
					pixels[(y_images[1] * width + x_images[1]) * 4 + 0] / 255.0f,
					pixels[(y_images[1] * width + x_images[1]) * 4 + 1] / 255.0f,
					pixels[(y_images[1] * width + x_images[1]) * 4 + 2] / 255.0f
				),
				make_float3(
					pixels[(y_images[2] * width + x_images[2]) * 4 + 0] / 255.0f,
					pixels[(y_images[2] * width + x_images[2]) * 4 + 1] / 255.0f,
					pixels[(y_images[2] * width + x_images[2]) * 4 + 2] / 255.0f
				),
				make_float3(
					pixels[(y_images[3] * width + x_images[3]) * 4 + 0] / 255.0f,
					pixels[(y_images[3] * width + x_images[3]) * 4 + 1] / 255.0f,
					pixels[(y_images[3] * width + x_images[3]) * 4 + 2] / 255.0f
				),
			};

			return lerp(
				lerp(sample_colors[0], sample_colors[1], left_right_t),
				lerp(sample_colors[2], sample_colors[3], left_right_t),
				bottom_top_t
			);
		}
		else
		{
			int x_image = (int)clamp((actual_uv.x * (float)(width - 1)), 0.0f, (float)(width - 1));
			int y_image = (int)clamp(((1.0f - actual_uv.y) * (float)(height - 1)), 0.0f, (float)(height - 1));

			return make_float3(
				pixels[(y_image * width + x_image) * 4 + 0] / 255.0f,
				pixels[(y_image * width + x_image) * 4 + 1] / 255.0f,
				pixels[(y_image * width + x_image) * 4 + 2] / 255.0f
			);
		}
	}
};

#endif // !__TEXTURE__