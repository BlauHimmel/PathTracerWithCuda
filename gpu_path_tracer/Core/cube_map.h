#pragma once

#ifndef __CUBE_MAP__
#define __CUBE_MAP__

#include "Math\basic_math.hpp"
#include "Math\cuda_math.hpp"

struct cube_map
{
	uchar* m_x_positive_map;
	uchar* m_x_negative_map;
	uchar* m_y_positive_map;
	uchar* m_y_negative_map;
	uchar* m_z_positive_map;
	uchar* m_z_negative_map;

	int length;

	__host__ __device__ float3 get_background_color(
		const float3& direction,
		bool use_sky_box,
		bool use_sky,
		bool use_bilinear
	)
	{
		if (use_sky_box)
		{
			if (use_bilinear)
			{
				float u, v;
				int index;
				convert_xyz_to_cube_uv(direction.x, direction.y, direction.z, index, u, v);
				float x_image_real = u * (float)(length - 1);
				float y_image_real = (1.0f - v) * (float)(length - 1);

				int floor_x_image = (int)clamp(floorf(x_image_real), 0.0f, (float)(length - 1));
				int ceil_x_image = (int)clamp(ceilf(x_image_real), 0.0f, (float)(length - 1));
				int floor_y_image = (int)clamp(floorf(y_image_real), 0.0f, (float)(length - 1));
				int ceil_y_image = (int)clamp(ceilf(y_image_real), 0.0f, (float)(length - 1));

				//0:left bottm	1:right bottom	2:left top	3:right top
				int x_images[4] = { floor_x_image, ceil_x_image, floor_x_image, ceil_x_image };
				int y_images[4] = { floor_y_image, floor_y_image, ceil_y_image, ceil_y_image };

				float left_right_t = x_image_real - floorf(x_image_real);
				float bottom_top_t = y_image_real - floorf(y_image_real);

				uchar* pixels = nullptr;
				if (index == 0) pixels = m_x_positive_map;
				else if (index == 1) pixels = m_x_negative_map;
				else if (index == 2) pixels = m_y_positive_map;
				else if (index == 3) pixels = m_y_negative_map;
				else if (index == 4) pixels = m_z_positive_map;
				else if (index == 5) pixels = m_z_negative_map;

				float3 sample_colors[4] = {
					make_float3(
						pixels[(y_images[0] * length + x_images[0]) * 4 + 0] / 255.0f,
						pixels[(y_images[0] * length + x_images[0]) * 4 + 1] / 255.0f,
						pixels[(y_images[0] * length + x_images[0]) * 4 + 2] / 255.0f
					),
					make_float3(
						pixels[(y_images[1] * length + x_images[1]) * 4 + 0] / 255.0f,
						pixels[(y_images[1] * length + x_images[1]) * 4 + 1] / 255.0f,
						pixels[(y_images[1] * length + x_images[1]) * 4 + 2] / 255.0f
					),
					make_float3(
						pixels[(y_images[2] * length + x_images[2]) * 4 + 0] / 255.0f,
						pixels[(y_images[2] * length + x_images[2]) * 4 + 1] / 255.0f,
						pixels[(y_images[2] * length + x_images[2]) * 4 + 2] / 255.0f
					),
					make_float3(
						pixels[(y_images[3] * length + x_images[3]) * 4 + 0] / 255.0f,
						pixels[(y_images[3] * length + x_images[3]) * 4 + 1] / 255.0f,
						pixels[(y_images[3] * length + x_images[3]) * 4 + 2] / 255.0f
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
				float u, v;
				int index;
				convert_xyz_to_cube_uv(direction.x, direction.y, direction.z, index, u, v);
				int x_image = (int)clamp((u * (float)(length - 1)), 0.0f, (float)(length - 1));
				int y_image = (int)clamp(((1.0f - v) * (float)(length - 1)), 0.0f, (float)(length - 1));

				uchar* pixels = nullptr;
				if (index == 0) pixels = m_x_positive_map;
				else if (index == 1) pixels = m_x_negative_map;
				else if (index == 2) pixels = m_y_positive_map;
				else if (index == 3) pixels = m_y_negative_map;
				else if (index == 4) pixels = m_z_positive_map;
				else if (index == 5) pixels = m_z_negative_map;

				return make_float3(
					pixels[(y_image * length + x_image) * 4 + 0] / 255.0f,
					pixels[(y_image * length + x_image) * 4 + 1] / 255.0f,
					pixels[(y_image * length + x_image) * 4 + 2] / 255.0f
				);
			}
		}

		if (use_sky)
		{
			float t = (dot(direction, make_float3(-0.41f, 0.41f, -0.82f)) + 1.0f) / 2.0f;
			float3 a = make_float3(0.15f, 0.3f, 0.5f);
			float3 b = make_float3(1.0f, 1.0f, 1.0f);
			return ((1.0f - t) * a + t * b) * 1.0f;
		}

		return make_float3(0.0f, 0.0f, 0.0f);
	}

};

#endif // !__CUBE_MAP__