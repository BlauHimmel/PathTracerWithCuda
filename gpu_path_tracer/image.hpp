#pragma once

#ifndef __IMAGE__
#define __IMAGE__

#include "basic_math.hpp"
#include "utilities.hpp"
#include <ctime>

struct image
{
	int width;
	int height;
	int pixel_count;
	int pass_counter;
	color* pixels_device;
	color256* pixels_256_device;

	clock_t start_clock;

	color get_pixel(int x, int y) const;
	void set_pixel(int x, int y, const color& color);
};

inline image* create_image(int width, int height)
{
	image* img = new image();
	img->width = width;
	img->height = height;
	img->pixel_count = width * height;
	img->pass_counter = 0;
	img->start_clock = clock();
	CUDA_CALL(cudaMallocManaged((void**)&img->pixels_device, img->pixel_count * sizeof(color)));
	CUDA_CALL(cudaMallocManaged((void**)&img->pixels_256_device, img->pixel_count * sizeof(color256)));

	return img;
}

inline void release_image(image* image)
{
	CUDA_CALL(cudaFree(image->pixels_device));
	CUDA_CALL(cudaFree(image->pixels_256_device));
	SAFE_DELETE(image);
}

inline void reset_image(image* image)
{
	image->pass_counter = 0;
	image->start_clock = clock();
}

inline float get_elapsed_second(image* image)
{
	return (static_cast<float>(clock()) - static_cast<float>(image->start_clock)) / static_cast<float>(CLOCKS_PER_SEC);
}

inline float get_fps(image* image)
{
	return static_cast<float>(image->pass_counter + 1.0f) / get_elapsed_second(image);
}

inline color image::get_pixel(int x, int y) const
{
	return pixels_device[y * width + x];
}

inline void image::set_pixel(int x, int y, const color& color)
{
	pixels_device[y * width + x].x = color.x;
	pixels_device[y * width + x].y = color.y;
	pixels_device[y * width + x].z = color.z;
}

#endif // !__IMAGE__
