#pragma once

#ifndef __IMAGE__
#define __IMAGE__

#include "basic_math.h"
#include "utilities.hpp"
#include <ctime>

struct image
{
	int width;
	int height;
	int pixel_count;
	int pass_counter;
	color* pixels;

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
	img->pixels = new color[img->pixel_count];
	img->pass_counter = 0;
	img->start_clock = clock();
	return img;
}

inline void release_image(image* image)
{
	SAFE_DELETE_ARRAY(image->pixels);
	SAFE_DELETE(image);
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
	return pixels[y * width + x];
}

inline void image::set_pixel(int x, int y, const color& color)
{
	pixels[y * width + x].x = color.x;
	pixels[y * width + x].y = color.y;
	pixels[y * width + x].z = color.z;
}

#endif // !__IMAGE__
