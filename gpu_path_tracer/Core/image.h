#pragma once

#ifndef __IMAGE__
#define __IMAGE__

#include "Math\basic_math.hpp"
#include "Others\utilities.hpp"
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

image* create_image(int width, int height);

void release_image(image* image);

void reset_image(image* image);

float get_elapsed_second(image* image);

float get_fps(image* image);


#endif // !__IMAGE__
