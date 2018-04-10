#include "Core\image.h"

color image::get_pixel(int x, int y) const
{
	return pixels_device[y * width + x];
}

void image::set_pixel(int x, int y, const color& color)
{
	pixels_device[y * width + x].x = color.x;
	pixels_device[y * width + x].y = color.y;
	pixels_device[y * width + x].z = color.z;
}

image* create_image(int width, int height)
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

void release_image(image* image)
{
	CUDA_CALL(cudaFree(image->pixels_device));
	CUDA_CALL(cudaFree(image->pixels_256_device));
	SAFE_DELETE(image);
}

void reset_image(image* image)
{
	image->pass_counter = 0;
	image->start_clock = clock();
}

float get_elapsed_second(image* image)
{
	return (static_cast<float>(clock()) - static_cast<float>(image->start_clock)) / static_cast<float>(CLOCKS_PER_SEC);
}

float get_fps(image* image)
{
	return static_cast<float>(image->pass_counter + 1.0f) / get_elapsed_second(image);
}

