#pragma once

#ifndef __IMAGE_LOADER__
#define __IMAGE_LOADER__

#include "lib\free_image\FreeImage.h"
#include "lib\lodepng\lodepng.h"
#include "Math\basic_math.hpp"
#include "Math\cuda_math.hpp"
#include <string>
#include <iostream>

class image_loader
{
	//4 bytes per pixel, ordered RGBA RGBA
public:

	static bool load_png(const std::string& filename, uint& width, uint& height, std::vector<uchar>& pixels);
	static bool load_bmp(const std::string& filename, uint& width, uint& height, std::vector<uchar>& pixels);
	static bool load_image(const std::string& filename, uint& width, uint& height, std::vector<uchar>& pixels);

private:

	static bool decode_bmp(
		const std::vector<uchar>& bmp,	
		std::vector<uchar>& image,		
		uint& width,					
		uint& height					
	);
};

#endif // !__IMAGE_LOADER__
