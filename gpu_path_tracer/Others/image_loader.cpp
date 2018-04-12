#include "Others\image_loader.h"

bool image_loader::load_png(const std::string& filename, uint& width, uint& height, std::vector<uchar>& pixels)
{
	std::vector<uchar> png;
	lodepng::load_file(png, filename);
	unsigned error = lodepng::decode(pixels, width, height, png);

	if (error)
	{
		std::cout << "[Error]Decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
		return false;
	}

	return true;
}

bool image_loader::load_bmp(const std::string& filename, uint& width, uint& height, std::vector<uchar>& pixels)
{
	std::vector<uchar> bmp;
	lodepng::load_file(bmp, filename);

	if (!decode_bmp(bmp, pixels, width, height))
	{
		return false;
	}

	return true;
}

bool image_loader::load_image(const std::string& filename, uint& width, uint& height, std::vector<uchar>& pixels)
{
	FreeImage_Initialise(TRUE);

	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFileType(filename.c_str());
	if (fif == FIF_UNKNOWN)
	{
		fif = FreeImage_GetFIFFromFilename(filename.c_str());
	}

	FIBITMAP* bitmap = nullptr;
	if ((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif))
	{
		bitmap = FreeImage_Load(fif, filename.c_str(), PNG_DEFAULT);
	}
	else
	{
		std::cout << "[Error]Unsupported image file format." << std::endl;
		FreeImage_DeInitialise();
		return false;
	}

	if (bitmap == nullptr) 
	{
		std::cout << "[Error]Failed to load image file." << std::endl;
		FreeImage_DeInitialise();
		return false;
	}
	
	width = FreeImage_GetWidth(bitmap);
	height = FreeImage_GetHeight(bitmap);

	FIBITMAP* temp_bitmap = bitmap;
	bitmap = FreeImage_ConvertTo24Bits(bitmap);
	FreeImage_Unload(temp_bitmap);

	uint pitch = FreeImage_GetPitch(bitmap);
	pixels.resize(width * height * 4);

	BYTE* pixels_buffer = FreeImage_GetBits(bitmap);
	pixels_buffer += (height - 1) * pitch;

	FREE_IMAGE_TYPE image_type = FreeImage_GetImageType(bitmap);
	uint bits_per_pixel = FreeImage_GetBPP(bitmap);

	uint index = 0;
	for (auto y = 0; y < height; y++)
	{
		BYTE* line_pixels = pixels_buffer;
		for (auto x = 0; x < width; x++)
		{
			pixels[index] = line_pixels[FI_RGBA_RED];	index++;
			pixels[index] = line_pixels[FI_RGBA_GREEN];	index++;
			pixels[index] = line_pixels[FI_RGBA_BLUE];	index++;
			pixels[index] = 255;						index++;
			line_pixels += 3;
		}
		pixels_buffer -= pitch;
	}

	FreeImage_Unload(bitmap);
	FreeImage_DeInitialise();
	return true;
}

bool image_loader::decode_bmp(const std::vector<uchar>& bmp, std::vector<uchar>& image, uint& width, uint& height)
{
	if (bmp.size() < 54)
	{
		//minimum BMP header size
		return false;
	}

	if (bmp[0] != 'B' || bmp[1] != 'M')
	{
		//It's not a BMP file if it doesn't start with marker 'BM'
		return false;
	}

	auto pixel_offset = bmp[10] + 256 * bmp[11];	//where the pixel data starts
													//read width and height from BMP header
	width = bmp[18] + bmp[19] * 256;
	height = bmp[22] + bmp[23] * 256;
	//read number of channels from BMP header
	if (bmp[28] != 24 && bmp[28] != 32)
	{
		//only 24-bit and 32-bit BMPs are supported.
		return false;
	}

	auto num_channels = bmp[28] / 8;

	//The amount of scanline bytes is width of image times channels, with extra bytes added if needed
	//to make it a multiple of 4 bytes.
	auto scanline_bytes = width * num_channels;
	if (scanline_bytes % 4 != 0)
	{
		scanline_bytes = (scanline_bytes / 4) * 4 + 4;
	}

	auto data_size = scanline_bytes * height;
	if (bmp.size() < data_size + pixel_offset)
	{
		//BMP file too small to contain all pixels
		return false;
	}

	image.resize(width * height * 4);

	/*
	There are 3 differences between BMP and the raw image buffer for LodePNG:
	-it's upside down
	-it's in BGR instead of RGB format (or BRGA instead of RGBA)
	-each scanline has padding bytes to make it a multiple of 4 if needed
	The 2D for loop below does all these 3 conversions at once.
	*/
	for (auto y = 0; y < height; y++)
	{
		for (auto x = 0; x < width; x++)
		{
			//pixel start byte position in the BMP
			auto bmp_pos = pixel_offset + (height - y - 1) * scanline_bytes + num_channels * x;
			//pixel start byte position in the new raw image
			auto new_pos = 4 * y * width + 4 * x;
			if (num_channels == 3)
			{
				image[new_pos + 0] = bmp[bmp_pos + 2]; //R
				image[new_pos + 1] = bmp[bmp_pos + 1]; //G
				image[new_pos + 2] = bmp[bmp_pos + 0]; //B
				image[new_pos + 3] = 255;            //A
			}
			else
			{
				image[new_pos + 0] = bmp[bmp_pos + 3]; //R
				image[new_pos + 1] = bmp[bmp_pos + 2]; //G
				image[new_pos + 2] = bmp[bmp_pos + 1]; //B
				image[new_pos + 3] = bmp[bmp_pos + 0]; //A
			}
		}
	}
	return true;
}
