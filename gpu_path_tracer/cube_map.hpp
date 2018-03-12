#pragma once

#ifndef __CUBE_MAP__
#define __CUBE_MAP__

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include "basic_math.h"
#include "lib\lodepng\lodepng.h"
#include "utilities.hpp"

struct cube_map
{
	uchar* m_x_positive_map;
	uchar* m_x_negative_map;
	uchar* m_y_positive_map;
	uchar* m_y_negative_map;
	uchar* m_z_positive_map;
	uchar* m_z_negative_map;

	int length;
};

class cube_map_loader
{
private:
	//============================================
	std::vector<uchar> m_x_positive_map;
	std::vector<uchar> m_x_negative_map;
	std::vector<uchar> m_y_positive_map;
	std::vector<uchar> m_y_negative_map;
	std::vector<uchar> m_z_positive_map;
	std::vector<uchar> m_z_negative_map;
	//============================================

	cube_map* m_cube_map_device = nullptr;
	
	bool m_is_loaded = false;
	int m_width, m_height;

public:
	bool load_data(
		const std::string& filename_x_positive,
		const std::string& filename_x_negative,
		const std::string& filename_y_positive,
		const std::string& filename_y_negative,
		const std::string& filename_z_positive,
		const std::string& filename_z_negative
	);
	void unload_data();

	cube_map* get_cube_map_device() const;

	bool create_cube_device_data();
	void release_cube_device_data();

private:
	bool decode_bmp(
		const std::vector<uchar>& bmp,	//in
		std::vector<uchar>& image,		//out
		int& width,						//out
		int& height						//out
	);

};

inline bool cube_map_loader::load_data(
	const std::string& filename_x_positive, 
	const std::string& filename_x_negative, 
	const std::string& filename_y_positive, 
	const std::string& filename_y_negative, 
	const std::string& filename_z_positive, 
	const std::string& filename_z_negative
)
{
	std::vector<uchar> bmp_buffer;
	int width, height;

	std::cout << "[Info]Loading file " << filename_x_positive << "...." << std::endl;
	lodepng::load_file(bmp_buffer, filename_x_positive);
	if (!decode_bmp(bmp_buffer, m_x_positive_map, m_width, m_height) || m_width != m_height)
	{
		m_x_positive_map.clear();
		m_x_positive_map.shrink_to_fit();
		return false;
	}
	std::cout << "[Info]Load file " << filename_x_positive << " succeeded." << std::endl;


	std::cout << "[Info]Loading file " << filename_x_negative << "...." << std::endl;
	lodepng::load_file(bmp_buffer, filename_x_negative);
	if (!decode_bmp(bmp_buffer, m_x_negative_map, width, height) || width != height || width != m_width || height != m_height)
	{
		m_x_positive_map.clear();
		m_x_negative_map.clear();
		m_x_positive_map.shrink_to_fit();
		m_x_negative_map.shrink_to_fit();
		return false;
	}
	std::cout << "[Info]Load file " << filename_x_negative << " succeeded." << std::endl;

	std::cout << "[Info]Loading file " << filename_y_positive << "...." << std::endl;
	lodepng::load_file(bmp_buffer, filename_y_positive);
	if (!decode_bmp(bmp_buffer, m_y_positive_map, width, height) || width != height || width != m_width || height != m_height)
	{
		m_x_positive_map.clear();
		m_x_negative_map.clear();
		m_y_positive_map.clear();
		m_x_positive_map.shrink_to_fit();
		m_x_negative_map.shrink_to_fit();
		m_y_positive_map.shrink_to_fit();
		return false;
	}
	std::cout << "[Info]Load file " << filename_y_positive << " succeeded." << std::endl;

	std::cout << "[Info]Loading file " << filename_y_negative << "...." << std::endl;
	lodepng::load_file(bmp_buffer, filename_y_negative);
	if (!decode_bmp(bmp_buffer, m_y_negative_map, width, height) || width != height || width != m_width || height != m_height)
	{
		m_x_positive_map.clear();
		m_x_negative_map.clear();
		m_y_positive_map.clear();
		m_y_negative_map.clear();
		m_x_positive_map.shrink_to_fit();
		m_x_negative_map.shrink_to_fit();
		m_y_positive_map.shrink_to_fit();
		m_y_negative_map.shrink_to_fit();
		return false;
	}
	std::cout << "[Info]Load file " << filename_y_negative << " succeeded." << std::endl;

	std::cout << "[Info]Loading file " << filename_z_positive << "...." << std::endl;
	lodepng::load_file(bmp_buffer, filename_z_positive);
	if (!decode_bmp(bmp_buffer, m_z_positive_map, width, height) || width != height || width != m_width || height != m_height)
	{
		m_x_positive_map.clear();
		m_x_negative_map.clear();
		m_y_positive_map.clear();
		m_y_negative_map.clear();
		m_z_positive_map.clear();
		m_x_positive_map.shrink_to_fit();
		m_x_negative_map.shrink_to_fit();
		m_y_positive_map.shrink_to_fit();
		m_y_negative_map.shrink_to_fit();
		m_z_positive_map.shrink_to_fit();
		return false;
	}
	std::cout << "[Info]Load file " << filename_z_positive << " succeeded." << std::endl;

	std::cout << "[Info]Loading file " << filename_z_negative << "...." << std::endl;
	lodepng::load_file(bmp_buffer, filename_z_negative);
	if (!decode_bmp(bmp_buffer, m_z_negative_map, width, height) || width != height || width != m_width || height != m_height)
	{
		m_x_positive_map.clear();
		m_x_negative_map.clear();
		m_y_positive_map.clear();
		m_y_negative_map.clear();
		m_z_positive_map.clear();
		m_z_negative_map.clear();
		m_x_positive_map.shrink_to_fit();
		m_x_negative_map.shrink_to_fit();
		m_y_positive_map.shrink_to_fit();
		m_y_negative_map.shrink_to_fit();
		m_z_positive_map.shrink_to_fit();
		m_z_negative_map.shrink_to_fit();
		return false;
	}
	std::cout << "[Info]Load file " << filename_z_negative << " succeeded." << std::endl;

	m_is_loaded = true;
	return true;
}

inline void cube_map_loader::unload_data()
{
	m_x_positive_map.clear();
	m_x_negative_map.clear();
	m_y_positive_map.clear();
	m_y_negative_map.clear();
	m_z_positive_map.clear();
	m_z_negative_map.clear();
	m_x_positive_map.shrink_to_fit();
	m_x_negative_map.shrink_to_fit();
	m_y_positive_map.shrink_to_fit();
	m_y_negative_map.shrink_to_fit();
	m_z_positive_map.shrink_to_fit();
	m_z_negative_map.shrink_to_fit();

	m_is_loaded = false;
}

inline cube_map * cube_map_loader::get_cube_map_device() const
{
	return m_cube_map_device;
}

inline bool cube_map_loader::create_cube_device_data()
{
	if (!m_is_loaded)
	{
		return false;
	}

	CUDA_CALL(cudaMallocManaged((void**)&m_cube_map_device, sizeof(cube_map)));

	CUDA_CALL(cudaMallocManaged((void**)&(m_cube_map_device->m_x_positive_map), m_width * m_height * 4 * sizeof(uchar)));
	CUDA_CALL(cudaMallocManaged((void**)&(m_cube_map_device->m_x_negative_map), m_width * m_height * 4 * sizeof(uchar)));
	CUDA_CALL(cudaMallocManaged((void**)&(m_cube_map_device->m_y_positive_map), m_width * m_height * 4 * sizeof(uchar)));
	CUDA_CALL(cudaMallocManaged((void**)&(m_cube_map_device->m_y_negative_map), m_width * m_height * 4 * sizeof(uchar)));
	CUDA_CALL(cudaMallocManaged((void**)&(m_cube_map_device->m_z_positive_map), m_width * m_height * 4 * sizeof(uchar)));
	CUDA_CALL(cudaMallocManaged((void**)&(m_cube_map_device->m_z_negative_map), m_width * m_height * 4 * sizeof(uchar)));

	CUDA_CALL(cudaMemcpy(m_cube_map_device->m_x_positive_map, m_x_positive_map.data(), m_width * m_height * 4 * sizeof(uchar), cudaMemcpyDefault));
	CUDA_CALL(cudaMemcpy(m_cube_map_device->m_x_negative_map, m_x_negative_map.data(), m_width * m_height * 4 * sizeof(uchar), cudaMemcpyDefault));
	CUDA_CALL(cudaMemcpy(m_cube_map_device->m_y_positive_map, m_y_positive_map.data(), m_width * m_height * 4 * sizeof(uchar), cudaMemcpyDefault));
	CUDA_CALL(cudaMemcpy(m_cube_map_device->m_y_negative_map, m_y_negative_map.data(), m_width * m_height * 4 * sizeof(uchar), cudaMemcpyDefault));
	CUDA_CALL(cudaMemcpy(m_cube_map_device->m_z_positive_map, m_z_positive_map.data(), m_width * m_height * 4 * sizeof(uchar), cudaMemcpyDefault));
	CUDA_CALL(cudaMemcpy(m_cube_map_device->m_z_negative_map, m_z_negative_map.data(), m_width * m_height * 4 * sizeof(uchar), cudaMemcpyDefault));

	m_cube_map_device->length = m_width;

	return true;
}

inline void cube_map_loader::release_cube_device_data()
{
	if (m_cube_map_device == nullptr)
	{
		return;
	}

	CUDA_CALL(cudaFree(m_cube_map_device->m_x_positive_map));
	CUDA_CALL(cudaFree(m_cube_map_device->m_x_negative_map));
	CUDA_CALL(cudaFree(m_cube_map_device->m_y_positive_map));
	CUDA_CALL(cudaFree(m_cube_map_device->m_y_negative_map));
	CUDA_CALL(cudaFree(m_cube_map_device->m_z_positive_map));
	CUDA_CALL(cudaFree(m_cube_map_device->m_z_negative_map));
	CUDA_CALL(cudaFree(m_cube_map_device));

	m_cube_map_device = nullptr;

	return;
}

inline bool cube_map_loader::decode_bmp(
	const std::vector<uchar>& bmp,	//in
	std::vector<uchar>& image,		//out
	int& width,						//out
	int& height						//out
)
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

#endif // !__CUBE_MAP__
