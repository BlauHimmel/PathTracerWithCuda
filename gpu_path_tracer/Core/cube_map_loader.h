#pragma once

#ifndef __CUBE_MAP_LOADER__
#define __CUBE_MAP_LOADER__

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <time.h>
#include "Math\basic_math.hpp"
#include "Core\cube_map.h"
#include "lib\lodepng\lodepng.h"
#include "Others\utilities.hpp"

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

#endif // !__CUBE_MAP_LOADER__
