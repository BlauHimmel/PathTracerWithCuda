#pragma once

#ifndef __CUBE_MAP__
#define __CUBE_MAP__

#include "basic_math.hpp"

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

#endif // !__CUBE_MAP__