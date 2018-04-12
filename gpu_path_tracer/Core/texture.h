#pragma once

#ifndef __TEXTURE__
#define __TEXTURE__

#include "Math\basic_math.hpp"


struct texture_wrapper
{
	int width;
	int height;
	uchar* pixels;
};

#endif // !__TEXTURE__