#pragma once

#ifndef __CONFIGURATION__
#define __CONFIGURATION__

struct configuration
{
	int block_size;
	int max_tracer_depth;
	float vector_bias_length;
	float energy_exist_threshold;
	float sss_threshold;
	bool use_sky_box;
	bool use_ground;
	bool use_bilinear;
};

#endif // !__CONFIGURATION__
