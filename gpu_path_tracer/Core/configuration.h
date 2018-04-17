#pragma once

#ifndef __CONFIGURATION__
#define __CONFIGURATION__

struct configuration
{
	int width;
	int height;
	bool use_fullscreen;
	int block_size;
	int max_block_size;
	int max_tracer_depth;
	float vector_bias_length;
	float energy_exist_threshold;
	float sss_threshold;
	bool use_sky_box;
	bool use_sky;
	bool use_bilinear;
	bool gamma_correction;
	bool use_anti_alias;
	float fov;
	int bvh_leaf_node_triangle_num;
	int bvh_bucket_max_divide_internal_num;
	int bvh_build_block_size;
};

#endif // !__CONFIGURATION__
