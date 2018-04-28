#pragma once

#ifndef __BVH_BUILD_CONFIG__
#define __BVH_BUILD_CONFIG__

#include <string>
#include <algorithm>

#undef BVH_MORTON_CODE_BUILD_OPENMP					//used by bvh_morton_code_cpu and bvh_morton_code_cuda

class bvh_build_config
{
public:
	static int bvh_leaf_node_triangle_num;
	static int bvh_bucket_max_divide_internal_num;	  //used by bvh_naive_cpu
	static int bvh_build_block_size;				  //used by bvh_morton_code_cuda
};

enum class bvh_build_method
{
	NAIVE_CPU,
	MORTON_CODE_CPU,
	MORTON_CODE_CUDA
};

extern bvh_build_method parse_bvh_build_method(std::string& text);

extern std::string to_string(bvh_build_method bvh_build);


#endif // !__BVH_BUILD_CONFIG__
