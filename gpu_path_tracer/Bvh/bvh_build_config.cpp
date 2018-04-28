#include "Bvh\bvh_build_config.h"

int bvh_build_config::bvh_leaf_node_triangle_num = 1;
int bvh_build_config::bvh_bucket_max_divide_internal_num = 12;
int bvh_build_config::bvh_build_block_size = 32;

bvh_build_method parse_bvh_build_method(std::string& text)
{
	std::transform(text.begin(), text.end(), text.begin(), tolower);

	if (text == "naivecpu")
	{
		return bvh_build_method::NAIVE_CPU;
	}

	if (text == "mortoncodecpu")
	{
		return bvh_build_method::MORTON_CODE_CPU;
	}

	if (text == "mortoncodecuda")
	{
		return bvh_build_method::MORTON_CODE_CUDA;
	}

	return bvh_build_method::NAIVE_CPU;
}

std::string to_string(bvh_build_method bvh_build)
{
	if (bvh_build == bvh_build_method::NAIVE_CPU)
	{
		return "NaiveCPU";
	}

	if (bvh_build == bvh_build_method::MORTON_CODE_CPU)
	{
		return "MortonCodeCPU";
	}

	if (bvh_build == bvh_build_method::MORTON_CODE_CUDA)
	{
		return "MortonCodeCUDA";
	}

	return "NaiveCPU";
}
