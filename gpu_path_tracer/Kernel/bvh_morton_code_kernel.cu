#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust\sort.h>
#include <exception>

#include "Math\basic_math.hpp"
#include "Math\cuda_math.hpp"
#include "Bvh\bvh.h"
#include "Core\configuration.h"
#include "Core\triangle.h"

struct bvh_node_morton_node_predicate
{
	bool operator()(const bvh_node& left, const bvh_node& right)
	{
		return left.morton_code < right.morton_code;
	}
};

__host__ __device__ uint expand_bits(
	uint value			//in
)
{
	value = (value * 0x00010001u) & 0xFF0000FFu;
	value = (value * 0x00000101u) & 0x0F00F00Fu;
	value = (value * 0x00000011u) & 0xC30C30C3u;
	value = (value * 0x00000005u) & 0x49249249u;
	return value;
}

/*
Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
From from http://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
*/
__host__ __device__ uint morton_code(
	const float3& point		//in
)
{
	float x = fminf(fmaxf(point.x * 1024.0f, 0.0f), 1023.0f);
	float y = fminf(fmaxf(point.y * 1024.0f, 0.0f), 1023.0f);
	float z = fminf(fmaxf(point.z * 1024.0f, 0.0f), 1023.0f);

	uint xx = expand_bits(static_cast<uint>(x));
	uint yy = expand_bits(static_cast<uint>(y));
	uint zz = expand_bits(static_cast<uint>(z));

	return xx * 4 + yy * 2 + zz;
}

/*
Counts the number of leading zero bits in a 32-bit integer.
From : http://embeddedgurus.com/state-space/2014/09/fast-deterministic-and-portable-counting-leading-zeros/
*/
__device__ uint clz(uint value)
{
	static const uchar clz_table[] =
	{
		32u, 31u, 30u, 30u, 29u, 29u, 29u, 29u,
		28u, 28u, 28u, 28u, 28u, 28u, 28u, 28u,
		27u, 27u, 27u, 27u, 27u, 27u, 27u, 27u,
		27u, 27u, 27u, 27u, 27u, 27u, 27u, 27u,
		26u, 26u, 26u, 26u, 26u, 26u, 26u, 26u,
		26u, 26u, 26u, 26u, 26u, 26u, 26u, 26u,
		26u, 26u, 26u, 26u, 26u, 26u, 26u, 26u,
		26u, 26u, 26u, 26u, 26u, 26u, 26u, 26u,
		25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
		25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
		25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
		25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
		25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
		25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
		25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
		25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
		24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u
	};

	uint n;

	if (value >= (1u << 16))
	{
		if (value >= (1u << 24))
		{
			n = 24u;
		}
		else
		{
			n = 16u;
		}
	}
	else
	{
		if (value >= (1u << 8))
		{
			n = 8u;
		}
		else
		{
			n = 0u;
		}
	}

	return (uint)(clz_table[value >> n]) - n;
}

__host__ __device__ void get_bounding_box(
	bounding_box& box,			//in out
	const float3& vertex0,		//in
	const float3& vertex1,		//in
	const float3& vertex2		//in
)
{
	box.left_bottom = make_float3(min(vertex0.x, vertex1.x, vertex2.x), min(vertex0.y, vertex1.y, vertex2.y), min(vertex0.z, vertex1.z, vertex2.z));
	box.right_top = make_float3(max(vertex0.x, vertex1.x, vertex2.x), max(vertex0.y, vertex1.y, vertex2.y), max(vertex0.z, vertex1.z, vertex2.z));
	box.centroid = 0.5f * (box.right_top + box.left_bottom);
}

__host__ __device__ void get_bounding_box(
	bounding_box& box,					//in out
	const float3& other_left_bottom,	//in
	const float3& other_right_top		//in
)
{
	box.left_bottom = other_left_bottom;
	box.right_top = other_right_top;
	box.centroid = 0.5f * (box.right_top + box.left_bottom);
}

__host__ __device__ void expand_to_fit_triangle(
	bounding_box& box,			//in out
	const float3& vertex0,		//in
	const float3& vertex1,		//in
	const float3& vertex2		//in
)
{
	box.left_bottom = fminf(box.left_bottom, make_float3(min(vertex0.x, vertex1.x, vertex2.x), min(vertex0.y, vertex1.y, vertex2.y), min(vertex0.z, vertex1.z, vertex2.z)));
	box.right_top = fmaxf(box.right_top, make_float3(max(vertex0.x, vertex1.x, vertex2.x), max(vertex0.y, vertex1.y, vertex2.y), max(vertex0.z, vertex1.z, vertex2.z)));
	box.centroid = 0.5f * (box.right_top + box.left_bottom);
}

__host__ __device__ void expand_to_fit_box(
	bounding_box& box,					//in out
	const float3& other_left_bottom,	//in
	const float3& other_right_top		//in
)
{
	box.left_bottom = fminf(box.left_bottom, other_left_bottom);
	box.right_top = fmaxf(box.right_top, other_right_top);
	box.centroid = 0.5f * (box.right_top + box.left_bottom);
}

/*
From the paper: Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees
*/
__device__ uint find_split(
	bvh_node_morton_code_cuda* sorted_leaf_nodes,	//in
	uint first_index,								//in
	uint last_index									//in
)
{
	uint first_morton_code = sorted_leaf_nodes[first_index].morton_code;
	uint last_morton_code = sorted_leaf_nodes[last_index].morton_code;

	if (first_morton_code == last_morton_code)
	{
		return first_index;
	}

	uint common_prefix_length = clz(first_morton_code ^ last_morton_code);
	uint split_index = first_index;
	uint step = last_index - first_index;

	do
	{
		step = (step + 1) >> 1;
		uint new_split_index = split_index + step;

		if (new_split_index < last_index)
		{
			uint split_morton_code = sorted_leaf_nodes[new_split_index].morton_code;
			uint split_prefix_length = clz(first_morton_code ^ split_morton_code);

			if (split_prefix_length > common_prefix_length)
			{
				split_index = new_split_index;
			}
		}


	} while (step > 1);

	return split_index;
}

/*
From the paper: Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees
*/
__device__ uint2 find_range(
	bvh_node_morton_code_cuda* sorted_leaf_nodes,	//in
	uint node_num,									//in
	uint index										//in
)
{
	if (index == 0)
	{
		return make_uint2(0, node_num - 1);
	}

	int direction;
	uint delta_min;
	uint initial_index = index;

	uint previous_morton_code = sorted_leaf_nodes[index - 1].morton_code;
	uint current_morton_code = sorted_leaf_nodes[index].morton_code;
	uint next_morton_code = sorted_leaf_nodes[index + 1].morton_code;

	if (previous_morton_code == current_morton_code && current_morton_code == next_morton_code)
	{
		while (index > 0 && index < node_num - 1)
		{
			index++;

			if (index >= node_num - 1)
			{
				break;
			}

			if (sorted_leaf_nodes[index].morton_code != sorted_leaf_nodes[index + 1].morton_code)
			{
				break;
			}
		}

		return make_uint2(initial_index, index);
	}
	else
	{
		uint prefix_common_length_left = clz(current_morton_code ^ previous_morton_code);
		uint prefix_common_length_right = clz(current_morton_code ^ next_morton_code);

		if (prefix_common_length_left > prefix_common_length_right)
		{
			direction = -1;
			delta_min = prefix_common_length_right;
		}
		else
		{
			direction = 1;
			delta_min = prefix_common_length_left;
		}
	}

	uint l_max = 2;
	uint test_index = index + l_max * direction;

	while (test_index < node_num && test_index >= 0 &&
		clz(current_morton_code ^ sorted_leaf_nodes[test_index].morton_code) > delta_min)
	{
		l_max *= 2;
		test_index = index + l_max * direction;
	}

	int l = 0;

	for (int divisor = 2; l_max / divisor >= 1; divisor *= 2)
	{
		int t = l_max / divisor;
		test_index = index + (l + t) * direction;

		if (test_index < node_num && test_index >= 0)
		{
			if (clz(current_morton_code ^ sorted_leaf_nodes[test_index].morton_code) > delta_min)
			{
				l = l + t;
			}
		}
	}

	if (direction == 1)
	{
		return make_uint2(index, index + l * direction);
	}
	else/* if (direction == -1) */
	{
		return make_uint2(index + l * direction, index);
	}
}

__device__ void generate_internal_node(
	bvh_node_morton_code_cuda* internal_nodes,			//in
	bvh_node_morton_code_cuda* leaf_nodes,				//in
	uint leaf_node_num,									//in
	uint index											//in
)
{
	uint2 range = find_range(leaf_nodes, leaf_node_num, index);
	uint split_index = find_split(leaf_nodes, range.x, range.y);

	int left_index;
	int right_index;
	bool is_left_leaf;
	bool is_right_leaf;

	if (split_index == range.x)
	{
		left_index = split_index;
		is_left_leaf = true;
		leaf_nodes[left_index].parent_index = index;
	}
	else
	{
		left_index = split_index;
		is_left_leaf = false;
		internal_nodes[left_index].parent_index = index;
	}

	if (split_index + 1 == range.y)
	{
		right_index = split_index + 1;
		is_right_leaf = true;
		leaf_nodes[right_index].parent_index = index;
	}
	else
	{
		right_index = split_index + 1;
		is_right_leaf = false;
		internal_nodes[right_index].parent_index = index;
	}

	internal_nodes[index].left_index = left_index;
	internal_nodes[index].right_index = right_index;
	internal_nodes[index].is_left_leaf = is_left_leaf;
	internal_nodes[index].is_right_leaf = is_right_leaf;
}

__global__ void __compute_triangle_bounding_box_kernel(
	triangle* triangles_device,							//in
	int triangle_num,									//in
	bvh_node_morton_code_cuda* triangle_nodes_device,	//in
	bounding_box* mesh_box,								//in
	int start_index,									//in
	int block_size										//in
)
{
	int block_x = blockIdx.x;
	int thread_x = threadIdx.x;

	int triangle_node_index = block_size * block_x + thread_x;
	bool is_index_valid = triangle_node_index < triangle_num;

	if (is_index_valid)
	{
		triangle_nodes_device[triangle_node_index].triangle_index = triangle_node_index + start_index;
		get_bounding_box(
			triangle_nodes_device[triangle_node_index].box,
			triangles_device[triangle_node_index].vertex0,
			triangles_device[triangle_node_index].vertex1,
			triangles_device[triangle_node_index].vertex2
		);
		triangle_nodes_device[triangle_node_index].morton_code = morton_code(
			(triangle_nodes_device[triangle_node_index].box.centroid - mesh_box->left_bottom) /
			(mesh_box->right_top - mesh_box->left_bottom)
		);
	}
}

__global__ void __generate_internal_node_kernel(
	bvh_node_morton_code_cuda* internal_nodes_device,	//in
	uint internal_node_num,								//in
	bvh_node_morton_code_cuda* leaf_nodes_device,		//in
	uint leaf_node_num,									//in
	int block_size										//in
)
{
	int block_x = blockIdx.x;
	int thread_x = threadIdx.x;

	int internal_node_index = block_size * block_x + thread_x;
	bool is_index_valid = internal_node_index < internal_node_num;

	if (is_index_valid)
	{
		generate_internal_node(internal_nodes_device, leaf_nodes_device, leaf_node_num, (uint)internal_node_index);
	}
}

extern "C" void compute_triangle_bounding_box_kernel(
	triangle* triangles_device,							//in
	int triangle_num,									//in
	bvh_node_morton_code_cuda* triangle_nodes_device,	//in
	bounding_box* mesh_box,								//in
	int start_index,									//in
	int block_size										//in
)
{
	int threads_num_per_block = block_size;
	int total_blocks_num_per_gird = (triangle_num + threads_num_per_block - 1) / threads_num_per_block;

	__compute_triangle_bounding_box_kernel <<<total_blocks_num_per_gird, threads_num_per_block >>> (
		triangles_device,
		triangle_num,
		triangle_nodes_device,
		mesh_box,
		start_index,
		block_size
		);

	CUDA_CALL(cudaDeviceSynchronize());
}

extern "C" void generate_internal_node_kernel(
	bvh_node_morton_code_cuda* internal_nodes_device,	//in
	uint internal_node_num,								//in
	bvh_node_morton_code_cuda* leaf_nodes_device,		//in
	uint leaf_node_num,									//in
	int block_size										//in
)
{
	int threads_num_per_block = block_size;
	int total_blocks_num_per_gird = (internal_node_num + threads_num_per_block - 1) / threads_num_per_block;

	__generate_internal_node_kernel <<<total_blocks_num_per_gird, threads_num_per_block>>> (
		internal_nodes_device,
		internal_node_num,
		leaf_nodes_device,
		leaf_node_num,
		block_size
		);

	CUDA_CALL(cudaDeviceSynchronize());
}