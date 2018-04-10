#pragma once

#ifndef __BVH__
#define __BVH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust\device_vector.h>
#include <vector>
#include <stack>
#include <algorithm>
#include <glm\glm.hpp>
#include <omp.h>

#include "Core\triangle.h"
#include "Others\utilities.hpp"
#include "Math\cuda_math.hpp"
#include "Core\configuration.h"
#include "Bvh\bvh_morton_code_kernel.h"
#include "Bvh\bounding_box.h"
#include "Bvh\bvh_node.h"

#undef BVH_MORTON_CODE_BUILD_OPENMP					//used by bvh_morton_code_cpu and bvh_morton_code_cuda			
#define BVH_BUILD_METHOD bvh_morton_code_cuda::

class bvh_build_config
{
public:
	static int bvh_leaf_node_triangle_num;            
	static int bvh_bucket_max_divide_internal_num;	  //used by bvh_naive_cpu
	static int bvh_build_block_size;				  //used by bvh_morton_code_cuda
};

namespace bvh_naive_cpu
{
	INTERNAL_FUNC bvh_node_device get_bvh_node_device(bvh_node* node);

	INTERNAL_FUNC void split_bounding_box(bvh_node* node, bounding_box* boxes, int start_index);

	API_ENTRY bvh_node* build_bvh(triangle* triangles, int triangle_num, int start_index);

	API_ENTRY void release_bvh(bvh_node* root_node);

	API_ENTRY bvh_node_device* build_bvh_device_data(bvh_node* root);

	API_ENTRY void update_bvh(
		const glm::mat4& initial_transform_mat,
		const glm::mat4& transform_mat,
		bvh_node_device* initial_root,
		bvh_node_device* transformed_root
	);
}

namespace bvh_morton_code_cpu
{
	bool bvh_node_morton_node_comparator(const bvh_node& left, const bvh_node& right);

	/*
		Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
		From from http://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
	*/
	INTERNAL_FUNC uint expand_bits(uint value);

	/*
		Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
		From from http://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
	*/
	INTERNAL_FUNC uint morton_code(const float3& point);

	/*
		Counts the number of leading zero bits in a 32-bit integer.
		From : http://embeddedgurus.com/state-space/2014/09/fast-deterministic-and-portable-counting-leading-zeros/
	*/
	INTERNAL_FUNC uint clz(uint value);

	/*
		From the paper: Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees
	*/
	INTERNAL_FUNC uint find_split(bvh_node* sorted_leaf_nodes, uint first_index, uint last_index);

	/*
		From the paper: Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees
	*/
	INTERNAL_FUNC uint2 find_range(bvh_node* sorted_leaf_nodes, uint node_num, uint index);

	INTERNAL_FUNC void generate_internal_node(bvh_node* internal_nodes, bvh_node* leaf_nodes, uint leaf_node_num, uint index);

	INTERNAL_FUNC void generate_bounding_box_for_internal_node(bvh_node* node);

	API_ENTRY bvh_node* build_bvh(triangle* triangles, int triangle_num, int start_index);

	API_ENTRY void release_bvh(bvh_node* root_node);

	API_ENTRY bvh_node_device* build_bvh_device_data(bvh_node* root);

	API_ENTRY void update_bvh(
		const glm::mat4& initial_transform_mat,
		const glm::mat4& transform_mat,
		bvh_node_device* initial_root,
		bvh_node_device* transformed_root
	);
}

namespace bvh_morton_code_cuda
{
	bool bvh_node_morton_node_comparator(const bvh_node_morton_code_cuda& left, const bvh_node_morton_code_cuda& right);

	INTERNAL_FUNC void generate_bounding_box_for_internal_node(
		bvh_node_morton_code_cuda* node, 
		bvh_node_morton_code_cuda* leaf_nodes, 
		bvh_node_morton_code_cuda* internal_nodes
	);

	API_ENTRY bvh_node* build_bvh(triangle* triangles, int triangle_num, int start_index);

	API_ENTRY void release_bvh(bvh_node* root_node);

	API_ENTRY bvh_node_device* build_bvh_device_data(bvh_node* root);

	API_ENTRY void update_bvh(
		const glm::mat4& initial_transform_mat,
		const glm::mat4& transform_mat,
		bvh_node_device* initial_root,
		bvh_node_device* transformed_root
	);
}

#endif // !__BVH__
