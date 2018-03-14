#pragma once

#ifndef __BVH__
#define __BVH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust\device_vector.h>

#include <vector>
#include <stack>

#include "triangle.hpp"
#include "utilities.hpp"
#include "cuda_math.hpp"

#define BVH_LEAF_NODE_TRIANGLE_NUM 6
#define BVH_BUCKET_MAX_DIVIDE_INTERNAL_NUM 12

struct bounding_box
{
	float3 left_bottom;
	float3 right_top;
	float3 centroid;

	bounding_box();
	bounding_box(const float3& left_bottom, const float3& right_top);
	void expand_to_fit_box(const float3& other_left_bottom, const float3& other_right_top);
	void expand_to_fit_triangle(const float3& vertex0, const float3& vertex1, const float3& vertex2);
	void get_bounding_box(const float3& other_left_bottom, const float3& other_right_top);
	void get_bounding_box(const float3& vertex0, const float3& vertex1, const float3& vertex2);
	float get_surface_area();
	float get_axis_length(int axis); //return- 0:x 1:y 2:z
};

struct bvh_node
{
	bounding_box box;
	uint morton_code;
	bvh_node* left = nullptr;
	bvh_node* right = nullptr;
	bool is_leaf = false;
	int traversal_index = -1;
	std::vector<int> triangle_indices;
};

struct bvh_node_device
{
	bounding_box box;
	int* triangle_indices = nullptr;	//length = BVH_LEAF_NODE_TRIANGLE_NUM, index equal -1 means no triangle
	bool is_leaf = false;
	int next_node_index = -1;
};

//For simplicity, I decide not to use the a series of classes here.

namespace bvh_naive_cpu
{
	INTERNAL_FUNC bvh_node_device get_bvh_node_device(bvh_node* node);

	INTERNAL_FUNC void split_bounding_box(bvh_node* node, bounding_box* boxes, int start_index);

	API_ENTRY bvh_node* build_bvh(triangle* triangles, int triangle_num, int start_index);

	API_ENTRY void release_bvh(bvh_node* root_node);

	API_ENTRY bvh_node_device* build_bvh_device_data(bvh_node* root);
}

namespace bvh_morton_code_cpu
{
	struct morton_node_predicate
	{
		bool operator()(const bvh_node& left, const bvh_node& right) const
		{
			return left.morton_code < right.morton_code;
		}
	};

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
	INTERNAL_FUNC uint find_split(uint* morton_codes, uint first_index, uint last_index);

	/*
		From the paper: Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees
	*/
	uint2 find_range(uint* morton_codes, uint morton_codes_size, uint index);
}

#endif // !__BVH__
