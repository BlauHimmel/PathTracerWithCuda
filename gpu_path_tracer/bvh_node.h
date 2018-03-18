#pragma once

#ifndef __BVH_NODE__
#define __BVH_NODE__

#include <vector>

#include "cuda_math.hpp"
#include "bounding_box.hpp"

struct bvh_node
{
	bounding_box box;
	bvh_node* left = nullptr;
	bvh_node* right = nullptr;
	bvh_node* parent = nullptr;	
	bool is_leaf = false;
	std::vector<int> triangle_indices;
	int traversal_index = -1;

	uint morton_code = 0;		//used by bvh_morton_code_cpu
	int is_visited = -1;		//used by bvh_morton_code_cpu
	int triangle_index = -1;	//used by bvh_morton_code_cpu
};

//Special intermediate structure only used for the construction of morton code based bvh on cuda
struct bvh_node_morton_code_cuda
{
	bounding_box box;
	int left_index = -1;
	int right_index = -1;
	int parent_index = -1;	
	bool is_left_leaf = false;
	bool is_right_leaf = false;
	bool is_leaf = false;
	int triangle_index = -1;
	uint morton_code = 0;
	int is_visited = -1;		
};

struct bvh_node_device
{
	bounding_box box;
	int* triangle_indices = nullptr;	//length = BVH_LEAF_NODE_TRIANGLE_NUM, index equal -1 means no triangle
	bool is_leaf = false;
	int next_node_index = -1;
};

#endif // !__BVH_NODE__
