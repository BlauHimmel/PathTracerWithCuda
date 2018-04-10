#pragma once

#ifndef __BVH_MORTON_CODE_KERNEL__
#define __BVH_MORTON_CODE_KERNEL__

#include "bvh_node.h"
#include "configuration.hpp"
#include "triangle.hpp"

extern "C" void compute_triangle_bounding_box_kernel(
	triangle* triangles_device,							//in
	int triangle_num,									//in
	bvh_node_morton_code_cuda* triangle_nodes_device,	//in
	bounding_box* mesh_box,								//in
	int start_index,									//in
	int block_size										//in
);

extern "C" void generate_internal_node_kernel(
	bvh_node_morton_code_cuda* internal_nodes_device,	//in
	uint internal_node_num,								//in
	bvh_node_morton_code_cuda* leaf_nodes_device,		//in
	uint leaf_node_num,									//in
	int block_size										//in
);

#endif //!__BVH_MORTON_CODE_KERNEL__