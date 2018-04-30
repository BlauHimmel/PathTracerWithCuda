#pragma once

#ifndef __PARALLEL_FUNCTION__
#define __PARALLEL_FUNCTION__

#include "Bvh\bvh_node.h"
#include "Bvh\bvh.h"
#include "Others\utilities.hpp"
#include <thrust\sort.h>
#include <thrust\device_vector.h>
#include <thrust\remove.h>

extern "C" void radix_sort(
	bvh_node_morton_code_cuda* arrays,			//in
	bvh_node_morton_code_cuda* sorted_arrays,	//out
	int number									//in
);

extern "C" int thread_shrink(
	int* energy_exist_pixels_device,	//in out
	int energy_exist_pixels_count		//in
);

#endif //! __PARALLEL_FUNCTION__
