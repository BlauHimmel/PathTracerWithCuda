#pragma once

#include <cuda_runtime.h>
#include <thrust\sort.h>
#include <thrust\device_vector.h>
#include <thrust\remove.h>
#include <thrust\execution_policy.h>

#include "Bvh\bvh_node.h"
#include "Bvh\bvh.h"
#include "Others\utilities.hpp"


struct bvh_node_morton_node_predicate
{
	__device__ bool operator()(const bvh_node_morton_code_cuda& left, const bvh_node_morton_code_cuda& right)
	{
		return left.morton_code < right.morton_code;
	}
};

extern "C" void radix_sort(
	bvh_node_morton_code_cuda* arrays,			//in
	bvh_node_morton_code_cuda* sorted_arrays,	//out
	int number									//in
)
{
	thrust::device_vector<bvh_node_morton_code_cuda> dev_arrays(arrays, arrays + number);
	thrust::sort(dev_arrays.begin(), dev_arrays.end(), bvh_node_morton_node_predicate());
	CUDA_CALL(cudaMemcpy(sorted_arrays, thrust::raw_pointer_cast(dev_arrays.data()), number * sizeof(bvh_node_morton_code_cuda), cudaMemcpyDefault));
	CUDA_CALL(cudaDeviceSynchronize());
}

struct is_negative_predicate
{
	__device__ bool operator()(int value)
	{
		return value < 0;
	}
};

extern "C" int thread_shrink(
	int* energy_exist_pixels_device,	//in out
	int energy_exist_pixels_count		//in
)
{
	auto energy_exist_pixels_end_on_device = thrust::remove_if(thrust::device, energy_exist_pixels_device, energy_exist_pixels_device + energy_exist_pixels_count, is_negative_predicate());
	return (int)(thrust::raw_pointer_cast(energy_exist_pixels_end_on_device) - energy_exist_pixels_device);
}