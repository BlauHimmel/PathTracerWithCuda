#pragma once

#ifndef __UTILITIES__
#define __UTILITIES__

#define API_ENTRY

#define INTERNAL_FUNC

#define CUDA_CALL(Statement)\
{\
	cudaError error = Statement;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "[Cuda]Error in file '%s' in line %i : %s.\n",\
			__FILE__, __LINE__, cudaGetErrorString(error));\
	}\
}\

#define SAFE_DELETE(Ptr)\
{\
	if (Ptr != nullptr)\
	{\
		delete Ptr;\
		Ptr = nullptr;\
	}\
}\

#define SAFE_DELETE_ARRAY(ArrayPtr)\
{\
	if (ArrayPtr != nullptr)\
	{\
		delete[] ArrayPtr;\
		ArrayPtr = nullptr;\
	}\
}\

#define TIME_COUNT_CALL_START()\
{\
clock_t __start_time, __end_time;\
__start_time = clock();\

#define TIME_COUNT_CALL_END(Time)\
__end_time = clock();\
Time = static_cast<double>(__end_time - __start_time) / CLOCKS_PER_SEC * 1000.0;\
}\

#define CHECK_PROPERTY(Category, Property, Token)\
if (Property.is_null())\
{\
	std::cout << "[Error]" << #Category << " property <" << Token << "> not defined!" << std::endl;\
	return false;\
}\

#endif // !__UTILITIES__
