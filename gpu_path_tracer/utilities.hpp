#pragma once

#ifndef __UTILITIES__
#define __UTILITIES__

#define CUDA_CALL(Statement)\
{\
	cudaError error = Statement;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",\
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


#endif // !__UTILITIES__
