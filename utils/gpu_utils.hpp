#ifdef USE_GPU
#pragma once
#include <cuda_runtime.h>
#define CUDA_ASSERT(ans) { checkCudaResult((ans), __FILE__, __LINE__); }

inline void checkCudaResult(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA_ASSERT: code:%d, %s %s:%d\n", code, cudaGetErrorString(code), file, line);
      if (abort) assert(false);
   }
}
#endif
