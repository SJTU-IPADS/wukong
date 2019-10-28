/*
 * Copyright (c) 2016 Shanghai Jiao Tong University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://ipads.se.sjtu.edu.cn/projects/wukong
 *
 */

#pragma once

#ifdef USE_GPU
#include <cuda_runtime.h>

#define CUDA_ASSERT(ans) { check_cuda_result((ans), __FILE__, __LINE__); }

inline void check_cuda_result(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA_ASSERT: code:%d, %s %s:%d\n",
		        code, cudaGetErrorString(code), file, line);
		if (abort) assert(false);
	}
}


#define CUDA_STREAM_SYNC(stream) (CUDA_ASSERT(cudaStreamSynchronize(stream)))
#define CUDA_DEVICE_SYNC (CUDA_ASSERT(cudaDeviceSynchronize()))

#define WUKONG_GPU_AGENT_TID (Global::num_proxies + Global::num_engines)

#define WUKONG_CUDA_NUM_THREADS 512
#define WUKONG_GPU_ELEM_SIZE sizeof(sid_t)

inline int WUKONG_GET_BLOCKS(const int n) {
	return (n + WUKONG_CUDA_NUM_THREADS - 1) / WUKONG_CUDA_NUM_THREADS;
}

#define WUKONG_GPU_RBUF_SIZE(num_elems) (WUKONG_GPU_ELEM_SIZE * num_elems)

#endif
