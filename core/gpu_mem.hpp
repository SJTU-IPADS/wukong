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
 *      http://ipads.se.sjtu.edu.cn/projects/wukong.html
 *
 */

#pragma once

#ifdef USE_GPU

#include "unit.hpp"
#include "gpu_utils.hpp"

class GPUMem {
private:
	int devid;
	int num_servers;
	int num_agents;

	// The Wukong's GPU memory layout: key/value cache | intermediate results | heap
	char *mem_gpu;
	uint64_t mem_gpu_sz;

	// FIXME: kvcache for gstore
	//

	// intermediate results (history table)
	char *buf; // #threads
	uint64_t buf_sz;
	uint64_t buf_off;

	// FIXME: head
	//

public:
	GPUMem(int devid, int num_servers, int num_agents)
		: devid(devid), num_servers(num_servers), num_agents(num_agents) {
		buf_sz = MiB2B(global_gpu_rdma_buf_size_mb);
		mem_gpu_sz = buf_sz * num_agents;

		CUDA_ASSERT(cudaSetDevice(devid));
		CUDA_ASSERT(cudaMalloc(&mem_gpu, mem_gpu_sz));
		CUDA_ASSERT(cudaMemset(mem_gpu, 0, mem_gpu_sz));

		buf_off = 0;
		buf = mem_gpu + buf_off;
		logstream(LOG_INFO) << "GPUMem: devid: " << devid
		                    << ", num_servers: " << num_servers
		                    << ", num_agents: " << num_agents << LOG_endl;
	}

	~GPUMem() { CUDA_ASSERT(cudaFree(mem_gpu)); }

	inline char *memory() { return mem_gpu; }
	inline uint64_t memory_size() { return mem_gpu_sz; }

	// intermediate results
	inline char *buffer(int tid) { return buf + buf_sz * (tid % num_agents); }
	inline uint64_t buffer_size() { return buf_sz; }
	inline uint64_t buffer_offset(int tid) { return buf_off + buf_sz * (tid % num_agents); }
};

#endif
