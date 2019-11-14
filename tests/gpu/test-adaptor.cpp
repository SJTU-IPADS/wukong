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

#include <boost/mpi.hpp>
#include <vector>
#include "global.hpp"
#include "config.hpp"
#include "mem.hpp"
#include "gpu/gpu_mem.hpp"
#include "rdma.hpp"
#include "adaptor.hpp"
#include "logger2.hpp"
#include "unit.hpp"

int main(int argc, char *argv[]) {
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    int sid = world.rank(); // server ID

    // load global configs
    load_config(string(argv[1]), world.size());

    // set the address file of host/cluster
    string host_fname = std::string(argv[2]);

    // allocate memory regions
    vector<RDMA::MemoryRegion> mrs;

    // CPU (host) memory
    Mem *mem = new Mem(Global::num_servers, Global::num_threads);
    logstream(LOG_INFO) << "#" << sid << ": allocate " << B2GiB(mem->size()) << "GB memory" << LOG_endl;
    RDMA::MemoryRegion mr_cpu = { RDMA::MemType::CPU, mem->address(), mem->size(), mem };
    mrs.push_back(mr_cpu);

    // GPU (device) memory
    int devid = 0; // FIXME: it means one GPU device?
    GPUMem *gpu_mem = new GPUMem(devid, Global::num_servers, Global::num_gpus);
    logstream(LOG_INFO) << "#" << sid << ": allocate " << B2GiB(gpu_mem->size()) << "GB GPU memory" << LOG_endl;
    RDMA::MemoryRegion mr_gpu = { RDMA::MemType::GPU, gpu_mem->address(), gpu_mem->size(), gpu_mem };
    mrs.push_back(mr_gpu);

    // init RDMA devices and connections
    RDMA_init(Global::num_servers, Global::num_threads, sid, mrs, host_fname);

    // init communication
    RDMA_Adaptor *rdma_adaptor = new RDMA_Adaptor(sid, mrs, Global::num_servers, Global::num_threads);

    // create proxies and engines
    vector<Adaptor *> adaptors;
    for (int tid = 0; tid < Global::num_threads; tid++) {
        Adaptor *adaptor = new Adaptor(tid, nullptr, rdma_adaptor);
        adaptors.push_back(adaptor);
    }

    if (sid == 0) {
        sid_t str[3] = {1, 4, 3};
        char *outbuf;
        CUDA_ASSERT(cudaMalloc(&outbuf, 10000));
        CUDA_ASSERT(cudaMemcpy(outbuf, str, 3 * sizeof(sid_t), cudaMemcpyHostToDevice));
        sid_t tmp[3];
        CUDA_ASSERT(cudaMemcpy(tmp, outbuf, 3 * sizeof(sid_t), cudaMemcpyDeviceToHost));
        logstream(LOG_ERROR) << "---I am sender, on GPU: " << tmp[0] << ", " << tmp[1] << ", " << tmp[2] << LOG_endl;
        // set type
        uint64_t test_type = SPARQL_HISTORY;
        CUDA_ASSERT(cudaMemcpy(gpu_mem->rdma_buf_type(0), &test_type, sizeof(uint64_t), cudaMemcpyHostToDevice));
        adaptors[0]->send_dev2host(1, 0, outbuf, 3 * sizeof(sid_t));
    } else {
        sid_t *tmp;
        Bundle b = adaptors[0]->recv();
        tmp = (sid_t *)b.data.c_str();
        logstream(LOG_ERROR) << "---type: " << b.type << ", " << tmp[0] << " " << tmp[1] << " " << tmp[2] << LOG_endl;
    }
    return 0;

}
