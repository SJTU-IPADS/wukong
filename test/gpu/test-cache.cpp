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


#include <map>
#include <boost/mpi.hpp>
#include <iostream>

#include "global.hpp"
#include "config.hpp"
#include "bind.hpp"
#include "mem.hpp"
#include "gpu/gpu_mem.hpp"
#include "gpu/gpu_cache.hpp"
#include "string_server.hpp"
#include "dgraph.hpp"
#include "console.hpp"
#include "rdma.hpp"
#include "comm/adaptor.hpp"
#include "stats.hpp"
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
    RDMA_Adaptor *rdma_adaptor = new RDMA_Adaptor(sid, mrs,
            Global::num_servers, Global::num_threads);
    TCP_Adaptor *tcp_adaptor = new TCP_Adaptor(sid, host_fname, Global::data_port_base,
            Global::num_servers, Global::num_threads);

    // init control communicaiton
    con_adaptor = new TCP_Adaptor(sid, host_fname, Global::ctrl_port_base,
                                  Global::num_servers, Global::num_proxies);

    // load string server (read-only, shared by all proxies and all engines)
    StringServer str_server(Global::input_folder);

    // load RDF graph (shared by all engines and proxies)
    DGraph dgraph(sid, mem, &str_server, Global::input_folder);

    // prepare statistics for SPARQL optimizer
    Stats stat(sid);
    if (Global::enable_planner) {
        if (Global::generate_statistics) {
            uint64_t t0 = timer::get_usec();
            dgraph.gstore.generate_statistic(stat);
            uint64_t t1 = timer::get_usec();
            logstream(LOG_EMPH)  << "generate_statistic using time: " << t1 - t0 << "usec" << LOG_endl;
            stat.gather_stat(con_adaptor);
        } else {
            // use the dataset name by default
            string fname = Global::input_folder + "/statfile";
            stat.load_stat_from_file(fname, con_adaptor);
        }
    }



    if (sid == 0) {
        vertex_t *d_va = (vertex_t *)gpu_mem->kvcache();
        edge_t *d_ea = (edge_t *)(gpu_mem->kvcache() + (gpu_mem->kvcache_size() * GStore::HD_RATIO / 100));
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        auto &rsmm = dgraph.gstore.get_rdf_seg_meta_map();
        segid_t seg(0, 2, OUT);
        GPUCache cache(gpu_mem, dgraph.gstore.vertices, dgraph.gstore.edges, rsmm);

        vector<segid_t> dummy;

        logstream(LOG_EMPH) << "Load segment: #key: " << rsmm[seg].num_keys << ", #buckets:" << rsmm[seg].num_buckets << LOG_endl;
        cache.load_segment(seg, seg, dummy, stream, false);
        CUDA_ASSERT(cudaDeviceSynchronize());
        char v[sizeof(vertex_t)];
        CUDA_ASSERT(cudaMemcpy(v, d_va, sizeof(vertex_t), cudaMemcpyDeviceToHost));
        logstream(LOG_EMPH) << "Load segment success." << LOG_endl;
    }

    return 0;

}
