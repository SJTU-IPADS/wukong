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
#include <iostream>

#include <boost/mpi.hpp>

#include "client/proxy.hpp"
#include "client/console.hpp"

#include "core/common/conflict.hpp"
#include "core/common/config.hpp"
#include "core/common/string_server.hpp"
#include "core/common/bind.hpp"
#include "core/common/rdma.hpp"
#include "core/common/mem.hpp"

#include "core/engine/engine.hpp"

#include "core/store/dgraph.hpp"

#include "core/network/adaptor.hpp"

#include "optimizer/stats.hpp"

// utils
#include "utils/unit.hpp"
#include "utils/logger2.hpp"

#ifdef USE_GPU

#include "gpu/gpu_mem.hpp"
#include "gpu/gpu_agent.hpp"
#include "gpu/gpu_engine.hpp"
#include "gpu/gpu_cache.hpp"
#include "gpu/gpu_stream.hpp"

void *agent_thread(void *arg)
{
    wukong::GPUAgent *agent = (wukong::GPUAgent *)arg;
    if (wukong::enable_binding && wukong::core_bindings.count(agent->tid) != 0)
        wukong::bind_to_core(wukong::core_bindings[agent->tid]);
    else
        wukong::bind_to_core(wukong::default_bindings[agent->tid % wukong::num_cores]);

    agent->run();
}
#endif  // end of USE_GPU

void *engine_thread(void *arg)
{
    wukong::Engine *engine = (wukong::Engine *)arg;
    if (wukong::enable_binding && wukong::core_bindings.count(engine->tid) != 0)
        wukong::bind_to_core(wukong::core_bindings[engine->tid]);
    else
        wukong::bind_to_core(wukong::default_bindings[engine->tid % wukong::num_cores]);

    engine->run();
}

void *proxy_thread(void *arg)
{
    wukong::Proxy *proxy = (wukong::Proxy *)arg;
    if (wukong::enable_binding && wukong::core_bindings.count(proxy->tid) != 0)
        wukong::bind_to_core(wukong::core_bindings[proxy->tid]);
    else
        wukong::bind_to_core(wukong::default_bindings[proxy->tid % wukong::num_cores]);

    // run the builtin console
    wukong::run_console(proxy);
}

static void
usage(char *fn)
{
    std::cout << "usage: " << fn << " <config_fname> <host_fname> [options]" << std::endl;
    std::cout << "options:" << std::endl;
    std::cout << "  -b binding : the file of core binding" << std::endl;
    std::cout << "  -c command : the one-shot command" << std::endl;
}

void print_badge(){
    std::cout << "====================================" << std::endl;
    std::cout << " _      __     __" << std::endl;
    std::cout << "| | /| / /_ __/ /_____  ___  ___ _" << std::endl;
    std::cout << "| |/ |/ / // /  '_/ _ \\/ _ \\/ _ `/" << std::endl;
    std::cout << "|__/|__/\\_,_/_/\\_\\\\___/_//_/\\_, / " << std::endl;
    std::cout << "                           /___/" << std::endl;
    std::cout << "====================================" << std::endl;
}

int
main(int argc, char *argv[])
{
    wukong::conflict_detector();

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    int sid = world.rank(); // server ID

    if (argc < 3) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    if (sid == 0)
        print_badge();

    // load global configs
    wukong::load_config(std::string(argv[1]), world.size());

    // set the address file of host/cluster
    std::string host_fname = std::string(argv[2]);

    // load CPU topology by hwloc
    wukong::load_node_topo();
    logstream(LOG_INFO) << "#" << sid << ": has " << wukong::num_cores << " cores." << LOG_endl;

    int c;
    while ((c = getopt(argc - 2, argv + 2, "b:c:")) != -1) {
        switch (c) {
        case 'b':
            wukong::enable_binding = wukong::load_core_binding(optarg);
            break;
        case 'c':
            wukong::enable_oneshot = true;
            wukong::oneshot_cmd = optarg;
            break;
        default:
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    // allocate memory regions
    std::vector<wukong::RDMA::MemoryRegion> mrs;

    // rdma broadcast memory
    std::vector<wukong::Broadcast_Mem *> bcast_mems;
    wukong::Broadcast_Mem *ss_bcast_mem = new wukong::Broadcast_Mem(wukong::Global::num_servers, wukong::Global::num_threads);
    bcast_mems.push_back(ss_bcast_mem);
    // CPU (host) memory
    wukong::Mem *mem = new wukong::Mem(wukong::Global::num_servers, wukong::Global::num_threads, bcast_mems);
    logstream(LOG_INFO) << "#" << sid << ": allocate " << B2GiB(mem->size())
                        << "GB memory" << LOG_endl;
    wukong::RDMA::MemoryRegion mr_cpu = { wukong::RDMA::MemType::CPU, mem->address(), mem->size(), mem };
    mrs.push_back(mr_cpu);

#ifdef USE_GPU
    // GPU (device) memory
    int devid = 0; // FIXME: it means one GPU device?
    wukong::GPUMem *gpu_mem = new wukong::GPUMem(devid, wukong::Global::num_servers, wukong::Global::num_gpus);
    logstream(LOG_INFO) << "#" << sid << ": allocate " << B2GiB(gpu_mem->size())
                        << "GB GPU memory" << LOG_endl;
    wukong::RDMA::MemoryRegion mr_gpu = { wukong::RDMA::MemType::GPU, gpu_mem->address(), gpu_mem->size(), gpu_mem };
    mrs.push_back(mr_gpu);
#endif

    // RDMA full-link communication
    int flink_nthreads = wukong::Global::num_proxies + wukong::Global::num_engines;
    // RDMA broadcast communication
    int bcast_nthreads = 2;
    int rdma_init_nthreads = flink_nthreads + bcast_nthreads;
    // init RDMA devices and connections
    wukong::RDMA_init(wukong::Global::num_servers, rdma_init_nthreads, sid, mrs, host_fname);

    // init communication
    wukong::RDMA_Adaptor *rdma_adaptor = new wukong::RDMA_Adaptor(sid, mrs,
            wukong::Global::num_servers, wukong::Global::num_threads);
    wukong::TCP_Adaptor *tcp_adaptor = new wukong::TCP_Adaptor(sid, host_fname, wukong::Global::data_port_base,
            wukong::Global::num_servers, wukong::Global::num_threads);

    // init control communicaiton
    wukong::con_adaptor = new wukong::TCP_Adaptor(sid, host_fname, wukong::Global::ctrl_port_base,
                                  wukong::Global::num_servers, wukong::Global::num_proxies);

    // load string server (read-only, shared by all proxies and all engines)
    wukong::StringServer str_server(wukong::Global::input_folder);

    // load RDF graph (shared by all engines and proxies)
    wukong::DGraph dgraph(sid, mem, &str_server, wukong::Global::input_folder);

    // prepare statistics for SPARQL optimizer
    wukong::Stats stats(sid);
    uint64_t t0, t1;
    if (wukong::Global::generate_statistics) {
        t0 = wukong::timer::get_usec();
        stats.generate_statistics(dgraph.gstore);
        t1 = wukong::timer::get_usec();
        logstream(LOG_EMPH)  << "generate statistics using time: " << t1 - t0 << "usec" << LOG_endl;
        stats.gather_stat(wukong::con_adaptor);
    } else {
        t0 = wukong::timer::get_usec();
        std::string fname = wukong::Global::input_folder + "/statfile";  // using default name
        stats.load_stat_from_file(fname, wukong::con_adaptor);
        t1 = wukong::timer::get_usec();
        logstream(LOG_EMPH)  << "load statistics using time: " << t1 - t0 << "usec" << LOG_endl;
    }
    // create proxies and engines
    for (int tid = 0; tid < wukong::Global::num_proxies + wukong::Global::num_engines; tid++) {
        wukong::Adaptor *adaptor = new wukong::Adaptor(tid, tcp_adaptor, rdma_adaptor);

        // TID: proxy = [0, #proxies), engine = [#proxies, #proxies + #engines)
        if (tid < wukong::Global::num_proxies) {
            wukong::Proxy *proxy = new wukong::Proxy(sid, tid, &str_server, &dgraph, adaptor, &stats);
            wukong::proxies.push_back(proxy);
        } else {
            wukong::Engine *engine = new wukong::Engine(sid, tid, &str_server, &dgraph, adaptor);
            wukong::engines.push_back(engine);
        }
    }

    // launch all proxies and engines
    pthread_t *threads  = new pthread_t[wukong::Global::num_threads];
    for (int tid = 0; tid < wukong::Global::num_proxies + wukong::Global::num_engines; tid++) {
        // TID: proxy = [0, #proxies), engine = [#proxies, #proxies + #engines)
        if (tid < wukong::Global::num_proxies)
            pthread_create(&(threads[tid]), NULL, proxy_thread,
                           (void *)wukong::proxies[tid]);
        else
            pthread_create(&(threads[tid]), NULL, engine_thread,
                           (void *)wukong::engines[tid - wukong::Global::num_proxies]);
    }

#ifdef USE_GPU
    logstream(LOG_INFO) << "#" << sid
                        << " #threads:" << wukong::Global::num_threads
                        << ", #proxies:" << wukong::Global::num_proxies
                        << ", #engines:" << wukong::Global::num_engines
                        << ", #agent:" << wukong::Global::num_gpus << LOG_endl;

    // create GPU agent
    wukong::GPUStreamPool stream_pool(32);
    wukong::GPUCache gpu_cache(gpu_mem, dgraph.gstore->vertices, dgraph.gstore->edges,
                       static_cast<wukong::StaticGStore *>(dgraph.gstore)->get_rdf_seg_metas());
    wukong::GPUEngine gpu_engine(sid, WUKONG_GPU_AGENT_TID, gpu_mem, &gpu_cache, &stream_pool, &dgraph);
    wukong::GPUAgent agent(sid, WUKONG_GPU_AGENT_TID, new wukong::Adaptor(WUKONG_GPU_AGENT_TID,
                   tcp_adaptor, rdma_adaptor), &gpu_engine);
    pthread_create(&(threads[WUKONG_GPU_AGENT_TID]), NULL, agent_thread, (void *)&agent);
#endif

    // wait to all threads termination
    for (size_t t = 0; t < wukong::Global::num_threads; t++) {
        if (int rc = pthread_join(threads[t], NULL)) {
            logger(LOG_ERROR, "return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

    /// TODO: exit gracefully (properly call MPI_Init() and MPI_Finalize(), delete all objects)
    return 0;
}
