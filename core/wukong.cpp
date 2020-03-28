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

#include "conflict.hpp"
#include "config.hpp"
#include "bind.hpp"
#include "mem.hpp"
#include "string_server.hpp"
#include "dgraph.hpp"
#include "proxy.hpp"
#include "console.hpp"
#include "rdma.hpp"
#include "optimizer/stats.hpp"

#include "engine/engine.hpp"
#include "comm/adaptor.hpp"

// utils
#include "unit.hpp"
#include "logger2.hpp"

#ifdef USE_GPU

#include "gpu/gpu_mem.hpp"
#include "gpu/gpu_agent.hpp"
#include "gpu/gpu_engine.hpp"
#include "gpu/gpu_cache.hpp"
#include "gpu/gpu_stream.hpp"

void *agent_thread(void *arg)
{
    GPUAgent *agent = (GPUAgent *)arg;
    if (enable_binding && core_bindings.count(agent->tid) != 0)
        bind_to_core(core_bindings[agent->tid]);
    else
        bind_to_core(default_bindings[agent->tid % num_cores]);

    agent->run();
}
#endif  // end of USE_GPU

void *engine_thread(void *arg)
{
    Engine *engine = (Engine *)arg;
    if (enable_binding && core_bindings.count(engine->tid) != 0)
        bind_to_core(core_bindings[engine->tid]);
    else
        bind_to_core(default_bindings[engine->tid % num_cores]);

    engine->run();
}

void *proxy_thread(void *arg)
{
    Proxy *proxy = (Proxy *)arg;
    if (enable_binding && core_bindings.count(proxy->tid) != 0)
        bind_to_core(core_bindings[proxy->tid]);
    else
        bind_to_core(default_bindings[proxy->tid % num_cores]);

    // run the builtin console
    run_console(proxy);
}

static void
usage(char *fn)
{
    cout << "usage: " << fn << " <config_fname> <host_fname> [options]" << endl;
    cout << "options:" << endl;
    cout << "  -b binding : the file of core binding" << endl;
    cout << "  -c command : the one-shot command" << endl;
}

int
main(int argc, char *argv[])
{
    conflict_detector();

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    int sid = world.rank(); // server ID

    if (argc < 3) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    // load global configs
    load_config(string(argv[1]), world.size());

    // set the address file of host/cluster
    string host_fname = std::string(argv[2]);

    // load CPU topology by hwloc
    load_node_topo();
    logstream(LOG_INFO) << "#" << sid << ": has " << num_cores << " cores." << LOG_endl;

    int c;
    while ((c = getopt(argc - 2, argv + 2, "b:c:")) != -1) {
        switch (c) {
        case 'b':
            enable_binding = load_core_binding(optarg);
            break;
        case 'c':
            enable_oneshot = true;
            oneshot_cmd = optarg;
            break;
        default:
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    // allocate memory regions
    vector<RDMA::MemoryRegion> mrs;

    // rdma broadcast memory
    vector<Broadcast_Mem *> bcast_mems;
    Broadcast_Mem *ss_bcast_mem = new Broadcast_Mem(Global::num_servers, Global::num_threads);
    bcast_mems.push_back(ss_bcast_mem);
    // CPU (host) memory
    Mem *mem = new Mem(Global::num_servers, Global::num_threads, bcast_mems);
    logstream(LOG_INFO) << "#" << sid << ": allocate " << B2GiB(mem->size())
                        << "GB memory" << LOG_endl;
    RDMA::MemoryRegion mr_cpu = { RDMA::MemType::CPU, mem->address(), mem->size(), mem };
    mrs.push_back(mr_cpu);

#ifdef USE_GPU
    // GPU (device) memory
    int devid = 0; // FIXME: it means one GPU device?
    GPUMem *gpu_mem = new GPUMem(devid, Global::num_servers, Global::num_gpus);
    logstream(LOG_INFO) << "#" << sid << ": allocate " << B2GiB(gpu_mem->size())
                        << "GB GPU memory" << LOG_endl;
    RDMA::MemoryRegion mr_gpu = { RDMA::MemType::GPU, gpu_mem->address(), gpu_mem->size(), gpu_mem };
    mrs.push_back(mr_gpu);
#endif

    // RDMA full-link communication
    int flink_nthreads = Global::num_proxies + Global::num_engines;
    // RDMA broadcast communication
    int bcast_nthreads = 2;
    int rdma_init_nthreads = flink_nthreads + bcast_nthreads;
    // init RDMA devices and connections
    RDMA_init(Global::num_servers, rdma_init_nthreads, sid, mrs, host_fname);

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
    Stats stats(sid);
    uint64_t t0, t1;
    if (Global::generate_statistics) {
        t0 = timer::get_usec();
        stats.generate_statistics(dgraph.gstore);
        t1 = timer::get_usec();
        logstream(LOG_EMPH)  << "generate statistics using time: " << t1 - t0 << "usec" << LOG_endl;
        stats.gather_stat(con_adaptor);
    } else {
        t0 = timer::get_usec();
        string fname = Global::input_folder + "/statfile";  // using default name
        stats.load_stat_from_file(fname, con_adaptor);
        t1 = timer::get_usec();
        logstream(LOG_EMPH)  << "load statistics using time: " << t1 - t0 << "usec" << LOG_endl;
    }
    // create proxies and engines
    for (int tid = 0; tid < Global::num_proxies + Global::num_engines; tid++) {
        Adaptor *adaptor = new Adaptor(tid, tcp_adaptor, rdma_adaptor);

        // TID: proxy = [0, #proxies), engine = [#proxies, #proxies + #engines)
        if (tid < Global::num_proxies) {
            Proxy *proxy = new Proxy(sid, tid, &str_server, &dgraph, adaptor, &stats);
            proxies.push_back(proxy);
        } else {
            Engine *engine = new Engine(sid, tid, &str_server, &dgraph, adaptor);
            engines.push_back(engine);
        }
    }

    // launch all proxies and engines
    pthread_t *threads  = new pthread_t[Global::num_threads];
    for (int tid = 0; tid < Global::num_proxies + Global::num_engines; tid++) {
        // TID: proxy = [0, #proxies), engine = [#proxies, #proxies + #engines)
        if (tid < Global::num_proxies)
            pthread_create(&(threads[tid]), NULL, proxy_thread,
                           (void *)proxies[tid]);
        else
            pthread_create(&(threads[tid]), NULL, engine_thread,
                           (void *)engines[tid - Global::num_proxies]);
    }

#ifdef USE_GPU
    logstream(LOG_INFO) << "#" << sid
                        << " #threads:" << Global::num_threads
                        << ", #proxies:" << Global::num_proxies
                        << ", #engines:" << Global::num_engines
                        << ", #agent:" << Global::num_gpus << LOG_endl;

    // create GPU agent
    GPUStreamPool stream_pool(32);
    GPUCache gpu_cache(gpu_mem, dgraph.gstore->vertices, dgraph.gstore->edges,
                       static_cast<StaticGStore *>(dgraph.gstore)->get_rdf_seg_metas());
    GPUEngine gpu_engine(sid, WUKONG_GPU_AGENT_TID, gpu_mem, &gpu_cache, &stream_pool, &dgraph);
    GPUAgent agent(sid, WUKONG_GPU_AGENT_TID, new Adaptor(WUKONG_GPU_AGENT_TID,
                   tcp_adaptor, rdma_adaptor), &gpu_engine);
    pthread_create(&(threads[WUKONG_GPU_AGENT_TID]), NULL, agent_thread, (void *)&agent);
#endif

    // wait to all threads termination
    for (size_t t = 0; t < Global::num_threads; t++) {
        if (int rc = pthread_join(threads[t], NULL)) {
            logger(LOG_ERROR, "return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

    /// TODO: exit gracefully (properly call MPI_Init() and MPI_Finalize(), delete all objects)
    return 0;
}
