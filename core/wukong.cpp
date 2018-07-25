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
#include "logger2.hpp"
#include <map>
#include <boost/mpi.hpp>
#include <iostream>

#include "config.hpp"
#include "bind.hpp"
#include "mem.hpp"
#ifdef USE_GPU
#include "gpu_mem.hpp"
#endif
#include "string_server.hpp"
#include "dgraph.hpp"
#include "engine.hpp"
#include "proxy.hpp"
#include "console.hpp"
#include "rdma.hpp"
#include "adaptor.hpp"

#include "unit.hpp"

#include "data_statistic.hpp"

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
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    int sid = world.rank(); // server ID
    int devid = 0;

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

    // allocate memory
    Mem *mem = new Mem(global_num_servers, global_num_threads);
    logstream(LOG_INFO)  << "#" << sid << ": allocate " << B2GiB(mem->memory_size()) << "GB memory" << LOG_endl;
    vector<RDMA::MemoryRegion> mrs;
    RDMA::MemoryRegion mr_cpu = { mem->memory(), mem->memory_size(), RDMA::MemType::CPU };
    mrs.push_back(mr_cpu);
    #ifdef USE_GPU
    GPUMem *gpu_mem = new GPUMem(devid, global_num_servers, global_num_gpus);
    logstream(LOG_INFO)  << "#" << sid << ": allocate " << B2GiB(gpu_mem->memory_size()) << "GB GPU memory" << LOG_endl;
    RDMA::MemoryRegion mr_gpu = { gpu_mem->memory(), gpu_mem->memory_size(), RDMA::MemType::GPU };
    mrs.push_back(mr_gpu);
    #endif
    // init RDMA devices and connections
    RDMA_init(global_num_servers, global_num_threads, sid, mrs, host_fname);

    // init communication
    RDMA_Adaptor *rdma_adaptor = new RDMA_Adaptor(sid, mem, global_num_servers, global_num_threads);
    TCP_Adaptor *tcp_adaptor = new TCP_Adaptor(sid, host_fname, global_num_threads, global_data_port_base);

    // load string server (read-only, shared by all proxies and all engines)
    String_Server str_server(global_input_folder);

    // load RDF graph (shared by all engines and proxies)
    DGraph dgraph(sid, mem, &str_server, global_input_folder);

    // init control communicaiton
    con_adaptor = new TCP_Adaptor(sid, host_fname, global_num_proxies, global_ctrl_port_base);

    // prepare statistics for SPARQL optimizer
    data_statistic stat(sid);
    if (global_enable_planner) {
        if (global_generate_statistics) {
            uint64_t t0 = timer::get_usec();
            dgraph.gstore.generate_statistic(stat);
            uint64_t t1 = timer::get_usec();
            logstream(LOG_EMPH)  << "generate_statistic using time: " << t1 - t0 << "usec" << LOG_endl;
            stat.gather_stat(con_adaptor);
        } else {
            // use the dataset name by default
            string fname = global_input_folder + "/statfile";
            stat.load_stat_from_file(fname, con_adaptor);
        }
    }

    // create proxies and engines
    ASSERT(global_num_threads == global_num_proxies + global_num_engines);
    for (int tid = 0; tid < global_num_threads; tid++) {
        Adaptor *adaptor = new Adaptor(tid, tcp_adaptor, rdma_adaptor);

        // TID: proxy = [0, #proxies), engine = [#proxies, #proxies + #engines)
        if (tid < global_num_proxies) {
            Proxy *proxy = new Proxy(sid, tid, &str_server, &dgraph, adaptor, &stat);
            proxies.push_back(proxy);
        } else {
            Engine *engine = new Engine(sid, tid, &str_server, &dgraph, adaptor);
            engines.push_back(engine);
        }
    }

    // launch all proxies and engines
    pthread_t *threads  = new pthread_t[global_num_threads];
    for (int tid = 0; tid < global_num_threads; tid++) {
        // TID: proxy = [0, #proxies), engine = [#proxies, #proxies + #engines)
        if (tid < global_num_proxies)
            pthread_create(&(threads[tid]), NULL, proxy_thread, (void *)proxies[tid]);
        else
            pthread_create(&(threads[tid]), NULL, engine_thread, (void *)engines[tid - global_num_proxies]);
    }

    // wait to all threads termination
    for (size_t t = 0; t < global_num_threads; t++) {
        if (int rc = pthread_join(threads[t], NULL)) {
            logger(LOG_ERROR, "return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

    /// TODO: exit gracefully (properly call MPI_Init() and MPI_Finalize(), delete all objects)
    return 0;
}
