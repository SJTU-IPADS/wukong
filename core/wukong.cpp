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

#include "config.hpp"
#include "bind.hpp"
#include "mem.hpp"
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
	cout << "INFO#" << sid << ": has " << num_cores << " cores." << endl;

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
	cout << "INFO#" << sid << ": allocate " << B2GiB(mem->memory_size()) << "GB memory" << endl;

	// init RDMA devices and connections
	RDMA_init(global_num_servers, global_num_threads,
	          sid, mem->memory(), mem->memory_size(), host_fname);

	// init data communication
	RDMA_Adaptor *rdma_adaptor = new RDMA_Adaptor(sid, mem, global_num_servers, global_num_threads);
	TCP_Adaptor *tcp_adaptor = new TCP_Adaptor(sid, host_fname, global_num_threads, global_data_port_base);

	// load string server (read-only, shared by all proxies)
	String_Server str_server(global_input_folder);

	// load RDF graph (shared by all engines)
	DGraph dgraph(sid, mem, &str_server, global_input_folder);

	// prepare data for planner
	data_statistic stat(tcp_adaptor, &world);
	if (global_enable_planner) {
		dgraph.gstore.generate_statistic(stat);
		stat.gather_data();
	}

	// init control communicaiton
	con_adaptor = new TCP_Adaptor(sid, host_fname, global_num_proxies, global_ctrl_port_base);

	// launch proxy and engine threads
	assert(global_num_threads == global_num_proxies + global_num_engines);
	pthread_t *threads  = new pthread_t[global_num_threads];
	for (int tid = 0; tid < global_num_threads; tid++) {
		Adaptor *adaptor = new Adaptor(tid, tcp_adaptor, rdma_adaptor);

		// TID: proxy = [0, #proxies), engine = [#proxies, #proxies + #engines)
		if (tid < global_num_proxies) {
			Proxy *proxy = new Proxy(sid, tid, &str_server, adaptor, &stat);
			pthread_create(&(threads[tid]), NULL, proxy_thread, (void *)proxy);
			proxies.push_back(proxy);
		} else {
			Engine *engine = new Engine(sid, tid, &dgraph, adaptor);
			pthread_create(&(threads[tid]), NULL, engine_thread, (void *)engine);
			engines.push_back(engine);
		}
	}

	// wait to all threads termination
	for (size_t t = 0; t < global_num_threads; t++) {
		int rc = pthread_join(threads[t], NULL);
		if (rc) {
			printf("ERROR: return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
	}

	/// TODO: exit gracefully (properly call MPI_Init() and MPI_Finalize(), delete all objects)
	return 0;
}
