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

#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <boost/mpi.hpp>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include "omp.h"

#include "config.hpp"
#include "graph_basic_types.hpp"
#include "rdma_resource.hpp"
#include "gstore.hpp"
#include "timer.hpp"

using namespace std;

class DGraph {
	static const int nthread_parallel_load = 20;

	int sid;

	RdmaResource *rdma;

	vector<vector<triple_t> > triple_spo;
	vector<vector<triple_t> > triple_ops;

	vector<uint64_t> num_triples;  // record #triples loaded from input data for each server

	void remove_duplicate(vector<triple_t>& elist) {
		if (elist.size() > 1) {
			uint64_t end = 1;
			for (uint64_t i = 1; i < elist.size(); i++) {
				if (elist[i].s == elist[i - 1].s &&
				        elist[i].p == elist[i - 1].p &&
				        elist[i].o == elist[i - 1].o) {
					continue;
				}
				elist[end] = elist[i];
				end++;
			}
			elist.resize(end);
		}
	}

	// NOTE: send_edge can be safely called by multiple threads, since the buffer
	//       is exclusively used by one thread.
	void send_edge(int tid, int dst_sid, uint64_t s, uint64_t p, uint64_t o) {
		// the RDMA buffer is first split into T partitions (T is the number of threads)
		// each partition is further split into S pieces (S is the number of servers)
		uint64_t buffer_sz = floor(rdma->get_buffer_size() / global_num_servers, sizeof(uint64_t));
		uint64_t *buffer = (uint64_t *)(rdma->get_buffer(tid) + buffer_sz * dst_sid);

		// the 1st uint64_t of buffer records #triples
		uint64_t n = buffer[0];

		// flush buffer if there is no enough space to buffer a new triple
		if ((1 + n * 3 + 3) * sizeof(uint64_t) > buffer_sz) {
			flush_edges(tid, dst_sid);
			n = buffer[0]; // reset, it should be 0
		}

		// buffer the triple and update the counter
		buffer[1 + n * 3 + 0] = s;
		buffer[1 + n * 3 + 1] = p;
		buffer[1 + n * 3 + 2] = o;
		buffer[0] = n + 1;
	}

	void flush_edges(int tid, int dst_sid) {
		uint64_t buffer_sz = floor(rdma->get_buffer_size() / global_num_servers, sizeof(uint64_t));
		uint64_t *buffer = (uint64_t *)(rdma->get_buffer(tid) + buffer_sz * dst_sid);

		// the 1st uint64_t of buffer records #new-triples
		uint64_t n = buffer[0];

		// the kvstore is split into S pieces (S is the number of servers).
		// hence, the kvstore can be directly RDMA write in parallel by all servers
		uint64_t kvstore_sz = floor(rdma->get_kvstore_size() / global_num_servers, sizeof(uint64_t));

		// serialize the RDMA WRITEs by multiple threads
		uint64_t exist = __sync_fetch_and_add(&num_triples[dst_sid], n);
		if ((1 + exist * 3 + n * 3) * sizeof(uint64_t) > kvstore_sz) {
			cout << "ERROR: no enough space to store input data!" << endl;
			cout << " kvstore size = " << kvstore_sz
			     << " #exist-triples = " << exist
			     << " #new-triples = " << n
			     << endl;
			assert(false);
		}

		// send triples and clear the buffer
		uint64_t offset = kvstore_sz * sid
		                  + 1 * sizeof(uint64_t)          // reserve the 1st uint64_t as #triples
		                  + exist * 3 * sizeof(uint64_t); // skip #exist-triples
		uint64_t length = n * 3 * sizeof(uint64_t);       // send #new-triples
		if (dst_sid != sid)
			rdma->RdmaWrite(tid, dst_sid, (char *)(buffer + 1), length, offset);
		else
			memcpy(rdma->get_kvstore() + offset, (char *)(buffer + 1), length);

		buffer[0] = 0; // clear the buffer
	}

	void load_data(vector<string>& fnames) {
		uint64_t t1 = timer::get_usec();

		int num_files = fnames.size();

		// ensure the file name list has the same order on all servers
		sort(fnames.begin(), fnames.end());

		// load input data and assign to different severs in parallel
		#pragma omp parallel for num_threads(global_num_engines)
		for (int i = 0; i < num_files; i++) {
			int localtid = omp_get_thread_num();

			// each server only load a part of files
			if (i % global_num_servers != sid) continue;

			if (boost::starts_with(fnames[i], "hdfs:")) {
				// files located on HDFS
				wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
				wukong::hdfs::fstream file(hdfs, fnames[i]);
				uint64_t s, p, o;
				while (file >> s >> p >> o) {
					int s_sid = mymath::hash_mod(s, global_num_servers);
					int o_sid = mymath::hash_mod(o, global_num_servers);
					if (s_sid == o_sid) {
						send_edge(localtid, s_sid, s, p, o);
					} else {
						send_edge(localtid, s_sid, s, p, o);
						send_edge(localtid, o_sid, s, p, o);
					}
				}
			} else {
				// files located on a shared filesystem (e.g., NFS)
				ifstream file(fnames[i].c_str());
				uint64_t s, p, o;
				while (file >> s >> p >> o) {
					int s_sid = mymath::hash_mod(s, global_num_servers);
					int o_sid = mymath::hash_mod(o, global_num_servers);
					if (s_sid == o_sid) {
						send_edge(localtid, s_sid, s, p, o);
					} else {
						send_edge(localtid, s_sid, s, p, o);
						send_edge(localtid, o_sid, s, p, o);
					}
				}
				file.close();
			}
		}

		// flush rest triples within each RDMA buffer
		for (int s = 0; s < global_num_servers; s++)
			for (int t = 0; t < global_num_engines; t++)
				flush_edges(t, s);

		// exchange #triples among all servers
		for (int s = 0; s < global_num_servers; s++) {

			uint64_t *buffer = (uint64_t *)rdma->get_buffer(0);
			buffer[0] = num_triples[s];

			uint64_t kvstore_sz = floor(rdma->get_kvstore_size() / global_num_servers, sizeof(uint64_t));
			uint64_t offset = kvstore_sz * sid;
			if (s != sid)
				rdma->RdmaWrite(0, s, (char*)buffer, sizeof(uint64_t), offset);
			else
				memcpy(rdma->get_kvstore() + offset, (char*)buffer, sizeof(uint64_t));
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// timing
		uint64_t t2 = timer::get_usec();
		cout << (t2 - t1) / 1000 << " ms for loading RDF data files" << endl;
	}

	// selectively load own partitioned data from allfiles
	void load_data_from_allfiles(vector<string> &fnames) {
		sort(fnames.begin(), fnames.end());
		int num_files = fnames.size();

		#pragma omp parallel for num_threads(global_num_engines)
		for (int i = 0; i < num_files; i++) {
			int localtid = omp_get_thread_num();
			uint64_t max_size = floor(rdma->get_kvstore_size() / global_num_engines, sizeof(uint64_t));
			uint64_t offset = max_size * localtid;
			uint64_t *local_buffer = (uint64_t *)(rdma->get_kvstore() + offset);

			// TODO: support HDFS
			ifstream file(fnames[i].c_str());
			uint64_t s, p, o;
			while (file >> s >> p >> o) {
				int s_mid = mymath::hash_mod(s, global_num_servers);
				int o_mid = mymath::hash_mod(o, global_num_servers);
				if (s_mid == sid || o_mid == sid) {
					*(local_buffer + (*local_buffer) * 3 + 1) = s;
					*(local_buffer + (*local_buffer) * 3 + 2) = p;
					*(local_buffer + (*local_buffer) * 3 + 3) = o;
					*local_buffer = *local_buffer + 1;
					if ((*local_buffer + 1) * 3 * sizeof(uint64_t) >= max_size) {
						cout << "[fail to execute load_data_from_allfiles] Out of memory" << endl;
						exit(-1);
					}
				}
			}
			file.close();
		}
	}

	void load_and_sync_data(vector<string> &files) {
		uint64_t t1 = timer::get_usec();

		/**
		 * load_data: load partial input files by each server and exchanges triples
		 *            according to graph partitioning
		 * load_data_from_allfiles: load all files by each server and select triples
		 *                          according to graph partitioning
		 *
		 * Trade-off: load_data_from_allfiles avoids network traffic and memory,
		 *            but it requires more I/O from distributed FS.
		 *
		 * Wukong adopts load_data_from_allfiles for slow network (w/o RDMA) and
		 *        adopts load_data for fast network (w/ RDMA).
		 *
		 * Tips: the buffer (registered memory) can be reused for further primitives for RDMA
		 *
		 */
		if (global_use_rdma)
			load_data(files);
		else
			load_data_from_allfiles(files);

		uint64_t total = 0;
		for (int id = 0; id < global_num_servers; id++) {
			uint64_t max_size = floor(rdma->get_kvstore_size() / global_num_servers, sizeof(uint64_t));
			uint64_t *recv_buffer = (uint64_t*)(rdma->get_kvstore() + max_size * id);
			total += *recv_buffer;
		}

		triple_spo.clear();
		triple_ops.clear();
		triple_spo.resize(nthread_parallel_load);
		triple_ops.resize(nthread_parallel_load);
		for (int i = 0; i < triple_spo.size(); i++) {
			triple_spo[i].reserve(total / nthread_parallel_load * 1.5);
			triple_ops[i].reserve(total / nthread_parallel_load * 1.5);
		}

		volatile int done = 0;
		#pragma omp parallel for num_threads(nthread_parallel_load)
		for (int t = 0; t < nthread_parallel_load; t++) {
			int local_count = 0;
			for (int mid = 0; mid < global_num_servers; mid++) {
				//recv from different machine
				uint64_t max_size = floor(rdma->get_kvstore_size() / global_num_servers, sizeof(uint64_t));
				uint64_t* recv_buffer = (uint64_t*)(rdma->get_kvstore() + max_size * mid);
				uint64_t num_edge = *recv_buffer;
				for (uint64_t i = 0; i < num_edge; i++) {
					uint64_t s = recv_buffer[1 + i * 3];
					uint64_t p = recv_buffer[1 + i * 3 + 1];
					uint64_t o = recv_buffer[1 + i * 3 + 2];
					if (mymath::hash_mod(s, global_num_servers) == sid) {
						int s_tableid = (s / global_num_servers) % nthread_parallel_load;
						if ( s_tableid == t)
							triple_spo[t].push_back(triple_t(s, p, o));
					}

					if (mymath::hash_mod(o, global_num_servers) == sid) {
						int o_tableid = (o / global_num_servers) % nthread_parallel_load;
						if ( o_tableid == t)
							triple_ops[t].push_back(triple_t(s, p, o));
					}

					local_count++;
					if (local_count == total / 100) {
						local_count = 0;
						int ret = __sync_fetch_and_add( &done, 1 );
						if ((ret + 1) % (nthread_parallel_load * 5) == 0)
							cout << "already aggregrate " << (ret + 1) / nthread_parallel_load << " %" << endl;
					}
				}
			}
			sort(triple_spo[t].begin(), triple_spo[t].end(), edge_sort_by_spo());
			sort(triple_ops[t].begin(), triple_ops[t].end(), edge_sort_by_ops());
			remove_duplicate(triple_spo[t]);
			remove_duplicate(triple_ops[t]);
		}

		uint64_t t2 = timer::get_usec();
		cout << (t2 - t1) / 1000 << " ms for aggregrate edges" << endl;
	}

	uint64_t inline floor(uint64_t original, uint64_t n) {
		assert(n != 0);
		return original - original % n;
	}

	uint64_t inline ceil(uint64_t original, uint64_t n) {
		assert(n != 0);
		if (original % n == 0)
			return original;
		return original - original % n + n;
	}

public:
	GStore gstore;

	DGraph(string dname, int sid, RdmaResource *rdma)
		: sid(sid), rdma(rdma), num_triples(global_num_servers) {
		vector<string> files; // ID-format data files

		if (boost::starts_with(dname, "hdfs:")) {
			if (!wukong::hdfs::has_hadoop()) {
				cout << "ERROR: attempting to load data files from HDFS "
				     << "but Wukong was built without HDFS."
				     << endl;
				exit(-1);
			}

			wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
			files = hdfs.list_files(dname);
		} else {
			// files located on a shared filesystem (e.g., NFS)
			DIR *dir = opendir(dname.c_str());
			if (dir == NULL) {
				cout << "ERORR: failed to open directory (" << dname
				     << ") at server " << sid << endl;
				exit(-1);
			}

			struct dirent *ent;
			while ((ent = readdir(dir)) != NULL) {
				if (ent->d_name[0] == '.')
					continue;

				string fname(dname + ent->d_name);
				// Assume the filenames of RDF data files (ID-format) start with 'id_'.
				/// TODO: move RDF data files and metadata files to different directories
				if (boost::starts_with(fname, dname + "id_"))
					files.push_back(fname);
			}
		}

		if (files.size() == 0) {
			cout << "ERORR: no files found in directory (" << dname
			     << ") at server " << sid << endl;
			assert(false);
		}

		load_and_sync_data(files);

		// NOTE: the local graph store must be initiated after load_and_sync_data
		gstore.init(rdma, sid);

		//#pragma omp parallel for num_threads(nthread_parallel_load)
		for (int t = 0; t < nthread_parallel_load; t++) {
			gstore.atomic_batch_insert(triple_spo[t], triple_ops[t]);
			vector<triple_t>().swap(triple_spo[t]);
			vector<triple_t>().swap(triple_ops[t]);
		}

		gstore.init_index_table();

		cout << "Server#" << sid << ": loading DGraph is finished." << endl;
		//gstore.print_memory_usage();
	}

	edge_t *get_edges_global(int tid, uint64_t vid, int direction, int predicate, int *size) {
		return gstore.get_edges_global(tid, vid, direction, predicate, size);
	}

	edge_t *get_edges_local(int tid, uint64_t vid, int direction, int predicate, int *size) {
		return gstore.get_edges_local(tid, vid, direction, predicate, size);
	}

	edge_t *get_index_edges_local(int tid, uint64_t vid, int direction, int *size) {
		return gstore.get_index_edges_local(tid, vid, direction, size);
	}
};
