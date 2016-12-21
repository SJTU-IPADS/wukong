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
	static const int nthread_parallel_load = 16;

	int sid;

	RdmaResource *rdma;

	vector<uint64_t> num_triples;  // record #triples loaded from input data for each server

	vector<vector<triple_t>> triple_spo;
	vector<vector<triple_t>> triple_ops;

	void dedup_triples(vector<triple_t> &triples) {
		if (triples.size() <= 1)
			return;

		uint64_t n = 1;
		for (uint64_t i = 1; i < triples.size(); i++) {
			if (triples[i].s == triples[i - 1].s
			        && triples[i].p == triples[i - 1].p
			        && triples[i].o == triples[i - 1].o)
				continue;

			triples[n++] = triples[i];
		}
		triples.resize(n);
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
		uint64_t kvs_sz = floor(rdma->get_kvs_size() / global_num_servers, sizeof(uint64_t));

		// serialize the RDMA WRITEs by multiple threads
		uint64_t exist = __sync_fetch_and_add(&num_triples[dst_sid], n);
		if ((1 + exist * 3 + n * 3) * sizeof(uint64_t) > kvs_sz) {
			cout << "ERROR: no enough space to store input data!" << endl;
			cout << " kvstore size = " << kvs_sz
			     << " #exist-triples = " << exist
			     << " #new-triples = " << n
			     << endl;
			assert(false);
		}

		// send triples and clear the buffer
		uint64_t offset = kvs_sz * sid
		                  + 1 * sizeof(uint64_t)          // reserve the 1st uint64_t as #triples
		                  + exist * 3 * sizeof(uint64_t); // skip #exist-triples
		uint64_t length = n * 3 * sizeof(uint64_t);       // send #new-triples
		if (dst_sid != sid)
			rdma->RdmaWrite(tid, dst_sid, (char *)(buffer + 1), length, offset);
		else
			memcpy(rdma->get_kvs() + offset, (char *)(buffer + 1), length);

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

		// flush the rest triples within each RDMA buffer
		for (int s = 0; s < global_num_servers; s++)
			for (int t = 0; t < global_num_engines; t++)
				flush_edges(t, s);

		// exchange #triples among all servers
		for (int s = 0; s < global_num_servers; s++) {

			uint64_t *buffer = (uint64_t *)rdma->get_buffer(0);
			buffer[0] = num_triples[s];

			uint64_t kvs_sz = floor(rdma->get_kvs_size() / global_num_servers, sizeof(uint64_t));
			uint64_t offset = kvs_sz * sid;
			if (s != sid)
				rdma->RdmaWrite(0, s, (char*)buffer, sizeof(uint64_t), offset);
			else
				memcpy(rdma->get_kvs() + offset, (char*)buffer, sizeof(uint64_t));
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// timing
		uint64_t t2 = timer::get_usec();
		cout << (t2 - t1) / 1000 << " ms for loading RDF data files" << endl;
	}

	// selectively load own partitioned data from all files
	void load_data_from_allfiles(vector<string> &fnames) {
		uint64_t t1 = timer::get_usec();

		sort(fnames.begin(), fnames.end());
		int num_files = fnames.size();

		#pragma omp parallel for num_threads(global_num_engines)
		for (int i = 0; i < num_files; i++) {
			int localtid = omp_get_thread_num();
			uint64_t max_size = floor(rdma->get_kvs_size() / global_num_engines, sizeof(uint64_t));
			uint64_t offset = max_size * localtid;
			uint64_t *local_buffer = (uint64_t *)(rdma->get_kvs() + offset);

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

		// timing
		uint64_t t2 = timer::get_usec();
		cout << (t2 - t1) / 1000 << " ms for loading RDF data files" << endl;
	}

	void aggregate_data() {
		uint64_t t1 = timer::get_usec();

		// calculate #triples on the kvstore from all servers
		uint64_t total = 0;
		uint64_t kvs_sz = floor(rdma->get_kvs_size() / global_num_servers, sizeof(uint64_t));
		for (int id = 0; id < global_num_servers; id++) {
			uint64_t *recv_buffer = (uint64_t *)(rdma->get_kvs() + kvs_sz * id);
			total += recv_buffer[0];
		}

		// pre-expand to avoid frequent reallocation (maybe imbalance)
		for (int i = 0; i < triple_spo.size(); i++) {
			triple_spo[i].reserve(total / nthread_parallel_load);
			triple_ops[i].reserve(total / nthread_parallel_load);
		}

		/* each thread will scan all triples (from all servers) and pickup certain triples.
		   It ensures that the triples belong to the same vertex will be stored in the same
		   triple_spo/ops. This will simplify the deduplication and insertion to gstore. */
		volatile int progress = 0;
		#pragma omp parallel for num_threads(nthread_parallel_load)
		for (int tid = 0; tid < nthread_parallel_load; tid++) {
			int pcnt = 0; // per thread count for print progress
			for (int id = 0; id < global_num_servers; id++) {
				uint64_t *kvs = (uint64_t*)(rdma->get_kvs() + kvs_sz * id);
				uint64_t n = kvs[0];
				for (uint64_t i = 0; i < n; i++) {
					uint64_t s = kvs[1 + i * 3 + 0];
					uint64_t p = kvs[1 + i * 3 + 1];
					uint64_t o = kvs[1 + i * 3 + 2];

					// out-edges
					if (mymath::hash_mod(s, global_num_servers) == sid)
						if ((s % nthread_parallel_load) == tid)
							triple_spo[tid].push_back(triple_t(s, p, o));

					// in-edges
					if (mymath::hash_mod(o, global_num_servers) == sid)
						if ((o % nthread_parallel_load) == tid)
							triple_ops[tid].push_back(triple_t(s, p, o));

					// print the progress (step = 5%) of aggregation
					if (++pcnt >= total * 0.05) {
						int now = __sync_add_and_fetch(&progress, 1);
						if (now % nthread_parallel_load == 0)
							cout << "already aggregrate " << (now / nthread_parallel_load) * 5 << "%" << endl;
						pcnt = 0;
					}
				}
			}

			sort(triple_spo[tid].begin(), triple_spo[tid].end(), edge_sort_by_spo());
			dedup_triples(triple_ops[tid]);

			sort(triple_ops[tid].begin(), triple_ops[tid].end(), edge_sort_by_ops());
			dedup_triples(triple_spo[tid]);
		}

		// timing
		uint64_t t2 = timer::get_usec();
		cout << (t2 - t1) / 1000 << " ms for aggregrate triples" << endl;
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
		: sid(sid), rdma(rdma), num_triples(global_num_servers),
		  triple_spo(nthread_parallel_load), triple_ops(nthread_parallel_load),
		  gstore(rdma, sid) {
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
		 */
		if (global_use_rdma)
			load_data(files);
		else
			load_data_from_allfiles(files);

		/**
		 * all triples are temporarily stored in kvstore.
		 * aggregate, sort and dedup the triples before inserting to gstore (kvstore)
		 */
		aggregate_data();

		// initiate gstore (kvstore) after loading and exchanging triples
		gstore.init();

		#pragma omp parallel for num_threads(nthread_parallel_load)
		for (int t = 0; t < nthread_parallel_load; t++) {
			gstore.atomic_batch_insert(triple_spo[t], triple_ops[t]);

			// release memory
			vector<triple_t>().swap(triple_spo[t]);
			vector<triple_t>().swap(triple_ops[t]);
		}

		gstore.init_index_table();

		cout << "Server#" << sid << ": loading DGraph is finished." << endl;
		// gstore.print_memory_usage();
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
