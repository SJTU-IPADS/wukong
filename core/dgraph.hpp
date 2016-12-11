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

	vector<vector<edge_triple> > triple_spo;
	vector<vector<edge_triple> > triple_ops;

	vector<uint64_t> nedges;

	void remove_duplicate(vector<edge_triple>& elist) {
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

	void inline send_edge(int localtid, int dst_sid, uint64_t s, uint64_t p, uint64_t o) {
		// The RDMA buffer is shared by all threads to communication to all servers
		uint64_t subslot_size = mymath::floor(rdma->get_slotsize() / global_num_servers, sizeof(uint64_t));
		uint64_t *local_buffer = (uint64_t *)(rdma->GetMsgAddr(localtid) + subslot_size * dst_sid);

		// The 1st uint64_t of buffer records the number of triples
		*(local_buffer + (*local_buffer) * 3 + 1) = s;
		*(local_buffer + (*local_buffer) * 3 + 2) = p;
		*(local_buffer + (*local_buffer) * 3 + 3) = o;
		*local_buffer = *local_buffer + 1;

		// Q: what does means of 10, reserve to what?
		if (((*local_buffer) * 3 + 10) * sizeof(uint64_t) >= subslot_size) {
			//full , should be flush!
			flush_edge(localtid, dst_sid);
		}
	}

	void flush_edge(int localtid, int dst_sid) {
		uint64_t subslot_size = mymath::floor(rdma->get_slotsize() / global_num_servers, sizeof(uint64_t));
		uint64_t *local_buffer = (uint64_t *) (rdma->GetMsgAddr(localtid) + subslot_size * dst_sid );
		uint64_t num_edge_to_send = *local_buffer;

		//clear and skip the number infomation
		*local_buffer = 0;
		local_buffer += 1;
		uint64_t max_size = mymath::floor(rdma->get_memorystore_size() / global_num_servers, sizeof(uint64_t));
		uint64_t old_num = __sync_fetch_and_add(&nedges[dst_sid], num_edge_to_send);
		if ((old_num + num_edge_to_send + 1) * 3 * sizeof(uint64_t) >= max_size) {
			cout << "old =" << old_num << endl;
			cout << "num_edge_to_send =" << num_edge_to_send << endl;
			cout << "max_size =" << max_size << endl;
			cout << "Don't have enough space to store data" << endl;
			exit(-1);
		}

		// we need to flush to the same offset of different machine
		uint64_t remote_offset = max_size * sid;
		remote_offset += (old_num * 3 + 1) * sizeof(uint64_t);
		uint64_t remote_length = num_edge_to_send * 3 * sizeof(uint64_t);
		if (dst_sid != sid) {
			rdma->RdmaWrite(localtid, dst_sid, (char*)local_buffer, remote_length, remote_offset);
		} else {
			memcpy(rdma->get_buffer() + remote_offset, (char*)local_buffer, remote_length);
		}
	}

	void load_data(vector<string>& file_vec) {
		uint64_t t1 = timer::get_usec();

		sort(file_vec.begin(), file_vec.end());
		int nfile = file_vec.size();
		volatile int finished_count = 0;

		#pragma omp parallel for num_threads(global_num_engines)
		for (int i = 0; i < nfile; i++) {
			int localtid = omp_get_thread_num();
			if (i % global_num_servers != sid)
				continue;

			if (boost::starts_with(file_vec[i], "hdfs:")) {
				// files located on HDFS
				wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
				wukong::hdfs::fstream file(hdfs, file_vec[i]);
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
				ifstream file(file_vec[i].c_str());
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

			// debug print
			int ret = __sync_fetch_and_add(&finished_count, 1);
			if (ret % 40 == 39) {
				cout << "server " << sid << " already load " << ret + 1 << " files" << endl;
			}
		}

		// flush rest triples within RDMA buffer
		for (int mid = 0; mid < global_num_servers; mid++)
			for (int i = 0; i < global_num_engines; i++)
				flush_edge(i, mid);

		// exchange #triples among all servers
		for (int mid = 0; mid < global_num_servers; mid++) {
			//after flush all data, we need to write the number of total edges;
			uint64_t *local_buffer = (uint64_t *) rdma->GetMsgAddr(0);
			*local_buffer = nedges[mid];
			uint64_t max_size = mymath::floor(rdma->get_memorystore_size() / global_num_servers, sizeof(uint64_t));
			uint64_t remote_offset = max_size * sid;
			if (mid != sid) {
				rdma->RdmaWrite(0, mid, (char*)local_buffer, sizeof(uint64_t), remote_offset);
			} else {
				memcpy(rdma->get_buffer() + remote_offset, (char*)local_buffer, sizeof(uint64_t));
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// timing
		uint64_t t2 = timer::get_usec();
		cout << (t2 - t1) / 1000 << " ms for loading RFD data files" << endl;
	}


	void load_data_from_allfiles(vector<string>& file_vec) {
		sort(file_vec.begin(), file_vec.end());
		int nfile = file_vec.size();

		#pragma omp parallel for num_threads(global_num_engines)
		for (int i = 0; i < nfile; i++) {
			int localtid = omp_get_thread_num();
			uint64_t max_size = mymath::floor(rdma->get_memorystore_size() / global_num_engines, sizeof(uint64_t));
			uint64_t offset = max_size * localtid;
			uint64_t* local_buffer = (uint64_t*)(rdma->get_buffer() + offset);

			// TODO: support HDFS
			ifstream file(file_vec[i].c_str());
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

	void load_and_sync_data(vector<string> &file_vec) {
		uint64_t t1 = timer::get_usec();

#ifdef HAS_RDMA
		load_data(file_vec);
#else
		load_data_from_allfiles(file_vec);
#endif

		int num_recv_block = global_num_servers;
		volatile int finished_count = 0;
		uint64_t total_count = 0;
		for (int mid = 0; mid < num_recv_block; mid++) {
			uint64_t max_size = mymath::floor(rdma->get_memorystore_size() / num_recv_block, sizeof(uint64_t));
			uint64_t offset = max_size * mid;
			uint64_t *recv_buffer = (uint64_t*)(rdma->get_buffer() + offset);
			total_count += *recv_buffer;
		}

		triple_spo.clear();
		triple_ops.clear();
		triple_spo.resize(nthread_parallel_load);
		triple_ops.resize(nthread_parallel_load);
		for (int i = 0; i < triple_spo.size(); i++) {
			triple_spo[i].reserve(total_count / nthread_parallel_load * 1.5);
			triple_ops[i].reserve(total_count / nthread_parallel_load * 1.5);
		}

		#pragma omp parallel for num_threads(nthread_parallel_load)
		for (int t = 0; t < nthread_parallel_load; t++) {
			int local_count = 0;
			for (int mid = 0; mid < num_recv_block; mid++) {
				//recv from different machine
				uint64_t max_size = mymath::floor(rdma->get_memorystore_size() / num_recv_block, sizeof(uint64_t));
				uint64_t offset = max_size * mid;
				uint64_t* recv_buffer = (uint64_t*)(rdma->get_buffer() + offset);
				uint64_t num_edge = *recv_buffer;
				for (uint64_t i = 0; i < num_edge; i++) {
					uint64_t s = recv_buffer[1 + i * 3];
					uint64_t p = recv_buffer[1 + i * 3 + 1];
					uint64_t o = recv_buffer[1 + i * 3 + 2];
					if (mymath::hash_mod(s, global_num_servers) == sid) {
						int s_tableid = (s / global_num_servers) % nthread_parallel_load;
						if ( s_tableid == t)
							triple_spo[t].push_back(edge_triple(s, p, o));
					}

					if (mymath::hash_mod(o, global_num_servers) == sid) {
						int o_tableid = (o / global_num_servers) % nthread_parallel_load;
						if ( o_tableid == t)
							triple_ops[t].push_back(edge_triple(s, p, o));
					}

					local_count++;
					if (local_count == total_count / 100) {
						local_count = 0;
						int ret = __sync_fetch_and_add( &finished_count, 1 );
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

public:
	GStore gstore;

	DGraph(int sid, RdmaResource *rdma, string dname)
		: sid(sid), rdma(rdma), nedges(global_num_servers) {
		vector<string> files; // ID-format data files

		// load the configure file of a batch mode execution
		if (boost::starts_with(dname, "hdfs:")) {
			if (!wukong::hdfs::has_hadoop()) {
				cout << "ERORR: attempting to load data files from HDFS "
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
				/// TODO: move RDF data files and mapping files to different directory
				if (boost::starts_with(fname, dname + "id_"))
					files.push_back(fname);
			}
		}

		if (files.size() == 0)
			cout << "ERORR: no files found in directory (" << dname
			     << ") at server " << sid << endl;

		load_and_sync_data(files);

		// NOTE: the local graph store must be initiated after load_and_sync_data
		gstore.init(rdma, global_num_servers, sid);

		//#pragma omp parallel for num_threads(nthread_parallel_load)
		for (int t = 0; t < nthread_parallel_load; t++) {
			gstore.atomic_batch_insert(triple_spo[t], triple_ops[t]);
			vector<edge_triple>().swap(triple_spo[t]);
			vector<edge_triple>().swap(triple_ops[t]);
		}

		gstore.init_index_table();

		cout << "Server#" << sid << ": loading DGraph is finished." << endl;
		//gstore.print_memory_usage();
	}

	edge *get_edges_global(int tid, uint64_t vid, int direction, int predicate, int* size) {
		return gstore.get_edges_global(tid, vid, direction, predicate, size);
	}

	edge *get_edges_local(int tid, uint64_t vid, int direction, int predicate, int* size) {
		return gstore.get_edges_local(tid, vid, direction, predicate, size);
	}

	edge *get_index_edges_local(int tid, uint64_t vid, int direction, int* size) {
		return gstore.get_index_edges_local(tid, vid, direction, size);
	}
};
