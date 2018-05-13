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

#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <unordered_set>
#include <vector>
#include <algorithm>
// #include <assert.h>
#include "assertion.hpp"
#include <boost/mpi.hpp>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include "omp.h"

#include "config.hpp"
#include "type.hpp"
#include "rdma.hpp"
#include "gstore.hpp"
#include "timer.hpp"

using namespace std;

struct triple_sort_by_spo {
	inline bool operator()(const triple_t &t1, const triple_t &t2) {
		if (t1.s < t2.s)
			return true;
		else if (t1.s == t2.s)
			if (t1.p < t2.p)
				return true;
			else if (t1.p == t2.p && t1.o < t2.o)
				return true;
		return false;
	}
};

struct triple_sort_by_ops {
	inline bool operator()(const triple_t &t1, const triple_t &t2) {
		if (t1.o < t2.o)
			return true;
		else if (t1.o == t2.o)
			if (t1.p < t2.p)
				return true;
			else if ((t1.p == t2.p) && (t1.s < t2.s))
				return true;
		return false;
	}
};

/**
 * Map the RDF model (e.g., triples, predicate) to Graph model (e.g., vertex, edge, index)
 */
class DGraph {
	int sid;

	Mem *mem;

	String_Server *str_server;

	vector<uint64_t> num_triples;  // record #triples loaded from input data for each server

	vector<vector<triple_t>> triple_spo;
	vector<vector<triple_t>> triple_ops;
	vector<vector<triple_attr_t>> triple_sav;

#if DYNAMIC_GSTORE
	boost::unordered_map<sid_t, sid_t> id2id;

	void flush_convertmap() { id2id.clear(); }

	void convert_sid(sid_t &sid) {
		if (id2id.find(sid) != id2id.end())
			sid = id2id[sid];
	}

	bool check_sid(const sid_t id) {
		if (!str_server->exist(id)) {
			logstream(LOG_WARNING) << "Unknown SID: " << id << LOG_endl;
			return false;
		}
		return true;
	}

	void dynamic_load_mappings(string dname) {
		DIR *dir = opendir(dname.c_str());

		if (dir == NULL) {
			logstream(LOG_ERROR) << "failed to open the directory of ID-mapping files ("
			                     << dname << ")." << LOG_endl;
			exit(-1);
		}

		struct dirent *ent;
		while ((ent = readdir(dir)) != NULL) {
			if (ent->d_name[0] == '.')
				continue;
			string fname(dname + ent->d_name);
			if (boost::ends_with(fname, "/str_index")
			        || boost::ends_with(fname, "/str_normal")) {
				logstream(LOG_INFO) << "loading ID-mapping file: " << fname << LOG_endl;
				ifstream file(fname.c_str());
				string str;
				sid_t id;
				while (file >> str >> id) {
					if (str_server->exist(str)) {
						id2id[id] = str_server->str2id[str];
					} else {
						if (boost::ends_with(fname, "/str_index"))
							id2id[id] = str_server->next_index_id ++;
						else
							id2id[id] = str_server->next_normal_id ++;
						str_server->str2id[str] = id2id[id];
						str_server->id2str[id2id[id]] = str;
					}
				}
				file.close();
			}
		}
	}
#endif // DYNAMIC_GSTORE

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

	void flush_triples(int tid, int dst_sid) {
		uint64_t buf_sz = floor(mem->buffer_size() / global_num_servers - sizeof(uint64_t), sizeof(sid_t));
		uint64_t *pn = (uint64_t *)(mem->buffer(tid) + (buf_sz + sizeof(uint64_t)) * dst_sid);
		sid_t *buf = (sid_t *)(pn + 1);

		// the 1st uint64_t of buffer records #new-triples
		uint64_t n = *pn;

		// the kvstore is temporally split into #servers pieces.
		// hence, the kvstore can be directly RDMA write in parallel by all servers
		uint64_t kvs_sz = floor(mem->kvstore_size() / global_num_servers - sizeof(uint64_t), sizeof(sid_t));

		// serialize the RDMA WRITEs by multiple threads
		uint64_t exist = __sync_fetch_and_add(&num_triples[dst_sid], n);
		if ((exist * 3 + n * 3) * sizeof(sid_t) > kvs_sz) {
			logstream(LOG_ERROR) << "no enough space to store input data!" << LOG_endl;
			logstream(LOG_ERROR) << " kvstore size = " << kvs_sz
			                     << " #exist-triples = " << exist
			                     << " #new-triples = " << n
			                     << LOG_endl;
			ASSERT(false);
		}

		// send triples and clear the buffer
		uint64_t off = (kvs_sz + sizeof(uint64_t)) * sid
		               + sizeof(uint64_t)           // reserve the 1st uint64_t as #triples
		               + exist * 3 * sizeof(sid_t); // skip #exist-triples
		uint64_t sz = n * 3 * sizeof(sid_t);        // send #new-triples
		if (dst_sid != sid) {
			RDMA &rdma = RDMA::get_rdma();
			rdma.dev->RdmaWrite(tid, dst_sid, (char *)buf, sz, off);
		} else {
			memcpy(mem->kvstore() + off, (char *)buf, sz);
		}

		// clear the buffer
		*pn = 0;
	}

	// send_triple can be safely called by multiple threads,
	// since the buffer is exclusively used by one thread.
	void send_triple(int tid, int dst_sid, sid_t s, sid_t p, sid_t o) {
		// the RDMA buffer is first split into #threads partitions
		// each partition is further split into #servers pieces
		// each piece: #triple, tirple, triple, . . .
		uint64_t buf_sz = floor(mem->buffer_size() / global_num_servers - sizeof(uint64_t), sizeof(sid_t));
		uint64_t *pn = (uint64_t *)(mem->buffer(tid) + (buf_sz + sizeof(uint64_t)) * dst_sid);
		sid_t *buf = (sid_t *)(pn + 1);

		// the 1st entry of buffer records #triples (suppose the )
		uint64_t n = *pn;

		// flush buffer if there is no enough space to buffer a new triple
		if ((n * 3 + 3) * sizeof(sid_t) > buf_sz) {
			flush_triples(tid, dst_sid);
			n = *pn; // reset, it should be 0
			ASSERT(n == 0);
		}

		// buffer the triple and update the counter
		buf[n * 3 + 0] = s;
		buf[n * 3 + 1] = p;
		buf[n * 3 + 2] = o;
		*pn = (n + 1);
	}

	int load_data(vector<string> &fnames) {
		// ensure the file name list has the same order on all servers
		sort(fnames.begin(), fnames.end());

		// load input data and assign to different severs in parallel
		int num_files = fnames.size();
		#pragma omp parallel for num_threads(global_num_engines)
		for (int i = 0; i < num_files; i++) {
			int localtid = omp_get_thread_num();

			// each server only load a part of files
			if (i % global_num_servers != sid) continue;

			auto lambda = [&](istream & file) {
				sid_t s, p, o;
				while (file >> s >> p >> o) {
					int s_sid = mymath::hash_mod(s, global_num_servers);
					int o_sid = mymath::hash_mod(o, global_num_servers);
					if (s_sid == o_sid) {
						send_triple(localtid, s_sid, s, p, o);
					} else {
						send_triple(localtid, s_sid, s, p, o);
						send_triple(localtid, o_sid, s, p, o);
					}
				}
			};

			if (boost::starts_with(fnames[i], "hdfs:")) {
				// files located on HDFS
				wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
				wukong::hdfs::fstream file(hdfs, fnames[i]);
				lambda(file);
				file.close();
			} else {
				// files located on a shared filesystem (e.g., NFS)
				ifstream file(fnames[i].c_str());
				lambda(file);
				file.close();
			}
		}

		// flush the rest triples within each RDMA buffer
		for (int s = 0; s < global_num_servers; s++)
			for (int t = 0; t < global_num_engines; t++)
				flush_triples(t, s);

		// exchange #triples among all servers
		for (int s = 0; s < global_num_servers; s++) {
			uint64_t *buf = (uint64_t *)mem->buffer(0);
			buf[0] = num_triples[s];

			uint64_t kvs_sz = floor(mem->kvstore_size() / global_num_servers, sizeof(sid_t));
			uint64_t offset = kvs_sz * sid;
			if (s != sid) {
				RDMA &rdma = RDMA::get_rdma();
				rdma.dev->RdmaWrite(0, s, (char*)buf, sizeof(uint64_t), offset);
			} else {
				memcpy(mem->kvstore() + offset, (char*)buf, sizeof(uint64_t));
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

		return global_num_servers;
	}

	// selectively load own partitioned data from all files
	int load_data_from_allfiles(vector<string> &fnames) {
		sort(fnames.begin(), fnames.end());

		int num_files = fnames.size();
		#pragma omp parallel for num_threads(global_num_engines)
		for (int i = 0; i < num_files; i++) {
			int localtid = omp_get_thread_num();
			uint64_t kvs_sz = floor(mem->kvstore_size() / global_num_engines - sizeof(uint64_t), sizeof(sid_t));
			uint64_t *pn = (uint64_t *)(mem->kvstore() + (kvs_sz + sizeof(uint64_t)) * localtid);
			sid_t *kvs = (sid_t *)(pn + 1);

			// the 1st uint64_t of kvs records #triples
			uint64_t n = *pn;

			auto lambda = [&](istream & file) {
				sid_t s, p, o;
				while (file >> s >> p >> o) {
					int s_sid = mymath::hash_mod(s, global_num_servers);
					int o_sid = mymath::hash_mod(o, global_num_servers);
					if ((s_sid == sid) || (o_sid == sid)) {
						ASSERT((n * 3 + 3) * sizeof(sid_t) <= kvs_sz);
						// buffer the triple and update the counter
						kvs[n * 3 + 0] = s;
						kvs[n * 3 + 1] = p;
						kvs[n * 3 + 2] = o;
						n++;
					}
				}
			};

			if (boost::starts_with(fnames[i], "hdfs:")) {
				// files located on HDFS
				wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
				wukong::hdfs::fstream file(hdfs, fnames[i]);
				lambda(file);
				file.close();
			} else {
				ifstream file(fnames[i].c_str());
				lambda(file);
				file.close();
			}
			*pn = n;
		}

		return global_num_engines;
	}

	// selectively load own partitioned data (attributes) from all files
	void load_attr_from_allfiles(vector<string> &fnames) {
		if (fnames.size() == 0)
			return; // no attributed files

		sort(fnames.begin(), fnames.end());

		//parallel load from all files
		int num_files = fnames.size();
		#pragma omp parallel for num_threads(global_num_engines)
		for (int i = 0; i < num_files; i++) {
			int localtid = omp_get_thread_num();
			auto load_attr = [&](istream & file) {
				sid_t s, a;
				attr_t v;
				int type;
				while (file >> s >> a >> type) {
					switch (type) {
					case 1:
						int i;
						file >> i;
						v = i;
						break;
					case 2:
						float f;
						file >> f;
						v = f;
						break;
					case 3:
						double d;
						file >> d;
						v = d;
						break;
					default:
						logstream(LOG_ERROR) << "Unsupported value type" << LOG_endl;
						break;
					}
					if (sid == mymath::hash_mod(s, global_num_servers))
						triple_sav[localtid].push_back(triple_attr_t(s, a, v));
				}
			};

			//load from hdfs or posix file
			if (boost::starts_with(fnames[i], "hdfs:")) {
				// files located on HDFS
				wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
				wukong::hdfs::fstream file(hdfs, fnames[i]);
				load_attr(file);
				file.close();
			} else {
				ifstream file(fnames[i].c_str());
				load_attr(file);
				file.close();
			}
		}
	}

	void aggregate_data(int num_partitions) {
		// calculate #triples on the kvstore from all servers
		uint64_t total = 0;
		uint64_t kvs_sz = floor(mem->kvstore_size() / num_partitions - sizeof(uint64_t), sizeof(sid_t));
		for (int i = 0; i < num_partitions; i++) {
			uint64_t *pn = (uint64_t *)(mem->kvstore() + (kvs_sz + sizeof(uint64_t)) * i);
			total += *pn; // the 1st uint64_t of kvs records #triples
		}

		// pre-expand to avoid frequent reallocation (maybe imbalance)
		for (int i = 0; i < triple_spo.size(); i++) {
			triple_spo[i].reserve(total / global_num_engines);
			triple_ops[i].reserve(total / global_num_engines);
		}

		// each thread will scan all triples (from all servers) and pickup certain triples.
		// It ensures that the triples belong to the same vertex will be stored in the same
		// triple_spo/ops. This will simplify the deduplication and insertion to gstore.
		volatile int progress = 0;
		#pragma omp parallel for num_threads(global_num_engines)
		for (int tid = 0; tid < global_num_engines; tid++) {
			int cnt = 0; // per thread count for print progress
			for (int id = 0; id < num_partitions; id++) {
				uint64_t *pn = (uint64_t *)(mem->kvstore() + (kvs_sz + sizeof(uint64_t)) * id);
				sid_t *kvs = (sid_t *)(pn + 1);

				// the 1st uint64_t of kvs records #triples
				uint64_t n = *pn;
				for (uint64_t i = 0; i < n; i++) {
					sid_t s = kvs[i * 3 + 0];
					sid_t p = kvs[i * 3 + 1];
					sid_t o = kvs[i * 3 + 2];

					// out-edges
					if (mymath::hash_mod(s, global_num_servers) == sid)
						if ((s % global_num_engines) == tid)
							triple_spo[tid].push_back(triple_t(s, p, o));

					// in-edges
					if (mymath::hash_mod(o, global_num_servers) == sid)
						if ((o % global_num_engines) == tid)
							triple_ops[tid].push_back(triple_t(s, p, o));

					// print the progress (step = 5%) of aggregation
					if (++cnt >= total * 0.05) {
						int now = __sync_add_and_fetch(&progress, 1);
						if (now % global_num_engines == 0)
							logstream(LOG_INFO) << "already aggregrate " << (now / global_num_engines) * 5 << "%" << LOG_endl;
						cnt = 0;
					}
				}
			}

			sort(triple_spo[tid].begin(), triple_spo[tid].end(), triple_sort_by_spo());
			dedup_triples(triple_ops[tid]);

			sort(triple_ops[tid].begin(), triple_ops[tid].end(), triple_sort_by_ops());
			dedup_triples(triple_spo[tid]);
		}
	}

	inline vector<string> list_files(string dname, string prefix) {
		if (boost::starts_with(dname, "hdfs:")) {
			if (!wukong::hdfs::has_hadoop()) {
				logstream(LOG_ERROR) << "attempting to load data files from HDFS "
				                     << "but Wukong was built without HDFS."
				                     << LOG_endl;
				exit(-1);
			}

			wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
			return vector<string>(hdfs.list_files(dname, prefix));
		} else {
			// files located on a shared filesystem (e.g., NFS)
			DIR *dir = opendir(dname.c_str());
			if (dir == NULL) {
				logstream(LOG_ERROR) << "failed to open directory (" << dname
				                     << ") at server " << sid << LOG_endl;
				exit(-1);
			}

			vector<string> files;
			struct dirent *ent;
			while ((ent = readdir(dir)) != NULL) {
				if (ent->d_name[0] == '.')
					continue;

				string fname(dname + ent->d_name);
				// Assume the fnames (ID-format) start with the prefix.
				if (boost::starts_with(fname, dname + prefix))
					files.push_back(fname);
			}
			return files;
		}

	}

	uint64_t inline floor(uint64_t original, uint64_t n) {
		ASSERT(n != 0);
		return original - original % n;
	}

	uint64_t inline ceil(uint64_t original, uint64_t n) {
		ASSERT(n != 0);
		if (original % n == 0)
			return original;
		return original - original % n + n;
	}

public:
	GStore gstore;

	DGraph(int sid, Mem *mem, String_Server *str_server, string dname)
		: sid(sid), str_server(str_server), mem(mem), gstore(sid, mem) {
		uint64_t start, end;

		num_triples.resize(global_num_servers);

		triple_spo.resize(global_num_engines);
		triple_ops.resize(global_num_engines);
		triple_sav.resize(global_num_engines);

		vector<string> dfiles(list_files(dname, "id_"));   // ID-format data files
		vector<string> afiles(list_files(dname, "attr_")); // ID-format attribute files

		if (dfiles.size() == 0) {
			logstream(LOG_WARNING) << "no data files found in directory (" << dname
			                       << ") at server " << sid << LOG_endl;
		} else {
			logstream(LOG_INFO) << dfiles.size() << " files and " << afiles.size()
			                    << " attributed files found in directory (" << dname
			                    << ") at server " << sid << LOG_endl;
		}

		// load_data: load partial input files by each server and exchanges triples
		//            according to graph partitioning
		// load_data_from_allfiles: load all files by each server and select triples
		//                          according to graph partitioning
		//
		// Trade-off: load_data_from_allfiles avoids network traffic and memory,
		//            but it requires more I/O from distributed FS.
		//
		// Wukong adopts load_data_from_allfiles for slow network (w/o RDMA) and
		//        adopts load_data for fast network (w/ RDMA).
		start = timer::get_usec();
		int num_partitons = 0;
		if (global_use_rdma)
			num_partitons = load_data(dfiles);
		else
			num_partitons = load_data_from_allfiles(dfiles);
		end = timer::get_usec();
		logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << " ms "
		                    << "for loading data files" << LOG_endl;

		// all triples are partitioned and temporarily stored in the kvstore on each server.
		// the kvstore is split into num_partitions partitions, each contains #triples and triples
		//
		// Wukong aggregates, sorts and dedups all triples before finally inserting them to gstore (kvstore)
		start = timer::get_usec();
		aggregate_data(num_partitons);
		end = timer::get_usec();
		logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << " ms "
		                    << "for aggregrating triples" << LOG_endl;

		// load attribute files
		start = timer::get_usec();
		load_attr_from_allfiles(afiles);
		end = timer::get_usec();
		logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << " ms "
		                    << "for loading attribute files" << LOG_endl;

		// initiate gstore (kvstore) after loading and exchanging triples (memory reused)
		gstore.refresh();


		start = timer::get_usec();
		#pragma omp parallel for num_threads(global_num_engines)
		for (int t = 0; t < global_num_engines; t++) {
			gstore.insert_normal(triple_spo[t], triple_ops[t], t);

			// release memory
			vector<triple_t>().swap(triple_spo[t]);
			vector<triple_t>().swap(triple_ops[t]);
		}
		end = timer::get_usec();
		logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
		                    << "for inserting normal data into gstore" << LOG_endl;

		start = timer::get_usec();
		#pragma omp parallel for num_threads(global_num_engines)
		for (int t = 0; t < global_num_engines; t++) {
			gstore.insert_vertex_attr(triple_sav[t], t);
			// release memory
			vector<triple_attr_t>().swap(triple_sav[t]);
		}
		end = timer::get_usec();
		logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
		                    << "for inserting attributes into gstore" << LOG_endl;

		start = timer::get_usec();
		gstore.insert_index();
		end = timer::get_usec();
		logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
		                    << "for inserting index data into gstore" << LOG_endl;

		logstream(LOG_INFO) << "#" << sid << ": loading DGraph is finished" << LOG_endl;
		gstore.print_mem_usage();
	}


#if DYNAMIC_GSTORE
	int64_t dynamic_load_data(string dname, bool check_dup) {
		dynamic_load_mappings(dname); // load ID-mapping files and construct id2id mapping

		vector<string> dfiles(list_files(dname, "id_"));   // ID-format data files
		vector<string> afiles(list_files(dname, "attr_")); // ID-format attribute files

		if (dfiles.size() == 0 && afiles.size() == 0) {
			logstream(LOG_WARNING) << "no files found in directory (" << dname
			                       << ") at server " << sid << LOG_endl;
			return 0;
		}

		logstream(LOG_INFO) << dfiles.size() << " data files and " << afiles.size()
		                    << " attribute files found in directory (" << dname
		                    << ") at server " << sid << LOG_endl;

		sort(dfiles.begin(), dfiles.end());

		int num_dfiles = dfiles.size();
		#pragma omp parallel for num_threads(global_num_engines)
		for (int i = 0; i < num_dfiles; i++) {
			int64_t cnt = 0;

			/// FIXME: support HDFS
			ifstream file(dfiles[i]);
			sid_t s, p, o;
			while (file >> s >> p >> o) {
				convert_sid(s); convert_sid(p); convert_sid(o); //convert origin ids to new ids
				/// FIXME: just check and print warning
				check_sid(s); check_sid(p); check_sid(o);

				if (sid == mymath::hash_mod(s, global_num_servers)) {
					gstore.insert_triple_out(triple_t(s, p, o), check_dup);
					cnt ++;
				}

				if (sid == mymath::hash_mod(o, global_num_servers)) {
					gstore.insert_triple_in(triple_t(s, p, o), check_dup);
					cnt ++;
				}
			}
			file.close();

			logstream(LOG_INFO) << "load " << cnt << " triples from file " << dfiles[i]
			                    << " at server " << sid << LOG_endl;
		}

		flush_convertmap(); //clean the id2id mapping

		sort(afiles.begin(), afiles.end());
		int num_afiles = afiles.size();
		#pragma omp parallel for num_threads(global_num_engines)
		for (int i = 0; i < num_afiles; i++) {
			int64_t cnt = 0;

			/// FIXME: support HDFS
			ifstream file(afiles[i]);
			sid_t s, a;
			attr_t v;
			int type;
			while (file >> s >> a >> type) {
				/// FIXME: just check and print warning
				check_sid(s); check_sid(a);

				switch (type) {
				case 1:
					int i;
					file >> i;
					v = i;
					break;
				case 2:
					float f;
					file >> f;
					v = f;
					break;
				case 3:
					double d;
					file >> d;
					v = d;
					break;
				default:
					logstream(LOG_ERROR) << "Unsupported value type" << LOG_endl;
					break;
				}

				if (sid == mymath::hash_mod(s, global_num_servers)) {
					/// Support attribute files
					// gstore.insert_triple_attribute(triple_sav_t(s, a, v));
					cnt ++;
				}
			}
			file.close();

			logstream(LOG_INFO) << "load " << cnt << " attributes from file " << afiles[i]
			                    << " at server " << sid << LOG_endl;
		}

		return 0;
	}
#endif

	int gstore_check(bool index_check, bool normal_check) {
		return gstore.gstore_check(index_check, normal_check);
	}

	// FIXME: rename the function by the term of RDF model (e.g., subject/object)
	edge_t *get_edges_global(int tid, sid_t vid, dir_t d, sid_t pid, uint64_t *sz) {
		return gstore.get_edges_global(tid, vid, d, pid, sz);
	}

	// FIXME: rename the function by the term of RDF model (e.g., subject/object)
	edge_t *get_index_edges_local(int tid, sid_t vid, dir_t d, uint64_t *sz) {
		return gstore.get_index_edges_local(tid, vid, d, sz);
	}

	// FIXME: rename the function by the term of attribute graph model (e.g., value)
	bool get_vertex_attr_global(int tid, sid_t vid, dir_t d, sid_t pid, attr_t &result) {
		return gstore.get_vertex_attr_global(tid, vid, d, pid, result);
	}
};
