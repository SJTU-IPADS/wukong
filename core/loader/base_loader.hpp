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

#include <boost/mpi.hpp>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "omp.h"

#include "global.hpp"
#include "type.hpp"
#include "rdma.hpp"

// loader
#include "loader_interface.hpp"
#include "store/gstore.hpp"

// utils
#include "timer.hpp"
#include "assertion.hpp"
#include "math.hpp"
#include "variant.hpp"

using namespace std;

class BaseLoader : public LoaderInterface {
protected:
    int sid;
    Mem *mem;
    StringServer *str_server;
    GStore *gstore;

    vector<uint64_t> num_triples;  // record #triples loaded from input data for each server

    virtual istream *init_istream(const string &src) = 0;
    virtual void close_istream(istream *stream) = 0;
    virtual vector<string> list_files(const string &src, string prefix) = 0;

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
        uint64_t buf_sz = floor(mem->buffer_size() / Global::num_servers - sizeof(uint64_t),
                                sizeof(sid_t));
        uint64_t *pn = (uint64_t *)(mem->buffer(tid) + (buf_sz + sizeof(uint64_t)) * dst_sid);
        sid_t *buf = (sid_t *)(pn + 1);

        // the 1st uint64_t of buffer records #new-triples
        uint64_t n = *pn;

        // the kvstore is temporally split into #servers pieces.
        // hence, the kvstore can be directly RDMA write in parallel by all servers
        uint64_t kvs_sz = floor(mem->kvstore_size() / Global::num_servers - sizeof(uint64_t),
                                sizeof(sid_t));

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
        // each piece: #triples, tirple, triple, . . .
        uint64_t buf_sz = floor(mem->buffer_size() / Global::num_servers - sizeof(uint64_t), sizeof(sid_t));
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

    int read_partial_exchange(vector<string> &fnames) {
        // ensure the file name list has the same order on all servers
        sort(fnames.begin(), fnames.end());

        auto lambda = [&](istream & file, int localtid) {
            sid_t s, p, o;
            while (file >> s >> p >> o) {
                int s_sid = wukong::math::hash_mod(s, Global::num_servers);
                int o_sid = wukong::math::hash_mod(o, Global::num_servers);
                if (s_sid == o_sid) {
                    send_triple(localtid, s_sid, s, p, o);
                } else {
                    send_triple(localtid, s_sid, s, p, o);
                    send_triple(localtid, o_sid, s, p, o);
                }
            }
        };

        // load input data and assign to different severs in parallel
        int num_files = fnames.size();
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < num_files; i++) {
            int localtid = omp_get_thread_num();

            // each server only load a part of files
            if (i % Global::num_servers != sid) continue;

            istream *file = init_istream(fnames[i]);
            lambda(*file, localtid);
            close_istream(file);
        }

        // flush the rest triples within each RDMA buffer
        for (int s = 0; s < Global::num_servers; s++)
            for (int t = 0; t < Global::num_engines; t++)
                flush_triples(t, s);

        // exchange #triples among all servers
        for (int s = 0; s < Global::num_servers; s++) {
            uint64_t *buf = (uint64_t *)mem->buffer(0);
            buf[0] = num_triples[s];

            uint64_t kvs_sz = floor(mem->kvstore_size() / Global::num_servers, sizeof(sid_t));
            uint64_t offset = kvs_sz * sid;
            if (s != sid) {
                RDMA &rdma = RDMA::get_rdma();
                rdma.dev->RdmaWrite(0, s, (char*)buf, sizeof(uint64_t), offset);
            } else {
                memcpy(mem->kvstore() + offset, (char*)buf, sizeof(uint64_t));
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        return Global::num_servers;
    }

    // selectively load own partitioned data from all files
    int read_all_files(vector<string> &fnames) {
        sort(fnames.begin(), fnames.end());

        auto lambda = [&](istream & file, uint64_t &n, uint64_t kvs_sz, sid_t * kvs) {
            sid_t s, p, o;
            while (file >> s >> p >> o) {
                int s_sid = wukong::math::hash_mod(s, Global::num_servers);
                int o_sid = wukong::math::hash_mod(o, Global::num_servers);
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

        int num_files = fnames.size();
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < num_files; i++) {
            int localtid = omp_get_thread_num();
            uint64_t kvs_sz = floor(mem->kvstore_size() / Global::num_engines - sizeof(uint64_t),
                                    sizeof(sid_t));
            uint64_t *pn = (uint64_t *)(mem->kvstore() + (kvs_sz + sizeof(uint64_t)) * localtid);
            sid_t *kvs = (sid_t *)(pn + 1);

            // the 1st uint64_t of kvs records #triples
            uint64_t n = *pn;

            istream *file = init_istream(fnames[i]);
            lambda(*file, n, kvs_sz, kvs);
            close_istream(file);

            *pn = n;
        }

        return Global::num_engines;
    }

    // selectively load own partitioned data (attributes) from all files
    void load_attr_from_allfiles(vector<string> &fnames, vector<vector<triple_attr_t>> &triple_sav) {
        if (fnames.size() == 0)
            return; // no attributed files

        sort(fnames.begin(), fnames.end());

        auto load_attr = [&](istream & file, int localtid) {
            sid_t s, a;
            attr_t v;
            int type;
            while (file >> s >> a >> type) {
                switch (type) {
                case INT_t: { int i; file >> i; v = i; break; }
                case FLOAT_t: { float f; file >> f; v = f; break; }
                case DOUBLE_t: { double d; file >> d; v = d; break; }
                default:
                    logstream(LOG_ERROR) << "Unsupported value type" << LOG_endl;
                    break;
                }

                if (sid == wukong::math::hash_mod(s, Global::num_servers))
                    triple_sav[localtid].push_back(triple_attr_t(s, a, v));
            }
        };

        //parallel load from all files
        int num_files = fnames.size();
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < num_files; i++) {
            int localtid = omp_get_thread_num();

            //load from hdfs or posix file
            istream *file = init_istream(fnames[i]);
            load_attr(*file, localtid);
            close_istream(file);
        }
    }

    void sort_attr(vector<vector<triple_attr_t>> &triple_sav) {
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int tid = 0; tid < Global::num_engines; tid++)
            sort(triple_sav[tid].begin(), triple_sav[tid].end(), triple_sort_by_asv());
    }

    void aggregate_data(int num_partitions,
                        vector<vector<triple_t>> &triple_pso,
                        vector<vector<triple_t>> &triple_pos) {
        // calculate #triples on the kvstore from all servers
        uint64_t total = 0;
        uint64_t kvs_sz = floor(mem->kvstore_size() / num_partitions - sizeof(uint64_t), sizeof(sid_t));
        for (int i = 0; i < num_partitions; i++) {
            uint64_t *pn = (uint64_t *)(mem->kvstore() + (kvs_sz + sizeof(uint64_t)) * i);
            total += *pn; // the 1st uint64_t of kvs records #triples
        }

        // pre-expand to avoid frequent reallocation (maybe imbalance)
        for (int i = 0; i < triple_pso.size(); i++) {
            triple_pso[i].reserve(total / Global::num_engines);
            triple_pos[i].reserve(total / Global::num_engines);
        }

        // each thread will scan all triples (from all servers) and pickup certain triples.
        // It ensures that the triples belong to the same vertex will be stored in the same
        // triple_pso/ops. This will simplify the deduplication and insertion to gstore.
        volatile uint64_t progress = 0;
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int tid = 0; tid < Global::num_engines; tid++) {
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
                    if (wukong::math::hash_mod(s, Global::num_servers) == sid)
                        if ((s % Global::num_engines) == tid)
                            triple_pso[tid].push_back(triple_t(s, p, o));

                    // in-edges
                    if (wukong::math::hash_mod(o, Global::num_servers) == sid)
                        if ((o % Global::num_engines) == tid)
                            triple_pos[tid].push_back(triple_t(s, p, o));

                    // print the progress (step = 5%) of aggregation
                    if (++cnt >= total * 0.05) {
                        uint64_t now = wukong::atomic::add_and_fetch(&progress, 1);
                        if (now % Global::num_engines == 0)
                            logstream(LOG_INFO) << "already aggregrate "
                                                << (now / Global::num_engines) * 5
                                                << "%" << LOG_endl;
                        cnt = 0;
                    }
                }
            }

#ifdef VERSATILE
            sort(triple_pso[tid].begin(), triple_pso[tid].end(), triple_sort_by_spo());
            sort(triple_pos[tid].begin(), triple_pos[tid].end(), triple_sort_by_ops());
#else
            sort(triple_pso[tid].begin(), triple_pso[tid].end(), triple_sort_by_pso());
            sort(triple_pos[tid].begin(), triple_pos[tid].end(), triple_sort_by_pos());
#endif
            dedup_triples(triple_pos[tid]);
            dedup_triples(triple_pso[tid]);

            triple_pos[tid].shrink_to_fit();
            triple_pso[tid].shrink_to_fit();
        }
    }

public:
    BaseLoader(int sid, Mem *mem, StringServer *str_server, GStore *gstore)
        : sid(sid), mem(mem), str_server(str_server), gstore(gstore) { }

    virtual ~BaseLoader() { }

    void load(const string &src,
              vector<vector<triple_t>> &triple_pso,
              vector<vector<triple_t>> &triple_pos,
              vector<vector<triple_attr_t>> &triple_sav) {
        uint64_t start, end;

        num_triples.resize(Global::num_servers);
        triple_pso.resize(Global::num_engines);
        triple_pos.resize(Global::num_engines);
        triple_sav.resize(Global::num_engines);

        vector<string> dfiles(list_files(src, "id_"));   // ID-format data files
        vector<string> afiles(list_files(src, "attr_")); // ID-format attribute files

        if (dfiles.size() == 0) {
            logstream(LOG_WARNING) << "no data files found in directory (" << src
                                   << ") at server " << sid << LOG_endl;
        } else {
            logstream(LOG_INFO) << dfiles.size() << " files and " << afiles.size()
                                << " attributed files found in directory (" << src
                                << ") at server " << sid << LOG_endl;
        }

        auto count_preds = [](const string str_idx_file) {
            string pred;
            int pid, count = 0;
            ifstream ifs(str_idx_file.c_str());
            while (ifs >> pred >> pid) {
                count++;
            }
            ifs.close();
            return count;
        };

        int num_normal_preds = count_preds(src + "str_index");
        if (num_normal_preds == 0)
            logstream(LOG_ERROR) << "Encoding file of predicates should be named as \"str_index\". Graph loading failed. Please quit and try again." << LOG_endl;
        else
            gstore->num_normal_preds = num_normal_preds - 1; // skip PREDICATE_ID
        if (Global::enable_vattr)
            gstore->num_attr_preds = count_preds(src + "str_attr_index");

        // read_partial_exchange: load partial input files by each server and exchanges triples
        //            according to graph partitioning
        // read_all_files: load all files by each server and select triples
        //                          according to graph partitioning
        //
        // Trade-off: read_all_files avoids network traffic and memory,
        //            but it requires more I/O from distributed FS.
        //
        // Wukong adopts read_all_files for slow network (w/o RDMA) and
        //        adopts read_partial_exchange for fast network (w/ RDMA).
        start = timer::get_usec();
        int num_partitons = 0;
        if (Global::use_rdma)
            num_partitons = read_partial_exchange(dfiles);
        else
            num_partitons = read_all_files(dfiles);
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << " ms "
                            << "for loading data files" << LOG_endl;

        // all triples are partitioned and temporarily stored in the kvstore on each server.
        // the kvstore is split into num_partitions partitions, each contains #triples and triples
        //
        // Wukong aggregates, sorts and dedups all triples before finally inserting
        // them to gstore (kvstore)
        start = timer::get_usec();
        aggregate_data(num_partitons, triple_pso, triple_pos);
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << " ms "
                            << "for aggregrating triples" << LOG_endl;

        // load attribute files
        start = timer::get_usec();
        load_attr_from_allfiles(afiles, triple_sav);
        sort_attr(triple_sav);
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << " ms "
                            << "for loading attribute files" << LOG_endl;

        // initiate gstore (kvstore) after loading and exchanging triples (memory reused)
        gstore->refresh();
    }
};
