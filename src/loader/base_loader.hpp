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

#include <dirent.h>
#include <omp.h>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/mpi.hpp>
#include <boost/unordered_map.hpp>

#include "core/common/global.hpp"
#include "core/common/rdma.hpp"
#include "core/common/type.hpp"

// loader
#include "loader_interface.hpp"

// utils
#include "utils/assertion.hpp"
#include "utils/math.hpp"
#include "utils/timer.hpp"

namespace wukong {

class BaseLoader : public LoaderInterface {
protected:
    int sid;
    Mem* mem;
    StringServer* str_server;

    std::vector<uint64_t> num_triples;  // record #triples loaded from input data for each server

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

    void dedup_triples(std::vector<triple_t>& triples) {
        if (triples.size() <= 1)
            return;

        uint64_t n = 1;
        for (uint64_t i = 1; i < triples.size(); i++) {
            if (triples[i] == triples[i - 1]) {
                continue;
            }

            triples[n++] = triples[i];
        }
        triples.resize(n);
    }

    void flush_triples(int tid, int dst_sid) {
        uint64_t buf_sz = floor(mem->buffer_size() / Global::num_servers - sizeof(uint64_t), sizeof(triple_t));
        uint64_t* pn = reinterpret_cast<uint64_t*>(mem->buffer(tid) + (buf_sz + sizeof(uint64_t)) * dst_sid);
        triple_t* buf = reinterpret_cast<triple_t*>(pn + 1);

        // the 1st uint64_t of buffer records #new-triples
        uint64_t n = *pn;

        // the kvstore is temporally split into #servers pieces.
        // hence, the kvstore can be directly RDMA write in parallel by all servers
        uint64_t kvs_sz = floor(mem->kvstore_size() / Global::num_servers - sizeof(uint64_t), sizeof(triple_t));

        // serialize the RDMA WRITEs by multiple threads
        uint64_t exist = __sync_fetch_and_add(&num_triples[dst_sid], n);

        uint64_t cur_sz = (exist + n) * sizeof(triple_t);
        if (cur_sz > kvs_sz) {
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
                        + exist * sizeof(triple_t); // skip #exist-triples
        
        uint64_t sz = n * sizeof(triple_t);        // send #new-triples

        if (dst_sid != sid) {
            RDMA& rdma = RDMA::get_rdma();
            rdma.dev->RdmaWrite(tid, dst_sid, reinterpret_cast<char*>(buf), sz, off);
        } else {
            memcpy(mem->kvstore() + off, reinterpret_cast<char*>(buf), sz);
        }

        // clear the buffer
        *pn = 0;
    }

    // send_triple can be safely called by multiple threads,
    // since the buffer is exclusively used by one thread.
    void send_triple(int tid, int dst_sid, triple_t triple) {
        // the RDMA buffer is first split into #threads partitions
        // each partition is further split into #servers pieces
        // each piece: #triples, tirple, triple, . . .
        uint64_t buf_sz = floor(mem->buffer_size() / Global::num_servers - sizeof(uint64_t), sizeof(triple_t));
        uint64_t* pn = reinterpret_cast<uint64_t*>(mem->buffer(tid) + (buf_sz + sizeof(uint64_t)) * dst_sid);
        triple_t* buf = reinterpret_cast<triple_t*>(pn + 1);

        // the 1st entry of buffer records #triples (suppose the )
        uint64_t n = *pn;

        // flush buffer if there is no enough space to buffer a new triple
        uint64_t cur_sz = (n + 1) * sizeof(triple_t);
        if (cur_sz > buf_sz) {
            flush_triples(tid, dst_sid);
            n = *pn;  // reset, it should be 0
            ASSERT(n == 0);
        }

        // buffer the triple and update the counter
        buf[n] = triple;
        *pn = (n + 1);
    }

    int read_partial_exchange(std::vector<std::string>& fnames) {
        // ensure the file name list has the same order on all servers
        std::sort(fnames.begin(), fnames.end());

        auto lambda = [&](std::istream& file, int localtid) {
            triple_t triple;
        #ifdef TRDF_MODE
            while (file >> triple.s >> triple.p >> triple.o >> triple.ts >> triple.te) {
        #else
            while (file >> triple.s >> triple.p >> triple.o) {
        #endif
                int s_sid = PARTITION(triple.s);
                int o_sid = PARTITION(triple.o);
                if (s_sid == o_sid) {
                    send_triple(localtid, s_sid, triple);
                } else {
                    send_triple(localtid, s_sid, triple);
                    send_triple(localtid, o_sid, triple);
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

            std::istream* file = init_istream(fnames[i]);
            lambda(*file, localtid);
            close_istream(file);
        }

        // flush the rest triples within each RDMA buffer
        for (int s = 0; s < Global::num_servers; s++)
            for (int t = 0; t < Global::num_engines; t++)
                flush_triples(t, s);

        // exchange #triples among all servers
        for (int s = 0; s < Global::num_servers; s++) {
            uint64_t* buf = reinterpret_cast<uint64_t*>(mem->buffer(0));
            buf[0] = num_triples[s];

            uint64_t kvs_sz = floor(mem->kvstore_size() / Global::num_servers - sizeof(uint64_t), sizeof(triple_t));
            uint64_t offset = (kvs_sz + sizeof(uint64_t)) * sid;
            if (s != sid) {
                RDMA& rdma = RDMA::get_rdma();
                rdma.dev->RdmaWrite(0, s, reinterpret_cast<char*>(buf), sizeof(uint64_t), offset);
            } else {
                memcpy(mem->kvstore() + offset, reinterpret_cast<char*>(buf), sizeof(uint64_t));
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        return Global::num_servers;
    }

    // selectively load own partitioned data from all files
    int read_all_files(std::vector<std::string>& fnames) {
        std::sort(fnames.begin(), fnames.end());

        auto lambda = [&](std::istream& file, uint64_t& n, uint64_t kvs_sz, triple_t* kvs) {
            triple_t triple;
        #ifdef TRDF_MODE
            while (file >> triple.s >> triple.p >> triple.o >> triple.ts >> triple.te) {
        #else
            while (file >> triple.s >> triple.p >> triple.o) {
        #endif
                int s_sid = PARTITION(triple.s);
                int o_sid = PARTITION(triple.o);
                if ((s_sid == sid) || (o_sid == sid)) {
                    ASSERT((n + 1) * sizeof(triple_t) <= kvs_sz);
                    // buffer the triple and update the counter
                    kvs[n] = triple;
                    n++;
                }
            }
        };


        int num_files = fnames.size();
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < num_files; i++) {
            int localtid = omp_get_thread_num();
            uint64_t kvs_sz = floor(mem->kvstore_size() / Global::num_engines - sizeof(uint64_t),
                                    sizeof(triple_t));
            uint64_t* pn = reinterpret_cast<uint64_t*>(mem->kvstore() + (kvs_sz + sizeof(uint64_t)) * localtid);
            triple_t* kvs = reinterpret_cast<triple_t*>(pn + 1);

            // the 1st uint64_t of kvs records #triples
            uint64_t n = *pn;

            std::istream* file = init_istream(fnames[i]);
            lambda(*file, n, kvs_sz, kvs);
            close_istream(file);

            *pn = n;
        }

        return Global::num_engines;
    }

    // selectively load own partitioned data (attributes) from all files
    void load_attr_from_allfiles(std::vector<std::string>& fnames, std::vector<std::vector<triple_attr_t>>& triple_sav) {
        if (fnames.size() == 0)
            return;  // no attributed files

        std::sort(fnames.begin(), fnames.end());

        auto load_attr = [&](std::istream& file, int localtid) {
            sid_t s, a;
            attr_t v;
            int type;
            while (file >> s >> a >> type) {
                switch (type) {
                case INT_t: {
                    int i;
                    file >> i;
                    v = i;
                    break;
                }
                case FLOAT_t: {
                    float f;
                    file >> f;
                    v = f;
                    break;
                }
                case DOUBLE_t: {
                    double d;
                    file >> d;
                    v = d;
                    break;
                }
                default:
                    logstream(LOG_ERROR) << "Unsupported value type" << LOG_endl;
                    break;
                }

                if (sid == PARTITION(s))
                    triple_sav[localtid].push_back(triple_attr_t(s, a, v));
            }
        };

        // parallel load from all files
        int num_files = fnames.size();
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < num_files; i++) {
            int localtid = omp_get_thread_num();

            // load from hdfs or posix file
            std::istream* file = init_istream(fnames[i]);
            load_attr(*file, localtid);
            close_istream(file);
        }
    }

    void sort_attr(std::vector<std::vector<triple_attr_t>>& triple_sav) {
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int tid = 0; tid < Global::num_engines; tid++)
            std::sort(triple_sav[tid].begin(), triple_sav[tid].end(), triple_sort_by_asv());
    }

    void aggregate_data(int num_partitions,
                        std::vector<std::vector<triple_t>>& triple_pso,
                        std::vector<std::vector<triple_t>>& triple_pos) {
        // calculate #triples on the kvstore from all servers
        uint64_t total = 0;
        uint64_t kvs_sz = floor(mem->kvstore_size() / num_partitions - sizeof(uint64_t), sizeof(triple_t));
        for (int i = 0; i < num_partitions; i++) {
            uint64_t* pn = reinterpret_cast<uint64_t*>(mem->kvstore() + (kvs_sz + sizeof(uint64_t)) * i);
            total += *pn;  // the 1st uint64_t of kvs records #triples
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
            int cnt = 0;  // per thread count for print progress
            for (int id = 0; id < num_partitions; id++) {
                uint64_t* pn = reinterpret_cast<uint64_t*>(mem->kvstore() + (kvs_sz + sizeof(uint64_t)) * id);
                triple_t* kvs = reinterpret_cast<triple_t*>(pn + 1);

                // the 1st uint64_t of kvs records #triples
                uint64_t n = *pn;
                for (uint64_t i = 0; i < n; i++) {
                    triple_t triple = kvs[i];

                    // out-edges
                    if (PARTITION(triple.s) == sid)
                        if ((triple.s % Global::num_engines) == tid)
                            triple_pso[tid].push_back(triple);

                    // in-edges
                    if (PARTITION(triple.o) == sid)
                        if ((triple.o % Global::num_engines) == tid)
                            triple_pos[tid].push_back(triple);

                    // print the progress (step = 5%) of aggregation
                    if (++cnt >= total * 0.05) {
                        uint64_t now = wukong::atomic::add_and_fetch(&progress, 1);
                        if (now % Global::num_engines == 0)
                            logstream(LOG_INFO) << "[Loader] triples already aggregrate "
                                                << (now / Global::num_engines) * 5
                                                << "%" << LOG_endl;
                        cnt = 0;
                    }
                }
            }
        }
    }

    // Load normal triples from all files, using read_partial_exchange or read_all_files.
    void load_triples_from_all(std::vector<std::string>& dfiles,
                               std::vector<std::vector<triple_t>>& triple_pso,
                               std::vector<std::vector<triple_t>>& triple_pos) {
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
        uint64_t start = timer::get_usec();
        int num_partitons = 0;
        if (Global::use_rdma)
            num_partitons = read_partial_exchange(dfiles);
        else
            num_partitons = read_all_files(dfiles);
        uint64_t end = timer::get_usec();
        logstream(LOG_INFO) << "[Loader] #" << sid << ": " << (end - start) / 1000 << " ms "
                            << "for loading data files" << LOG_endl;

        // all triples are partitioned and temporarily stored in the kvstore on each server.
        // the kvstore is split into num_partitions partitions, each contains #triples and triples
        //
        // Wukong aggregates all triples before finally inserting them to gstore (kvstore)
        start = timer::get_usec();
        aggregate_data(num_partitons, triple_pso, triple_pos);
        end = timer::get_usec();
        logstream(LOG_INFO) << "[Loader] #" << sid << ": " << (end - start) / 1000 << " ms "
                            << "for aggregrating triples" << LOG_endl;
    }

    // Load preprocessed data from selected files.
    void load_triples_from_selected(const std::string& src, std::vector<std::string>& fnames,
                                    std::vector<std::vector<triple_t>>& triple_pso,
                                    std::vector<std::vector<triple_t>>& triple_pos) {
        uint64_t start = timer::get_usec();

        size_t triple_cnt = get_triple_cnt(src);
        for (int i = 0; i < triple_pso.size(); i++) {
            triple_pso[i].reserve(triple_cnt / Global::num_servers / Global::num_engines);
            triple_pos[i].reserve(triple_cnt / Global::num_servers / Global::num_engines);
        }

        auto load_one_file = [&](bool by_s, std::istream& file,
                                 std::vector<triple_t>& triple_pso, std::vector<triple_t>& triple_pos) {
            triple_t triple;
        #ifdef TRDF_MODE
            while (file >> triple.s >> triple.p >> triple.o >> triple.ts >> triple.te) {
        #else
            while (file >> triple.s >> triple.p >> triple.o) {
        #endif
                if (by_s)    // out-edges
                    triple_pso.push_back(triple);
                else        // in-edges
                    triple_pos.push_back(triple);
            }
        };

        int num_files = fnames.size();
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < num_files; i++) {
            int localtid = omp_get_thread_num();

            size_t spos = fnames[i].find_last_of('_') + 1;
            size_t len = fnames[i].find('.') - spos;
            int pid = stoi(fnames[i].substr(spos, len));
            if (PARTITION(pid) == sid) {
                std::istream* file = init_istream(fnames[i]);
                bool by_s = (fnames[i].find("spo") != std::string::npos);
                load_one_file(by_s, *file, triple_pso[localtid], triple_pos[localtid]);
                close_istream(file);
            }
        }
        uint64_t end = timer::get_usec();
        logstream(LOG_INFO) << "[Loader] #" << sid << ": " << (end - start) / 1000 << " ms "
                            << "for loading triples" << LOG_endl;
    }

    void sort_normal_triples(std::vector<std::vector<triple_t>>& triple_pso,
                             std::vector<std::vector<triple_t>>& triple_pos) {
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int tid = 0; tid < Global::num_engines; tid++) {
#ifdef VERSATILE
            std::sort(triple_pso[tid].begin(), triple_pso[tid].end(), triple_sort_by_spo());
            std::sort(triple_pos[tid].begin(), triple_pos[tid].end(), triple_sort_by_ops());
#else
            std::sort(triple_pso[tid].begin(), triple_pso[tid].end(), triple_sort_by_pso());
            std::sort(triple_pos[tid].begin(), triple_pos[tid].end(), triple_sort_by_pos());
#endif
            dedup_triples(triple_pos[tid]);
            dedup_triples(triple_pso[tid]);

            triple_pos[tid].shrink_to_fit();
            triple_pso[tid].shrink_to_fit();
        }
    }

    bool is_preprocessed(const std::string& src) {
        auto info_file = list_files(src, "metadata");
        if (info_file.size() == 1) {
            std::istream* file = init_istream(src + "/metadata");
            int num = 0;
            if ((*file) >> num) {
                // Preprocessed data can be re-hashed into #num_servers partitions if num is times of num_servers.
                if (num >= Global::num_servers && num % Global::num_servers == 0)
                    return true;
                // Partitions is not times of num_servers(e.g. partitions = 16, num_servers = 3), report error.
                logstream(LOG_ERROR) << "Partitions of preprocessed data should be times of num_servers. Please re-preprocess data or use raw data instead." << LOG_endl;
                exit(-1);
            }
            close_istream(file);
        }
        return false;
    }

    size_t get_triple_cnt(const std::string& src) {
        std::istream* file = init_istream(src + "/metadata");
        size_t num, triples;
        ASSERT(*file >> num >> triples);
        close_istream(file);
        return triples;
    }

public:
    BaseLoader(int sid, Mem* mem, StringServer* str_server)
        : sid(sid), mem(mem), str_server(str_server) {}

    virtual ~BaseLoader() {}

    virtual std::istream* init_istream(const std::string& src) = 0;
    virtual void close_istream(std::istream* stream) = 0;
    virtual std::vector<std::string> list_files(const std::string& src, std::string prefix) = 0;

    void load(const std::string& src,
              std::vector<std::vector<triple_t>>& triple_pso,
              std::vector<std::vector<triple_t>>& triple_pos,
              std::vector<std::vector<triple_attr_t>>& triple_sav) {
        uint64_t start, end;

        num_triples.resize(Global::num_servers);
        triple_pso.resize(Global::num_engines);
        triple_pos.resize(Global::num_engines);
        triple_sav.resize(Global::num_engines);

        // ID-format data files
        std::vector<std::string> dfiles(list_files(src, "id_"));
        // ID-format attribute files
        std::vector<std::string> afiles(list_files(src, "attr_"));

        if (dfiles.size() == 0) {
            logstream(LOG_WARNING) << "[Loader] no data files found in directory (" << src
                                   << ") at server " << sid << LOG_endl;
        } else {
            logstream(LOG_INFO) << "[Loader] " << dfiles.size() << " files and " << afiles.size()
                                << " attributed files found in directory (" << src
                                << ") at server " << sid << LOG_endl;
        }

        // load_triples_from_all: load triples from all the input files
        // load_triples_from_selected: load triples from selected input files which is preprocessed
        //
        // Trade-off: load_triples_from_selected is faster than load_triples_from_all,
        //            but it requires preprocessing and specific conditions
        //
        // Wukong adopts load_normal_from_selected for well-preprocessed data which meets is_preprocessed()
        //        and adopts load_normal_from_all for other input data.
        if (is_preprocessed(src))
            load_triples_from_selected(src, dfiles, triple_pso, triple_pos);
        else
            load_triples_from_all(dfiles, triple_pso, triple_pos);

        // Wukong sorts and dedups all triples before finally inserting them to gstore (kvstore)
        sort_normal_triples(triple_pso, triple_pos);

        // load attribute files
        start = timer::get_usec();
        load_attr_from_allfiles(afiles, triple_sav);
        sort_attr(triple_sav);
        end = timer::get_usec();
        logstream(LOG_INFO) << "[Loader] #" << sid << ": " << (end - start) / 1000 << " ms "
                            << "for loading attribute files" << LOG_endl;
    }
};

}  // namespace wukong
