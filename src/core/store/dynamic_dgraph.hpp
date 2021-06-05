/*
 * Copyright (c) 2021 Shanghai Jiao Tong University.
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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "core/store/dgraph.hpp"

// #include "loader/dynamic_loader.hpp"

namespace wukong {

/**
 * @brief Dynamic RDF Graph
 * 
 * support to dynamically insert triples
 * 
 */
class DynamicRDFGraph : public DGraph {
protected:
    // DynamicLoader *dynamic_loader;

public:
    DynamicRDFGraph(int sid, Mem* mem, StringServer* str_server)
        : DGraph(sid, mem, str_server) {
        this->gstore = std::make_shared<DynamicKVStore<ikey_t, iptr_t, edge_t>>(sid, mem);
        // this->dynamic_loader = new DynamicLoader(sid, str_server, static_cast<DynamicKVStore *>(gstore));
    }

    ~DynamicRDFGraph() {
        // if(dynamic_loader) delete dynamic_loader;
    }

    int dynamic_load_data(std::string dname, bool check_dup) override {
        uint64_t start, end;

        std::shared_ptr<BaseLoader> loader;
        // load from hdfs or posix file
        if (boost::starts_with(dname, "hdfs:"))
            loader = std::make_shared<HDFSLoader>(sid, mem, str_server);
        else
            loader = std::make_shared<PosixLoader>(sid, mem, str_server);

        // step 1: load ID-mapping files and construct id2id mapping
        // TODO

        // step 2: list files to load

        // ID-format data files
        std::vector<std::string> dfiles(loader->list_files(dname, "id_"));
        // ID-format attribute files
        std::vector<std::string> afiles(loader->list_files(dname, "attr_"));

        if (dfiles.size() == 0 && afiles.size() == 0) {
            logstream(LOG_WARNING) << "no files found in directory (" << dname
                                   << ") at server " << sid << LOG_endl;
            return 0;
        }
        logstream(LOG_INFO) << dfiles.size() << " data files and "
                            << afiles.size() << " attribute files found in directory ("
                            << dname << ") at server " << sid << LOG_endl;

        std::sort(dfiles.begin(), dfiles.end());

        int num_dfiles = dfiles.size();

        auto read_file = [&](std::istream& file, uint64_t& cnt, int tid, bool check_dup) {
            sid_t s, p, o;
            while (file >> s >> p >> o) {
                if (this->sid == PARTITION(s)) {
                    insert_triple_out(triple_t(s, p, o), check_dup, tid);
                    cnt++;
                }

                if (this->sid == PARTITION(o)) {
                    insert_triple_in(triple_t(s, p, o), check_dup, tid);
                    cnt++;
                }
            }
        };

        // step 3: load triples into gstore
        start = timer::get_usec();
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < num_dfiles; i++) {
            uint64_t cnt = 0;
            int tid = omp_get_thread_num();
            std::istream* file = loader->init_istream(dfiles[i]);
            read_file(*file, cnt, tid, check_dup);
            loader->close_istream(file);
            logstream(LOG_INFO) << "load " << cnt << " triples from file " << dfiles[i]
                                << " at server " << sid << LOG_endl;
        }
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting into gstore" << LOG_endl;

        // step 4: load attribute triples
        // TODO

        return 0;
    }

    void init_gstore(std::vector<std::vector<triple_t>>& triple_pso,
                     std::vector<std::vector<triple_t>>& triple_pos,
                     std::vector<std::vector<triple_attr_t>>& triple_sav) override {
        uint64_t start, end;

        start = timer::get_usec();
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int t = 0; t < Global::num_engines; t++) {
            insert_normal(t, triple_pso[t], triple_pos[t]);

            // release memory
            std::vector<triple_t>().swap(triple_pso[t]);
            std::vector<triple_t>().swap(triple_pos[t]);
        }
        end = timer::get_usec();
        logstream(LOG_INFO) << "[DynamicRDFGraph] #" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting normal data into gstore" << LOG_endl;

        start = timer::get_usec();
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int t = 0; t < Global::num_engines; t++) {
            insert_attr(triple_sav[t], t);

            // release memory
            std::vector<triple_attr_t>().swap(triple_sav[t]);
        }
        end = timer::get_usec();
        logstream(LOG_INFO) << "[DynamicRDFGraph] #" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting attributes into gstore" << LOG_endl;

        start = timer::get_usec();
        insert_index();
        end = timer::get_usec();
        logstream(LOG_INFO) << "[DynamicRDFGraph] #" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting index data into gstore" << LOG_endl;
    }

    void insert_triple_out(const triple_t& triple, bool check_dup, int tid) {
        bool dedup_or_isdup = check_dup;
        bool nodup = false;
        if (triple.p == TYPE_ID) {
            // for TYPE_ID condition, dedup is always needed
            // for LUBM benchmark, maybe for others,too.
            dedup_or_isdup = true;
            ikey_t key = ikey_t(triple.s, triple.p, OUT);
        #ifdef TRDF_MODE
            edge_t value = edge_t(triple.o, triple.ts, triple.te);
        #else
            edge_t value = edge_t(triple.o);
        #endif
            // <1> vid's type (7) [need dedup]
            if (this->gstore->insert_key_value(key, value, dedup_or_isdup, tid)) {
#ifdef VERSATILE
                key = ikey_t(triple.s, PREDICATE_ID, OUT);
            #if TRDF_MODE
                value = edge_t(triple.p, triple.ts, triple.te);
            #else
                value = edge_t(triple.p);
            #endif
                // key and its buddy_key should be used to
                // identify the exist of corresponding index
                ikey_t buddy_key = ikey_t(triple.s, PREDICATE_ID, IN);
                // <2> vid's predicate, value is TYPE_ID (*8) [dedup from <1>]
                if (this->gstore->insert_key_value(key, value, nodup, tid) && !this->gstore->check_key_exist(buddy_key)) {
                    key = ikey_t(0, TYPE_ID, IN);
                #if TRDF_MODE
                    value = edge_t(triple.s, triple.ts, triple.te);
                #else
                    value = edge_t(triple.s);
                #endif
                    // <3> the index to vid (*3) [dedup from <2>]
                    this->gstore->insert_key_value(key, value, nodup, tid);
                }
#endif  // end of VERSATILE
            }
            if (!dedup_or_isdup) {
                key = ikey_t(0, triple.o, IN);
            #ifdef TRDF_MODE
                value = edge_t(triple.s, triple.ts, triple.te);
            #else
                value = edge_t(triple.s);
            #endif
                // <4> type-index (2) [if <1>'s result is not dup, this is not dup, too]
                if (this->gstore->insert_key_value(key, value, nodup, tid)) {
#ifdef VERSATILE
                    key = ikey_t(0, TYPE_ID, OUT);
                #if TRDF_MODE
                    value = edge_t(triple.o, triple.ts, triple.te);
                #else
                    value = edge_t(triple.o);
                #endif
                    // <5> index to this type (*4) [dedup from <4>]
                    this->gstore->insert_key_value(key, value, nodup, tid);
#endif  // end of VERSATILE
                }
            }
        } else {
            ikey_t key = ikey_t(triple.s, triple.p, OUT);
        #ifdef TRDF_MODE
            edge_t value = edge_t(triple.o, triple.ts, triple.te);
        #else
            edge_t value = edge_t(triple.o);
        #endif
            // <6> vid's ngbrs w/ predicate (6) [need dedup]
            if (this->gstore->insert_key_value(key, value, dedup_or_isdup, tid)) {
                key = ikey_t(0, triple.p, IN);
            #ifdef TRDF_MODE
                value = edge_t(triple.s, triple.ts, triple.te);
            #else
                value = edge_t(triple.s);
            #endif
                // key and its buddy_key should be used to
                // identify the exist of corresponding index
                ikey_t buddy_key = ikey_t(0, triple.p, OUT);
                // <7> predicate-index (1) [dedup from <6>]
                if (this->gstore->insert_key_value(key, value, nodup, tid) && !this->gstore->check_key_exist(buddy_key)) {
#ifdef VERSATILE
                    key = ikey_t(0, PREDICATE_ID, OUT);
                #if TRDF_MODE
                    value = edge_t(triple.p, triple.ts, triple.te);
                #else
                    value = edge_t(triple.p);
                #endif
                    // <8> the index to predicate (*5) [dedup from <7>]
                    this->gstore->insert_key_value(key, value, nodup, tid);
#endif  // end of VERSATILE
                }
#ifdef VERSATILE
                key = ikey_t(triple.s, PREDICATE_ID, OUT);
            #if TRDF_MODE
                value = edge_t(triple.p, triple.ts, triple.te);
            #else
                value = edge_t(triple.p);
            #endif
                // key and its buddy_key should be used to
                // identify the exist of corresponding index
                buddy_key = ikey_t(triple.s, PREDICATE_ID, IN);
                // <9> vid's predicate (*8) [dedup from <6>]
                if (this->gstore->insert_key_value(key, value, nodup, tid) && !this->gstore->check_key_exist(buddy_key)) {
                    key = ikey_t(0, TYPE_ID, IN);
                #if TRDF_MODE
                    value = edge_t(triple.s, triple.ts, triple.te);
                #else
                    value = edge_t(triple.s);
                #endif
                    // <10> the index to vid (*3) [dedup from <9>]
                    this->gstore->insert_key_value(key, value, nodup, tid);
                }
#endif  // end of VERSATILE
            }
        }
    }

    void insert_triple_in(const triple_t& triple, bool check_dup, int tid) {
        bool dedup_or_isdup = check_dup;
        bool nodup = false;
        // skip type triples
        if (triple.p == TYPE_ID) return;
        ikey_t key = ikey_t(triple.o, triple.p, IN);
    #ifdef TRDF_MODE
        edge_t value = edge_t(triple.s, triple.ts, triple.te);
    #else
        edge_t value = edge_t(triple.s);
    #endif
        // <1> vid's ngbrs w/ predicate (6) [need dedup]
        if (this->gstore->insert_key_value(key, value, dedup_or_isdup, tid)) {
            // key doesn't exist before
            key = ikey_t(0, triple.p, OUT);
        #ifdef TRDF_MODE
            value = edge_t(triple.o, triple.ts, triple.te);
        #else
            value = edge_t(triple.o);
        #endif
            // key and its buddy_key should be used
            // to identify the exist of corresponding index
            ikey_t buddy_key = ikey_t(0, triple.p, IN);
            // <2> predicate-index (1) [dedup from <1>]
            if (this->gstore->insert_key_value(key, value, nodup, tid) && !this->gstore->check_key_exist(buddy_key)) {
#ifdef VERSATILE
                key = ikey_t(0, PREDICATE_ID, OUT);
            #if TRDF_MODE
                value = edge_t(triple.p, triple.ts, triple.te);
            #else
                value = edge_t(triple.p);
            #endif
                // <3> the index to predicate (*5) [dedup from <2>]
                this->gstore->insert_key_value(key, value, nodup, tid);
#endif  // end of VERSATILE
            }
#ifdef VERSATILE
            key = ikey_t(triple.o, PREDICATE_ID, IN);
        #if TRDF_MODE
            value = edge_t(triple.p, triple.ts, triple.te);
        #else
            value = edge_t(triple.p);
        #endif
            // key and its buddy_key should be used to
            // identify the exist of corresponding index
            buddy_key = ikey_t(triple.o, PREDICATE_ID, OUT);
            // <4> vid's predicate (*8) [dedup from <1>]
            if (this->gstore->insert_key_value(key, value, nodup, tid) && !this->gstore->check_key_exist(buddy_key)) {
                key = ikey_t(0, TYPE_ID, IN);
            #ifdef TRDF_MODE
                value = edge_t(triple.o, triple.ts, triple.te);
            #else
                value = edge_t(triple.o);
            #endif
                // <5> the index to vid (*3) [dedup from <4>]
                this->gstore->insert_key_value(key, value, nodup, tid);
            }
#endif  // end of VERSATILE
        }
    }
};

}  // namespace wukong
