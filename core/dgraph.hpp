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

#include "loader/posix_loader.hpp"
#include "loader/hdfs_loader.hpp"
#include "gstore.hpp"
#ifdef DYNAMIC_GSTORE
#include "loader/dynamic_loader.hpp"
#endif

using namespace std;

/**
 * Map the RDF model (e.g., triples, predicate) to Graph model (e.g., vertex, edge, index)
 */
class DGraph {
private:
    int sid;
    BaseLoader *loader;


public:
    GStore *gstore;
#ifdef DYNAMIC_GSTORE
    DynamicLoader *dynamic_loader;
#endif

    DGraph(int sid, Mem *mem, String_Server *str_server, string dname): sid(sid) {
        gstore = new GStore(sid, mem);
        #ifdef DYNAMIC_GSTORE
            dynamic_loader = new DynamicLoader(sid, str_server, gstore);
        #endif
        //load from hdfs or posix file
        if (boost::starts_with(dname, "hdfs:"))
            loader = new HDFSLoader(sid, mem, str_server, gstore);
        else
            loader = new PosixLoader(sid, mem, str_server, gstore);

        uint64_t start, end;

        vector<vector<triple_t>> triple_pso;
        vector<vector<triple_t>> triple_pos;
        vector<vector<triple_attr_t>> triple_sav;

        start = timer::get_usec();
        loader->load(dname, triple_pso, triple_pos, triple_sav);
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for loading triples from disk to memory." << LOG_endl;

#ifdef USE_GPU

        start = timer::get_usec();
        // merge triple_pso and triple_pos into a map
        gstore->init_triples_map(triple_pso, triple_pos);
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for merging triple_pso and triple_pos." << LOG_endl;

        start = timer::get_usec();
        gstore->init_segment_metas(triple_pso, triple_pos);
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for initializing predicate segment statistics." << LOG_endl;

        start = timer::get_usec();
        auto& predicates = gstore->get_all_predicates();
        logstream(LOG_DEBUG) << "#" << sid << ": all_predicates: " << predicates.size() << LOG_endl;
        #pragma omp parallel for num_threads(global_num_engines)
        for (int i = 0; i < predicates.size(); i++) {
            int localtid = omp_get_thread_num();
            sid_t pid = predicates[i];
            gstore->insert_triples_to_segment(localtid, segid_t(0, pid, OUT));
            gstore->insert_triples_to_segment(localtid, segid_t(0, pid, IN));
        }
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting triples as segments into gstore" << LOG_endl;

        gstore->finalize_segment_metas();
        gstore->free_triples_map();

        // synchronize segment metadata among servers
        extern TCP_Adaptor *con_adaptor;
        gstore->sync_metadata(con_adaptor);

#else   // !USE_GPU

        start = timer::get_usec();
        #pragma omp parallel for num_threads(global_num_engines)
        for (int t = 0; t < global_num_engines; t++) {
            gstore->insert_normal(triple_pso[t], triple_pos[t], t);

            // release memory
            vector<triple_t>().swap(triple_pso[t]);
            vector<triple_t>().swap(triple_pos[t]);
        }
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting normal data into gstore" << LOG_endl;

        start = timer::get_usec();
        #pragma omp parallel for num_threads(global_num_engines)
        for (int t = 0; t < global_num_engines; t++) {
            gstore->insert_attr(triple_sav[t], t);

            // release memory
            vector<triple_attr_t>().swap(triple_sav[t]);
        }
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting attributes into gstore" << LOG_endl;

        start = timer::get_usec();
        gstore->insert_index();
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting index data into gstore" << LOG_endl;

#endif  // end of USE_GPU

        logstream(LOG_INFO) << "#" << sid << ": loading DGraph is finished" << LOG_endl;
        print_graph_stat();

        gstore->print_mem_usage();
    }

    ~DGraph() {
        delete gstore;
        delete loader;
#ifdef DYNAMIC_GSTORE
        delete dynamic_loader;
#endif
    }

    int gstore_check(bool index_check, bool normal_check) {
        return gstore->gstore_check(index_check, normal_check);
    }

    edge_t *get_triples(int tid, sid_t vid, sid_t pid, dir_t d, uint64_t &sz) {
        return gstore->get_edges(tid, vid, pid, d, sz);
    }

    edge_t *get_index(int tid, sid_t pid, dir_t d, uint64_t &sz) {
        return gstore->get_edges(tid, 0, pid, d, sz);
    }

    // return attribute value (has_value == true)
    attr_t get_attr(int tid, sid_t vid, sid_t pid, dir_t d, bool &has_value) {
        uint64_t sz = 0;
        int type = 0;
        attr_t r;

        // get the pointer of edge
        edge_t *edge_ptr = gstore->get_edges(tid, vid, pid, d, sz, type);
        if (edge_ptr == NULL) {
            has_value = false; // not found
            return r;
        }

        // get the value of attribute by type
        switch (type) {
        case INT_t:
            r = *((int *)(edge_ptr));
            break;
        case FLOAT_t:
            r = *((float *)(edge_ptr));
            break;
        case DOUBLE_t:
            r = *((double *)(edge_ptr));
            break;
        default:
            logstream(LOG_ERROR) << "Unsupported value type." << LOG_endl;
            break;
        }

        has_value = true;
        return r;
    }

    void print_graph_stat() {
#ifdef VERSATILE
        /// (*3)  key = [  0 |      TYPE_ID |     IN]  value = [vid0, vid1, ..]  i.e., all local objects/subjects
        /// (*4)  key = [  0 |      TYPE_ID |    OUT]  value = [pid0, pid1, ..]  i.e., all local types
        /// (*5)  key = [  0 | PREDICATE_ID |    OUT]  value = [pid0, pid1, ..]  i.e., all local predicates
        uint64_t sz = 0;

        gstore->get_edges(0, 0, TYPE_ID, IN, sz);
        logstream(LOG_INFO) << "#vertices: " << sz << LOG_endl;

        gstore->get_edges(0, 0, TYPE_ID, OUT, sz);
        logstream(LOG_INFO) << "#types: " << sz << LOG_endl;

        gstore->get_edges(0, 0, PREDICATE_ID, OUT, sz);
        logstream(LOG_INFO) << "#predicates: " << sz << " (not including types)" << LOG_endl;
#endif // end of VERSATILE
    }
};
