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

#ifdef DYNAMIC_GSTORE
#include "store/dynamic_gstore.hpp"
#include "loader/dynamic_loader.hpp"
#else
#include "store/static_gstore.hpp"
#endif

#include "store/gchecker.hpp"

using namespace std;

/**
 * Map the RDF model (e.g., triples, predicate) to Graph model (e.g., vertex, edge, index)
 */
class DGraph {
private:
    int sid;
    BaseLoader *loader;
    GChecker *checker;


public:
    GStore *gstore;
#ifdef DYNAMIC_GSTORE
    DynamicLoader *dynamic_loader;
#endif

    DGraph(int sid, Mem *mem, StringServer *str_server, string dname): sid(sid) {
#ifdef DYNAMIC_GSTORE
        gstore = new DynamicGStore(sid, mem);
        dynamic_loader = new DynamicLoader(sid, str_server, static_cast<DynamicGStore *>(gstore));
#else
        gstore = new StaticGStore(sid, mem);
#endif
        checker = new GChecker(gstore);
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

        start = timer::get_usec();
        gstore->init(triple_pso, triple_pos, triple_sav);
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for initializing gstore." << LOG_endl;

        logstream(LOG_INFO) << "#" << sid << ": loading DGraph is finished" << LOG_endl;
        print_graph_stat();

        gstore->print_mem_usage();
    }

    ~DGraph() {
        delete gstore;
        delete checker;
        delete loader;
#ifdef DYNAMIC_GSTORE
        delete dynamic_loader;
#endif
    }

    int gstore_check(bool index_check, bool normal_check) {
        return checker->gstore_check(index_check, normal_check);
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
