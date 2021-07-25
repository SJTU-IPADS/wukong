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

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

// loader
#include "loader/hdfs_loader.hpp"
#include "loader/posix_loader.hpp"

// store
#include "core/store/gchecker.hpp"
#include "core/store/static_kvstore.hpp"
#include "core/store/dynamic_kvstore.hpp"

namespace wukong {

using RDFStore = KVStore<ikey_t, iptr_t, edge_t>;

/**
 * @brief RDFGraph
 * 
 * Map the RDFGraph model (e.g., triples, predicate) to KVS model (e.g., key, value)
 * Graph store adopts clustring chaining key/value store (see paper: DrTM SOSP'15)
 *
 *  encoding rules of KVStore
 *  subject/object (vid) >= 2^NBITS_IDX, 2^NBITS_IDX > predicate/type (p/tid) >= 2^1,
 *  PREDICATE_ID = 0, TYPE_ID = 1
 *  IN = 0, OUT = 1
 *
 *  Empty key
 *  (0)   key = [  0 |            0 |      0]  value = [vid0, vid1, ..]  i.e., init
 *  INDEX key/value pair
 *  (1)   key = [  0 |          pid | IN/OUT]  value = [vid0, vid1, ..]  i.e., predicate-index
 *  (2)   key = [  0 |          tid |     IN]  value = [vid0, vid1, ..]  i.e., type-index
 *  (3*)  key = [  0 |      TYPE_ID |     IN]  value = [vid0, vid1, ..]  i.e., all local objects/subjects
 *  (4*)  key = [  0 |      TYPE_ID |    OUT]  value = [pid0, pid1, ..]  i.e., all local types
 *  (5*)  key = [  0 | PREDICATE_ID |    OUT]  value = [pid0, pid1, ..]  i.e., all local predicates
 *  NORMAL key/value pair
 *  (6)   key = [vid |          pid | IN/OUT]  value = [vid0, vid1, ..]  i.e., vid's ngbrs w/ predicate
 *  (7)   key = [vid |      TYPE_ID |    OUT]  value = [tid0, tid1, ..]  i.e., vid's all types
 *  (8*)  key = [vid | PREDICATE_ID | IN/OUT]  value = [pid0, pid1, ..]  i.e., vid's all predicates
 *
 *  < S,  P, ?O>  ?O : (6)
 *  <?S,  P,  O>  ?S : (6)
 *  < S,  1, ?T>  ?T : (7)
 *  <?S,  1,  T>  ?S : (2)
 *  < S, ?P,  O>  ?P : (8)
 *
 *  <?S,  P, ?O>  ?S : (1)
 *                ?O : (1)
 *  <?S,  1, ?O>  ?O : (4)
 *                ?S : (4) +> (2)
 *  < S, ?P, ?O>  ?P : (8) AND exist(7)
 *                ?O : (8) AND exist(7) +> (6)
 *  <?S, ?P,  O>  ?P : (8)
 *                ?S : (8) +> (6)
 *  <?S, ?P,  T>  ?P : exist(2)
 *                ?S : (2)
 *
 *  <?S, ?P, ?O>  ?S : (3)
 *                ?O : (3) AND (4)
 *                ?P : (5)
 *                ?S ?P ?O : (3) +> (7) AND (8) +> (6)
 */
class DGraph {
protected:
    using tbb_unordered_set = tbb::concurrent_unordered_set<sid_t>;
    using tbb_edge_hash_map = tbb::concurrent_hash_map<sid_t, std::vector<edge_t>>;
    using tbb_triple_hash_map = tbb::concurrent_hash_map<ikey_t, std::vector<triple_t>, ikey_Hasher>;
    using tbb_triple_attr_hash_map = tbb::concurrent_hash_map<ikey_t, std::vector<triple_attr_t>, ikey_Hasher>;

    int sid;
    KVMem kv_mem;

    // PREDICATE_ID+TYPE_ID+edge_predicates+type_predicates
    std::vector<sid_t> predicates;
    // attribute_predicate
    std::vector<sid_t> attributes;
    // edge_predicates (including TYPE_ID)
    std::vector<sid_t> edge_predicates;
    // type_predicates
    std::vector<sid_t> type_predicates;

    // key: id of vertex/edge attribute, value: type(INT_t, FLOAT_t, DOUBLE_t) + size(>=1)
    std::map<sid_t, std::pair<data_type, int>> attr_type_dim_map;

    tbb_edge_hash_map pidx_in_map;   // predicate-index (IN)
    tbb_edge_hash_map pidx_out_map;  // predicate-index (OUT)
    tbb_edge_hash_map tidx_map;      // type-index

#ifdef VERSATILE
    /**
     * These sets and maps will be freed after being inserted into entry region
     * description                     key
     * 1*. v_set, all local entities   [0|TYPE_ID|IN]
     * 2*. t_set, all local types      [0|TYPE_ID|OUT]
     * 3*. p_set, all local predicates [0|PREDICATE_ID|OUT]
     * 4*. out_preds, vid's OUT preds  [vid|PREDICATE_ID|OUT]
     * 5*. in_preds, vid's IN preds    [vid|PREDICATE_ID|IN]
     */
    tbb_unordered_set v_set;
    tbb_unordered_set p_set;
    tbb_unordered_set t_set;
#endif  // VERSATILE

    virtual void init_gstore(std::vector<std::vector<triple_t>>& triple_pso,
                             std::vector<std::vector<triple_t>>& triple_pos,
                             std::vector<std::vector<triple_attr_t>>& triple_sav) = 0;

public:
    std::shared_ptr<RDFStore> gstore;

    DGraph(int sid, KVMem kv_mem)
        : sid(sid), kv_mem(kv_mem){}

    void load(std::string dname) {
        uint64_t start, end;

        // load from hdfs or posix file
        std::shared_ptr<BaseLoader> loader;
        BaseLoader::LoaderMem loader_mem = {
            .global_buf = kv_mem.kvs, .global_buf_sz = kv_mem.kvs_sz,
            .local_buf = kv_mem.rrbuf, .local_buf_sz = kv_mem.rrbuf_sz
        };
        if (boost::starts_with(dname, "hdfs:"))
            loader = std::make_shared<HDFSLoader>(sid, loader_mem);
        else
            loader = std::make_shared<PosixLoader>(sid, loader_mem);

        /* load triples from disk */
        std::vector<std::vector<triple_t>> triple_pso;
        std::vector<std::vector<triple_t>> triple_pos;
        std::vector<std::vector<triple_attr_t>> triple_sav;
        start = timer::get_usec();
        loader->load(dname, triple_pso, triple_pos, triple_sav);
        end = timer::get_usec();
        logstream(LOG_INFO) << "[Loader] #" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for loading triples from disk to memory." << LOG_endl;

        auto count_preds = [this](const std::string str_idx_file, bool is_attr = false) {
            std::string pred;
            int pid;
            std::ifstream ifs(str_idx_file.c_str());
            while (ifs >> pred >> pid) {
                if (!is_attr) {
                    this->predicates.push_back(pid);
                } else {
                    this->attributes.push_back(pid);
                    this->attr_type_dim_map.insert(
                        std::make_pair(pid, std::make_pair(SID_t, -1)));
                }
            }
            ifs.close();
        };

        count_preds(dname + "str_index");
        if (this->predicates.size() <= 1) {
            logstream(LOG_ERROR) << "Encoding file of predicates should be named as \"str_index\". Graph loading failed. Please quit and try again." << LOG_endl;
        }
        if (Global::enable_vattr)
            count_preds(dname + "str_attr_index", true);

        // initiate gstore (kvstore) after loading and exchanging triples (memory reused)
        gstore->refresh();

        /* initialize gstore with triples */
        start = timer::get_usec();
        init_gstore(triple_pso, triple_pos, triple_sav);
        end = timer::get_usec();
        logstream(LOG_INFO) << "[RDFGraph] #" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for initializing gstore." << LOG_endl;

        logstream(LOG_INFO) << "[RDFGraph] #" << sid << ": loading RDFGraph is finished" << LOG_endl;

        print_graph_stat();
    }

    virtual ~DGraph() {}

    // return total num of preds, including normal and attr
    inline int get_num_normal_preds() const { return this->predicates.size() - 1; }
    inline int get_num_normal_preds_without_TYPEID() const { return this->predicates.size() - 2; }
    inline int get_num_edge_predicate() const { return this->edge_predicates.size(); }
    inline int get_num_type_predicate() const { return this->type_predicates.size(); }
    inline int get_num_attr_preds() const { return this->attributes.size(); }
    inline std::vector<sid_t> get_edge_predicates() const { return this->edge_predicates; }
    inline std::vector<sid_t> get_type_predicates() const { return this->type_predicates; }
    inline data_type get_attribute_type(sid_t pid) const {
        auto iter = this->attr_type_dim_map.find(pid);
        ASSERT(iter != this->attr_type_dim_map.end());
        return iter->second.first;
    }

    int gstore_check(bool index_check, bool normal_check) {
        GChecker checker(gstore);
        return checker.gstore_check(index_check, normal_check);
    }

    virtual edge_t* get_triples(int tid, sid_t vid, sid_t pid, dir_t d, uint64_t& sz) {
        return gstore->get_values(tid, PARTITION(vid), ikey_t(vid, pid, d), sz);
    }

    virtual edge_t* get_index(int tid, sid_t pid, dir_t d, uint64_t& sz) {
        // index vertex should be 0 and always local
        return gstore->get_values(tid, this->sid, ikey_t(0, pid, d), sz);
    }

    // return attribute value (has_value == true)
    virtual attr_t get_attr(int tid, sid_t vid, sid_t pid, dir_t d, bool& has_value) {
        uint64_t sz = 0;
        attr_t r;

        // get the pointer of edge
        data_type type = this->get_attribute_type(pid);
        edge_t* edge_ptr = reinterpret_cast<edge_t*>(gstore->get_values(tid, PARTITION(vid), ikey_t(vid, pid, d), sz));
        if (edge_ptr == NULL) {
            has_value = false;  // not found
            return r;
        }

        // get the value of attribute by type
        switch (type) {
        case INT_t:
            r = *(reinterpret_cast<int*>(edge_ptr));
            break;
        case FLOAT_t:
            r = *(reinterpret_cast<float*>(edge_ptr));
            break;
        case DOUBLE_t:
            r = *(reinterpret_cast<double*>(edge_ptr));
            break;
        default:
            logstream(LOG_ERROR) << "Unsupported value type." << LOG_endl;
            break;
        }

        has_value = true;
        return r;
    }

    virtual int dynamic_load_data(std::string dname, bool check_dup) {}

    virtual void print_graph_stat() {
        gstore->print_mem_usage();

#ifdef VERSATILE
        /// (*3)  key = [  0 |      TYPE_ID |     IN]  value = [vid0, vid1, ..]  i.e., all local objects/subjects
        /// (*4)  key = [  0 |      TYPE_ID |    OUT]  value = [pid0, pid1, ..]  i.e., all local types
        /// (*5)  key = [  0 | PREDICATE_ID |    OUT]  value = [pid0, pid1, ..]  i.e., all local predicates
        uint64_t sz = 0;

        gstore->get_values(0, this->sid, ikey_t(0, TYPE_ID, IN), sz);
        logstream(LOG_INFO) << "[RDF] #vertices: " << sz << LOG_endl;

        gstore->get_values(0, this->sid, ikey_t(0, TYPE_ID, OUT), sz);
        logstream(LOG_INFO) << "[RDF] #types: " << sz << LOG_endl;

        gstore->get_values(0, this->sid, ikey_t(0, PREDICATE_ID, OUT), sz);
        logstream(LOG_INFO) << "[RDF] #predicates: " << sz << " (not including types)" << LOG_endl;
#endif  // end of VERSATILE

        logstream(LOG_INFO) << "[RDF] #predicates: " << this->predicates.size() << LOG_endl;
        logstream(LOG_INFO) << "[RDF] #edge_predicates: " << this->edge_predicates.size() << LOG_endl;
        logstream(LOG_INFO) << "[RDF] #type_predicates: " << this->type_predicates.size() << LOG_endl;
        logstream(LOG_INFO) << "[RDF] #attributes: " << this->attributes.size() << LOG_endl;
    }
};

}  // namespace wukong
