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

#include "core/common/string_server.hpp"
#include "core/store/dgraph.hpp"

namespace wukong {

/**
 * @brief Normal RDF Graph(Support dynamic)
 * 
 */
class RDFGraph : public DGraph {
    bool dynamic;

    void collect_idx_info(RDFStore::slot_t& slot) {
        sid_t vid = slot.key.vid;
        sid_t pid = slot.key.pid;

        uint64_t sz = slot.ptr.size;
        uint64_t off = slot.ptr.off;

        if (slot.key.dir == IN) {
            if (pid == PREDICATE_ID) {
            } else if (pid == TYPE_ID) {
                // (IN) type triples should be skipped
                ASSERT(false);
            } else {  // predicate-index (OUT) vid
                tbb_edge_hash_map::accessor a;
                pidx_out_map.insert(a, pid);
            #if TRDF_MODE
                a->second.push_back(edge_t(vid, TIMESTAMP_MIN, TIMESTAMP_MAX));
            #else
                a->second.push_back(edge_t(vid));
            #endif
            }
        } else {
            if (pid == PREDICATE_ID) {
            } else if (pid == TYPE_ID) {
#ifdef VERSATILE
                // every subject/object has at least one predicate or one type
                v_set.insert(vid);  // collect all local subjects w/ type
#endif
                // type-index (IN) vid
                for (uint64_t e = 0; e < sz; e++) {
                    tbb_edge_hash_map::accessor a;
                    tidx_map.insert(a, this->gstore->values[off + e].val);
                #if TRDF_MODE
                    a->second.push_back(edge_t(vid, this->gstore->values[off + e].ts, this->gstore->values[off + e].te));
                #else
                    a->second.push_back(edge_t(vid));
                #endif
#ifdef VERSATILE
                    t_set.insert(this->gstore->values[off + e].val);  // collect all local types
#endif
                }
            } else {  // predicate-index (IN) vid
                tbb_edge_hash_map::accessor a;
                pidx_in_map.insert(a, pid);
            #if TRDF_MODE
                a->second.push_back(edge_t(vid, TIMESTAMP_MIN, TIMESTAMP_MAX));
            #else
                a->second.push_back(edge_t(vid));
            #endif
            }
        }
    }

    /// skip all TYPE triples (e.g., <http://www.Department0.University0.edu> rdf:type ub:University)
    /// because Wukong treats all TYPE triples as index vertices. In addition, the triples in triple_pos
    /// has been sorted by the vid of object, and IDs of types are always smaller than normal vertex IDs.
    /// Consequently, all TYPE triples are aggregated at the beggining of triple_pos
    void insert_normal(int tid, std::vector<triple_t>& pso, std::vector<triple_t>& pos) {
        // treat type triples as index vertices
        uint64_t type_triples = 0;
        while (type_triples < pos.size() && is_tpid(pos[type_triples].o))
            type_triples++;

#ifdef VERSATILE
        /// The following code is used to support a rare case where the predicate is unknown
        /// (e.g., <http://www.Department0.University0.edu> ?P ?O). Each normal vertex should
        /// add two key/value pairs with a reserved ID (i.e., PREDICATE_ID) as the predicate
        /// to store the IN and OUT lists of its predicates.
        /// e.g., key=(vid, PREDICATE_ID, IN/OUT), val=(predicate0, predicate1, ...)
        ///
        /// NOTE, it is disabled by default in order to save memory.
        std::vector<sid_t> predicates;
#endif  // end of VERSATILE

        uint64_t s = 0;
        while (s < pso.size()) {
            // predicate-based key (subject + predicate)
            uint64_t e = s + 1;
            while ((e < pso.size()) && (pso[s].s == pso[e].s) && (pso[s].p == pso[e].p)) {
                e++;
            }

            // allocate entries
            uint64_t off = this->gstore->alloc_entries(e - s, tid);

            // insert subject & predicate
            uint64_t slot_id = this->gstore->insert_key(
                ikey_t(pso[s].s, pso[s].p, OUT), iptr_t(e - s, off));

            // insert objects
            for (uint64_t i = s; i < e; i++) {
            #if TRDF_MODE
                this->gstore->values[off++] = edge_t(pso[i].o, pso[i].ts, pso[i].te);
            #else
                this->gstore->values[off++] = edge_t(pso[i].o);
            #endif
            }

            collect_idx_info(this->gstore->slots[slot_id]);

#ifdef VERSATILE
            // add a new predicate
            predicates.push_back(pso[s].p);

            // insert a special PREDICATE triple (OUT)
            // store predicates of a vertex
            if (e >= pso.size() || pso[s].s != pso[e].s) {
                // every subject/object has at least one predicate or one type
                v_set.insert(pso[s].s);  // collect all local objects w/ predicate

                // allocate entries
                uint64_t sz = predicates.size();
                uint64_t off = this->gstore->alloc_entries(sz, tid);

                // insert subject
                uint64_t slot_id = this->gstore->insert_key(
                    ikey_t(pso[s].s, PREDICATE_ID, OUT),
                    iptr_t(sz, off));

                // insert predicates
                for (auto const& p : predicates) {
                #if TRDF_MODE
                    this->gstore->values[off++] = edge_t(p, TIMESTAMP_MIN, TIMESTAMP_MAX);
                #else
                    this->gstore->values[off++] = edge_t(p);
                #endif
                    p_set.insert(p);  // collect all local predicates
                }

                predicates.clear();
            }
#endif  // end of VERSATILE

            s = e;
        }

        s = type_triples;  // skip type triples
        while (s < pos.size()) {
            // predicate-based key (object + predicate)
            uint64_t e = s + 1;
            while ((e < pos.size()) && (pos[s].o == pos[e].o) && (pos[s].p == pos[e].p)) {
                e++;
            }

            // allocate entries
            uint64_t off = this->gstore->alloc_entries(e - s, tid);

            // insert object
            uint64_t slot_id = this->gstore->insert_key(
                ikey_t(pos[s].o, pos[s].p, IN),
                iptr_t(e - s, off));

            // insert values
            for (uint64_t i = s; i < e; i++) {
            #if TRDF_MODE
                this->gstore->values[off++] = edge_t(pos[i].s, pos[i].ts, pos[i].te);
            #else
                this->gstore->values[off++] = edge_t(pos[i].s);
            #endif
            }

            collect_idx_info(this->gstore->slots[slot_id]);

#ifdef VERSATILE
            // add a new predicate
            predicates.push_back(pos[s].p);

            // insert a special PREDICATE triple (OUT)
            if (e >= pos.size() || pos[s].o != pos[e].o) {
                // every subject/object has at least one predicate or one type
                v_set.insert(pos[s].o);  // collect all local subjects w/ predicate

                // allocate entries
                uint64_t sz = predicates.size();
                uint64_t off = this->gstore->alloc_entries(sz, tid);

                // insert subject
                uint64_t slot_id = this->gstore->insert_key(
                    ikey_t(pos[s].o, PREDICATE_ID, IN),
                    iptr_t(sz, off));

                // insert predicate
                for (auto const& p : predicates) {
                #if TRDF_MODE
                    this->gstore->values[off++] = edge_t(p, TIMESTAMP_MIN, TIMESTAMP_MAX);
                #else
                    this->gstore->values[off++] = edge_t(p);
                #endif
                    p_set.insert(p);  // collect all local predicates
                }

                predicates.clear();
            }
#endif  // end of VERSATILE
            s = e;
        }
    }

    // insert attributes
    void insert_attr(std::vector<triple_attr_t>& attrs, int64_t tid) {
        for (auto const& attr : attrs) {
            // allocate entries
            int type = boost::apply_visitor(variant_type(), attr.v);
            uint64_t sz = (get_sizeof(type) - 1) / sizeof(edge_t) + 1;  // get the ceil size;
            uint64_t off = this->gstore->alloc_entries(sz, tid);

            // insert subject
            uint64_t slot_id = this->gstore->insert_key(
                ikey_t(attr.s, attr.a, OUT),
                iptr_t(sz, off, type));

            // insert values (attributes)
            switch (type) {
            case INT_t:
                *reinterpret_cast<int*>(this->gstore->values + off) = boost::get<int>(attr.v);
                break;
            case FLOAT_t:
                *reinterpret_cast<float*>(this->gstore->values + off) = boost::get<float>(attr.v);
                break;
            case DOUBLE_t:
                *reinterpret_cast<double*>(this->gstore->values + off) = boost::get<double>(attr.v);
                break;
            default:
                logstream(LOG_ERROR) << "Unsupported value type of attribute" << LOG_endl;
            }
        }
    }

    virtual void insert_index() {
        uint64_t t1 = timer::get_usec();

        // insert type-index & predicate-idnex edges in parallel
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < 3; i++) {
            if (i == 0) insert_index_map(tidx_map, IN);
            if (i == 1) insert_index_map(pidx_in_map, IN);
            if (i == 2) insert_index_map(pidx_out_map, OUT);
        }

        // init edge_predicates and type_predicates
        std::set<sid_t> type_pred_set;
        std::set<sid_t> edge_pred_set;
        for (auto& e : tidx_map) type_pred_set.insert(e.first);
        for (auto& e : pidx_in_map) edge_pred_set.insert(e.first);
        for (auto& e : pidx_out_map) edge_pred_set.insert(e.first);
        edge_pred_set.insert(TYPE_ID);
        this->type_predicates.assign(type_pred_set.begin(), type_pred_set.end());
        this->edge_predicates.assign(edge_pred_set.begin(), edge_pred_set.end());

        tbb_edge_hash_map().swap(pidx_in_map);
        tbb_edge_hash_map().swap(pidx_out_map);
        tbb_edge_hash_map().swap(tidx_map);

#ifdef VERSATILE
        insert_idx_set(v_set, TYPE_ID, IN);
        insert_idx_set(t_set, TYPE_ID, OUT);
        insert_idx_set(p_set, PREDICATE_ID, OUT);

        tbb_unordered_set().swap(v_set);
        tbb_unordered_set().swap(t_set);
        tbb_unordered_set().swap(p_set);
#endif

        uint64_t t2 = timer::get_usec();
        logstream(LOG_DEBUG) << (t2 - t1) / 1000 << " ms for inserting index data into gstore" << LOG_endl;
    }

    void insert_index_map(tbb_edge_hash_map& map, dir_t d) {
        for (auto const& e : map) {
            // alloc entries
            sid_t pid = e.first;
            uint64_t sz = e.second.size();
            uint64_t off = this->gstore->alloc_entries(sz, 0);

            // insert index key
            uint64_t slot_id = this->gstore->insert_key(
                ikey_t(0, pid, d),
                iptr_t(sz, off));

            // insert subjects/objects
            for (auto const& edge : e.second)
                this->gstore->values[off++] = edge;
        }
    }

#ifdef VERSATILE
    // insert {v/t/p}_set into gstore
    void insert_idx_set(tbb_unordered_set& set, sid_t tpid, dir_t d) {
        // alloc entries
        uint64_t sz = set.size();
        uint64_t off = this->gstore->alloc_entries(sz, 0);

        // insert index key
        uint64_t slot_id = this->gstore->insert_key(
            ikey_t(0, tpid, d),
            iptr_t(sz, off));

        // insert index value
        for (auto const& e : set) {
        #if TRDF_MODE
            edge_t edge(e, TIMESTAMP_MIN, TIMESTAMP_MAX);
        #else
            edge_t edge(e);
        #endif
            this->gstore->values[off++] = edge;
        }
    }
#endif  // VERSATILE


public:
    RDFGraph(int sid, KVMem kv_mem, bool dynamic)
        : DGraph(sid, kv_mem), dynamic(dynamic) {
        if (!dynamic)  // static
            this->gstore = std::make_shared<StaticKVStore<ikey_t, iptr_t, edge_t>>(sid, kv_mem);
        else  // dynamic
            this->gstore = std::make_shared<DynamicKVStore<ikey_t, iptr_t, edge_t>>(sid, kv_mem);
    }

    ~RDFGraph() {}

    void init_gstore(std::vector<std::vector<triple_t>>& triple_pso,
                     std::vector<std::vector<triple_t>>& triple_pos,
                     std::vector<std::vector<triple_attr_t>>& triple_sav) override {
        uint64_t start, end;

        /* insert normal nodes inot gstore */
        start = timer::get_usec();
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int t = 0; t < Global::num_engines; t++) {
            insert_normal(t, triple_pso[t], triple_pos[t]);

            // release memory
            std::vector<triple_t>().swap(triple_pso[t]);
            std::vector<triple_t>().swap(triple_pos[t]);
        }
        end = timer::get_usec();
        if (this->dynamic)
            logstream(LOG_INFO) << "[DynamicRDFGraph] #";
        else
            logstream(LOG_INFO) << "[StaticRDFGraph] #";
        logstream(LOG_INFO) << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting normal data into gstore" << LOG_endl;

        /* insert attributes into gstore */
        start = timer::get_usec();
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int t = 0; t < Global::num_engines; t++) {
            insert_attr(triple_sav[t], t);

            // release memory
            std::vector<triple_attr_t>().swap(triple_sav[t]);
        }
        end = timer::get_usec();
        if (this->dynamic)
            logstream(LOG_INFO) << "[DynamicRDFGraph] #";
        else
            logstream(LOG_INFO) << "[StaticRDFGraph] #";
        logstream(LOG_INFO) << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting attributes into gstore" << LOG_endl;

        /* insert index into gstore */
        start = timer::get_usec();
        insert_index();
        end = timer::get_usec();
        if (this->dynamic)
            logstream(LOG_INFO) << "[DynamicRDFGraph] #";
        else
            logstream(LOG_INFO) << "[StaticRDFGraph] #";
        logstream(LOG_INFO) << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting index data into gstore" << LOG_endl;
    }

    int dynamic_load_data(std::string dname, bool check_dup) override {
        // check dynamic attr
        ASSERT_EQ(this->dynamic, true);

        uint64_t start, end;

        std::shared_ptr<BaseLoader> loader;
        // load from hdfs or posix file
        BaseLoader::LoaderMem loader_mem = {
            .global_buf = kv_mem.kvs, .global_buf_sz = kv_mem.kvs_sz,
            .local_buf = kv_mem.rrbuf, .local_buf_sz = kv_mem.rrbuf_sz
        };
        if (boost::starts_with(dname, "hdfs:"))
            loader = std::make_shared<HDFSLoader>(sid, loader_mem);
        else
            loader = std::make_shared<PosixLoader>(sid, loader_mem);

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

    void insert_triple_out(const triple_t& triple, bool check_dup, int tid) {
        // check dynamic attr
        ASSERT_EQ(this->dynamic, true);
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
        // check dynamic attr
        ASSERT_EQ(this->dynamic, true);
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
