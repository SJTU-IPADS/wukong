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
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/store/dgraph.hpp"

namespace wukong {

/**
 * @brief Segment-based RDF Graph
 *
 * Segments
 * #total: #normal pred(TYPE_ID not included) * 2(IN/OUT)
 *       + 1(TYPE_ID|OUT)
 *       + 1(index OUT, including predicate index OUT, all local types* and preds*)
 *       + 1(index IN, including predicate index IN, type index and all local entities*)
 *       + 2(vid's all predicates* IN/OUT)
 *       + #attr pred
 *
 * description              key                         segid                   num
 * 1. normal segments       [vid|pid|IN/OUT]            [0|pid|IN/OUT]          2 * #normal pred
 * 2. vid's all types       [vid|TYPE_ID|OUT]           [0|TYPE_ID|OUT]         1
 * 3. predicate index OUT   [0|pid|OUT]
 *    p_set*                [0|PREDICATE_ID|OUT]
 *    t_set*                [0|TYPEID|OUT]              [1|PREDICATE_ID|OUT]    1
 * 4. predicate index IN    [0|pid|IN]
 *    type index            [0|typeid|IN]
 *    v_set*                [0|TYPE_ID|IN]              [1|PREDICATE_ID|IN]     1
 * 5*. vid's all predicates [vid|PREDICATE_ID|IN/OUT]   [0|PREDICATE_ID|IN/OUT] 2
 * 6^. attr segments        [vid|pid|OUT]               [0|pid|OUT]             #attr pred
 */
class SegmentRDFGraph : public DGraph {
protected:
    struct cnt_t {
        cnt_t() {
            in = 0ul;
            out = 0ul;
        }

        cnt_t(const cnt_t& cnt) {
            in = cnt.in.load();
            out = cnt.out.load();
        }

        std::atomic<uint64_t> in, out;
    };

    // Index arrays which stores start offset of each predicate(attr)'s data.
    using triple_map_t = std::vector<std::unordered_map<sid_t, size_t>>;

    /**
     * automatically alloc buckets to segments, which will use all header region
     * you should turn it off if:
     * 1. you want to dynamically insert new predicates
     * 2. you don't want to use up all header region
     */
    bool global_auto_bkt_alloc = true;

    // bucket allocation offset for segments (temporal variable)
    uint64_t main_hdr_off = 0;

    // used for ext blocks allocation and ext bucket acquire
    pthread_spinlock_t seg_ext_locks[RDFStore::NUM_LOCKS];

    // key: server id, value: segment metadata of the server
    using SegMetaMap = std::map<segid_t, rdf_seg_meta_t>;

    // global meta map (multiple engines will access this map)
    tbb::concurrent_unordered_map<int, SegMetaMap> shared_rdf_seg_meta_map;
    // local meta map
    SegMetaMap rdf_seg_meta_map;

    // number of segments
    uint64_t num_segments;

    /**
     * minimum num of buckets per segment
     * usage: allocate buckets to empty segments (num_keys = 0)
     */
    uint64_t min_buckets_per_seg = 1;

    /**
     * @brief allocate the bucket for segments (statically)
     * 
     * @param seg segment meta data
     * @param segid segment id
     * @param total_num_keys total number of the keys in this segment
     */
    void alloc_buckets_to_seg(rdf_seg_meta_t& seg, segid_t segid, uint64_t total_num_keys) {
        // allocate buckets in main-header region to segments
        uint64_t nbuckets;
        if (seg.num_keys == 0) {
            nbuckets = 0;
        } else if (global_auto_bkt_alloc) {
            // deduct some buckets from total to prevent overflow
            static uint64_t num_free_buckets = this->gstore->num_buckets - num_segments;
            static double total_ratio_ = 0.0;

            double ratio = static_cast<double>(seg.num_keys) / total_num_keys;
            nbuckets = ratio * num_free_buckets;
            total_ratio_ += ratio;
            logger(LOG_DEBUG,
                   "Seg[%lu|%lu|%lu]: "
                   "#keys: %lu, nbuckets: %lu, bucket_off: %lu, "
                   "ratio: %f, total_ratio: %f",
                   segid.index, segid.pid, segid.dir,
                   seg.num_keys, nbuckets, main_hdr_off,
                   ratio, total_ratio_);
        } else {
            nbuckets = seg.num_keys * (100.0 / (Global::est_load_factor * RDFStore::ASSOCIATIVITY));
        }
        seg.num_buckets = std::max(nbuckets, min_buckets_per_seg);
        logger(LOG_DEBUG,
               "Seg[%lu|%lu|%lu]: "
               "#keys: %lu, nbuckets: %lu, bucket_off: %lu, ",
               segid.index, segid.pid, segid.dir,
               seg.num_keys, seg.num_buckets, main_hdr_off);

        seg.bucket_start = main_hdr_off;
        main_hdr_off += seg.num_buckets;
        ASSERT(main_hdr_off <= this->gstore->num_buckets);

        // allocate buckets in indirect-header region to segments
        // #buckets : #extended buckets = 1 : 0.15
        if (seg.num_buckets > 0) {
            uint64_t nbuckets = 0;
#ifdef USE_GPU
            nbuckets = EXT_BUCKET_EXTENT_LEN(seg.num_buckets);
#else
            nbuckets = EXT_BUCKET_EXTENT_LEN;
#endif
            uint64_t start_off = this->gstore->alloc_ext_buckets(nbuckets);
            seg.add_ext_buckets(ext_bucket_extent_t(nbuckets, start_off));
        }
    }

    /**
     * @brief allocate the value entries for segments (statically)
     * 
     * @param seg segment meta data
     * @param total_num_values total number of the values in this segment
     */
    void alloc_entries_to_seg(rdf_seg_meta_t& seg) {
        seg.edge_start = this->gstore->alloc_entries(seg.num_edges);
    }

    /**
     * @brief Allocate value space in given segment.
     * 
     * NOTICE: This function is not thread-safe.
     * 
     * @param num_values number of values
     * @param tid caller thread id
     * @param seg target segment
     * @return uint64_t value offset
     */
    uint64_t alloc_entries_in_seg(uint64_t num_values, int tid = 0, rdf_seg_meta_t* seg = nullptr) {
        ASSERT(seg != nullptr);
        // Init offset if first visited.
        if (seg->edge_off == 0)
            seg->edge_off = seg->edge_start;
        uint64_t orig = seg->edge_off;
        seg->edge_off += num_values;
        ASSERT(seg->edge_off <= seg->edge_start + seg->num_edges);
        return orig;
    }

    /**
     * @brief insert key to a slot in segment
     * 
     * @param key 
     * @param ptr 
     * @return uint64_t slot id
     */
    uint64_t insert_key_to_seg(ikey_t key, iptr_t ptr) {
        uint64_t bucket_id = this->gstore->bucket_local(key, &rdf_seg_meta_map[segid_t(key)]);
        uint64_t slot_id = bucket_id * RDFStore::ASSOCIATIVITY;
        uint64_t lock_id = bucket_id % RDFStore::NUM_LOCKS;
        uint64_t seg_ext_lock_id = segid_t(key).hash() % RDFStore::NUM_LOCKS;

        bool found = false;
        pthread_spin_lock(&this->gstore->bucket_locks[lock_id]);
        while (slot_id < this->gstore->num_slots) {
            // the last slot of each bucket is always reserved for pointer to indirect header
            /// TODO: add type info to slot and reuse the last slot to store key
            /// TODO: key.vid is reused to store the bucket_id of indirect header rather than ptr.off,
            ///       since the is_empty() is not robust.
            for (int i = 0; i < RDFStore::ASSOCIATIVITY - 1; i++, slot_id++) {
                if (this->gstore->slots[slot_id].key == key) {
                    logstream(LOG_ERROR) << "conflict at slot["
                                         << slot_id << "] of bucket["
                                         << bucket_id << "], "
                                         << key.to_string() << ", "
                                         << this->gstore->slots[slot_id].key.to_string()
                                         << LOG_endl;
                    ASSERT(false);
                }

                // insert to an empty slot
                if (this->gstore->slots[slot_id].key.is_empty()) {
                    this->gstore->slots[slot_id].key = key;
                    this->gstore->slots[slot_id].ptr = ptr;
                    goto done;
                }
            }

            // whether the bucket_ext (indirect-header region) is used
            if (!this->gstore->slots[slot_id].key.is_empty()) {
                // continue and jump to next bucket
                slot_id = this->gstore->slots[slot_id].key.vid * RDFStore::ASSOCIATIVITY;
                continue;
            }

            // allocate and link a new indirect header
            pthread_spin_lock(&seg_ext_locks[seg_ext_lock_id]);
            rdf_seg_meta_t& seg = rdf_seg_meta_map[segid_t(key)];
            uint64_t ext_bucket_id = seg.get_ext_bucket();
            if (ext_bucket_id == 0) {
                uint64_t nbuckets = 0;
#ifdef USE_GPU
                nbuckets = EXT_BUCKET_EXTENT_LEN(seg.num_buckets);
#else
                nbuckets = EXT_BUCKET_EXTENT_LEN;
#endif
                uint64_t start_off = this->gstore->alloc_ext_buckets(nbuckets);
                seg.add_ext_buckets(ext_bucket_extent_t(nbuckets, start_off));
                ext_bucket_id = seg.get_ext_bucket();
            }
            pthread_spin_unlock(&seg_ext_locks[seg_ext_lock_id]);
            this->gstore->slots[slot_id].key.vid = ext_bucket_id;

            // move to a new bucket_ext
            slot_id = this->gstore->slots[slot_id].key.vid * RDFStore::ASSOCIATIVITY;
            // insert to the first slot
            this->gstore->slots[slot_id].key = key;
            this->gstore->slots[slot_id].ptr = ptr;
            goto done;
        }
    done:
        pthread_spin_unlock(&this->gstore->bucket_locks[lock_id]);
        ASSERT_LT(slot_id, this->gstore->num_slots);
        return slot_id;
    }
    
    /**
     * @brief insert rdf index
     * 
     * insert {predicate index OUT, t_set*, p_set*}
     * or {predicate index IN, type index, v_set*}
     * 
     * @param pidx_map 
     * @param tidx_map 
     * @param d 
     * @param tid 
     */
    void insert_idx(const tbb_edge_hash_map &pidx_map, 
                    const tbb_edge_hash_map &tidx_map, 
                    dir_t d, int tid = 0) {
        tbb_edge_hash_map::const_accessor ca;
        rdf_seg_meta_t& segment = rdf_seg_meta_map[segid_t(1, PREDICATE_ID, d)];
        // it is possible that num_edges = 0 if loading an empty dataset
        // ASSERT(segment.num_edges > 0);

        uint64_t off = segment.edge_start;

        for (int i = 0; i < this->edge_predicates.size(); i++) {
            sid_t pid = this->edge_predicates[i];
            bool success = pidx_map.find(ca, pid);
            if (!success)
                continue;

            uint64_t sz = ca->second.size();
            ASSERT(sz <= segment.num_edges);

            logger(LOG_DEBUG, "insert_pidx[%s]: key: [%lu|%lu|%lu] sz: %lu",
                   (d == IN) ? "IN" : "OUT", 0, pid, d, sz);

            uint64_t slot_id = this->insert_key_to_seg(
                ikey_t(0, pid, d),
                iptr_t(sz, off));

            for (auto const& edge : ca->second)
                this->gstore->values[off++] = edge;

            ASSERT(off <= segment.edge_start + segment.num_edges);
        }
        // type index
        if (d == IN) {
            for (auto const& e : tidx_map) {
                sid_t pid = e.first;
                uint64_t sz = e.second.size();
                ASSERT(sz <= segment.num_edges);
                logger(LOG_DEBUG, "insert_tidx: pid: %lu, sz: %lu", pid, sz);

                uint64_t slot_id = this->insert_key_to_seg(
                    ikey_t(0, pid, IN),
                    iptr_t(sz, off));

                for (auto const& edge : e.second)
                    this->gstore->values[off++] = edge;

                ASSERT(off <= segment.edge_start + segment.num_edges);
            }
        }
#ifdef VERSATILE
        if (d == IN) {
            // all local entities, key: [0 | TYPE_ID | IN]
            insert_idx_set(v_set, off, TYPE_ID, IN);
            tbb_unordered_set().swap(v_set);
        } else {
            // all local types, key: [0 | TYPE_ID | OUT]
            insert_idx_set(t_set, off, TYPE_ID, OUT);
            tbb_unordered_set().swap(t_set);
            // all local predicates, key: [0 | PREDICATE_ID | OUT]
            insert_idx_set(p_set, off, PREDICATE_ID, OUT);
            tbb_unordered_set().swap(p_set);
        }
#endif  // VERSATILE
    }

#ifdef VERSATILE
    // predicate count(OUT/IN) for each vid
    tbb::concurrent_hash_map<sid_t, uint64_t> vp_meta[2];

    void alloc_vp_edges(dir_t d) {
        rdf_seg_meta_t& seg = rdf_seg_meta_map[segid_t(0, PREDICATE_ID, d)];
        uint64_t off = 0, sz = 0;
        for (auto iter = vp_meta[d].begin(); iter != vp_meta[d].end(); iter++) {
            sz = iter->second;
            iter->second = off;
            off += sz;
        }
        ASSERT(off == seg.num_edges);
    }

    /**
     * @brief insert VERSATILE-related data (vid's preds) into gstore
     * 
     * @param tid 
     * @param pso 
     * @param pos 
     */
    void insert_vp(int tid, const std::vector<triple_t>& pso, const std::vector<triple_t>& pos) {
        std::vector<sid_t> preds;
        uint64_t s = 0;
        auto const& out_seg = rdf_seg_meta_map[segid_t(0, PREDICATE_ID, OUT)];
        auto const& in_seg = rdf_seg_meta_map[segid_t(0, PREDICATE_ID, IN)];

        while (s < pso.size()) {
            // predicate-based key (subject + predicate)
            uint64_t e = s + 1;
            while ((e < pso.size()) && (pso[s].s == pso[e].s) && (pso[s].p == pso[e].p)) { e++; }

            preds.push_back(pso[s].p);

            // insert a vp key-value pair (OUT)
            if (e >= pso.size() || pso[s].s != pso[e].s) {
                // allocate entries
                uint64_t sz = preds.size();
                tbb::concurrent_hash_map<sid_t, uint64_t>::accessor a;
                vp_meta[OUT].find(a, pso[s].s);
                uint64_t off = out_seg.edge_start + a->second;
                a.release();

                // insert subject
                uint64_t slot_id = this->insert_key_to_seg(
                    ikey_t(pso[s].s, PREDICATE_ID, OUT),
                    iptr_t(sz, off));

                // insert predicates
                for (auto const& p : preds) {
                #if TRDF_MODE
                    this->gstore->values[off++] = edge_t(p, TIMESTAMP_MIN, TIMESTAMP_MAX);
                #else
                    this->gstore->values[off++] = edge_t(p);
                #endif
                }

                preds.clear();
            }
            s = e;
        }

        // treat type triples as index vertices
        uint64_t type_triples = 0;
        while (type_triples < pos.size() && is_tpid(pos[type_triples].o))
            type_triples++;

        // skip type triples
        s = type_triples;
        while (s < pos.size()) {
            // predicate-based key (object + predicate)
            uint64_t e = s + 1;
            while ((e < pos.size()) && (pos[s].o == pos[e].o) && (pos[s].p == pos[e].p)) { e++; }

            // add a new predicate
            preds.push_back(pos[s].p);

            // insert a vp key-value pair (IN)
            if (e >= pos.size() || pos[s].o != pos[e].o) {
                // allocate entries
                uint64_t sz = preds.size();
                tbb::concurrent_hash_map<sid_t, uint64_t>::accessor a;
                vp_meta[IN].find(a, pos[s].o);
                uint64_t off = in_seg.edge_start + a->second;
                a.release();

                // insert object
                uint64_t slot_id = this->insert_key_to_seg(
                    ikey_t(pos[s].o, PREDICATE_ID, IN),
                    iptr_t(sz, off));

                // insert predicates
                for (auto const& p : preds) {
                #if TRDF_MODE
                    this->gstore->values[off++] = edge_t(p, TIMESTAMP_MIN, TIMESTAMP_MAX);
                #else
                    this->gstore->values[off++] = edge_t(p);
                #endif
                }

                preds.clear();
            }
            s = e;
        }
    }
#endif  // VERSATILE

    /**
     * @brief Build index of each predicate's data to accelerate insert triples.
     * 
     * @param triples normal triple data
     * @return triple_map_t built index
     */
    triple_map_t init_triple_map(const std::vector<std::vector<triple_t>>& triples) {
        triple_map_t triple_map(Global::num_engines);

        #pragma omp parallel for num_threads(Global::num_engines)
        for (int tid = 0; tid < Global::num_engines; tid++) {
            const std::vector<triple_t>& local = triples[tid];
            for (size_t i = 0; i < local.size();) {
                sid_t pid = local[i].p;
                triple_map[tid].emplace(pid, i);
                // skip current predicate's data
                for (; i < local.size() && local[i].p == pid; i++) {}
            }
        }
        return triple_map;
    }

    /**
     * @brief Build index of each attr's data to accelerate insert attributes.
     * 
     * @param triples attr triple data
     * @return triple_map_t built index
     */
    triple_map_t init_triple_map(const std::vector<std::vector<triple_attr_t>>& triples) {
        triple_map_t triple_map(Global::num_engines);

        #pragma omp parallel for num_threads(Global::num_engines)
        for (int tid = 0; tid < Global::num_engines; tid++) {
            const std::vector<triple_attr_t>& local = triples[tid];
            for (size_t i = 0; i < local.size();) {
                sid_t aid = local[i].a;
                triple_map[tid].emplace(aid, i);
                // skip current attr's data
                for (; i < local.size() && local[i].a == aid; i++) {}
            }
        }
        return triple_map;
    }

    /**
     * @brief Helper function of insert_triples.
     * 
     * Insert normal predicate-based triples and return start offset of next triples.
     * 
     * @param tid caller thread id
     * @param triples triple data
     * @param seg target segment
     * @param s start offset
     * @param dir direction
     * @return size_t end offset
     */
    size_t insert_normal_triples(int tid, const std::vector<triple_t>& triples,
                                 rdf_seg_meta_t& seg, size_t s, dir_t dir) {
        ikey_t key;
        size_t e = s + 1;  // end offset
        if (dir == OUT) {  // pso triples
            // find end of triples with current (subject + predicate)
            while ((e < triples.size()) && (triples[s].s == triples[e].s) && (triples[s].p == triples[e].p)) { e++; }
            // allocate a vertex
            key = ikey_t(triples[s].s, triples[s].p, OUT);

        } else {  // pos triples
            while ((e < triples.size()) && (triples[s].o == triples[e].o) && (triples[s].p == triples[e].p)) { e++; }
            key = ikey_t(triples[s].o, triples[s].p, IN);
        }
        // alloc entries
        uint64_t off = alloc_entries_in_seg(e - s, tid, &seg);
        // insert vertex
        uint64_t slot_id = this->insert_key_to_seg(
            key,
            iptr_t(e - s, off));

        // insert values
        for (uint32_t i = s; i < e; i++) {
        #if TRDF_MODE
            this->gstore->values[off++] = (dir == OUT) ? edge_t(triples[i].o, triples[i].ts, triples[i].te)
                                                       : edge_t(triples[i].s, triples[i].ts, triples[i].te);
        #else
            this->gstore->values[off++] = (dir == OUT) ? edge_t(triples[i].o)
                                                       : edge_t(triples[i].s);
        #endif
        }

        collect_idx_info(this->gstore->slots[slot_id]);
        return e;
    }

    /**
     * @brief Insert triples beloging to the segment identified by segid to store
     * 
     * Notes: This function only insert triples belonging to normal segment
     * 
     * @param tid caller thread id
     * @param segid target segment id
     * @param triple_maps triple index
     * @param triples normal triple data
     */
    void insert_triples(int tid, segid_t segid, const triple_map_t& triple_maps,
                        const std::vector<std::vector<triple_t>>& triples) {
        auto& segment = rdf_seg_meta_map[segid];
        sid_t pid = segid.pid;
        uint64_t off = segment.edge_start;

        ASSERT(segid.index == 0);
        if (segment.num_edges == 0) {
            logger(LOG_DEBUG, "Thread(%d): abort! segment(%d|%d|%d) is empty.\n",
                   tid, segid.index, segid.pid, segid.dir);
            return;
        }

        for (int i = 0; i < Global::num_engines; i++) {
            auto& pmap = triple_maps[i];
            auto& vec = triples[i];

            // current thread's local triples has predicate data
            auto it = pmap.find(pid);
            if (it != pmap.end()) {
                size_t s = it->second;
                while (s < vec.size() && vec[s].p == pid) {
                    if (segid.dir == IN) {
                        // pos triples skip type triples
                        while (s < vec.size() && vec[s].p == pid && is_tpid(vec[s].o))
                            s++;
                    }
                    s = insert_normal_triples(tid, vec, segment, s, segid.dir);
                }
            }
        }
    }
    
    /**
     * @brief Insert attr triples beloging to the segment identified by segid to store
     * 
     * @param tid caller thread id
     * @param segid target segment id
     * @param triple_maps triple index
     * @param attr_triples attribute triple data
     */
    void insert_attr(int tid, segid_t segid, const triple_map_t& triple_maps,
                     const std::vector<std::vector<triple_attr_t>>& attr_triples) {
        auto& segment = rdf_seg_meta_map[segid];
        sid_t aid = segid.pid;
        int type = attr_type_dim_map[aid].first;
        uint64_t sz = (get_sizeof(type) - 1) / sizeof(edge_t) + 1;  // get the ceil size;

        if (segment.num_edges == 0) {
            logger(LOG_DEBUG, "Segment(%d|%d|%d) is empty.\n",
                   segid.index, segid.pid, segid.dir);
            return;
        }

        for (int i = 0; i < Global::num_engines; i++) {
            auto& pmap = triple_maps[i];
            auto& asv = attr_triples[i];
            // current thread's local triples has attr data
            auto it = pmap.find(aid);
            if (it != pmap.end()) {
                size_t s = it->second;
                while (s < asv.size() && asv[s].a == aid) {
                    // allocate entries
                    uint64_t off = alloc_entries_in_seg(sz, tid, &segment);
                    // insert subject
                    uint64_t slot_id = this->insert_key_to_seg(
                        ikey_t(asv[s].s, asv[s].a, OUT),
                        iptr_t(sz, off, type));

                    // insert values (attributes)
                    switch (type) {
                    case INT_t:
                        *reinterpret_cast<int*>(this->gstore->values + off) = boost::get<int>(asv[s].v);
                        break;
                    case FLOAT_t:
                        *reinterpret_cast<float*>(this->gstore->values + off) = boost::get<float>(asv[s].v);
                        break;
                    case DOUBLE_t:
                        *reinterpret_cast<double*>(this->gstore->values + off) = boost::get<double>(asv[s].v);
                        break;
                    default:
                        logstream(LOG_ERROR) << "Unsupported value type of attribute" << LOG_endl;
                    }
                    s++;
                }
            }
        }
    }

    /**
     * @brief init metadata for each segment
     * 
     * @param triple_pso pso triple data
     * @param triple_pos pos triple data
     * @param triple_sav attribute triple data
     */
    void init_seg_metas(const std::vector<std::vector<triple_t>>& triple_pso,
                        const std::vector<std::vector<triple_t>>& triple_pos,
                        const std::vector<std::vector<triple_attr_t>>& triple_sav) {
        /**
         * count(|pred| means number of local predicates):
         * 1. normal vertices [vid|pid|IN/OUT], key: pid, cnt_t: in&out, #item: |pred|
         * 2. vid's all types [vid|TYPE_ID(1)|OUT], key: TYPE_ID(1), cnt_t: out, #item: contained above
         * 3*. vid's all predicates [vid|PREDICATE_ID(0)|IN/OUT], key: PREDICATE_ID(0), cnt_t: in&out, #item: 1
         * 4^. attr triples [vid|pid|out], key: pid, cnt_t: out, #item: |attrpred|
         */
        std::map<sid_t, cnt_t> normal_cnt_map;

        /**
         * count(|pred| means number of local predicates):
         * 1. predicate index [0|pid|IN/OUT], key: pid, cnt_t: in&out, #item: |pred|
         * 2. type index [0|typeid|IN], key: typeid, cnt_t: in, #item: |type|
         */
        std::map<sid_t, cnt_t> index_cnt_map;

        // initialization
        for (sid_t pred : this->predicates) {
            index_cnt_map.insert(std::make_pair(pred, cnt_t()));
            normal_cnt_map.insert(std::make_pair(pred, cnt_t()));
            rdf_seg_meta_map.insert(std::make_pair(segid_t(0, pred, IN), rdf_seg_meta_t()));
            rdf_seg_meta_map.insert(std::make_pair(segid_t(0, pred, OUT), rdf_seg_meta_t()));
        }

        for (sid_t attr : this->attributes) {
            index_cnt_map.insert(std::make_pair(attr, cnt_t()));
            normal_cnt_map.insert(std::make_pair(attr, cnt_t()));
            rdf_seg_meta_map.insert(std::make_pair(segid_t(0, attr, IN), rdf_seg_meta_t()));
            rdf_seg_meta_map.insert(std::make_pair(segid_t(0, attr, OUT), rdf_seg_meta_t()));
        }

        // init index segment
        rdf_seg_meta_map.insert(std::make_pair(segid_t(1, PREDICATE_ID, IN), rdf_seg_meta_t()));
        rdf_seg_meta_map.insert(std::make_pair(segid_t(1, PREDICATE_ID, OUT), rdf_seg_meta_t()));

        #pragma omp parallel for num_threads(Global::num_engines)
        for (int tid = 0; tid < Global::num_engines; tid++) {
            const std::vector<triple_t>& pso = triple_pso[tid];
            const std::vector<triple_t>& pos = triple_pos[tid];
            const std::vector<triple_attr_t>& sav = triple_sav[tid];

            uint64_t s = 0;
            while (s < pso.size()) {
                uint64_t e = s + 1;

                while ((e < pso.size()) && (pso[s].s == pso[e].s) && (pso[s].p == pso[e].p)) {
                    // count #edge of type-idx (IN)
                    if (pso[e].p == TYPE_ID && is_tpid(pso[e].o)) {
#ifdef VERSATILE
                        t_set.insert(pso[e].o);
#endif
                        index_cnt_map[pso[e].o].in++;
                    }
                    e++;
                }

#ifdef VERSATILE
                v_set.insert(pso[s].s);
                p_set.insert(pso[s].p);

                // vid's preds count
                tbb::concurrent_hash_map<sid_t, uint64_t>::accessor oa;
                if (vp_meta[OUT].insert(oa, pso[s].s))
                    oa->second = 1;
                else
                    oa->second += 1;
                oa.release();
                // count vid's all predicates OUT (number of value in the segment)
                normal_cnt_map[PREDICATE_ID].out++;
#endif

                // count #edge of predicate
                normal_cnt_map[pso[s].p].out += (e - s);

                // count #edge of predicate-idx
                index_cnt_map[pso[s].p].in++;

                // count #edge of type-idx
                if (pso[s].p == TYPE_ID && is_tpid(pso[s].o)) {
#ifdef VERSATILE
                    t_set.insert(pso[s].o);
#endif
                    index_cnt_map[pso[s].o].in++;
                }
                s = e;
            }

            // skip type triples
            uint64_t type_triples = 0;
            triple_t tp;
            while (type_triples < pos.size() && is_tpid(pos[type_triples].o)) {
                type_triples++;
            }

            s = type_triples;
            while (s < pos.size()) {
                // predicate-based key (object + predicate)
                uint64_t e = s + 1;
                while ((e < pos.size()) && (pos[s].o == pos[e].o) && (pos[s].p == pos[e].p)) {
                    e++;
                }
#ifdef VERSATILE
                v_set.insert(pos[s].o);
                p_set.insert(pos[s].p);
                tbb::concurrent_hash_map<sid_t, uint64_t>::accessor ia;
                if (vp_meta[IN].insert(ia, pos[s].o))
                    ia->second = 1;
                else
                    ia->second += 1;
                ia.release();
                // count vid's all predicates IN (number of value in the segment)
                normal_cnt_map[PREDICATE_ID].in++;
#endif

                // count #edge of predicate
                normal_cnt_map[pos[s].p].in += (e - s);
                index_cnt_map[pos[s].p].out++;
                s = e;
            }

            s = 0;
            while (s < sav.size()) {
                uint64_t e = s + 1;

                while ((e < sav.size()) && (sav[s].s == sav[e].s) && (sav[s].a == sav[e].a)) {
                    e++;
                }

                // count #edge of predicate
                normal_cnt_map[sav[s].a].out += (e - s);

                s = e;
            }
        }

        // count the total number of keys
        uint64_t total_num_keys = 0;
#ifdef VERSATILE
        logger(LOG_DEBUG, "pid: %d: normal: #IN: %lu, #OUT: %lu", PREDICATE_ID,
               normal_cnt_map[PREDICATE_ID].in.load(), normal_cnt_map[PREDICATE_ID].out.load());
        total_num_keys += vp_meta[OUT].size() + vp_meta[IN].size();
        total_num_keys += 3; // vset, tset, pset
#endif
        for (sid_t pred : this->predicates) {
            if (pred == PREDICATE_ID) continue;
            logger(LOG_DEBUG, "pid: %d: normal: #IN: %lu, #OUT: %lu; index: #ALL: %lu, #IN: %lu, #OUT: %lu",
                   pred, normal_cnt_map[pred].in.load(), normal_cnt_map[pred].out.load(),
                   (index_cnt_map[pred].in.load() + index_cnt_map[pred].out.load()),
                   index_cnt_map[pred].in.load(), index_cnt_map[pred].out.load());
            /**
             * this if-sentence checks if the pred is a typeid
             * if the sentence is true, it means i is a normal predicate
             * index_cnt_map[pred] stores #edges of pred's predicate index (IN, OUT)
             * whose sum equals to #keys of the predicate
             */
            if (normal_cnt_map[pred].in.load() + normal_cnt_map[pred].out.load() > 0) {
                logstream(LOG_DEBUG) << "edge_predicates:" << pred << LOG_endl;
                this->edge_predicates.push_back(pred);
                total_num_keys += (index_cnt_map[pred].in.load() + index_cnt_map[pred].out.load());
            } else if (index_cnt_map[pred].in.load() > 0) {
                logstream(LOG_DEBUG) << "type_predicate:" << pred << LOG_endl;
                this->type_predicates.push_back(pred);
            }
        }

        for (sid_t attr : this->attributes) {
            total_num_keys += normal_cnt_map[attr].out.load();
        }

        /**
         * #predicate index = #edge_predicates * 2
         * #type index = #type_predicates
         */
        total_num_keys += this->edge_predicates.size() * 2 + this->type_predicates.size();

        // allocate buckets and entries to segments
        rdf_seg_meta_t& idx_out_seg = rdf_seg_meta_map[segid_t(1, PREDICATE_ID, OUT)];
        rdf_seg_meta_t& idx_in_seg = rdf_seg_meta_map[segid_t(1, PREDICATE_ID, IN)];

#ifdef VERSATILE
        // vid's all predicates OUT
        rdf_seg_meta_t& pred_out_seg = rdf_seg_meta_map[segid_t(0, PREDICATE_ID, OUT)];
        pred_out_seg.num_edges = normal_cnt_map[PREDICATE_ID].out.load();
        alloc_entries_to_seg(pred_out_seg);
        pred_out_seg.num_keys = vp_meta[OUT].size();
        alloc_buckets_to_seg(pred_out_seg, segid_t(0, PREDICATE_ID, OUT), total_num_keys);
        // vid's all predicates IN
        rdf_seg_meta_t& pred_in_seg = rdf_seg_meta_map[segid_t(0, PREDICATE_ID, IN)];
        pred_in_seg.num_edges = normal_cnt_map[PREDICATE_ID].in.load();
        alloc_entries_to_seg(pred_in_seg);
        pred_in_seg.num_keys = vp_meta[IN].size();
        alloc_buckets_to_seg(pred_in_seg, segid_t(0, PREDICATE_ID, IN), total_num_keys);

        #pragma omp parallel for num_threads(2)
        for (int tid = 0; tid < 2; tid++) {
            alloc_vp_edges((dir_t) tid);
        }

        // all local entities ([0|TYPE_ID|IN])
        idx_in_seg.num_edges += v_set.size();
        idx_in_seg.num_keys += 1;
        // all local types ([0|TYPEID|OUT])
        idx_out_seg.num_edges += t_set.size();
        idx_out_seg.num_keys += 1;
        // all local predicates ([0|PREDICATE_ID|OUT])
        idx_out_seg.num_edges += p_set.size();
        idx_out_seg.num_keys += 1;

        logstream(LOG_DEBUG) << "s_set: " << pred_out_seg.num_keys << ", o_set: " << pred_in_seg.num_keys
                             << ", v_set: " << v_set.size() << ", p_set: " << p_set.size() << ", t_set: " << t_set.size() << LOG_endl;
#endif

        for (sid_t pred : this->predicates) {
            if (pred == PREDICATE_ID) continue;
            rdf_seg_meta_t& out_seg = rdf_seg_meta_map[segid_t(0, pred, OUT)];
            rdf_seg_meta_t& in_seg = rdf_seg_meta_map[segid_t(0, pred, IN)];

            out_seg.num_edges = normal_cnt_map[pred].out.load();
            in_seg.num_edges = normal_cnt_map[pred].in.load();

            idx_out_seg.num_edges += index_cnt_map[pred].out.load();
            idx_in_seg.num_edges += index_cnt_map[pred].in.load();

            // normal pred segment
            uint64_t normal_nkeys[2] = {index_cnt_map[pred].out, index_cnt_map[pred].in};
            out_seg.num_keys = (out_seg.num_edges == 0) ? 0 : normal_nkeys[OUT];
            in_seg.num_keys = (in_seg.num_edges == 0) ? 0 : normal_nkeys[IN];

            // allocate space for edges in entry-region
            alloc_entries_to_seg(out_seg);
            alloc_entries_to_seg(in_seg);

            alloc_buckets_to_seg(out_seg, segid_t(0, pred, OUT), total_num_keys);
            alloc_buckets_to_seg(in_seg, segid_t(0, pred, IN), total_num_keys);

            logger(LOG_DEBUG,
                   "Predicate[%d]: normal: OUT[#keys: %lu, #buckets: %lu, #edges: %lu] "
                   "IN[#keys: %lu, #buckets: %lu, #edges: %lu];",
                   pred, out_seg.num_keys, out_seg.num_buckets, out_seg.num_edges,
                   in_seg.num_keys, in_seg.num_buckets, in_seg.num_edges);
        }

        for (sid_t attr : this->attributes) {
            rdf_seg_meta_t& out_seg = rdf_seg_meta_map[segid_t(0, attr, OUT)];
            rdf_seg_meta_t& in_seg = rdf_seg_meta_map[segid_t(0, attr, IN)];

            out_seg.num_edges = normal_cnt_map[attr].out.load();
            in_seg.num_edges = normal_cnt_map[attr].in.load();

            idx_out_seg.num_edges += index_cnt_map[attr].out.load();
            idx_in_seg.num_edges += index_cnt_map[attr].in.load();

            // attribute segment
            out_seg.num_keys = out_seg.num_edges;
            in_seg.num_keys = 0;
            // calculate the number of edge_t needed to store 1 value
            uint64_t sz = (get_sizeof(attr_type_dim_map[attr].first) - 1) / sizeof(edge_t) + 1;  // get the ceil size;
            out_seg.num_edges = out_seg.num_edges * sz;

            // allocate space for edges in entry-region
            alloc_entries_to_seg(out_seg);
            alloc_entries_to_seg(in_seg);

            alloc_buckets_to_seg(out_seg, segid_t(0, attr, OUT), total_num_keys);
            alloc_buckets_to_seg(in_seg, segid_t(0, attr, IN), total_num_keys);

            logger(LOG_DEBUG,
                   "Predicate[%d]: normal: OUT[#keys: %lu, #buckets: %lu, #edges: %lu] "
                   "IN[#keys: %lu, #buckets: %lu, #edges: %lu];",
                   attr, out_seg.num_keys, out_seg.num_buckets, out_seg.num_edges,
                   in_seg.num_keys, in_seg.num_buckets, in_seg.num_edges);
        }

        alloc_entries_to_seg(idx_out_seg);
        idx_out_seg.num_keys += this->edge_predicates.size() - 1;
        alloc_buckets_to_seg(idx_out_seg, segid_t(1, PREDICATE_ID, OUT), total_num_keys);

        alloc_entries_to_seg(idx_in_seg);
        idx_in_seg.num_keys += this->edge_predicates.size() + this->type_predicates.size() - 1;
        alloc_buckets_to_seg(idx_in_seg, segid_t(1, PREDICATE_ID, IN), total_num_keys);

        logger(LOG_DEBUG,
               "index: OUT[#keys: %lu, #buckets: %lu, #values: %lu], "
               "IN[#keys: %lu, #buckets: %lu, #values: %lu], bucket_off: %lu\n",
               idx_out_seg.num_keys, idx_out_seg.num_buckets, idx_out_seg.num_edges,
               idx_in_seg.num_keys, idx_in_seg.num_buckets, idx_in_seg.num_edges, main_hdr_off);
    }

    /**
     * @brief collect index data for a slot
     * 
     * @param slot 
     */
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
                // type-index (IN) -> vid_list
                for (uint64_t e = 0; e < sz; e++) {
                    tbb_edge_hash_map::accessor a;
                    tidx_map.insert(a, this->gstore->values[off + e].val);
                #if TRDF_MODE
                    a->second.push_back(edge_t(vid, this->gstore->values[off + e].ts, this->gstore->values[off + e].te));
                #else
                    a->second.push_back(edge_t(vid));
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

#ifdef VERSATILE
    /**
     * @brief insert {v/t/p}_set into gstore
     * 
     * @param set {v/t/p}_set
     * @param off value offset in segment
     * @param pid predicate
     * @param d direction
     */
    void insert_idx_set(const tbb_unordered_set& set, uint64_t& off, sid_t pid, dir_t d) {
        uint64_t sz = set.size();

        uint64_t slot_id = this->insert_key_to_seg(
            ikey_t(0, pid, d),
            iptr_t(sz, off));

        for (auto const& value : set) {
        #if TRDF_MODE
            edge_t edge(value, TIMESTAMP_MIN, TIMESTAMP_MAX);
        #else
            edge_t edge(value);
        #endif
            this->gstore->values[off++] = edge;
        }
    }
#endif  // VERSATILE

    /**
     * @brief send segment meta data to other servers
     * 
     * @param tcp_ad 
     */
    void send_seg_meta(TCP_Adaptor* tcp_ad) {
        std::stringstream ss;
        std::string str;
        boost::archive::binary_oarchive oa(ss);
        SyncSegmentMetaMsg msg(rdf_seg_meta_map);

        msg.sender_sid = sid;
        oa << msg;

        // send pred_metas to other servers
        for (int i = 0; i < Global::num_servers; ++i) {
            if (i == sid)
                continue;
            tcp_ad->send(i, 0, ss.str());
            logstream(LOG_INFO) << "#" << sid << " sends segment metadata to server " << i << LOG_endl;
        }
    }

    /**
     * @brief recv segment meta data from other servers
     * 
     * @param tcp_ad 
     */
    void recv_seg_meta(TCP_Adaptor* tcp_ad) {
        std::string str;
        // receive Global::num_servers - 1 messages
        for (int i = 0; i < Global::num_servers; ++i) {
            if (i == sid)
                continue;
            std::stringstream ss;
            str = tcp_ad->recv(0);
            ss << str;
            boost::archive::binary_iarchive ia(ss);
            SyncSegmentMetaMsg msg;
            ia >> msg;

            if (shared_rdf_seg_meta_map.find(msg.sender_sid) != shared_rdf_seg_meta_map.end())
                shared_rdf_seg_meta_map[msg.sender_sid] = msg.data;
            else
                shared_rdf_seg_meta_map.insert(std::make_pair(msg.sender_sid, msg.data));
            logstream(LOG_INFO) << "#" << sid
                                << " receives segment metadata from server " << msg.sender_sid
                                << LOG_endl;
        }
    }

    /**
     * @brief re-adjust attributes of segments (for GPU)
     */
    void finalize_seg_metas() {
        uint64_t nbuckets_per_blk = MiB2B(Global::gpu_key_blk_size_mb) / (sizeof(RDFStore::slot_t) * RDFStore::ASSOCIATIVITY);
        uint64_t nentries_per_blk = MiB2B(Global::gpu_value_blk_size_mb) / sizeof(edge_t);

        // set the number of cache blocks needed by each segment
        for (auto& e : rdf_seg_meta_map) {
            e.second.num_key_blks = ceil((static_cast<double>(e.second.get_total_num_buckets())) / nbuckets_per_blk);
            e.second.num_value_blks = ceil((static_cast<double>(e.second.num_edges)) / nentries_per_blk);
        }
    }

    /**
     * @brief release memory after gstore init
     */
    void finalize_init() {
        tbb_edge_hash_map().swap(pidx_in_map);
        tbb_edge_hash_map().swap(pidx_out_map);
        tbb_edge_hash_map().swap(tidx_map);

#ifdef VERSATILE
        for (int i = 0; i < 2; i++) {
            tbb::concurrent_hash_map<sid_t, uint64_t>().swap(vp_meta[i]);
        }
#endif  // VERSATILE
    }

public:
    SegmentRDFGraph(int sid, KVMem kv_mem)
        : DGraph(sid, kv_mem) {
        this->gstore = std::make_shared<StaticKVStore<ikey_t, iptr_t, edge_t>>(sid, kv_mem);
        for (int i = 0; i < RDFStore::NUM_LOCKS; i++) {
            pthread_spin_init(&seg_ext_locks[i], 0);
        }
    }

    ~SegmentRDFGraph() {}

    edge_t* get_triples(int tid, sid_t vid, sid_t pid, dir_t d, uint64_t& sz) override {
        rdf_seg_meta_t* seg;
        int dst_sid = PARTITION(vid);
        if (dst_sid == sid) {
            seg = &rdf_seg_meta_map[segid_t(ikey_t(vid, pid, d))];
        } else {
            seg = &shared_rdf_seg_meta_map[dst_sid][segid_t(ikey_t(vid, pid, d))];
        }
        return gstore->get_values(tid, dst_sid, ikey_t(vid, pid, d), sz, seg);
    }

    edge_t* get_index(int tid, sid_t pid, dir_t d, uint64_t& sz) override {
        // index vertex should be 0 and always local
        rdf_seg_meta_t* seg = &rdf_seg_meta_map[segid_t(ikey_t(0, pid, d))];
        return gstore->get_values(tid, this->sid, ikey_t(0, pid, d), sz, seg);
    }

    // return attribute value (has_value == true)
    attr_t get_attr(int tid, sid_t vid, sid_t pid, dir_t d, bool& has_value) override {
        uint64_t sz = 0;
        attr_t r;

        rdf_seg_meta_t* seg;
        int dst_sid = PARTITION(vid);
        if (dst_sid == sid) {
            seg = &rdf_seg_meta_map[segid_t(ikey_t(vid, pid, d))];
        } else {
            seg = &shared_rdf_seg_meta_map[dst_sid][segid_t(ikey_t(vid, pid, d))];
        }

        // get the pointer of edge
        data_type type = this->get_attribute_type(pid);
        edge_t* edge_ptr = gstore->get_values(tid, PARTITION(vid), ikey_t(vid, pid, d), sz, seg);
        if (edge_ptr == nullptr) {
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

    inline const std::map<segid_t, rdf_seg_meta_t>& get_rdf_seg_metas() {
        return rdf_seg_meta_map;
    }

    void sync_metadata() {
        extern TCP_Adaptor* con_adaptor;
        send_seg_meta(con_adaptor);
        recv_seg_meta(con_adaptor);
    }

    void init_gstore(std::vector<std::vector<triple_t>>& triple_pso,
                     std::vector<std::vector<triple_t>>& triple_pos,
                     std::vector<std::vector<triple_attr_t>>& triple_sav) override {
        this->num_segments = get_num_normal_preds() * PREDICATE_NSEGS + INDEX_NSEGS + this->get_num_attr_preds();

        uint64_t start, end;
        start = timer::get_usec();
        init_seg_metas(triple_pso, triple_pos, triple_sav);
        end = timer::get_usec();
        logstream(LOG_INFO) << "[SegmentRDFGraph] #" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for initializing predicate segment statistics." << LOG_endl;

#ifdef VERSATILE
        start = timer::get_usec();
#pragma omp parallel for num_threads(Global::num_engines)
        for (int tid = 0; tid < Global::num_engines; tid++) {
            insert_vp(tid, triple_pso[tid], triple_pos[tid]);
            // Re-sort triples array by pso to accelarate normal triples insert.
            std::sort(triple_pso[tid].begin(), triple_pso[tid].end(), triple_sort_by_pso());
            std::sort(triple_pos[tid].begin(), triple_pos[tid].end(), triple_sort_by_pos());
        }
        end = timer::get_usec();
        logstream(LOG_INFO) << "[SegmentRDFGraph] #" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting vid's predicates." << LOG_endl;
#endif  // VERSATILE

        start = timer::get_usec();
        // insert normal triples
        auto out_triple_map = init_triple_map(triple_pso);
        auto in_triple_map = init_triple_map(triple_pos);

        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < this->edge_predicates.size(); i++) {
            int localtid = omp_get_thread_num();
            sid_t pid = this->edge_predicates[i];
            insert_triples(localtid, segid_t(0, pid, OUT), out_triple_map, triple_pso);
            insert_triples(localtid, segid_t(0, pid, IN), in_triple_map, triple_pos);
        }

        #pragma omp parallel for num_threads(Global::num_engines)
        for (int t = 0; t < Global::num_engines; t++) {
            // release memory
            std::vector<triple_t>().swap(triple_pso[t]);
            std::vector<triple_t>().swap(triple_pos[t]);
        }

        end = timer::get_usec();
        logstream(LOG_INFO) << "[SegmentRDFGraph] #" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting normal triples as segments into gstore" << LOG_endl;

        start = timer::get_usec();
        auto attr_map = init_triple_map(triple_sav);
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < this->attributes.size(); i++) {
            int localtid = omp_get_thread_num();
            insert_attr(localtid, segid_t(0, this->attributes[i], OUT), attr_map, triple_sav);
        }

        #pragma omp parallel for num_threads(Global::num_engines)
        for (int t = 0; t < Global::num_engines; t++) {
            // release memory
            std::vector<triple_attr_t>().swap(triple_sav[t]);
        }

        end = timer::get_usec();
        logstream(LOG_INFO) << "[SegmentRDFGraph] #" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting attr triples as segments into gstore" << LOG_endl;

        start = timer::get_usec();
        // insert type-index edges in parallel
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < 2; i++) {
            if (i == 0)
                insert_idx(pidx_in_map, tidx_map, IN);
            else
                insert_idx(pidx_out_map, tidx_map, OUT);
        }

        end = timer::get_usec();
        logstream(LOG_INFO) << "[SegmentRDFGraph] #" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting index triples as segments into gstore" << LOG_endl;

        finalize_seg_metas();
        finalize_init();

        // synchronize segment metadata among servers
        sync_metadata();
    }

    virtual void print_graph_stat() {
        gstore->print_mem_usage();

#ifdef VERSATILE
        /// (*3)  key = [  0 |      TYPE_ID |     IN]  value = [vid0, vid1, ..]  i.e., all local objects/subjects
        /// (*4)  key = [  0 |      TYPE_ID |    OUT]  value = [pid0, pid1, ..]  i.e., all local types
        /// (*5)  key = [  0 | PREDICATE_ID |    OUT]  value = [pid0, pid1, ..]  i.e., all local predicates
        uint64_t sz = 0;

        gstore->get_values(0, this->sid, ikey_t(0, TYPE_ID, IN), sz,
                           &rdf_seg_meta_map[segid_t(ikey_t(0, TYPE_ID, IN))]);
        logstream(LOG_INFO) << "[SegmentRDF] #vertices: " << sz << LOG_endl;

        gstore->get_values(0, this->sid, ikey_t(0, TYPE_ID, OUT), sz,
                           &rdf_seg_meta_map[segid_t(ikey_t(0, TYPE_ID, OUT))]);
        logstream(LOG_INFO) << "[SegmentRDF] #types: " << sz << LOG_endl;

        gstore->get_values(0, this->sid, ikey_t(0, PREDICATE_ID, OUT), sz,
                           &rdf_seg_meta_map[segid_t(ikey_t(0, PREDICATE_ID, OUT))]);
        logstream(LOG_INFO) << "[SegmentRDF] #predicates: " << sz << " (not including types)" << LOG_endl;
#endif  // end of VERSATILE

        logstream(LOG_INFO) << "[SegmentRDF] #predicates: " << this->predicates.size() << LOG_endl;
        logstream(LOG_INFO) << "[SegmentRDF] #edge_predicates: " << this->edge_predicates.size() << LOG_endl;
        logstream(LOG_INFO) << "[SegmentRDF] #type_predicates: " << this->type_predicates.size() << LOG_endl;
        logstream(LOG_INFO) << "[SegmentRDF] #attributes: " << this->attributes.size() << LOG_endl;
    }
};

}  // namespace wukong
