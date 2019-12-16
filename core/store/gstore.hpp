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

#include <stdint.h>
#include <vector>
#include <queue>
#include <iostream>
#include <pthread.h>
#include <boost/unordered_set.hpp>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_unordered_map.h>
#include <unordered_set>
#include <unordered_map>
#include <atomic>

#include "global.hpp"
#include "rdma.hpp"
#include "type.hpp"

#include "store/vertex.hpp"
#include "store/meta.hpp"
#include "store/cache.hpp"
#include "comm/tcp_adaptor.hpp"

// utils
#include "math.hpp"
#include "timer.hpp"
#include "unit.hpp"
#include "variant.hpp"

using namespace std;

/**
 * Map the Graph model (e.g., vertex, edge, index) to KVS model (e.g., key, value)
 * Graph store adopts clustring chaining key/value store (see paper: DrTM SOSP'15)
 *
 *  encoding rules of GStore
 *  subject/object (vid) >= 2^NBITS_IDX, 2^NBITS_IDX > predicate/type (p/tid) >= 2^1,
 *  TYPE_ID = 1, PREDICATE_ID = 0, OUT = 1, IN = 0
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
/**
 * Segment-based GStore
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
 *    v_set*                [0|TYPE_ID|IN]              [0|PREDICATE_ID|IN]     1
 * 5*. vid's all predicates [vid|PREDICATE_ID|IN/OUT]   [0|PREDICATE_ID|IN/OUT] 2
 * 6^. attr segments        [vid|pid|OUT]               [0|pid|OUT]             #attr pred
 */
class GStore {
    friend class Stats;
    friend class GChecker;

protected:
    struct cnt_t {
        cnt_t() {
            in = 0ul;
            out = 0ul;
        }

        cnt_t(const cnt_t &cnt) {
            in = cnt.in.load();
            out = cnt.out.load();
        }

        atomic<uint64_t> in, out;
    };

    typedef tbb::concurrent_unordered_set<sid_t> tbb_unordered_set;
    typedef tbb::concurrent_hash_map<sid_t, vector<sid_t>> tbb_hash_map;
    typedef tbb::concurrent_hash_map<ikey_t, vector<triple_t>, ikey_Hasher> tbb_triple_hash_map;
    typedef tbb::concurrent_hash_map<ikey_t, vector<triple_attr_t>, ikey_Hasher> tbb_triple_attr_hash_map;

    static const int NUM_LOCKS = 1024;

    /**
     * automatically alloc buckets to segments, which will use all header region
     * you should turn it off if:
     * 1. you want to dynamically insert new predicates
     * 2. you don't want to use up all header region
     */
    bool global_auto_bkt_alloc = true;

    int sid;

    Mem *mem;

    uint64_t num_slots;       // 1 bucket = ASSOCIATIVITY slots
    uint64_t num_buckets;     // main-header region (static)
    uint64_t num_buckets_ext; // indirect-header region (dynamical)
    uint64_t last_ext;
    pthread_spinlock_t bucket_locks[NUM_LOCKS]; // lock virtualization (see paper: vLokc CGO'13)
    pthread_spinlock_t bucket_ext_lock;
    pthread_spinlock_t seg_ext_locks[NUM_LOCKS]; // used for ext blocks allocation and ext bucket acquire

    uint64_t num_entries;     // entry region (dynamical)

    tbb_hash_map pidx_in_map;  // predicate-index (IN)
    tbb_hash_map pidx_out_map; // predicate-index (OUT)
    tbb_hash_map tidx_map;     // type-index

    RDMA_Cache rdma_cache;

    // multiple engines will access shared_rdf_seg_meta_map
    // key: server id, value: segment metadata of the server
    tbb::concurrent_unordered_map <int, map<segid_t, rdf_seg_meta_t> > shared_rdf_seg_meta_map;
    std::map<segid_t, rdf_seg_meta_t> rdf_seg_meta_map;

    // id of all local normal preds, free after gstore init
    vector<sid_t> all_local_preds;
    // id of all local attr preds, free after gstore init
    tbb_unordered_set attr_set;
    // key: id of attr preds, value: type(SID_t, INT_t, FLOAT_t, DOUBLE_t), free after gstore init
    tbb::concurrent_unordered_map<sid_t, int> attr_type_map;

    //used to alloc buckets
    uint64_t main_hdr_off = 0;

    // number of segments
    uint64_t num_segments;
    /**
     * minimum num of buckets per segment
     * usage: allocate buckets to empty segments (num_keys = 0)
     */
    uint64_t min_buckets_per_seg = 1;

#ifdef VERSATILE
    /**
     * These sets and maps will be freed after being inserted into edges
     * description                     key
     * 1*. v_set, all local entities   [0|TYPE_ID|IN]
     * 2*. t_set, all local types      [0|TYPE_ID|OUT]
     * 3*. p_set, all local predicates [0|PREDICATE_ID|OUT]
     * 4*. out_preds, vid's OUT preds  [vid|PREDICATE_ID|OUT]
     * 5*. in_preds, vid's IN preds    [vid|PREDICATE_ID|IN]
     */
    tbb_unordered_set v_set;
    tbb_unordered_set t_set;
    tbb_unordered_set p_set;
    tbb::concurrent_hash_map<sid_t, uint64_t> vp_meta[2];

    // insert VERSATILE-related data into gstore
    virtual void insert_vp(int tid, const vector<triple_t> &pso, const vector<triple_t> &pos) = 0;

    virtual void alloc_vp_edges(dir_t d) = 0;
#endif // VERSATILE

    // allocate space to store edges of given size. Return offset of allocated space.
    virtual uint64_t alloc_edges(uint64_t n, int tid, rdf_seg_meta_t *seg) = 0;

    virtual uint64_t alloc_edges_to_seg(uint64_t num_edges) = 0;

    virtual void insert_idx(const tbb_hash_map &pidx_map, const tbb_hash_map &tidx_map,
                            dir_t d, int tid = 0) = 0;

    // check the validation of given edges according to given vertex.
    virtual bool edge_is_valid(vertex_t &v, edge_t *edge_ptr) = 0;

    // get the capacity of edges
    virtual uint64_t get_edge_sz(const vertex_t &v) = 0;

    // get bucket_id according to key
    uint64_t bucket_local(ikey_t key) {
        uint64_t bucket_id;
        auto &seg = rdf_seg_meta_map[segid_t(key)];
        ASSERT(seg.num_buckets > 0);
        bucket_id = seg.bucket_start + key.hash() % seg.num_buckets;
        return bucket_id;
    }

    uint64_t bucket_remote(ikey_t key, int dst_sid) {
        uint64_t bucket_id;
        auto &remote_meta_map = shared_rdf_seg_meta_map[dst_sid];
        auto &seg = remote_meta_map[segid_t(key)];
        ASSERT(seg.num_buckets > 0);
        bucket_id = seg.bucket_start + key.hash() % seg.num_buckets;
        return bucket_id;
    }

    // Get edges of given vertex from dst_sid by RDMA read.
    edge_t *rdma_get_edges(int tid, int dst_sid, vertex_t &v) {
        ASSERT(Global::use_rdma);

        char *buf = mem->buffer(tid);
        uint64_t r_off = num_slots * sizeof(vertex_t) + v.ptr.off * sizeof(edge_t);
        // the size of edges
        uint64_t r_sz = get_edge_sz(v);
        uint64_t buf_sz = mem->buffer_size();
        ASSERT(r_sz < buf_sz); // enough space to host the edges

        RDMA &rdma = RDMA::get_rdma();
        rdma.dev->RdmaRead(tid, dst_sid, buf, r_sz, r_off);
        return (edge_t *)buf;
    }

    // Attention: not thread safe. The safety is guarenteed by caller
    bool get_slot_id(ikey_t key, uint64_t &res) {
        uint64_t bucket_id = bucket_local(key);
        while (true) {
            for (int i = 0; i < ASSOCIATIVITY; i++) {
                uint64_t slot_id = bucket_id * ASSOCIATIVITY + i;
                if (i < ASSOCIATIVITY - 1) {
                    //data part
                    if (vertices[slot_id].key == key) {
                        //we found it
                        res = slot_id;
                        return true;
                    }
                } else {
                    if (vertices[slot_id].key.is_empty())
                        return false; // not found

                    bucket_id = vertices[slot_id].key.vid; // move to next bucket
                    break; // break for-loop
                }
            }
        }
    }

    // Get remote vertex of given key. This func will fail if RDMA is disabled.
    vertex_t get_vertex_remote(int tid, ikey_t key) {
        int dst_sid = wukong::math::hash_mod(key.vid, Global::num_servers);
        uint64_t bucket_id = bucket_remote(key, dst_sid);
        vertex_t vert;

        // FIXME: wukong doesn't support to directly get remote vertex/edge without RDMA
        ASSERT(Global::use_rdma);

        // check cache
        if (rdma_cache.lookup(key, vert))
            return vert;

        // get vertex by RDMA
        char *buf = mem->buffer(tid);
        uint64_t buf_sz = mem->buffer_size();
        while (true) {
            uint64_t off = bucket_id * ASSOCIATIVITY * sizeof(vertex_t);
            uint64_t sz = ASSOCIATIVITY * sizeof(vertex_t);
            ASSERT(sz < buf_sz); // enough space to host the vertices

            RDMA &rdma = RDMA::get_rdma();
            rdma.dev->RdmaRead(tid, dst_sid, buf, sz, off);
            vertex_t *verts = (vertex_t *)buf;
            for (int i = 0; i < ASSOCIATIVITY; i++) {
                if (i < ASSOCIATIVITY - 1) {
                    if (verts[i].key == key) {
                        rdma_cache.insert(verts[i]);
                        return verts[i]; // found
                    }
                } else {
                    if (verts[i].key.is_empty())
                        return vertex_t(); // not found

                    bucket_id = verts[i].key.vid; // move to next bucket
                    break; // break for-loop
                }
            }
        }
    }

    // Get local vertex of given key.
    vertex_t get_vertex_local(int tid, ikey_t key) {
        uint64_t bucket_id = bucket_local(key);
        while (true) {
            for (int i = 0; i < ASSOCIATIVITY; i++) {
                uint64_t slot_id = bucket_id * ASSOCIATIVITY + i;
                if (i < ASSOCIATIVITY - 1) {
                    //data part
                    if (vertices[slot_id].key == key) {
                        //we found it
                        return vertices[slot_id];
                    }
                } else {
                    if (vertices[slot_id].key.is_empty())
                        return vertex_t(); // not found

                    bucket_id = vertices[slot_id].key.vid; // move to next bucket
                    break; // break for-loop
                }
            }
        }
    }

    // Get remote edges according to given vid, pid, d.
    // @sz: size of return edges
    edge_t *get_edges_remote(int tid, sid_t vid, sid_t pid, dir_t d, uint64_t &sz,
                             int &type = *(int *)NULL) {
        ikey_t key = ikey_t(vid, pid, d);
        vertex_t v = get_vertex_remote(tid, key);

        if (v.key.is_empty()) {
            sz = 0;
            return NULL; // not found
        }

        // remote edges
        int dst_sid = wukong::math::hash_mod(vid, Global::num_servers);
        edge_t *edge_ptr = rdma_get_edges(tid, dst_sid, v);
        while (!edge_is_valid(v, edge_ptr)) { // check cache validation
            // invalidate cache and try again
            rdma_cache.invalidate(key);
            v = get_vertex_remote(tid, key);
            edge_ptr = rdma_get_edges(tid, dst_sid, v);
        }

        sz = v.ptr.size;
        if (&type != NULL)
            type = v.ptr.type;
        return edge_ptr;
    }

    // Get local edges according to given vid, pid, d.
    // @sz: size of return edges
    edge_t *get_edges_local(int tid, sid_t vid, sid_t pid, dir_t d, uint64_t &sz,
                            int &type = *(int *)NULL) {
        ikey_t key = ikey_t(vid, pid, d);
        vertex_t v = get_vertex_local(tid, key);

        if (v.key.is_empty()) {
            sz = 0;
            return NULL; // not found
        }

        // local edges
        edge_t *edge_ptr = &(edges[v.ptr.off]);

        sz = v.ptr.size;
        if (&type != NULL)
            type = v.ptr.type;
        return edge_ptr;
    }

    // Allocate extended buckets
    // @n: number of extended buckets to allocate
    // @return: start offset of allocated extended buckets
    uint64_t alloc_ext_buckets(uint64_t n) {
        uint64_t orig;
        pthread_spin_lock(&bucket_ext_lock);
        orig = last_ext;
        last_ext += n;
        if (last_ext >= num_buckets_ext) {
            logstream(LOG_ERROR) << "out of indirect-header region." << LOG_endl;
            ASSERT(last_ext < num_buckets_ext);
        }
        pthread_spin_unlock(&bucket_ext_lock);
        return num_buckets + orig;
    }

    void alloc_buckets_to_seg(rdf_seg_meta_t &seg, segid_t segid, uint64_t total_num_keys) {
        // allocate buckets in main-header region to segments
        uint64_t nbuckets;
        if (seg.num_keys == 0) {
            nbuckets = 0;
        } else if (global_auto_bkt_alloc) {
            // deduct some buckets from total to prevent overflow
            static uint64_t num_free_buckets = num_buckets - num_segments;
            static double total_ratio_ = 0.0;

            double ratio = static_cast<double>(seg.num_keys) / total_num_keys;
            nbuckets = ratio * num_free_buckets;
            total_ratio_ += ratio;
            logger(LOG_DEBUG, "Seg[%lu|%lu|%lu]: "
                   "#keys: %lu, nbuckets: %lu, bucket_off: %lu, "
                   "ratio: %f, total_ratio: %f",
                   segid.index, segid.pid, segid.dir,
                   seg.num_keys, nbuckets, main_hdr_off,
                   ratio, total_ratio_);
        } else {
            nbuckets = seg.num_keys * (100.0 / (Global::est_load_factor * ASSOCIATIVITY));
        }
        seg.num_buckets = std::max(nbuckets, min_buckets_per_seg);
        logger(LOG_DEBUG, "Seg[%lu|%lu|%lu]: "
               "#keys: %lu, nbuckets: %lu, bucket_off: %lu, ",
               segid.index, segid.pid, segid.dir,
               seg.num_keys, seg.num_buckets, main_hdr_off);

        seg.bucket_start = main_hdr_off;
        main_hdr_off += seg.num_buckets;
        ASSERT(main_hdr_off <= num_buckets);

        // allocate buckets in indirect-header region to segments
        // #buckets : #extended buckets = 1 : 0.15
        if (seg.num_buckets > 0) {
            uint64_t nbuckets = 0;
#ifdef USE_GPU
            nbuckets = EXT_BUCKET_EXTENT_LEN(seg.num_buckets);
#else
            nbuckets = EXT_BUCKET_EXTENT_LEN;
#endif
            uint64_t start_off = alloc_ext_buckets(nbuckets);
            seg.add_ext_buckets(ext_bucket_extent_t(nbuckets, start_off));
        }
    }

    // Index arrays which stores start offset of each predicate(attr)'s data.
    typedef vector<unordered_map<sid_t, size_t>> triple_map_t;
    // Build index of each predicate's data to accelerate insert triples.
    triple_map_t init_triple_map(const vector<vector<triple_t>> &triples) {
        triple_map_t triple_map(Global::num_engines);

        #pragma omp parallel for num_threads(Global::num_engines)
        for (int tid = 0; tid < Global::num_engines; tid++) {
            const vector<triple_t> &local = triples[tid];
            for (size_t i = 0; i < local.size();) {
                sid_t pid = local[i].p;
                triple_map[tid].emplace(pid, i);
                // skip current predicate's data
                for (; i < local.size() && local[i].p == pid; i++);
            }
        }
        return triple_map;
    }

    // Build index of each attr's data to accelerate insert triples attr.
    triple_map_t init_triple_map(const vector<vector<triple_attr_t>> &triples) {
        triple_map_t triple_map(Global::num_engines);

        #pragma omp parallel for num_threads(Global::num_engines)
        for (int tid = 0; tid < Global::num_engines; tid++) {
            const vector<triple_attr_t> &local = triples[tid];
            for (size_t i = 0; i < local.size();) {
                sid_t aid = local[i].a;
                triple_map[tid].emplace(aid, i);
                // skip current attr's data
                for (; i < local.size() && local[i].a == aid; i++);
            }
        }
        return triple_map;
    }

    /**
     * Helper function of insert_triples.
     * Insert normal predicate-based triples and return start offset of next triples.
     * @param s: start offset
     */
    size_t insert_normal_triples(int tid, const vector<triple_t> &triples,
                                 rdf_seg_meta_t &seg, size_t s, dir_t dir) {
        ikey_t key;
        size_t e = s + 1;  // end offset
        if (dir == OUT) {  // pso triples
            // find end of triples with current (subject + predicate)
            while ((e < triples.size())
                    && (triples[s].s == triples[e].s)
                    && (triples[s].p == triples[e].p)) { e++; }
            // allocate a vertex
            key = ikey_t(triples[s].s, triples[s].p, OUT);

        } else {           // pos triples
            while ((e < triples.size())
                    && (triples[s].o == triples[e].o)
                    && (triples[s].p == triples[e].p)) { e++; }
            key = ikey_t(triples[s].o, triples[s].p, IN);
        }
        // alloc edges
        uint64_t off = alloc_edges(e - s, tid, &seg);
        // insert vertex
        uint64_t slot_id = insert_key(key);
        iptr_t ptr = iptr_t(e - s, off);
        vertices[slot_id].ptr = ptr;

        // insert edges
        for (uint32_t i = s; i < e; i++)
            edges[off++].val = (dir == OUT) ? triples[i].o : triples[i].s;

        collect_idx_info(slot_id);
        return e;
    }

    /**
     * Insert triples beloging to the segment identified by segid to store
     * Notes: This function only insert triples belonging to normal segment
     */
    void insert_triples(int tid, segid_t segid, const triple_map_t &triple_maps,
            const vector<vector<triple_t>> &triples) {
        auto &segment = rdf_seg_meta_map[segid];
        sid_t pid = segid.pid;
        uint64_t off = segment.edge_start;

        ASSERT(segid.index == 0);
        if (segment.num_edges == 0) {
            logger(LOG_DEBUG, "Thread(%d): abort! segment(%d|%d|%d) is empty.\n",
                   tid, segid.index, segid.pid, segid.dir);
            return;
        }

        for (int i = 0; i < Global::num_engines; i++) {
            auto &pmap = triple_maps[i];
            auto &vec = triples[i];

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
     * Insert attr triples beloging to the segment identified by segid to store
     */
    void insert_attr(int tid, segid_t segid, const triple_map_t &triple_maps,
            const vector<vector<triple_attr_t>> &attr_triples) {
        auto &segment = rdf_seg_meta_map[segid];
        sid_t aid = segid.pid;
        int type = attr_type_map[aid];
        uint64_t sz = (get_sizeof(type) - 1) / sizeof(edge_t) + 1;   // get the ceil size;

        if (segment.num_edges == 0) {
            logger(LOG_DEBUG, "Segment(%d|%d|%d) is empty.\n",
                   segid.index, segid.pid, segid.dir);
            return;
        }

        for (int i = 0; i < Global::num_engines; i++) {
            auto &pmap = triple_maps[i];
            auto &asv = attr_triples[i];
            // current thread's local triples has attr data
            auto it = pmap.find(aid);
            if (it != pmap.end()) {
                size_t s = it->second;
                while (s < asv.size() && asv[s].a == aid) {
                    // allocate a vertex
                    ikey_t key = ikey_t(asv[s].s, asv[s].a, OUT);
                    // allocate edges
                    uint64_t off = alloc_edges(sz, tid, &segment);
                    // insert vertex
                    uint64_t slot_id = insert_key(key);
                    iptr_t ptr = iptr_t(sz, off, type);
                    vertices[slot_id].ptr = ptr;

                    // insert edges
                    switch (type) {
                        case INT_t:
                            *(int *)(edges + off) = boost::get<int>(asv[s].v);
                            break;
                        case FLOAT_t:
                            *(float *)(edges + off) = boost::get<float>(asv[s].v);
                            break;
                        case DOUBLE_t:
                            *(double *)(edges + off) = boost::get<double>(asv[s].v);
                            break;
                        default:
                            logstream(LOG_ERROR) << "Unsupported value type of attribute" << LOG_endl;
                    }
                    s++;
                }
            }
        }
    }

    // init metadata for each segment
    void init_seg_metas(const vector<vector<triple_t>> &triple_pso,
                        const vector<vector<triple_t>> &triple_pos,
                        const vector<vector<triple_attr_t>> &triple_sav) {
        /**
         * count(|pred| means number of local predicates):
         * 1. normal vertices [vid|pid|IN/OUT], key: pid, cnt_t: in&out, #item: |pred|
         * 2. vid's all types [vid|TYPE_ID(1)|OUT], key: TYPE_ID(1), cnt_t: out, #item: contained above
         * 3*. vid's all predicates [vid|PREDICATE_ID(0)|IN/OUT], key: PREDICATE_ID(0), cnt_t: in&out, #item: 1
         * 4^. attr triples [vid|pid|out], key: pid, cnt_t: out, #item: |attrpred|
         */
        map<sid_t, cnt_t> normal_cnt_map;

        /**
         * count(|pred| means number of local predicates):
         * 1. predicate index [0|pid|IN/OUT], key: pid, cnt_t: in&out, #item: |pred|
         * 2. type index [0|typeid|IN], key: typeid, cnt_t: in, #item: |type|
         */
        map<sid_t, cnt_t> index_cnt_map;

        // initialization
        for (int i = 0; i <= get_num_preds(); ++i) {
            index_cnt_map.insert(make_pair(i, cnt_t()));
            normal_cnt_map.insert(make_pair(i, cnt_t()));
            rdf_seg_meta_map.insert(make_pair(segid_t(0, i, IN), rdf_seg_meta_t()));
            rdf_seg_meta_map.insert(make_pair(segid_t(0, i, OUT), rdf_seg_meta_t()));
        }
        // init index segment
        rdf_seg_meta_map.insert(make_pair(segid_t(1, PREDICATE_ID, IN), rdf_seg_meta_t()));
        rdf_seg_meta_map.insert(make_pair(segid_t(1, PREDICATE_ID, OUT), rdf_seg_meta_t()));

        #pragma omp parallel for num_threads(Global::num_engines)
        for (int tid = 0; tid < Global::num_engines; tid++) {
            const vector<triple_t> &pso = triple_pso[tid];
            const vector<triple_t> &pos = triple_pos[tid];
            const vector<triple_attr_t> &sav = triple_sav[tid];

            uint64_t s = 0;
            while (s < pso.size()) {
                uint64_t e = s + 1;

                while ((e < pso.size())
                        && (pso[s].s == pso[e].s)
                        && (pso[s].p == pso[e].p))  {
                    // count #edge of type-idx
                    if (pso[e].p == TYPE_ID && is_tpid(pso[e].o)) {
#ifdef VERSATILE
                        t_set.insert(pso[e].o);
#endif
                        index_cnt_map[ pso[e].o ].in++;
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
                normal_cnt_map[ pso[s].p ].out += (e - s);

                // count #edge of predicate-idx
                index_cnt_map[ pso[s].p ].in++;

                // count #edge of type-idx
                if (pso[s].p == TYPE_ID && is_tpid(pso[s].o)) {
#ifdef VERSATILE
                    t_set.insert(pso[s].o);
#endif
                    index_cnt_map[ pso[s].o ].in++;
                }
                s = e;
            }

            uint64_t type_triples = 0;
            triple_t tp;
            while (type_triples < pos.size() && is_tpid(pos[type_triples].o)) {
                type_triples++;
            }

            s = type_triples; // skip type triples
            while (s < pos.size()) {
                // predicate-based key (object + predicate)
                uint64_t e = s + 1;
                while ((e < pos.size())
                        && (pos[s].o == pos[e].o)
                        && (pos[s].p == pos[e].p)) {
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
                normal_cnt_map[ pos[s].p ].in += (e - s);
                index_cnt_map[ pos[s].p ].out++;
                s = e;
            }

            s = 0;
            while (s < sav.size()) {
                uint64_t e = s + 1;

                while ((e < sav.size()) && (sav[s].s == sav[e].s) && (sav[s].a == sav[e].a))
                    e++;

                // count #edge of predicate
                normal_cnt_map[ sav[s].a ].out += (e - s);
                attr_set.insert(sav[s].a);
                attr_type_map.insert(std::make_pair(sav[s].a, boost::apply_visitor(variant_type(), sav[s].v)));

                s = e;
            }
        }

        // count the total number of keys
        uint64_t total_num_keys = 0, num_typeid = 0;
#ifdef VERSATILE
        logger(LOG_DEBUG, "pid: %d: normal: #IN: %lu, #OUT: %lu", PREDICATE_ID,
               normal_cnt_map[PREDICATE_ID].in.load(), normal_cnt_map[PREDICATE_ID].out.load());
        total_num_keys += vp_meta[OUT].size() + vp_meta[IN].size();
#endif
        for (int i = 1; i <= get_num_preds(); ++i) {
            logger(LOG_DEBUG, "pid: %d: normal: #IN: %lu, #OUT: %lu; index: #ALL: %lu, #IN: %lu, #OUT: %lu",
                   i, normal_cnt_map[i].in.load(), normal_cnt_map[i].out.load(),
                   (index_cnt_map[i].in.load() + index_cnt_map[i].out.load()),
                   index_cnt_map[i].in.load(), index_cnt_map[i].out.load());
            /**
             * this if-sentence checks if the i is a typeid
             * if the sentence is true, it means i is a normal predicate
             * index_cnt_map[i] stores #edges of pid's predicate index (IN, OUT)
             * whose sum equals to #keys of the predicate
             */
            if (attr_set.find(i) != attr_set.end()) {
                total_num_keys += normal_cnt_map[i].out.load();
            } else if (normal_cnt_map[i].in.load() + normal_cnt_map[i].out.load() > 0) {
                all_local_preds.push_back(i);
                total_num_keys += (index_cnt_map[i].in.load()
                                   + index_cnt_map[i].out.load());
            } else if (index_cnt_map[i].in.load() > 0) {
                num_typeid++;
            }
        }
        /**
         * #predicate index = #normal predicate * 2
         * #type index = #typeid
         */
        total_num_keys += all_local_preds.size() * 2 + num_typeid;

        // allocate buckets and edges to segments
        rdf_seg_meta_t &idx_out_seg = rdf_seg_meta_map[segid_t(1, PREDICATE_ID, OUT)];
        rdf_seg_meta_t &idx_in_seg = rdf_seg_meta_map[segid_t(1, PREDICATE_ID, IN)];

#ifdef VERSATILE
        // vid's all predicates OUT
        rdf_seg_meta_t &pred_out_seg = rdf_seg_meta_map[segid_t(0, PREDICATE_ID, OUT)];
        pred_out_seg.num_edges = normal_cnt_map[PREDICATE_ID].out.load();
        pred_out_seg.edge_start = alloc_edges_to_seg(pred_out_seg.num_edges);
        pred_out_seg.num_keys = vp_meta[OUT].size();
        alloc_buckets_to_seg(pred_out_seg, segid_t(0, PREDICATE_ID, OUT), total_num_keys);
        // vid's all predicates IN
        rdf_seg_meta_t &pred_in_seg = rdf_seg_meta_map[segid_t(0, PREDICATE_ID, IN)];
        pred_in_seg.num_edges = normal_cnt_map[PREDICATE_ID].in.load();
        pred_in_seg.edge_start = alloc_edges_to_seg(pred_in_seg.num_edges);
        pred_in_seg.num_keys = vp_meta[IN].size();
        alloc_buckets_to_seg(pred_in_seg, segid_t(0, PREDICATE_ID, IN), total_num_keys);

        #pragma omp parallel for num_threads(2)
        for (int tid = 0; tid < 2; tid++) {
            alloc_vp_edges((dir_t)tid);
        }

        // all local entities
        idx_in_seg.num_edges += v_set.size();
        idx_in_seg.num_keys += 1;
        // all local types
        idx_out_seg.num_edges += t_set.size();
        idx_out_seg.num_keys += 1;
        // all local predicates
        idx_out_seg.num_edges += p_set.size();
        idx_out_seg.num_keys += 1;

        logstream(LOG_DEBUG) <<  "s_set: " << pred_out_seg.num_keys << ", o_set: " << pred_in_seg.num_keys
                             << ", v_set: " << v_set.size() << ", p_set: " << p_set.size() << ", t_set: " << t_set.size() << LOG_endl;
#endif

        for (sid_t pid = 1; pid <= get_num_preds(); ++pid) {
            rdf_seg_meta_t &out_seg = rdf_seg_meta_map[segid_t(0, pid, OUT)];
            rdf_seg_meta_t &in_seg = rdf_seg_meta_map[segid_t(0, pid, IN)];

            out_seg.num_edges = normal_cnt_map[pid].out.load();
            in_seg.num_edges = normal_cnt_map[pid].in.load();

            idx_out_seg.num_edges += index_cnt_map[pid].out.load();
            idx_in_seg.num_edges += index_cnt_map[pid].in.load();

            if (attr_set.find(pid) != attr_set.end()) {
                // attribute segment
                out_seg.num_keys = out_seg.num_edges;
                in_seg.num_keys = 0;
                // calculate the number of edge_t needed to store 1 value
                uint64_t sz = (attr_type_map[pid] - 1) / sizeof(edge_t) + 1;   // get the ceil size;
                out_seg.num_edges = out_seg.num_edges * sz;
            } else {
                // normal pred segment
                uint64_t normal_nkeys[2] = {index_cnt_map[pid].out, index_cnt_map[pid].in};
                out_seg.num_keys = (out_seg.num_edges == 0) ? 0 : normal_nkeys[OUT];
                in_seg.num_keys  = (in_seg.num_edges == 0)  ? 0 : normal_nkeys[IN];
            }

            // allocate space for edges in entry-region
            out_seg.edge_start = alloc_edges_to_seg(out_seg.num_edges);
            in_seg.edge_start = alloc_edges_to_seg(in_seg.num_edges);

            alloc_buckets_to_seg(out_seg, segid_t(0, pid, OUT), total_num_keys);
            alloc_buckets_to_seg(in_seg, segid_t(0, pid, IN), total_num_keys);

            logger(LOG_DEBUG, "Predicate[%d]: normal: OUT[#keys: %lu, #buckets: %lu, #edges: %lu] "
                   "IN[#keys: %lu, #buckets: %lu, #edges: %lu];",
                   pid, out_seg.num_keys, out_seg.num_buckets, out_seg.num_edges,
                   in_seg.num_keys, in_seg.num_buckets, in_seg.num_edges);

        }

        idx_out_seg.edge_start = alloc_edges_to_seg(idx_out_seg.num_edges);
        idx_out_seg.num_keys = all_local_preds.size();
        alloc_buckets_to_seg(idx_out_seg, segid_t(1, PREDICATE_ID, OUT), total_num_keys);

        idx_in_seg.edge_start = alloc_edges_to_seg(idx_in_seg.num_edges);
        idx_in_seg.num_keys = all_local_preds.size() + num_typeid;
        alloc_buckets_to_seg(idx_in_seg, segid_t(1, PREDICATE_ID, IN), total_num_keys);

        logger(LOG_DEBUG, "index: OUT[#keys: %lu, #buckets: %lu, #edges: %lu], "
               "IN[#keys: %lu, #buckets: %lu, #edges: %lu], bucket_off: %lu\n",
               idx_out_seg.num_keys, idx_out_seg.num_buckets, idx_out_seg.num_edges,
               idx_in_seg.num_keys, idx_in_seg.num_buckets, idx_in_seg.num_edges, main_hdr_off);
    }

    // insert key to a slot
    uint64_t insert_key(ikey_t key, bool check_dup = true) {
        uint64_t bucket_id = bucket_local(key);
        uint64_t slot_id = bucket_id * ASSOCIATIVITY;
        uint64_t lock_id = bucket_id % NUM_LOCKS;
        uint64_t seg_ext_lock_id = segid_t(key).hash() % NUM_LOCKS;

        bool found = false;
        pthread_spin_lock(&bucket_locks[lock_id]);
        while (slot_id < num_slots) {
            // the last slot of each bucket is always reserved for pointer to indirect header
            /// TODO: add type info to slot and reuse the last slot to store key
            /// TODO: key.vid is reused to store the bucket_id of indirect header rather than ptr.off,
            ///       since the is_empty() is not robust.
            for (int i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++) {
                //ASSERT(vertices[slot_id].key != key); // no duplicate key
                if (vertices[slot_id].key == key) {
                    if (check_dup) {
                        logstream(LOG_ERROR) << "conflict at slot["
                                             << slot_id << "] of bucket["
                                             << bucket_id << "], "
                                             << key.to_string() << ", "
                                             << vertices[slot_id].key.to_string()
                                             << LOG_endl;
                        ASSERT(false);
                    } else {
                        goto done;
                    }
                }

                // insert to an empty slot
                if (vertices[slot_id].key.is_empty()) {
                    vertices[slot_id].key = key;
                    goto done;
                }
            }

            // whether the bucket_ext (indirect-header region) is used
            if (!vertices[slot_id].key.is_empty()) {
                slot_id = vertices[slot_id].key.vid * ASSOCIATIVITY;
                continue; // continue and jump to next bucket
            }

            // allocate and link a new indirect header
            pthread_spin_lock(&seg_ext_locks[seg_ext_lock_id]);
            rdf_seg_meta_t &seg = rdf_seg_meta_map[segid_t(key)];
            uint64_t ext_bucket_id = seg.get_ext_bucket();
            if (ext_bucket_id == 0) {
                uint64_t nbuckets = 0;
#ifdef USE_GPU
                nbuckets = EXT_BUCKET_EXTENT_LEN(seg.num_buckets);
#else
                nbuckets = EXT_BUCKET_EXTENT_LEN;
#endif
                uint64_t start_off = alloc_ext_buckets(nbuckets);
                seg.add_ext_buckets(ext_bucket_extent_t(nbuckets, start_off));
                ext_bucket_id = seg.get_ext_bucket();
            }
            pthread_spin_unlock(&seg_ext_locks[seg_ext_lock_id]);
            vertices[slot_id].key.vid = ext_bucket_id;

            slot_id = vertices[slot_id].key.vid * ASSOCIATIVITY; // move to a new bucket_ext
            vertices[slot_id].key = key; // insert to the first slot
            goto done;
        }
done:
        pthread_spin_unlock(&bucket_locks[lock_id]);
        ASSERT(slot_id < num_slots);
        return slot_id;
    }

    void collect_idx_info(uint64_t slot_id) {
        sid_t vid = vertices[slot_id].key.vid;
        sid_t pid = vertices[slot_id].key.pid;
        uint64_t sz = vertices[slot_id].ptr.size;
        uint64_t off = vertices[slot_id].ptr.off;

        if (vertices[slot_id].key.dir == IN) {
            if (pid == PREDICATE_ID) {
            } else if (pid == TYPE_ID) {
                ASSERT(false); // (IN) type triples should be skipped
            } else { // predicate-index (OUT) vid
                tbb_hash_map::accessor a;
                pidx_out_map.insert(a, pid);
                a->second.push_back(vid);
            }
        } else {
            if (pid == PREDICATE_ID) {
            } else if (pid == TYPE_ID) {
                // type-index (IN) -> vid_list
                for (uint64_t e = 0; e < sz; e++) {
                    tbb_hash_map::accessor a;
                    tidx_map.insert(a, edges[off + e].val);
                    a->second.push_back(vid);
                }
            } else { // predicate-index (IN) vid
                tbb_hash_map::accessor a;
                pidx_in_map.insert(a, pid);
                a->second.push_back(vid);
            }
        }
    }

#ifdef VERSATILE
    // insert {v/t/p}_set into gstore
    void insert_idx_set(const tbb_unordered_set &set, uint64_t &off, sid_t pid, dir_t d) {
        uint64_t sz = set.size();

        ikey_t key = ikey_t(0, pid, d);
        uint64_t slot_id = insert_key(key);
        iptr_t ptr = iptr_t(sz, off);
        vertices[slot_id].ptr = ptr;

        for (auto const &e : set)
            edges[off++].val = e;
    }
#endif // VERSATILE

    void send_seg_meta(TCP_Adaptor *tcp_ad) {
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

    void recv_seg_meta(TCP_Adaptor *tcp_ad) {
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
                shared_rdf_seg_meta_map.insert(make_pair(msg.sender_sid, msg.data));
            logstream(LOG_INFO) << "#" << sid
                                << " receives segment metadata from server " << msg.sender_sid
                                << LOG_endl;
        }
    }

    // re-adjust attributes of segments
    void finalize_seg_metas() {
        uint64_t nbuckets_per_blk = MiB2B(Global::gpu_key_blk_size_mb) / (sizeof(vertex_t) * ASSOCIATIVITY);
        uint64_t nentries_per_blk = MiB2B(Global::gpu_value_blk_size_mb) / sizeof(edge_t);

        // set the number of cache blocks needed by each segment
        for (auto &e : rdf_seg_meta_map) {
            e.second.num_key_blks = ceil(((double) e.second.get_total_num_buckets()) / nbuckets_per_blk);
            e.second.num_value_blks = ceil(((double) e.second.num_edges) / nentries_per_blk);
        }
    }

    // release memory after gstore init
    void finalize_init() {
        vector<sid_t>().swap(all_local_preds);
        tbb_unordered_set().swap(attr_set);
        tbb::concurrent_unordered_map<sid_t, int>().swap(attr_type_map);
        tbb_hash_map().swap(pidx_in_map);
        tbb_hash_map().swap(pidx_out_map);
        tbb_hash_map().swap(tidx_map);
#ifdef VERSATILE
        for (int i = 0; i < 2; i++) {
            tbb::concurrent_hash_map<sid_t, uint64_t>().swap(vp_meta[i]);
        }
#endif // VERSATILE
    }

public:
    vertex_t *vertices;
    edge_t *edges;

    static const int ASSOCIATIVITY = 8;  // the associativity of slots in each bucket

    // Memory Usage (estimation):
    //   header region: |vertex| = 128-bit; #verts = (#S + #O) * AVG(#P) ～= #T
    //   entry region:    |edge| =  32-bit; #edges = #T * 2 + (#S + #O) * AVG(#P) ～= #T * 3
    //
    //                                      (+VERSATILE)
    //                                      #verts += #S + #O
    //                                      #edges += (#S + #O) * AVG(#P) ~= #T
    //
    // main-header / (main-header + indirect-header)
    static const int MHD_RATIO = 80;
    // header * 100 / (header + entry)
    static const int HD_RATIO = (128 * 100 / (128 + 3 * std::numeric_limits<sid_t>::digits));

    // number of predicates in the whole dataset
    int num_normal_preds = 0;
    int num_attr_preds = 0;

    // return total num of preds, including normal and attr
    inline int get_num_preds() const { return num_normal_preds + num_attr_preds; }

    inline const std::map<segid_t, rdf_seg_meta_t> &get_rdf_seg_metas() { return rdf_seg_meta_map; }

    virtual ~GStore() {}

    virtual void init(vector<vector<triple_t>> &triple_pso,
                      vector<vector<triple_t>> &triple_pos,
                      vector<vector<triple_attr_t>> &triple_sav) = 0;
    virtual void refresh() = 0;

    /**
     * GStore: key (main-header and indirect-header region) | value (entry region)
     * head region is a cluster chaining hash-table (with associativity)
     * entry region is a varying-size array
     */
    GStore(int sid, Mem *mem): sid(sid), mem(mem) {
        uint64_t header_region = mem->kvstore_size() * HD_RATIO / 100;
        uint64_t entry_region = mem->kvstore_size() - header_region;

        // header region
        num_slots = header_region / sizeof(vertex_t);
        num_buckets = wukong::math::hash_prime_u64((num_slots / ASSOCIATIVITY) * MHD_RATIO / 100);
        num_buckets_ext = (num_slots / ASSOCIATIVITY) - num_buckets;
        // entry region
        num_entries = entry_region / sizeof(edge_t);

        vertices = (vertex_t *)(mem->kvstore());
        edges = (edge_t *)(mem->kvstore() + num_slots * sizeof(vertex_t));

        pthread_spin_init(&bucket_ext_lock, 0);
        for (int i = 0; i < NUM_LOCKS; i++) {
            pthread_spin_init(&bucket_locks[i], 0);
            pthread_spin_init(&seg_ext_locks[i], 0);
        }

        // print gstore usage
        logstream(LOG_INFO) << "gstore = ";
        logstream(LOG_INFO) << mem->kvstore_size() << " bytes " << LOG_endl;
        logstream(LOG_INFO) << "  header region: " << num_slots << " slots"
                            << " (main = " << num_buckets
                            << ", indirect = " << num_buckets_ext << ")" << LOG_endl;
        logstream(LOG_INFO) << "  entry region: " << num_entries << " entries" << LOG_endl;
    }

    // FIXME: refine return value with type of subject/object
    edge_t *get_edges(int tid, sid_t vid, sid_t pid, dir_t d, uint64_t &sz,
                      int &type = *(int *)NULL) {
        // index vertex should be 0 and always local
        if (vid == 0)
            return get_edges_local(tid, 0, pid, d, sz);

        // normal vertex
        if (wukong::math::hash_mod(vid, Global::num_servers) == sid)
            return get_edges_local(tid, vid, pid, d, sz, type);
        else
            return get_edges_remote(tid, vid, pid, d, sz, type);
    }

    void sync_metadata() {
        extern TCP_Adaptor *con_adaptor;
        send_seg_meta(con_adaptor);
        recv_seg_meta(con_adaptor);
    }

    virtual void print_mem_usage() {
        uint64_t used_slots = 0;
        uint64_t used_edges = 0;
        for (uint64_t x = 0; x < num_buckets; x++) {
            uint64_t slot_id = x * ASSOCIATIVITY;
            for (int y = 0; y < ASSOCIATIVITY - 1; y++, slot_id++) {
                if (vertices[slot_id].key.is_empty())
                    continue;
                used_slots++;
                used_edges += vertices[slot_id].ptr.size;
            }
        }

        logstream(LOG_INFO) << "main header: " << B2MiB(num_buckets * ASSOCIATIVITY * sizeof(vertex_t))
                            << " MB (" << num_buckets * ASSOCIATIVITY << " slots)" << LOG_endl;
        logstream(LOG_INFO) << "\tused: " << 100.0 * used_slots / (num_buckets * ASSOCIATIVITY)
                            << " % (" << used_slots << " slots)" << LOG_endl;
        logstream(LOG_INFO) << "\tchain: " << 100.0 * num_buckets / (num_buckets * ASSOCIATIVITY)
                            << " % (" << num_buckets << " slots)" << LOG_endl;

        used_slots = 0;
        for (uint64_t x = num_buckets; x < num_buckets + last_ext; x++) {
            uint64_t slot_id = x * ASSOCIATIVITY;
            for (int y = 0; y < ASSOCIATIVITY - 1; y++, slot_id++) {
                if (vertices[slot_id].key.is_empty())
                    continue;
                used_edges += vertices[slot_id].ptr.size;
            }
        }

        logstream(LOG_INFO) << "indirect header: " << B2MiB(num_buckets_ext * ASSOCIATIVITY * sizeof(vertex_t))
                            << " MB (" << num_buckets_ext * ASSOCIATIVITY << " slots)" << LOG_endl;
        logstream(LOG_INFO) << "\talloced: " << 100.0 * last_ext / num_buckets_ext
                            << " % (" << last_ext << " buckets)" << LOG_endl;
        logstream(LOG_INFO) << "\tused: " << 100.0 * used_slots / (num_buckets_ext * ASSOCIATIVITY)
                            << " % (" << used_slots << " slots)" << LOG_endl;

        logstream(LOG_INFO) << "entry: " << B2MiB(num_entries * sizeof(edge_t))
                            << " MB (" << num_entries << " entries)" << LOG_endl;
        logstream(LOG_INFO) << "\tused edges: " << B2MiB(used_edges * sizeof(edge_t))
                            << " MB (" << used_edges << " edges)" << LOG_endl;
    }
};
