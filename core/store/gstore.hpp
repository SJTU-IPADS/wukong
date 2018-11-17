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
#include <atomic>

#include "global.hpp"
#include "rdma.hpp"
#include "type.hpp"

#include "store/vertex.hpp"

// utils
#include "math.hpp"
#include "timer.hpp"
#include "unit.hpp"
#include "atomic.hpp"
#include "variant.hpp"


using namespace std;

/* Cache remote vertex(location) of the given key, eleminating one RDMA read.
 * This only works when RDMA enabled.
 */
class RDMA_Cache {
    static const int NUM_BUCKETS = 1 << 20;
    static const int ASSOCIATIVITY = 8;  // the associativity of items in each bucket

    // cache line
    struct item_t {
        vertex_t v;
        uint64_t expire_time;  // only work when DYNAMIC_GSTORE=on
        uint32_t cnt;  // keep track of visit cnt of v

        /// Each cache line use a version to detect reader-writer conflict.
        /// Version == 0, when an insertion occurs.
        /// Init value is set to 1.
        /// Version always increments after an insertion.
        uint32_t version;

        item_t() : expire_time(0), cnt(0), version(1) { }
    };

    // bucket whose items share the same index
    struct bucket_t {
        item_t items[ASSOCIATIVITY]; // item list
    };
    bucket_t *hashtable;

    uint64_t lease;  // only work when DYNAMIC_GSTORE=on

public:
    RDMA_Cache() {
        // init hashtable
        size_t mem_size = sizeof(bucket_t) * NUM_BUCKETS;
        hashtable = new bucket_t[NUM_BUCKETS];
        logstream(LOG_INFO) << "cache allocate " << mem_size << " memory" << LOG_endl;
    }

    /* Lookup a vertex in cache according to the given key.*/
    bool lookup(ikey_t key, vertex_t &ret) {
        if (!global_enable_caching)
            return false;

        int idx = key.hash() % NUM_BUCKETS; // find bucket
        item_t *items = hashtable[idx].items;
        bool found = false;
        uint32_t ver;

        // lookup vertex in item list
        for (int i = 0; i < ASSOCIATIVITY; i++) {
            if (items[i].v.key == key) {
                while (true) {
                    ver = items[i].version;
                    // re-check
                    if (items[i].v.key == key) {
#ifdef DYNAMIC_GSTORE
                        if (timer::get_usec() < items[i].expire_time) {
                            ret = items[i].v;
                            items[i].cnt++;
                            found = true;
                        }
#else
                        ret = items[i].v;
                        items[i].cnt++;
                        found = true;
#endif
                        asm volatile("" ::: "memory");

                        // check version
                        if (ver != 0 && items[i].version == ver)
                            return found;
                    } else
                        return false;
                }
            }
        }
        return false;
    } // end of lookup

    /* Insert a vertex into cache. */
    void insert(vertex_t &v) {
        if (!global_enable_caching)
            return;

        int idx = v.key.hash() % NUM_BUCKETS;
        item_t *items = hashtable[idx].items;

        uint64_t min_cnt;
        int pos = -1;  // position to insert v

        while (true) {
            for (int i = 0; i < ASSOCIATIVITY; i++) {
                if (items[i].v.key == v.key || items[i].v.key.is_empty()) {
                    pos = i;
                    break;
                }
            }
            // no free cache line
            // pick one to replace (LFU)
            if (pos == -1) {
                min_cnt = 1ul << 63;
                for (int i = 0; i < ASSOCIATIVITY; i++) {
                    if (items[i].cnt < min_cnt) {
                        min_cnt = items[i].cnt;
                        pos = i;
                        if (min_cnt == 0)
                            break;
                    }
                }
            }

            volatile uint32_t old_ver = items[pos].version;
            if (old_ver != 0) {
                uint32_t ret_ver = wukong::atomic::compare_and_swap(&items[pos].version, old_ver, 0);
                if (ret_ver == old_ver) {
#ifdef DYNAMIC_GSTORE
                    // Do not reset visit cnt for the same vertex
                    items[pos].cnt = (items[pos].v.key == v.key) ? items[pos].cnt : 0;
                    items[pos].v = v;
                    items[pos].expire_time = timer::get_usec() + lease;
#else
                    items[pos].cnt = 0;
                    items[pos].v = v;
#endif
                    asm volatile("" ::: "memory");
                    ret_ver = wukong::atomic::compare_and_swap(&items[pos].version, 0, old_ver + 1);
                    assert(ret_ver == 0);
                    return;
                }
            }
        }
    } // end of insert

#ifdef DYNAMIC_GSTORE
    /* Set lease.*/
    void set_lease(uint64_t _lease) { lease = _lease; }

    /**
     * Invalidate cache item of the given key.
     * Only work when the corresponding vertex exists.
     */
    void invalidate(ikey_t key) {
        if (!global_enable_caching)
            return;

        int idx = key.hash() % NUM_BUCKETS;
        item_t *items = hashtable[idx].items;
        for (int i = 0; i < ASSOCIATIVITY; i++) {
            if (items[i].v.key == key) {

                /// Version is not checked and set here
                /// since inconsistent expire time does not cause staleness.
                /// The only possible overhead is
                /// an extra update of vertex and expire time.
                items[i].expire_time = timer::get_usec();
                return;
            }
        }
    }
#endif
};

/**
 * Map the Graph model (e.g., vertex, edge, index) to KVS model (e.g., key, value)
 * Graph store adopts clustring chaining key/value store (see paper: DrTM SOSP'15)
 */
class GStore {
    friend class data_statistic;
    friend class GChecker;

protected:
    static const int NUM_LOCKS = 1024;

    int sid;

    Mem *mem;

    vertex_t *vertices;
    uint64_t num_slots;       // 1 bucket = ASSOCIATIVITY slots
    uint64_t num_buckets;     // main-header region (static)
    uint64_t num_buckets_ext; // indirect-header region (dynamical)
    uint64_t last_ext;
    pthread_spinlock_t bucket_locks[NUM_LOCKS]; // lock virtualization (see paper: vLokc CGO'13)
    pthread_spinlock_t bucket_ext_lock;
    uint64_t vcount;          // the count of accesses to vertices

    edge_t *edges;
    uint64_t num_entries;     // entry region (dynamical)
    uint64_t ecount;          // the count of accesses to edges

    typedef tbb::concurrent_hash_map<sid_t, vector<sid_t>> tbb_hash_map;
    tbb_hash_map pidx_in_map;  // predicate-index (IN)
    tbb_hash_map pidx_out_map; // predicate-index (OUT)
    tbb_hash_map tidx_map;     // type-index

    RDMA_Cache rdma_cache;


    // get bucket_id according to key
    virtual uint64_t bucket_local(ikey_t key) = 0;
    virtual uint64_t bucket_remote(ikey_t key, int dst_sid) = 0;

    // Allocate space to store edges of given size. Return offset of allocated space.
    virtual uint64_t alloc_edges(uint64_t n, int64_t tid = 0) = 0;

    // Check the validation of given edges according to given vertex.
    virtual bool edge_is_valid(vertex_t &v, edge_t *edge_ptr) = 0;

    // Get edges of given vertex from dst_sid by RDMA read.
    virtual edge_t *rdma_get_edges(int tid, int dst_sid, vertex_t &v) = 0;

    // insert key to a slot
    virtual uint64_t insert_key(ikey_t key, bool check_dup = true) = 0;


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

    // TODO check necessity
    void insert_index_map(tbb_hash_map &map, dir_t d) {
        for (auto const &e : map) {
            sid_t pid = e.first;
            uint64_t sz = e.second.size();
            uint64_t off = alloc_edges(sz);

            ikey_t key = ikey_t(0, pid, d);
            uint64_t slot_id = insert_key(key);
            iptr_t ptr = iptr_t(sz, off);
            vertices[slot_id].ptr = ptr;

            for (auto const &vid : e.second)
                edges[off++].val = vid;
        }
    }

    // Get remote vertex of given key. This func will fail if RDMA is disabled.
    vertex_t get_vertex_remote(int tid, ikey_t key) {
        int dst_sid = wukong::math::hash_mod(key.vid, global_num_servers);
        uint64_t bucket_id = bucket_remote(key, dst_sid);
        vertex_t vert;

        // FIXME: wukong doesn't support to directly get remote vertex/edge without RDMA
        ASSERT(global_use_rdma);

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
    } // end of get_vertex_remote

    // Get local vertex of given key.
    vertex_t get_vertex_local(int tid, ikey_t key) {
        uint64_t bucket_id = bucket_local(key);;
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
        int dst_sid = wukong::math::hash_mod(vid, global_num_servers);
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

    // TODO
#ifdef VERSATILE
    typedef tbb::concurrent_unordered_set<sid_t> tbb_unordered_set;

    tbb_unordered_set v_set; // all of subjects and objects
    tbb_unordered_set t_set; // all of types
    tbb_unordered_set p_set; // all of predicates

    void insert_index_set(tbb_unordered_set &set, sid_t tpid, dir_t d) {
        uint64_t sz = set.size();
        uint64_t off = alloc_edges(sz);

        ikey_t key = ikey_t(0, tpid, d);
        uint64_t slot_id = insert_key(key);
        iptr_t ptr = iptr_t(sz, off);
        vertices[slot_id].ptr = ptr;

        for (auto const &e : set)
            edges[off++].val = e;
    }
#endif // VERSATILE

public:
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

    /// encoding rules of GStore
    /// subject/object (vid) >= 2^NBITS_IDX, 2^NBITS_IDX > predicate/type (p/tid) >= 2^1,
    /// TYPE_ID = 1, PREDICATE_ID = 0, OUT = 1, IN = 0
    ///
    /// Empty key
    /// (0)   key = [  0 |            0 |      0]  value = [vid0, vid1, ..]  i.e., init
    /// INDEX key/value pair
    /// (1)   key = [  0 |          pid | IN/OUT]  value = [vid0, vid1, ..]  i.e., predicate-index
    /// (2)   key = [  0 |          tid |     IN]  value = [vid0, vid1, ..]  i.e., type-index
    /// (*3)  key = [  0 |      TYPE_ID |     IN]  value = [vid0, vid1, ..]  i.e., all local objects/subjects
    /// (*4)  key = [  0 |      TYPE_ID |    OUT]  value = [pid0, pid1, ..]  i.e., all local types
    /// (*5)  key = [  0 | PREDICATE_ID |    OUT]  value = [pid0, pid1, ..]  i.e., all local predicates
    /// NORMAL key/value pair
    /// (6)   key = [vid |          pid | IN/OUT]  value = [vid0, vid1, ..]  i.e., vid's ngbrs w/ predicate
    /// (7)   key = [vid |      TYPE_ID |    OUT]  value = [tid0, tid1, ..]  i.e., vid's all types
    /// (*8)  key = [vid | PREDICATE_ID | IN/OUT]  value = [pid0, pid1, ..]  i.e., vid's all predicates
    ///
    /// < S,  P, ?O>  ?O : (6)
    /// <?S,  P,  O>  ?S : (6)
    /// < S,  1, ?T>  ?T : (7)
    /// <?S,  1,  T>  ?S : (2)
    /// < S, ?P,  O>  ?P : (8)
    ///
    /// <?S,  P, ?O>  ?S : (1)
    ///               ?O : (1)
    /// <?S,  1, ?O>  ?O : (4)
    ///               ?S : (4) +> (2)
    /// < S, ?P, ?O>  ?P : (8) AND exist(7)
    ///               ?O : (8) AND exist(7) +> (6)
    /// <?S, ?P,  O>  ?P : (8)
    ///               ?S : (8) +> (6)
    /// <?S, ?P,  T>  ?P : exist(2)
    ///               ?S : (2)
    ///
    /// <?S, ?P, ?O>  ?S : (3)
    ///               ?O : (3) AND (4)
    ///               ?P : (5)
    ///               ?S ?P ?O : (3) +> (7) AND (8) +> (6)

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
        for (int i = 0; i < NUM_LOCKS; i++)
            pthread_spin_init(&bucket_locks[i], 0);

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
        if (wukong::math::hash_mod(vid, global_num_servers) == sid)
            return get_edges_local(tid, vid, pid, d, sz, type);
        else
            return get_edges_remote(tid, vid, pid, d, sz, type);
    }

    virtual void print_mem_usage() {
        // TODO
        uint64_t used_slots = 0;
        for (uint64_t x = 0; x < num_buckets; x++) {
            uint64_t slot_id = x * ASSOCIATIVITY;
            for (int y = 0; y < ASSOCIATIVITY - 1; y++, slot_id++) {
                if (vertices[slot_id].key.is_empty())
                    continue;
                used_slots++;
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
                used_slots++;
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
    }
};
