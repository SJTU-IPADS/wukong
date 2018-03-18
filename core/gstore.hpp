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

#include <stdint.h> // uint64_t
#include <vector>
#include <queue>
#include <iostream>
#include <pthread.h>
#include <boost/unordered_set.hpp>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_set.h>

#include "config.hpp"
#include "rdma.hpp"
#include "data_statistic.hpp"
#include "type.hpp"
#include "buddy_malloc.hpp"

#include "mymath.hpp"
#include "timer.hpp"
#include "unit.hpp"
#include "variant.hpp"

using namespace std;

enum { NBITS_DIR = 1 };
enum { NBITS_IDX = 17 }; // equal to the size of t/pid
enum { NBITS_VID = (64 - NBITS_IDX - NBITS_DIR) }; // 0: index vertex, ID: normal vertex

// reserve two special index IDs (predicate and type)
enum { PREDICATE_ID = 0, TYPE_ID = 1 };

static inline bool is_tpid(ssid_t id) { return (id > 1) && (id < (1 << NBITS_IDX)); }

static inline bool is_vid(ssid_t id) { return id >= (1 << NBITS_IDX); }

/**
 * predicate-base key/value store
 * key: vid | t/pid | direction
 * value: v/t/pid list
 */
struct ikey_t {
uint64_t dir : NBITS_DIR; // direction
uint64_t pid : NBITS_IDX; // predicate
uint64_t vid : NBITS_VID; // vertex

    ikey_t(): vid(0), pid(0), dir(0) { }

    ikey_t(sid_t v, sid_t p, dir_t d): vid(v), pid(p), dir(d) {
        assert((vid == v) && (dir == d) && (pid == p)); // no key truncate
    }

    bool operator == (const ikey_t &key) {
        if ((vid == key.vid) && (pid == key.pid) && (dir == key.dir))
            return true;
        return false;
    }

    bool operator != (const ikey_t &key) { return !(operator == (key)); }

    bool is_empty() { return ((vid == 0) && (pid == 0) && (dir == 0)); }

    void print() { cout << "[" << vid << "|" << pid << "|" << dir << "]" << endl; }

    uint64_t hash() {
        uint64_t r = 0;
        r += vid;
        r <<= NBITS_IDX;
        r += pid;
        r <<= NBITS_DIR;
        r += dir;
        return mymath::hash_u64(r); // the standard hash is too slow (i.e., std::hash<uint64_t>()(r))
    }
};

// 64-bit internal pointer
//   NBITS_SIZE: the max number of edges (edge_t) for a single vertex (256M)
//   NBITS_PTR: the max number of edges (edge_t) for the entire gstore (16GB)
//   NBITS_TYPE: sid(0), int(1), float(2), double(4)
enum { NBITS_SIZE = 28 };
enum { NBITS_PTR  = 34 };
enum { NBITS_TYPE =  2 };

struct iptr_t {
uint64_t size: NBITS_SIZE;
uint64_t off: NBITS_PTR;
uint64_t type: NBITS_TYPE;

    iptr_t(): size(0), off(0), type(0) { }

    iptr_t(uint64_t s, uint64_t o, uint64_t t = 0): size(s), off(o), type(t) {
        // no truncated
        assert ((size == s) && (off == o) && (type == t));
    }

    bool operator == (const iptr_t &ptr) {
        if ((size == ptr.size) && (off == ptr.off) && (type == ptr.type))
            return true;
        return false;
    }

    bool operator != (const iptr_t &ptr) {
        return !(operator == (ptr));
    }
};

// 128-bit vertex (key)
struct vertex_t {
    ikey_t key; // 64-bit: vertex | predicate | direction
    iptr_t ptr; // 64-bit: size | offset
};

// 32-bit edge (value)
struct edge_t {
    sid_t val;  // vertex ID

    edge_t &operator = (const edge_t &e) {
        if (this != &e) val = e.val;
        return *this;
    }
};

/**
 * Map the Graph model (e.g., vertex, edge, index) to KVS model (e.g., key, value)
 */
class GStore {
private:
    /// TODO: use more clever cache structure with lock-free implementation
    class RDMA_Cache {
        struct Item {
            pthread_spinlock_t lock;
            vertex_t v;

#if DYNAMIC_GSTORE
            uint64_t expire_time;  // expiring time for dynamic cache
#endif

            Item() {
                pthread_spin_init(&lock, 0);
            }
        };

        static const int NUM_ITEMS = 100000;
        Item items[NUM_ITEMS];
        uint64_t lease;   // lease for dynamic cache

    public:
        RDMA_Cache() { }
        RDMA_Cache(uint64_t lease): lease(lease) { }
        bool lookup(ikey_t key, vertex_t &ret) {
            if (!global_enable_caching)
                return false;

            int idx = key.hash() % NUM_ITEMS;
            bool found = false;
            pthread_spin_lock(&(items[idx].lock));
            if (items[idx].v.key == key) {

#if DYNAMIC_GSTORE
                if (timer::get_usec() < items[idx].expire_time) {
                    ret = items[idx].v;
                    found = true;
                }
#else
                ret = items[idx].v;
                found = true;
#endif
            }
            pthread_spin_unlock(&(items[idx].lock));
            return found;
        }

        void insert(vertex_t &v) {
            if (!global_enable_caching)
                return;
            int idx = v.key.hash() % NUM_ITEMS;
            pthread_spin_lock(&items[idx].lock);

#if DYNAMIC_GSTORE
            items[idx].expire_time = timer::get_usec() + lease;
#endif
            items[idx].v = v;
            pthread_spin_unlock(&items[idx].lock);
        }

        void invalidate(ikey_t key) {
            if (!global_enable_caching)
                return;

            int idx = key.hash() % NUM_ITEMS;
            pthread_spin_lock(&(items[idx].lock));
            if (items[idx].v.key == key)
                items[idx].v.key = ikey_t();
            pthread_spin_unlock(&items[idx].lock);
        }
    };

    static const int NUM_LOCKS = 1024;

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

    int sid;
    Mem *mem;

    vertex_t *vertices;
    uint64_t num_slots;       // 1 bucket = ASSOCIATIVITY slots
    uint64_t num_buckets;     // main-header region (static)
    pthread_spinlock_t bucket_locks[NUM_LOCKS]; // lock virtualization (see paper: vLokc CGO'13)

    uint64_t num_buckets_ext; // indirect-header region (dynamical)
    uint64_t last_ext;
    pthread_spinlock_t bucket_ext_lock;

    bool check_key_exist(ikey_t key) {
        uint64_t bucket_id = key.hash() % num_buckets;
        uint64_t slot_id = bucket_id * ASSOCIATIVITY;
        uint64_t lock_id = bucket_id % NUM_LOCKS;

        pthread_spin_lock(&bucket_locks[lock_id]);
        while (slot_id < num_slots) {
            // the last slot of each bucket is always reserved for pointer to indirect header
            /// TODO: add type info to slot and reuse the last slot to store key
            /// TODO: key.vid is reused to store the bucket_id of indirect header rather than ptr.off,
            ///       since the is_empty() is not robust.
            for (int i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++) {
                //assert(vertices[slot_id].key != key); // no duplicate key
                if (vertices[slot_id].key == key) {
                    pthread_spin_unlock(&bucket_locks[lock_id]);
                    return true;
                }

                // insert to an empty slot
                if (vertices[slot_id].key.is_empty()) {
                    pthread_spin_unlock(&bucket_locks[lock_id]);
                    return false;
                }
            }
            // whether the bucket_ext (indirect-header region) is used
            if (!vertices[slot_id].key.is_empty()) {
                slot_id = vertices[slot_id].key.vid * ASSOCIATIVITY;
                continue; // continue and jump to next bucket
            }
            pthread_spin_unlock(&bucket_locks[lock_id]);
            return false;
        }
    }

    // cluster chaining hash-table (see paper: DrTM SOSP'15)
    uint64_t insert_key(ikey_t key, bool check_dup = true) {
        uint64_t bucket_id = key.hash() % num_buckets;
        uint64_t slot_id = bucket_id * ASSOCIATIVITY;
        uint64_t lock_id = bucket_id % NUM_LOCKS;

        bool found = false;
        pthread_spin_lock(&bucket_locks[lock_id]);
        while (slot_id < num_slots) {
            // the last slot of each bucket is always reserved for pointer to indirect header
            /// TODO: add type info to slot and reuse the last slot to store key
            /// TODO: key.vid is reused to store the bucket_id of indirect header rather than ptr.off,
            ///       since the is_empty() is not robust.
            for (int i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++) {
                //assert(vertices[slot_id].key != key); // no duplicate key
                if (vertices[slot_id].key == key) {
                    if (check_dup) {
                        key.print();
                        vertices[slot_id].key.print();
                        cout << "ERROR: conflict at slot["
                             << slot_id << "] of bucket["
                             << bucket_id << "]" << endl;
                        assert(false);
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
            pthread_spin_lock(&bucket_ext_lock);
            if (last_ext >= num_buckets_ext) {
                cout << "ERROR: out of indirect-header region." << endl;
                assert(last_ext < num_buckets_ext);
            }
            vertices[slot_id].key.vid = num_buckets + (last_ext++);
            pthread_spin_unlock(&bucket_ext_lock);

            slot_id = vertices[slot_id].key.vid * ASSOCIATIVITY; // move to a new bucket_ext
            vertices[slot_id].key = key; // insert to the first slot
            goto done;
        }
done:
        pthread_spin_unlock(&bucket_locks[lock_id]);
        assert(slot_id < num_slots);
        assert(vertices[slot_id].key == key);
        return slot_id;
    }


    edge_t *edges;
    uint64_t num_entries;     // entry region (dynamical)

#if DYNAMIC_GSTORE
    // manage the memory of edges(val)
    Malloc_Interface *edge_allocator;

    /// A size flag is put into the tail of edges (in the entry region) for dynamic cache.
    /// NOTE: the (remote) edges accessed by (local) RDMA cache are valid
    ///       if and only if the size flag of edges is consistent with the size within the pointer.
    static const uint64_t INVALID_EDGES = 1 << NBITS_SIZE;

    inline uint64_t b2e(uint64_t sz) { return sz / sizeof(edge_t); } //byte_to_edge
    inline uint64_t e2b(uint64_t sz) { return sz * sizeof(edge_t); } //edge_to_byte
    inline uint64_t blksz(uint64_t sz) { return b2e(edge_allocator->sz_to_blksz(e2b(sz))); } //unit: edge

    inline void insert_sz(uint64_t n, uint64_t sz, uint64_t off) {
        uint64_t blk_sz = blksz(sz + 1);   // reserve one space for sz
        edges[off + blk_sz - 1].val = n;  // store num of edges in array's tail
    }

    inline bool edge_is_valid(vertex_t &v, edge_t *edge_ptr) {
        if (!global_enable_caching)
            return true;

        uint64_t blk_sz = blksz(v.ptr.size + 1);  // reserve one space for the size of edges
        return (edge_ptr[blk_sz - 1].val == v.ptr.size);   // check whether the sizes are consistent
    }

    /// Defer blk's free operation for dynamic cache.
    /// Pending the free operation when blk is to be collected by add_pending_free()
    /// When allocating a new blk by alloc_edges(), check if pending free's lease expires
    /// and collect free space by sweep_free().
    uint64_t lease;
    struct free_blk {
        uint64_t off;
        uint64_t expire_time;
        free_blk(uint64_t off, uint64_t expire_time): off(off), expire_time(expire_time) { }
    };
    queue<free_blk> free_queue;
    pthread_spinlock_t free_queue_lock;

    inline void add_pending_free(iptr_t ptr) {
        uint64_t expire_time = timer::get_usec() + lease;
        free_blk blk(ptr.off, expire_time);

        pthread_spin_lock(&free_queue_lock);
        free_queue.push(blk);
        pthread_spin_unlock(&free_queue_lock);
    }

    inline void sweep_free() {
        pthread_spin_lock(&free_queue_lock);
        while (!free_queue.empty()) {
            free_blk blk = free_queue.front();
            if (timer::get_usec() < blk.expire_time)
                break;
            edge_allocator->free(e2b(blk.off));
            free_queue.pop();
        }
        pthread_spin_unlock(&free_queue_lock);
    }

    inline uint64_t alloc_edges(uint64_t n, int64_t tid = -1) {
        if (global_enable_caching)
            sweep_free();
        uint64_t sz = e2b(n + 1); // reserve one space for sz
        uint64_t off = b2e(edge_allocator->malloc(sz, tid));
        insert_sz(n, n, off);
        return off;
    }
#else // NOT DYNAMIC_GSTORE
    uint64_t last_entry;
    pthread_spinlock_t entry_lock;

    uint64_t alloc_edges(uint64_t n, int64_t tid = -1) {
        uint64_t orig;
        pthread_spin_lock(&entry_lock);
        orig = last_entry;
        last_entry += n;
        if (last_entry >= num_entries) {
            cout << "ERROR: out of entry region." << endl;
            assert(last_entry < num_entries);
        }
        pthread_spin_unlock(&entry_lock);
        return orig;
    }
#endif // DYNAMIC_GSTORE


#if DYNAMIC_GSTORE
    bool is_dup(vertex_t *v, uint64_t value) {
        int size = v->ptr.size;
        for (int i = 0; i < size; i++)
            if (edges[v->ptr.off + i].val == value)
                return true;
        return false;
    }

    bool insert_vertex_edge(ikey_t key, uint64_t value, bool &dedup_or_isdup) {
        uint64_t bucket_id = key.hash() % num_buckets;
        uint64_t lock_id = bucket_id % NUM_LOCKS;
        uint64_t v_ptr = insert_key(key, false);
        vertex_t *v = &vertices[v_ptr];
        pthread_spin_lock(&bucket_locks[lock_id]);
        if (v->ptr.size == 0) {
            uint64_t off = alloc_edges(1);
            edges[off].val = value;
            vertices[v_ptr].ptr = iptr_t(1, off);
            pthread_spin_unlock(&bucket_locks[lock_id]);
            dedup_or_isdup = false;
            return true;
        } else {

            if (dedup_or_isdup && is_dup(v, value)) {
                pthread_spin_unlock(&bucket_locks[lock_id]);
                return false;
            }
            dedup_or_isdup = false;

            uint64_t need_size = v->ptr.size + 1;
            if (blksz(v->ptr.size) - 1 < need_size) { // need to realloc new blk
                iptr_t old_ptr = v->ptr;

                uint64_t off = alloc_edges(need_size);
                memcpy(&edges[off], &edges[old_ptr.off], e2b(old_ptr.size));
                edges[off + old_ptr.size].val = value;
                insert_sz(INVALID_EDGES, old_ptr.size, old_ptr.off);  // invalidate old blk
                v->ptr = iptr_t(need_size, off);

                if (global_enable_caching)
                    add_pending_free(old_ptr);
                else
                    edge_allocator->free(e2b(old_ptr.off));
            } else {
                insert_sz(need_size, need_size, v->ptr.off);
                edges[v->ptr.off + v->ptr.size].val = value;
                v->ptr.size = need_size;
            }

            pthread_spin_unlock(&bucket_locks[lock_id]);
            return false;
        }
    }
#endif // DYNAMIC_GSTORE

    typedef tbb::concurrent_hash_map<sid_t, vector<sid_t>> tbb_hash_map;

    tbb_hash_map pidx_in_map; // predicate-index (IN)
    tbb_hash_map pidx_out_map; // predicate-index (OUT)
    tbb_hash_map tidx_map; // type-index

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


    RDMA_Cache rdma_cache;

    inline edge_t *rdma_get_edges(int tid, int dst_sid, vertex_t &v) {
        assert(global_use_rdma);

        char *buf = mem->buffer(tid);
        uint64_t r_off  = num_slots * sizeof(vertex_t) + v.ptr.off * sizeof(edge_t);

#if DYNAMIC_GSTORE
        uint64_t r_sz = blksz(v.ptr.size + 1) * sizeof(edge_t);  //get whole blk
#else
        uint64_t r_sz = v.ptr.size * sizeof(edge_t);
#endif

        RDMA &rdma = RDMA::get_rdma();
        rdma.dev->RdmaRead(tid, dst_sid, buf, r_sz, r_off);
        edge_t *result_ptr = (edge_t *)buf;
        return result_ptr;
    }

    vertex_t get_vertex_remote(int tid, ikey_t key) {
        int dst_sid = mymath::hash_mod(key.vid, global_num_servers);
        uint64_t bucket_id = key.hash() % num_buckets;
        vertex_t vert;

        // Currently, we don't support to directly get remote vertex/edge without RDMA
        // TODO: implement it w/o RDMA
        assert(global_use_rdma);

        if (rdma_cache.lookup(key, vert))
            return vert; // found

        char *buf = mem->buffer(tid);
        while (true) {
            uint64_t off = bucket_id * ASSOCIATIVITY * sizeof(vertex_t);
            uint64_t sz = ASSOCIATIVITY * sizeof(vertex_t);

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

    vertex_t get_vertex_local(int tid, ikey_t key) {
        uint64_t bucket_id = key.hash() % num_buckets;
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

    edge_t *get_edges_remote(int tid, sid_t vid, dir_t d, sid_t pid, uint64_t *sz) {
        int dst_sid = mymath::hash_mod(vid, global_num_servers);
        ikey_t key = ikey_t(vid, pid, d);
        edge_t *edge_ptr;
        vertex_t v;

#if DYNAMIC_GSTORE
        v = get_vertex_remote(tid, key);
        if (v.key.is_empty()) {
            *sz = 0;
            return NULL; // not found
        }

        edge_ptr = rdma_get_edges(tid, dst_sid, v);

        while (!edge_is_valid(v, edge_ptr)) { // edge is not valid
            rdma_cache.invalidate(key);
            v = get_vertex_remote(tid, key);
            edge_ptr = rdma_get_edges(tid, dst_sid, v);
        }
#else // NOT DYNAMIC_GSTORE
        v = get_vertex_remote(tid, key);
        if (v.key.is_empty()) {
            *sz = 0;
            return NULL; // not found
        }

        edge_ptr = rdma_get_edges(tid, dst_sid, v);
#endif // DYNAMIC_GSTORE

        *sz = v.ptr.size;
        return edge_ptr;
    }

    edge_t *get_edges_local(int tid, sid_t vid, dir_t d, sid_t pid, uint64_t *sz) {
        ikey_t key = ikey_t(vid, pid, d);
        vertex_t v = get_vertex_local(tid, key);

        if (v.key.is_empty()) {
            *sz = 0;
            return NULL;
        }

        *sz = v.ptr.size;
        uint64_t off = v.ptr.off;
        return &(edges[off]);
    }

    bool get_vertex_attr_remote(int tid, sid_t vid, dir_t d, sid_t pid, attr_t &r) {
        int dst_sid = mymath::hash_mod(vid, global_num_servers);
        ikey_t key = ikey_t(vid, pid, d);
        edge_t *edge_ptr;
        vertex_t v;

#if DYNAMIC_GSTORE
        v = get_vertex_remote(tid, key);
        if (v.key.is_empty()) {
            return false; // not found
        }

        edge_ptr = rdma_get_edges(tid, dst_sid, v);

        while (!edge_is_valid(v, edge_ptr)) { // edge is not valid
            rdma_cache.invalidate(key);
            v = get_vertex_remote(tid, key);
            edge_ptr = rdma_get_edges(tid, dst_sid, v);
        }
#else // NOT DYNAMIC_GSTORE
        v = get_vertex_remote(tid, key);
        if (v.key.is_empty()) {
            return false; // not found
        }

        edge_ptr = rdma_get_edges(tid, dst_sid, v);
#endif // DYNAMIC_GSTORE

        uint64_t r_off  = num_slots * sizeof(vertex_t) + v.ptr.off * sizeof(edge_t);
        switch (v.ptr.type) {
        case INT_t:
            r = *((int *)(&(edges[r_off])));
            break;
        case FLOAT_t:
            r = *((float *)(&(edges[r_off])));
            break;
        case DOUBLE_t:
            r = *((double *)(&(edges[r_off])));
            break;
        default:
            cout << "Not support value type" << endl;
            break;
        }

        return true;
    }

    bool get_vertex_attr_local(int tid, sid_t vid, dir_t d, sid_t pid, attr_t &r) {
        ikey_t key = ikey_t(vid, pid, d);
        vertex_t v = get_vertex_local(tid, key);

        if (v.key.is_empty())
            return false; // not found

        uint64_t off = v.ptr.off;
        switch (v.ptr.type) {
        case INT_t:
            r = *((int *)(&(edges[off])));
            break;
        case FLOAT_t:
            r = *((float *)(&(edges[off])));
            break;
        case DOUBLE_t:
            r = *((double *)(&(edges[off])));
            break;
        default:
            cout << "Not support value type" << endl;
            break;
        }

        return true;
    }

public:
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

    /// GStore: key (main-header and indirect-header region) | value (entry region)
    ///         head region is a cluster chaining hash-table (with associativity)
    ///         entry region is a varying-size array
    GStore(int sid, Mem *mem): sid(sid), mem(mem) {
        uint64_t header_region = mem->kvstore_size() * HD_RATIO / 100;
        uint64_t entry_region = mem->kvstore_size() - header_region;

        // header region
        num_slots = header_region / sizeof(vertex_t);
        num_buckets = mymath::hash_prime_u64((num_slots / ASSOCIATIVITY) * MHD_RATIO / 100);
        num_buckets_ext = (num_slots / ASSOCIATIVITY) - num_buckets;

        // entry region
        num_entries = entry_region / sizeof(edge_t);
#if DYNAMIC_GSTORE
        edge_allocator = new Buddy_Malloc();
        pthread_spin_init(&free_queue_lock, 0);
        lease = SEC(120);
        rdma_cache = RDMA_Cache(lease);
#else
        pthread_spin_init(&entry_lock, 0);
#endif

        cout << "INFO: gstore = " << mem->kvstore_size() << " bytes " << std::endl
             << "      header region: " << num_slots << " slots"
             << " (main = " << num_buckets << ", indirect = " << num_buckets_ext << ")" << std::endl
             << "      entry region: " << num_entries << " entries" << std::endl;

        vertices = (vertex_t *)(mem->kvstore());
        edges = (edge_t *)(mem->kvstore() + num_slots * sizeof(vertex_t));

        pthread_spin_init(&bucket_ext_lock, 0);
        for (int i = 0; i < NUM_LOCKS; i++)
            pthread_spin_init(&bucket_locks[i], 0);
    }

    void refresh() {
        #pragma omp parallel for num_threads(global_num_engines)
        for (uint64_t i = 0; i < num_slots; i++) {
            vertices[i].key = ikey_t();
            vertices[i].ptr = iptr_t();
        }

        last_ext = 0;

#if DYNAMIC_GSTORE
        edge_allocator->init((void *)edges, num_entries * sizeof(edge_t), global_num_engines);
#else
        last_entry = 0;
#endif
    }

    /// skip all TYPE triples (e.g., <http://www.Department0.University0.edu> rdf:type ub:University)
    /// because Wukong treats all TYPE triples as index vertices. In addition, the triples in triple_ops
    /// has been sorted by the vid of object, and IDs of types are always smaller than normal vertex IDs.
    /// Consequently, all TYPE triples are aggregated at the beggining of triple_ops
    void insert_normal(vector<triple_t> &spo, vector<triple_t> &ops, int tid) {
        // treat type triples as index vertices
        uint64_t type_triples = 0;
        while (type_triples < ops.size() && is_tpid(ops[type_triples].o))
            type_triples++;

#ifdef VERSATILE
        /// The following code is used to support a rare case where the predicate is unknown
        /// (e.g., <http://www.Department0.University0.edu> ?P ?O). Each normal vertex should
        /// add two key/value pairs with a reserved ID (i.e., PREDICATE_ID) as the predicate
        /// to store the IN and OUT lists of its predicates.
        /// e.g., key=(vid, PREDICATE_ID, IN/OUT), val=(predicate0, predicate1, ...)
        ///
        /// NOTE, it is disabled by default in order to save memory.
        vector<sid_t> predicates;
#endif // VERSATILE

        uint64_t s = 0;
        while (s < spo.size()) {
            // predicate-based key (subject + predicate)
            uint64_t e = s + 1;
            while ((e < spo.size())
                    && (spo[s].s == spo[e].s)
                    && (spo[s].p == spo[e].p))  { e++; }

            // allocate a vertex and edges
            ikey_t key = ikey_t(spo[s].s, spo[s].p, OUT);
            uint64_t off = alloc_edges(e - s, tid);

            // insert a vertex
            uint64_t slot_id = insert_key(key);
            iptr_t ptr = iptr_t(e - s, off);
            vertices[slot_id].ptr = ptr;

            // insert edges
            for (uint64_t i = s; i < e; i++)
                edges[off++].val = spo[i].o;

#ifdef VERSATILE
            // add a new predicate
            predicates.push_back(spo[s].p);

            // insert a special PREDICATE triple (OUT)
            if (e >= spo.size() || spo[s].s != spo[e].s) {
                // allocate a vertex and edges
                ikey_t key = ikey_t(spo[s].s, PREDICATE_ID, OUT);
                uint64_t sz = predicates.size();
                uint64_t off = alloc_edges(sz, tid);

                // insert a vertex
                uint64_t slot_id = insert_key(key);
                iptr_t ptr = iptr_t(sz, off);
                vertices[slot_id].ptr = ptr;

                // insert edges
                for (auto const &p : predicates)
                    edges[off++].val = p;

                predicates.clear();
            }
#endif // VERSATILE

            s = e;
        }

        s = type_triples; // skip type triples
        while (s < ops.size()) {
            // predicate-based key (object + predicate)
            uint64_t e = s + 1;
            while ((e < ops.size())
                    && (ops[s].o == ops[e].o)
                    && (ops[s].p == ops[e].p)) { e++; }

            // allocate a vertex and edges
            ikey_t key = ikey_t(ops[s].o, ops[s].p, IN);
            uint64_t off = alloc_edges(e - s, tid);

            // insert a vertex
            uint64_t slot_id = insert_key(key);
            iptr_t ptr = iptr_t(e - s, off);
            vertices[slot_id].ptr = ptr;

            // insert edges
            for (uint64_t i = s; i < e; i++)
                edges[off++].val = ops[i].s;

#ifdef VERSATILE
            // add a new predicate
            predicates.push_back(ops[s].p);

            // insert a special PREDICATE triple (OUT)
            if (e >= ops.size() || ops[s].o != ops[e].o) {
                // allocate a vertex and edges
                ikey_t key = ikey_t(ops[s].o, PREDICATE_ID, IN);
                uint64_t sz = predicates.size();
                uint64_t off = alloc_edges(sz, tid);

                // insert a vertex
                uint64_t slot_id = insert_key(key);
                iptr_t ptr = iptr_t(sz, off);
                vertices[slot_id].ptr = ptr;

                // insert edges
                for (auto const &p : predicates)
                    edges[off++].val = p;

                predicates.clear();
            }
#endif // VERSATILE
            s = e;
        }
    }

    void insert_index() {
        uint64_t t1 = timer::get_usec();

        cout << " start (parallel) prepare index info " << endl;

#if DYNAMIC_GSTORE
        edge_allocator->merge_freelists();
#endif
        // scan raw data to generate index data in parallel
        #pragma omp parallel for num_threads(global_num_engines)
        for (uint64_t bucket_id = 0; bucket_id < num_buckets + last_ext; bucket_id++) {
            uint64_t slot_id = bucket_id * ASSOCIATIVITY;
            for (int i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++) {
                // skip empty slot
                if (vertices[slot_id].key.is_empty()) break;

                sid_t vid = vertices[slot_id].key.vid;
                sid_t pid = vertices[slot_id].key.pid;

                uint64_t sz = vertices[slot_id].ptr.size;
                uint64_t off = vertices[slot_id].ptr.off;

                if (vertices[slot_id].key.dir == IN) {
                    if (pid == PREDICATE_ID) {
#ifdef VERSATILE
                        // every subject/object has at least one predicate or one type
                        v_set.insert(vid); // collect all local objects w/ predicate
                        for (uint64_t e = 0; e < sz; e++)
                            p_set.insert(edges[off + e].val); // collect all local predicates
#endif
                    } else if (pid == TYPE_ID) {
                        assert(false); // (IN) type triples should be skipped
                    } else { // predicate-index (OUT) vid
                        tbb_hash_map::accessor a;
                        pidx_out_map.insert(a, pid);
                        a->second.push_back(vid);
                    }
                } else {
                    if (pid == PREDICATE_ID) {
#ifdef VERSATILE
                        // every subject/object has at least one predicate or one type
                        v_set.insert(vid); // collect all local subjects w/ predicate
                        for (uint64_t e = 0; e < sz; e++)
                            p_set.insert(edges[off + e].val); // collect all local predicates
#endif
                    } else if (pid == TYPE_ID) {
#ifdef VERSATILE
                        // every subject/object has at least one predicate or one type
                        v_set.insert(vid); // collect all local subjects w/ type
#endif
                        // type-index (IN) vid
                        for (uint64_t e = 0; e < sz; e++) {
                            tbb_hash_map::accessor a;
                            tidx_map.insert(a, edges[off + e].val);
                            a->second.push_back(vid);
#ifdef VERSATILE
                            t_set.insert(edges[off + e].val); // collect all local types
#endif
                        }
                    } else { // predicate-index (IN) vid
                        tbb_hash_map::accessor a;
                        pidx_in_map.insert(a, pid);
                        a->second.push_back(vid);
                    }
                }
            }
        }
        uint64_t t2 = timer::get_usec();
        cout << (t2 - t1) / 1000 << " ms for (parallel) prepare index info" << endl;

        /// TODO: parallelize index insertion

        // add type/predicate index vertices
        insert_index_map(tidx_map, IN);
        insert_index_map(pidx_in_map, IN);
        insert_index_map(pidx_out_map, OUT);

        tbb_hash_map().swap(pidx_in_map);
        tbb_hash_map().swap(pidx_out_map);
        tbb_hash_map().swap(tidx_map);

#ifdef VERSATILE
        insert_index_set(v_set, TYPE_ID, IN);
        insert_index_set(t_set, TYPE_ID, OUT);
        insert_index_set(p_set, PREDICATE_ID, OUT);

        tbb_unordered_set().swap(v_set);
        tbb_unordered_set().swap(t_set);
        tbb_unordered_set().swap(p_set);
#endif

        uint64_t t3 = timer::get_usec();
        cout << (t3 - t2) / 1000 << " ms for insert index data into gstore" << endl;
    }

#if DYNAMIC_GSTORE
    void insert_triple_out(const triple_t &triple, bool check_dup) {
        bool dedup_or_isdup = check_dup;
        bool nodup = false;
        if (triple.p == TYPE_ID) {
            dedup_or_isdup = true;  // for TYPE_ID condition, dedup is always needed for LUBM benchmark, maybe for others,too.
            ikey_t key = ikey_t(triple.s, triple.p, OUT);
            if (insert_vertex_edge(key, triple.o, dedup_or_isdup)) { // <1> vid's type (7) [need dedup]
#ifdef VERSATILE
                key = ikey_t(triple.s, PREDICATE_ID, OUT);
                ikey_t buddy_key = ikey_t(triple.s, PREDICATE_ID, IN);
                if (insert_vertex_edge(key, triple.p, nodup) && !check_key_exist(buddy_key)) { // <2> vid's predicate, value is TYPE_ID (*8) [dedup from <1>]
                    key = ikey_t(0, TYPE_ID, IN);
                    insert_vertex_edge(key, triple.s, nodup); // <3> the index to vid (*3) [dedup from <2>]
                }
#endif // VERSATILE
            }
            if (!dedup_or_isdup) {
                key = ikey_t(0, triple.o, IN);
                if (insert_vertex_edge(key, triple.s, nodup)) { // <4> type-index (2) [if <1>'s result is not dup, this is not dup, too]
#ifdef VERSATILE
                    key = ikey_t(0, TYPE_ID, OUT);
                    insert_vertex_edge(key, triple.o, nodup);// <5> index to this type (*4) [dedup from <4>]
#endif // VERSATILE
                }
            }
        } else {
            ikey_t key = ikey_t(triple.s, triple.p, OUT);
            if (insert_vertex_edge(key, triple.o, dedup_or_isdup)) {  // <6> vid's ngbrs w/ predicate (6) [need dedup]
                key = ikey_t(0, triple.p, IN);
                ikey_t buddy_key = ikey_t(0, triple.p, OUT);
                if (insert_vertex_edge(key, triple.s, nodup) && !check_key_exist(buddy_key)) { // <7> predicate-index (1) [dedup from <6>]
#ifdef VERSATILE
                    key = ikey_t(0, PREDICATE_ID, OUT);
                    insert_vertex_edge(key, triple.p, nodup); // <8> the index to predicate (*5) [dedup from <7>]
#endif // VERSATILE
                }
#ifdef VERSATILE
                key = ikey_t(triple.s, PREDICATE_ID, OUT);
                buddy_key = ikey_t(triple.s, PREDICATE_ID, IN);
                if (insert_vertex_edge(key, triple.p, nodup) && !check_key_exist(buddy_key)) { // <9> vid's predicate (*8) [dedup from <6>]
                    key = ikey_t(0, TYPE_ID, IN);
                    insert_vertex_edge(key, triple.s, nodup); // <10> the index to vid (*3) [dedup from <9>]
                }
#endif // VERSATILE
            }
        }
    }

    void insert_triple_in(const triple_t &triple, bool check_dup) {
        bool dedup_or_isdup = check_dup;
        bool nodup = false;
        if (triple.p == TYPE_ID) // skipped
            return;
        ikey_t key = ikey_t(triple.o, triple.p, IN);
        if (insert_vertex_edge(key, triple.s, dedup_or_isdup)) {  // <1> vid's ngbrs w/ predicate (6) [need dedup]
            // key doesn't exist before
            key = ikey_t(0, triple.p, OUT);
            ikey_t buddy_key = ikey_t(0, triple.p, IN);
            if (insert_vertex_edge(key, triple.o, nodup) && !check_key_exist(buddy_key)) { // <2> predicate-index (1) [dedup from <1>]
#ifdef VERSATILE
                key = ikey_t(0, PREDICATE_ID, OUT);
                insert_vertex_edge(key, triple.p, nodup); // <3> the index to predicate (*5) [dedup from <2>]
#endif // VERSATILE
            }
#ifdef VERSATILE
            key = ikey_t(triple.o, PREDICATE_ID, IN);
            buddy_key = ikey_t(triple.o, PREDICATE_ID, OUT);
            if (insert_vertex_edge(key, triple.p, nodup) && !check_key_exist(buddy_key)) { // <4> vid's predicate (*8) [dedup from <1>]
                key = ikey_t(0, TYPE_ID, IN);
                insert_vertex_edge(key, triple.o, nodup); // <5> the index to vid (*3) [dedup from <4>]
            }
#endif // VERSATILE
        }
    }

#endif // DYNAMIC_GSTORE

    uint64_t ivertex_num = 0;
    uint64_t nvertex_num = 0;

    void outdir_idx_to_normal(ikey_t key, bool check) {
        if (!check)
            return;
        ivertex_num ++;
        uint64_t vsz = 0;
        edge_t *vres = get_edges_local(0, key.vid, (dir_t)key.dir, key.pid, &vsz); //(1)[IN]/(2)
        for (int i = 0; i < vsz; i++) {
            uint64_t tsz = 0;
            edge_t *tres = get_edges_local(0, vres[i].val, OUT, TYPE_ID, &tsz);
            bool found = false;
            for (int j = 0; j < tsz; j++) {     //(2)
                if (tres[j].val == key.pid && !found)
                    found = true;
                else if (tres[j].val == key.pid && found) {
                    cout << "Error: " << "In the value part of normal key/value pair [ " << key.vid
                         << " | TYPE_ID | OUT] there is duplicate type " << key.pid << endl;
                }
            }
            if (tsz != 0 && !found) {  //(1)[IN]
                if (get_vertex_local(0, ikey_t(vres[i].val, key.pid, OUT)).key.is_empty()) {
                    cout << "Error: if " << key.pid << " is type id, then there is no type "
                         << key.pid << "in normal key/value pair [" << key.vid << " | TYPE_ID | OUT] 's value part" << endl;
                    cout << "And if " << key.pid << " is predicate id, then there is no key called "
                         << "[ " << vres[i].val << " | " << key.pid << " | " << "] exist" << endl;
                }
            }
        }
    }

    void indir_pidx_to_np(ikey_t key, bool check) {
        if (!check)
            return;
        ivertex_num ++;
        uint64_t vsz = 0;
        edge_t *vres = get_edges_local(0, key.vid, (dir_t)key.dir, key.pid, &vsz);//predicate index
        for (int i = 0; i < vsz; i++) {
            if (get_vertex_local(0, ikey_t(vres[i].val, key.pid, IN)).key.is_empty()) {
                cout << "Error: key " << " [ " << vres[i].val << " | "
                     << key.pid << " | " << " IN ] does not exist." << endl;
            }
        }
    }

    void nt_to_tidx(ikey_t key, bool check) {
        if (!check)
            return;
        nvertex_num ++;
        uint64_t tsz = 0;
        edge_t *tres = get_edges_local(0, key.vid, (dir_t)key.dir, key.pid, &tsz);
        for (int i = 0; i < tsz; i++) {
            uint64_t vsz = 0;
            edge_t *vres = get_edges_local(0, 0, IN, tres[i].val, &vsz);
            bool found = false;
            for (int j = 0; j < vsz; j++) {
                if (vres[j].val == key.vid && !found)
                    found = true;
                else if (vres[j].val == key.vid && found) {
                    cout << "Error: " << "In the value part of type index [ 0 | " << tres[i].val
                         << " | IN ]" << " there is duplicate value " << key.vid << endl;
                }
            }
            if (!found) {
                cout << "Error: " << "In the value part of type index [ 0 | " << tres[i].val
                     << " | IN ]" << " there is no value " << key.vid << endl;
            }
        }
    }

    void np_to_pidx(ikey_t key, dir_t dir, bool check) {
        if (!check)
            return;
        nvertex_num ++;
        uint64_t vsz = 0;
        edge_t *vres = get_edges_local(0, 0, dir, key.pid, &vsz);
        bool found = false;
        for (int i = 0; i < vsz; i++) {
            if (vres[i].val == key.vid && !found)
                found = true;
            else if (vres[i].val == key.vid && found) {
                cout << "Error: " << "In the value part of predicate index [ 0 | " << key.pid
                     << " | " << dir << " ]" << " there is duplicate value " << key.vid << endl;
                break;
            }
        }
        if (!found) {
            cout << "Error: " << "In the value part of predicate index [ 0 | " << key.pid
                 << " | " << dir << " ]" << " there is no value " << key.vid << endl;
        }
    }

#ifdef VERSATILE
    void outdir_idx_vercheck(ikey_t key, bool check) {
        if (!check)
            return;
        uint64_t vsz = 0;
        edge_t *vres = get_edges_local(0, 0, OUT, TYPE_ID, &vsz); //(1)[IN]/(2)
        bool found = false;
        for (int i = 0; i < vsz; i++) {
            if (vres[i].val == key.pid && !found)
                found = true;
            else if (vres[i].val == key.pid && found) {
                cout << "Error: " << "In the value part of all local types [ 0 | TYPE_ID | OUT ]"
                     << " there is duplicate value " << key.pid << endl;
            }
        }
        if (!found) {
            uint64_t psz = 0;
            edge_t *pres = get_edges_local(0, 0, OUT, PREDICATE_ID, &psz);
            bool found = false;
            for (int i = 0; i < psz; i++) {
                if (pres[i].val == key.pid && !found)
                    found = true;
                else if (pres[i].val == key.pid && found) {
                    cout << "Error: " << "In the value part of all local predicates [ 0 | PREDICATE_ID | OUT ]"
                         << " there is duplicate value " << key.pid << endl;
                    break;
                }
            }
            if (!found) {
                cout << "Error: " << "if " << key.pid << "is predicate, in the value part of all local predicates [ 0 | PREDICATE_ID | OUT ]"
                     << " there is no value " << key.pid << endl;
                cout << "Error: " << "if " << key.pid << " is type, in the value part of all local types [ 0 | TYPE_ID | OUT ]"
                     << " there is no value " << key.pid << endl;
            }
            uint64_t vsz = 0;
            edge_t *vres = get_edges_local(0, 0, IN, key.pid, &vsz);
            if (vsz == 0) {
                cout << "Error: " << "if " << key.pid << " is type, in the value part of all local types [ 0 | TYPE_ID | OUT ]"
                     << " there is no value " << key.pid << endl;
                return;
            }
            for (int i = 0; i < vsz; i++) {
                found = false;
                uint64_t sosz = 0;
                edge_t *sores = get_edges_local(0, 0, IN, TYPE_ID, &sosz);
                for (int j = 0; j < sosz; j++) {
                    if (sores[j].val == vres[i].val && !found)
                        found = true;
                    else if (sores[j].val == vres[i].val && found) {
                        cout << "Error: " << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                             << " there is duplicate value " << vres[i].val << endl;
                        break;
                    }
                }
                if (!found) {
                    cout << "Error: " << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                         << " there is no value " << vres[i].val << endl;
                }
                found = false;
                uint64_t p2sz = 0;
                edge_t *p2res = get_edges_local(0, vres[i].val, OUT, PREDICATE_ID, &p2sz);
                for (int j = 0; j < p2sz; j++) {
                    if (p2res[j].val == key.pid && !found)
                        found = true;
                    else if (p2res[j].val == key.pid && found) {
                        cout << "Error: " << "In the value part of " << vres[i].val << "'s all predicates [ "
                             << vres[i].val << " | PREDICATE_ID | OUT ], there is duplicate value " << key.pid << endl;
                        break;
                    }
                }
                if (!found) {
                    cout << "Error: " << "In the value part of " << vres[i].val << "'s all predicates [ "
                         << vres[i].val << " | PREDICATE_ID | OUT ], there is no value " << key.pid << endl;
                }
            }
        }
    }

    void indir_pidx_vercheck(ikey_t key, bool check) {
        if (!check)
            return;
        uint64_t psz = 0;
        edge_t *pres = get_edges_local(0, 0, OUT, PREDICATE_ID, &psz);
        bool found = false;
        for (int i = 0; i < psz; i++) {
            if (pres[i].val == key.pid && !found)
                found = true;
            else if (pres[i].val == key.pid && found) {
                cout << "Error: " << "In the value part of all local predicates [ 0 | PREDICATE_ID | OUT ]"
                     << " there is duplicate value " << key.pid << endl;
                break;
            }
        }
        if (!found) {
            cout << "Error: " << "In the value part of all local predicates [ 0 | PREDICATE_ID | OUT ]"
                 << " there is no value " << key.pid << endl;
        }
        uint64_t vsz = 0;
        edge_t *vres = get_edges_local(0, 0, OUT, key.pid, &vsz);
        for (int i = 0; i < vsz; i++) {
            found = false;
            uint64_t sosz = 0;
            edge_t *sores = get_edges_local(0, 0, IN, TYPE_ID, &sosz);
            for (int j = 0; j < sosz; j++) {
                if (sores[j].val == vres[i].val && !found)
                    found = true;
                else if (sores[j].val == vres[i].val && found) {
                    cout << "Error: " << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                         << " there is duplicate value " << vres[i].val << endl;
                    break;
                }
            }
            if (!found) {
                cout << "Error: " << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                     << " there is no value " << vres[i].val << endl;
            }
            found = false;
            uint64_t psz = 0;
            edge_t *pres = get_edges_local(0, vres[i].val, IN, PREDICATE_ID, &psz);
            for (int j = 0; j < psz; j++) {
                if (pres[j].val == key.pid && !found)
                    found = true;
                else if (pres[j].val == key.pid && found) {
                    cout << "Error: " << "In the value part of " << vres[i].val << "'s all predicates [ "
                         << vres[i].val << "PREDICATE_ID | IN ], there is duplicate value " << key.pid << endl;
                    break;
                }
            }
            if (!found) {
                cout << "Error: " << "In the value part of " << vres[i].val << "'s all predicates [ "
                     << vres[i].val << "PREDICATE_ID | IN ], there is no value " << key.pid << endl;
            }
        }
    }

    void nt_vercheck(ikey_t key, bool check) {
        if (!check)
            return;
        bool found = false;
        uint64_t psz = 0;
        edge_t *pres = get_edges_local(0, key.vid, OUT, PREDICATE_ID, &psz);
        for (int i = 0; i < psz; i++) {
            if (pres[i].val == key.pid && !found)
                found = true;
            else if (pres[i].val == key.pid && found) {
                cout << "Error: " << "In the value part of " << key.vid << "'s all predicates [ "
                     << key.vid << "PREDICATE_ID | OUT ], there is duplicate value " << key.pid << endl;
                break;
            }
        }
        if (!found) {
            cout << "Error: " << "In the value part of " << key.vid << "'s all predicates [ "
                 << key.vid << "PREDICATE_ID | OUT ], there is no value " << key.pid << endl;
        }
        found = false;
        uint64_t ossz = 0;
        edge_t *osres = get_edges_local(0, 0, IN, key.pid, &ossz);
        for (int i = 0; i < ossz; i++) {
            if (osres[i].val == key.vid && !found)
                found = true;
            else if (osres[i].val == key.vid && found) {
                cout << "Error: " << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                     << " there is duplicate value " << key.vid << endl;
                break;
            }
        }
        if (!found) {
            cout << "Error: " << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                 << " there is no value " << key.vid << endl;
        }
    }
#endif // VERSATILE

    void check_on_vertex(ikey_t key, bool index_check, bool normal_check) {
        if (key.vid == 0 && is_tpid(key.pid) && key.dir == IN) {
            outdir_idx_to_normal(key, index_check); // (2)/(1)[IN] -->(7)/(6)
#ifdef VERSATILE
            outdir_idx_vercheck(key, index_check); //
#endif
        } else if (key.vid == 0 && is_tpid(key.pid) && key.dir == OUT) {
            indir_pidx_to_np(key, index_check); // (1)[OUT]-->(6)
#ifdef VERSATILE
            indir_pidx_vercheck(key, index_check); //
#endif
        } else if (is_vid(key.vid) && key.pid == TYPE_ID && key.dir == OUT) {
            nt_to_tidx(key, normal_check); // (7)-->(2)
#ifdef VERSATILE
            nt_vercheck(key, index_check); //
#endif
        } else if (is_vid(key.vid) && is_tpid(key.pid) && key.dir == OUT) {
            np_to_pidx(key, IN, normal_check); // (6)[OUT]-->(1)
        } else if (is_vid(key.vid) && is_tpid(key.pid) && key.dir == IN) {
            np_to_pidx(key, OUT, normal_check); // (6)[IN]-->(1)
        }
    }

    int gstore_check(bool index_check, bool normal_check) {
        cout << "Graph storage intergity check has started on server " << sid << endl;
        ivertex_num = 0;
        nvertex_num = 0;
        for (uint64_t bucket_id = 0; bucket_id < num_buckets + num_buckets_ext; bucket_id++) {
            uint64_t slot_id = bucket_id * ASSOCIATIVITY;
            for (int i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++) {
                if (!vertices[slot_id].key.is_empty()) {
                    check_on_vertex(vertices[slot_id].key, index_check, normal_check);
                }
            }
        }
        cout << "Server " << sid << " has checked " << ivertex_num << " index vertices"
             << " and " << nvertex_num << " normal vertices." << endl;
        return 0;
    }

    // FIXME: refine parameters with vertex_t
    edge_t *get_edges_global(int tid, sid_t vid, dir_t d, sid_t pid, uint64_t *sz) {
        if (mymath::hash_mod(vid, global_num_servers) == sid)
            return get_edges_local(tid, vid, d, pid, sz);
        else
            return get_edges_remote(tid, vid, d, pid, sz);
    }

    edge_t *get_index_edges_local(int tid, sid_t pid, dir_t d, uint64_t *sz) {
        // the vid of index vertex should be 0
        return get_edges_local(tid, 0, d, pid, sz);
    }

    // insert vertex attributes
    void insert_vertex_attr(vector<triple_attr_t> &attrs, int64_t tid) {
        for (auto const &attr : attrs) {
            // allocate a vertex and edges
            ikey_t key = ikey_t(attr.s, attr.a, OUT);
            int type = boost::apply_visitor(get_type, attr.v);
            uint64_t sz = (get_sizeof(type) - 1) / sizeof(edge_t) + 1;   // get the ceil size;
            uint64_t off = alloc_edges(sz, tid);

            // insert a vertex
            uint64_t slot_id = insert_key(key);
            iptr_t ptr = iptr_t(sz, off, type);
            vertices[slot_id].ptr = ptr;

            // insert edges (attributes)
            switch (type) {
            case INT_t:
                *(int *)(edges + off) = boost::get<int>(attr.v);
                break;
            case FLOAT_t:
                *(float *)(edges + off) = boost::get<float>(attr.v);
                break;
            case DOUBLE_t:
                *(double *)(edges + off) = boost::get<double>(attr.v);
                break;
            default:
                cout << "Unsupported value type of attribute" << endl;
            }
        }
    }

    // get vertex attributes
    bool get_vertex_attr_global(int tid, sid_t vid, dir_t d , sid_t pid, attr_t &r) {
        if (sid == mymath::hash_mod(vid, global_num_servers))
            return get_vertex_attr_local(tid, vid, d, pid, r);
        else
            return get_vertex_attr_remote(tid, vid, d, pid, r);
    }

    // prepare data for planner
    void generate_statistic(data_statistic & stat) {
        for (uint64_t bucket_id = 0; bucket_id < num_buckets + num_buckets_ext; bucket_id++) {
            uint64_t slot_id = bucket_id * ASSOCIATIVITY;
            for (int i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++) {
                // skip empty slot
                if (vertices[slot_id].key.is_empty()) continue;

                sid_t vid = vertices[slot_id].key.vid;
                sid_t pid = vertices[slot_id].key.pid;

                uint64_t off = vertices[slot_id].ptr.off;
                if (pid == PREDICATE_ID) continue; // skip for index vertex

                unordered_map<ssid_t, int> &ptcount = stat.predicate_to_triple;
                unordered_map<ssid_t, int> &pscount = stat.predicate_to_subject;
                unordered_map<ssid_t, int> &pocount = stat.predicate_to_object;
                unordered_map<ssid_t, int> &tyscount = stat.type_to_subject;
                unordered_map<ssid_t, vector<direct_p> > &ipcount = stat.id_to_predicate;

                if (vertices[slot_id].key.dir == IN) {
                    uint64_t sz = vertices[slot_id].ptr.size;

                    // triples only count from one direction
                    if (ptcount.find(pid) == ptcount.end())
                        ptcount[pid] = sz;
                    else
                        ptcount[pid] += sz;

                    // count objects
                    if (pocount.find(pid) == pocount.end())
                        pocount[pid] = 1;
                    else
                        pocount[pid]++;

                    // count in predicates for specific id
                    ipcount[vid].push_back(direct_p(IN, pid));
                } else {
                    // count subjects
                    if (pscount.find(pid) == pscount.end())
                        pscount[pid] = 1;
                    else
                        pscount[pid]++;

                    // count out predicates for specific id
                    ipcount[vid].push_back(direct_p(OUT, pid));

                    // count type predicate
                    if (pid == TYPE_ID) {
                        uint64_t sz = vertices[slot_id].ptr.size;
                        uint64_t off = vertices[slot_id].ptr.off;

                        for (uint64_t j = 0; j < sz; j++) {
                            //src may belongs to multiple types
                            sid_t obid = edges[off + j].val;

                            if (tyscount.find(obid) == tyscount.end())
                                tyscount[obid] = 1;
                            else
                                tyscount[obid]++;

                            if (pscount.find(obid) == pscount.end())
                                pscount[obid] = 1;
                            else
                                pscount[obid]++;

                            ipcount[vid].push_back(direct_p(OUT, obid));
                        }
                    }
                }
            }
        }

        //cout<<"sizeof predicate_to_triple = "<<stat.predicate_to_triple.size()<<endl;
        //cout<<"sizeof predicate_to_subject = "<<stat.predicate_to_subject.size()<<endl;
        //cout<<"sizeof predicate_to_object = "<<stat.predicate_to_object.size()<<endl;
        //cout<<"sizeof type_to_subject = "<<stat.type_to_subject.size()<<endl;
        //cout<<"sizeof id_to_predicate = "<<stat.id_to_predicate.size()<<endl;

        unordered_map<pair<ssid_t, ssid_t>, four_num, boost::hash<pair<int, int>>> &ppcount = stat.correlation;

        // do statistic for correlation
        for (unordered_map<ssid_t, vector<direct_p> >::iterator it = stat.id_to_predicate.begin();
                it != stat.id_to_predicate.end(); it++ ) {
            ssid_t vid = it->first;
            vector<direct_p> &vec = it->second;

            for (uint64_t i = 0; i < vec.size(); i++) {
                for (uint64_t j = i + 1; j < vec.size(); j++) {
                    ssid_t p1, d1, p2, d2;
                    if (vec[i].p < vec[j].p) {
                        p1 = vec[i].p;
                        d1 = vec[i].dir;
                        p2 = vec[j].p;
                        d2 = vec[j].dir;
                    } else {
                        p1 = vec[j].p;
                        d1 = vec[j].dir;
                        p2 = vec[i].p;
                        d2 = vec[i].dir;
                    }

                    if (d1 == OUT && d2 == OUT)
                        ppcount[make_pair(p1, p2)].out_out++;

                    if (d1 == OUT && d2 == IN)
                        ppcount[make_pair(p1, p2)].out_in++;

                    if (d1 == IN && d2 == IN)
                        ppcount[make_pair(p1, p2)].in_in++;

                    if (d1 == IN && d2 == OUT)
                        ppcount[make_pair(p1, p2)].in_out++;
                }
            }
        }
        //cout << "sizeof correlation = " << stat.correlation.size() << endl;
        cout << "INFO#" << sid << ": generating stats is finished." << endl;
    }

    // analysis and debuging
    void print_mem_usage() {
        uint64_t used_slots = 0;
        for (uint64_t x = 0; x < num_buckets; x++) {
            uint64_t slot_id = x * ASSOCIATIVITY;
            for (int y = 0; y < ASSOCIATIVITY - 1; y++, slot_id++) {
                if (vertices[slot_id].key.is_empty())
                    continue;
                used_slots++;
            }
        }

        cout << "main header: " << B2MiB(num_buckets * ASSOCIATIVITY * sizeof(vertex_t))
             << " MB (" << num_buckets * ASSOCIATIVITY << " slots)" << endl;
        cout << "\tused: " << 100.0 * used_slots / (num_buckets * ASSOCIATIVITY)
             << " % (" << used_slots << " slots)" << endl;
        cout << "\tchain: " << 100.0 * num_buckets / (num_buckets * ASSOCIATIVITY)
             << " % (" << num_buckets << " slots)" << endl;

        used_slots = 0;
        for (uint64_t x = num_buckets; x < num_buckets + last_ext; x++) {
            uint64_t slot_id = x * ASSOCIATIVITY;
            for (int y = 0; y < ASSOCIATIVITY - 1; y++, slot_id++) {
                if (vertices[slot_id].key.is_empty())
                    continue;
                used_slots++;
            }
        }

        cout << "indirect header: " << B2MiB(num_buckets_ext * ASSOCIATIVITY * sizeof(vertex_t))
             << " MB (" << num_buckets_ext * ASSOCIATIVITY << " slots)" << endl;
        cout << "\talloced: " << 100.0 * last_ext / num_buckets_ext
             << " % (" << last_ext << " buckets)" << endl;
        cout << "\tused: " << 100.0 * used_slots / (num_buckets_ext * ASSOCIATIVITY)
             << " % (" << used_slots << " slots)" << endl;

        cout << "entry: " << B2MiB(num_entries * sizeof(edge_t))
             << " MB (" << num_entries << " entries)" << endl;
#if DYNAMIC_GSTORE
        edge_allocator->print_memory_usage();
#else
        cout << "\tused: " << 100.0 * last_entry / num_entries
             << " % (" << last_entry << " entries)" << endl;
#endif

        uint64_t sz = 0;
        get_edges_local(0, 0, IN, TYPE_ID, &sz);
        cout << "#vertices: " << sz << endl;
        get_edges_local(0, 0, OUT, TYPE_ID, &sz);
        cout << "#predicates: " << sz << endl;
    }
};
