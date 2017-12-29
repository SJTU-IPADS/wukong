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
    class RDMA_Cache {
        struct Item {
            pthread_spinlock_t lock;
            vertex_t v;
            Item() {
                pthread_spin_init(&lock, 0);
            }
        };

        static const int NUM_ITEMS = 100000;
        Item items[NUM_ITEMS];

    public:
        /// TODO: use more clever cache structure with lock-free implementation
        bool lookup(ikey_t key, vertex_t &ret) {
            if (!global_enable_caching)
                return false;

            int idx = key.hash() % NUM_ITEMS;
            bool found = false;
            pthread_spin_lock(&(items[idx].lock));
            if (items[idx].v.key == key) {
                ret = items[idx].v;
                found = true;
            }
            pthread_spin_unlock(&(items[idx].lock));
            return found;
        }

        void insert(vertex_t &v) {
            if (!global_enable_caching)
                return;

            int idx = v.key.hash() % NUM_ITEMS;
            pthread_spin_lock(&items[idx].lock);
            items[idx].v = v;
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

    uint64_t realloc_edges(uint64_t off, uint64_t n) {
        uint64_t ptr = off * sizeof(edge_t);
        uint64_t sz = n * sizeof(edge_t);
        return edge_allocator->realloc(ptr, sz) / sizeof(edge_t);
    }

    uint64_t alloc_edges(uint64_t n, int64_t tid = -1) {
        uint64_t sz = n * sizeof(edge_t);
        return edge_allocator->malloc(sz, tid) / sizeof(edge_t);
    }
#else
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
#endif


#if DYNAMIC_GSTORE
    bool insert_vertex_edge(ikey_t key, uint64_t value) {
        uint64_t bucket_id = key.hash() % num_buckets;
        uint64_t lock_id = bucket_id % NUM_LOCKS;
        uint64_t v_ptr = insert_key(key, false);
        vertex_t *v = &vertices[v_ptr];
        pthread_spin_lock(&bucket_locks[lock_id]);
        if (v->ptr.size == 0) {
            uint64_t off = alloc_edges(1);
            vertices[v_ptr].ptr = iptr_t(1, off);
            edges[off].val = value;
            pthread_spin_unlock(&bucket_locks[lock_id]);
            return true;
        } else {
            uint64_t need_size = v->ptr.size + 1;
            uint64_t off = realloc_edges(v->ptr.off, need_size);
            v->ptr = iptr_t(need_size, off);
            off = v->ptr.off + v->ptr.size - 1;
            edges[off].val = value;
            pthread_spin_unlock(&bucket_locks[lock_id]);
            return false;
        }
    }
#endif

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
#endif

    RDMA_Cache rdma_cache;

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
        vertex_t v = get_vertex_remote(tid, key);

        if (v.key.is_empty()) {
            *sz = 0;
            return NULL; // not found
        }

        // Currently, we don't support to directly get remote vertex/edge without RDMA
        // TODO: implement it w/o RDMA
        assert(global_use_rdma);

        char *buf = mem->buffer(tid);
        uint64_t r_off  = num_slots * sizeof(vertex_t) + v.ptr.off * sizeof(edge_t);
        uint64_t r_sz = v.ptr.size * sizeof(edge_t);

        RDMA &rdma = RDMA::get_rdma();
        rdma.dev->RdmaRead(tid, dst_sid, buf, r_sz, r_off);
        edge_t *result_ptr = (edge_t *)buf;

        *sz = v.ptr.size;
        return result_ptr;
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
        vertex_t v = get_vertex_remote(tid, key);

        if (v.key.is_empty())
            return false; // not found

        // TODO: implement it w/o RDMA
        assert(global_use_rdma);

        char *buf = mem->buffer(tid);
        uint64_t r_off = num_slots * sizeof(vertex_t) + v.ptr.off * sizeof(edge_t);
        uint64_t r_sz = v.ptr.size * sizeof(edge_t);

        RDMA &rdma = RDMA::get_rdma();
        rdma.dev->RdmaRead(tid, dst_sid, buf, r_sz, r_off);
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
    // encoding rules of GStore
    // subject/object (vid) >= 2^NBITS_IDX, 2^NBITS_IDX > predicate/type (p/tid) >= 2^1,
    // TYPE_ID = 1, PREDICATE_ID = 0, OUT = 1, IN = 0
    //
    // Empty key
    // (0)   key = [  0 |            0 |      0]  value = [vid0, vid1, ..]  i.e., init
    // INDEX key/value pair
    // (1)   key = [  0 |          pid | IN/OUT]  value = [vid0, vid1, ..]  i.e., predicate-index
    // (2)   key = [  0 |          tid |     IN]  value = [vid0, vid1, ..]  i.e., type-index
    // (*3)  key = [  0 |      TYPE_ID |     IN]  value = [vid0, vid1, ..]  i.e., all local objects/subjects
    // (*4)  key = [  0 |      TYPE_ID |    OUT]  value = [pid0, pid1, ..]  i.e., all local types
    // (*5)  key = [  0 | PREDICATE_ID |    OUT]  value = [pid0, pid1, ..]  i.e., all local predicates
    // NORMAL key/value pair
    // (6)   key = [vid |          pid | IN/OUT]  value = [vid0, vid1, ..]  i.e., vid's ngbrs w/ predicate
    // (7)   key = [vid |      TYPE_ID |    OUT]  value = [tid0, tid1, ..]  i.e., vid's all types
    // (*8)  key = [vid | PREDICATE_ID | IN/OUT]  value = [pid0, pid1, ..]  i.e., vid's all predicates
    //
    // < S,  P, ?O>  ?O : (6)
    // <?S,  P,  O>  ?S : (6)
    // < S,  1, ?T>  ?T : (7)
    // <?S,  1,  T>  ?S : (2)
    // < S, ?P,  O>  ?P : (8)
    //
    // <?S,  P, ?O>  ?S : (1)
    //               ?O : (1)
    // <?S,  1, ?O>  ?O : (4)
    //               ?S : (4) +> (2)
    // < S, ?P, ?O>  ?P : (8) AND exist(7)
    //               ?O : (8) AND exist(7) +> (6)
    // <?S, ?P,  O>  ?P : (8)
    //               ?S : (8) +> (6)
    // <?S, ?P,  T>  ?P : exist(2)
    //               ?S : (2)
    //
    // <?S, ?P, ?O>  ?S : (3)
    //               ?O : (3) AND (4)
    //               ?P : (5)
    //               ?S ?P ?O : (3) +> (7) AND (8) +> (6)

    // GStore: key (main-header and indirect-header region) | value (entry region)
    //         head region is a cluster chaining hash-table (with associativity)
    //         entry region is a varying-size array
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

    // skip all TYPE triples (e.g., <http://www.Department0.University0.edu> rdf:type ub:University)
    // because Wukong treats all TYPE triples as index vertices. In addition, the triples in triple_ops
    // has been sorted by the vid of object, and IDs of types are always smaller than normal vertex IDs.
    // Consequently, all TYPE triples are aggregated at the beggining of triple_ops
    void insert_normal(vector<triple_t> &spo, vector<triple_t> &ops, int tid) {
        // treat type triples as index vertices
        uint64_t type_triples = 0;
        while (type_triples < ops.size() && is_tpid(ops[type_triples].o))
            type_triples++;

#ifdef VERSATILE
        // The following code is used to support a rare case where the predicate is unknown
        // (e.g., <http://www.Department0.University0.edu> ?P ?O). Each normal vertex should
        // add two key/value pairs with a reserved ID (i.e., PREDICATE_ID) as the predicate
        // to store the IN and OUT lists of its predicates.
        // e.g., key=(vid, PREDICATE_ID, IN/OUT), val=(predicate0, predicate1, ...)
        //
        // NOTE, it is disabled by default in order to save memory.
        vector<sid_t> predicates;
#endif

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
#endif

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
#endif
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

#if DYNAMIC_GSTORE    //with bug, fix it later
    void insert_triple_out(const triple_t &triple) {
        // (s, ->, p) ---> normal vertex(o)
        // (s, ->, 0) ---> predicts
        // (p, <-, 0) ---> normal vertex(s)
        ikey_t key = ikey_t(triple.s, triple.p, OUT);
        if (insert_vertex_edge(key, triple.o)) {
            // key doesn't exist before
            if (triple.p == PREDICATE_ID) {
#ifdef VERSATILE
                key = ikey_t(0, TYPE_ID, IN);
                insert_vertex_edge(key, triple.s);
                key = ikey_t(0, TYPE_ID, OUT);
                insert_vertex_edge(key, triple.o);
#endif
            } else if (triple.p == TYPE_ID) {
                key = ikey_t(0, triple.o, IN);
                insert_vertex_edge(key, triple.s);
            } else {
                key = ikey_t(0, triple.p, IN);
                insert_vertex_edge(key, triple.o); // triple.s
            }
        }
    }

    void insert_triple_in(const triple_t &triple) {
        // (o, <-, p) ---> normal vertex(s)
        // (o, <-, 0) ---> predicts
        // (p, ->, 0) ---> normal vertex(o)
        ikey_t key = ikey_t(triple.o, triple.p, IN);
        if (insert_vertex_edge(key, triple.s)) {
            // key doesn't exist before
            if (triple.p == PREDICATE_ID) {
#ifdef VERSATILE
                key = ikey_t(0, TYPE_ID, IN);
                insert_vertex_edge(key, triple.o);
                key = ikey_t(0, TYPE_ID, OUT);
                insert_vertex_edge(key, triple.s);
#endif
            } else if (triple.p == TYPE_ID) {  // insert unused
                return;
            } else {
                key = ikey_t(0, triple.p, OUT);
                insert_vertex_edge(key, triple.s); // triple.o
            }
        }
    }
#endif

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

        unordered_map<pair<ssid_t, ssid_t>, four_num, boost::hash<pair<int, int> > > &ppcount = stat.correlation;

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
