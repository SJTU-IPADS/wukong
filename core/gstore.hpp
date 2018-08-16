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
#include <atomic>


#include "global.hpp"
#include "rdma.hpp"
#include "data_statistic.hpp"
#include "type.hpp"

#include "mm/malloc_interface.hpp"
#include "mm/jemalloc.hpp"
#include "mm/buddy_malloc.hpp"

#ifdef USE_GPU
#include "rdf_meta.hpp"
#endif // USE_GPU

#include "math.hpp"
#include "timer.hpp"
#include "unit.hpp"
#include "variant.hpp"

#include "atomic.hpp"
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
        ASSERT((vid == v) && (dir == d) && (pid == p)); // no key truncate
    }

    bool operator == (const ikey_t &key) const {
        if ((vid == key.vid) && (pid == key.pid) && (dir == key.dir))
            return true;
        return false;
    }

    bool operator != (const ikey_t &key) const { return !(operator == (key)); }

    bool is_empty() { return ((vid == 0) && (pid == 0) && (dir == 0)); }

    void print_key() { cout << "[" << vid << "|" << pid << "|" << dir << "]" << endl; }

    uint64_t hash() const {
        uint64_t r = 0;
        r += vid;
        r <<= NBITS_IDX;
        r += pid;
        r <<= NBITS_DIR;
        r += dir;
        return wukong::math::hash_u64(r); // the standard hash is too slow (i.e., std::hash<uint64_t>()(r))
    }
};

struct ikey_Hasher {
    static size_t hash(const ikey_t &k) {
        return k.hash();
    }

    static bool equal(const ikey_t &x, const ikey_t &y) {
        return x.operator == (y);
    }
};

// 64-bit internal pointer
//   NBITS_SIZE: the max number of edges (edge_t) for a single vertex (256M)
//   NBITS_PTR: the max number of edges (edge_t) for the entire gstore (16GB)
//   NBITS_TYPE: the type of edge, used for attribute triple, sid(0), int(1), float(2), double(4)
enum { NBITS_SIZE = 28 };
enum { NBITS_PTR  = 34 };
enum { NBITS_TYPE =  2 };

struct iptr_t {
uint64_t size: NBITS_SIZE;
uint64_t off: NBITS_PTR;
uint64_t type: NBITS_TYPE;

    iptr_t(): size(0), off(0), type(0) { }

    // the default type is sid(type = 0)
    iptr_t(uint64_t s, uint64_t o, uint64_t t = 0): size(s), off(o), type(t) {
        // no truncated
        ASSERT ((size == s) && (off == o) && (type == t));
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
#if DYNAMIC_GSTORE
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
                    if(items[i].v.key == v.key || items[i].v.key.is_empty()) {
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
                            if(min_cnt == 0)
                                break;
                        }
                    }
                }

                volatile uint32_t old_ver = items[pos].version;
                if (old_ver != 0) {
                    uint32_t ret_ver = wukong::atomic::compare_and_swap(&items[pos].version, old_ver, 0);
                    if (ret_ver == old_ver) {
#if DYNAMIC_GSTORE
                        // Do not reset visit cnt for the same vertex
                        items[pos].cnt = (items[pos].v.key == v.key) ? items[pos].cnt : 0;
                        items[pos].v = v;
                        items[pos].expire_time = timer::get_usec() + lease;
#else
                        items[pos].v = v;
                        items[pos].cnt = 0;
#endif

                        asm volatile("" ::: "memory");
                        ret_ver = wukong::atomic::compare_and_swap(&items[pos].version, 0, old_ver + 1);
                        assert(ret_ver == 0);
                        return;
                    }
                } 
            }
        } // end of insert

#if DYNAMIC_GSTORE
        /* Set lease.*/
        void set_lease(uint64_t lease) {
            this->lease = lease;
        }

        /* Invalidate cache item of the given key.
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


    uint64_t bucket_local(ikey_t key) {
        uint64_t bucket_id;

#ifdef USE_GPU
        // the smallest pid is 1
        auto &seg = rdf_segment_meta_map[segid_t((key.vid == 0 ? 1 : 0), key.pid, key.dir)];
        ASSERT(seg.num_buckets > 0);
        bucket_id = seg.bucket_start + key.hash() % seg.num_buckets;
#else
        bucket_id = key.hash() % num_buckets;
#endif // end of USE_GPU

        return bucket_id;
    }

    uint64_t bucket_remote(ikey_t key, int dst_sid) {
        uint64_t bucket_id;

#ifdef USE_GPU
        auto &remote_meta_map = shared_rdf_segment_meta_map[dst_sid];
        auto &seg = remote_meta_map[segid_t((key.vid == 0 ? 1 : 0), key.pid, key.dir)];
        ASSERT(seg.num_buckets > 0);
        bucket_id = seg.bucket_start + key.hash() % seg.num_buckets;
#else
        bucket_id = key.hash() % num_buckets;
#endif // end of USE_GPU

        return bucket_id;
    }

    // cluster chaining hash-table (see paper: DrTM SOSP'15)
    uint64_t insert_key(ikey_t key, bool check_dup = true) {
        uint64_t bucket_id = bucket_local(key);
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
                //ASSERT(vertices[slot_id].key != key); // no duplicate key
                if (vertices[slot_id].key == key) {
                    if (check_dup) {
                        key.print_key();
                        vertices[slot_id].key.print_key();
                        logstream(LOG_ERROR) << "conflict at slot["
                            << slot_id << "] of bucket["
                            << bucket_id << "]" << LOG_endl;
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
#ifdef USE_GPU
            rdf_segment_meta_t &seg = rdf_segment_meta_map[segid_t((key.vid == 0 ? 1 : 0), key.pid, key.dir)];
            uint64_t ext_bucket_id = seg.get_ext_bucket();
            if (ext_bucket_id == 0) {
                uint64_t nbuckets = EXT_BUCKET_EXTENT_LEN(seg.num_buckets);
                uint64_t start_off = alloc_ext_buckets(nbuckets);
                seg.add_ext_buckets(ext_bucket_extent_t(nbuckets, start_off));
                ext_bucket_id = seg.get_ext_bucket();
            }
            vertices[slot_id].key.vid = ext_bucket_id;
#else // !USE_GPU
            vertices[slot_id].key.vid = alloc_ext_buckets(1);
#endif  // end of USE_GPU

            slot_id = vertices[slot_id].key.vid * ASSOCIATIVITY; // move to a new bucket_ext
            vertices[slot_id].key = key; // insert to the first slot
            goto done;
        }
done:
        pthread_spin_unlock(&bucket_locks[lock_id]);
        ASSERT(slot_id < num_slots);
        ASSERT(vertices[slot_id].key == key);
        return slot_id;
    }

    edge_t *edges;
    uint64_t num_entries;     // entry region (dynamical)

#ifdef DYNAMIC_GSTORE

    // manage the memory of edges(val)
    MAInterface *edge_allocator;

    /// A size flag is put into the tail of edges (in the entry region) for dynamic cache.
    /// NOTE: the (remote) edges accessed by (local) RDMA cache are valid
    ///       if and only if the size flag of edges is consistent with the size within the pointer.

    static const uint64_t INVALID_EDGES = 1 << NBITS_SIZE; // flag indicates invalidate edges

    // Convert given byte units to edge units.
    inline uint64_t b2e(uint64_t sz) { return sz / sizeof(edge_t); }
    // Convert given edge uints to byte units.
    inline uint64_t e2b(uint64_t sz) { return sz * sizeof(edge_t); }
    // Return exact block size of given size in edge unit.
    inline uint64_t blksz(uint64_t ptr) { return b2e(edge_allocator->block_size(e2b(ptr))); }

    /* Insert size flag size flag in edges.
     * @flag: size flag to insert
     * @sz: actual size of edges
     * @off: offset of edges
    */
    inline void insert_sz(uint64_t flag, uint64_t off) {
        uint64_t blk_sz = blksz(off);   // reserve one space for flag
        edges[off + blk_sz - 1].val = flag;
    }

    /* Check the validation of given edge according to given vertex.
     * The edge is valid only when the size flag of edge is consistent with the size within the vertex.
     */
    inline bool edge_is_valid(vertex_t &v, edge_t *edge_ptr) {
        if (!global_enable_caching)
            return true;

        uint64_t blk_sz = blksz(v.ptr.off);  // reserve one space for flag
        return (edge_ptr[blk_sz - 1].val == v.ptr.size);
    }

    /// Defer blk's free operation for dynamic cache.
    /// Pend the free operation when blk is to be collected by add_pending_free()
    /// When allocating a new blk by alloc_edges(), check if pending free's lease expires
    /// and collect free space by sweep_free().
    uint64_t lease;

    // block deferred to be freed
    struct free_blk {
        uint64_t off;
        uint64_t expire_time;
        free_blk(uint64_t off, uint64_t expire_time): off(off), expire_time(expire_time) { }
    };
    queue<free_blk> free_queue;
    pthread_spinlock_t free_queue_lock;

    // Pend the free operation of given block.
    inline void add_pending_free(iptr_t ptr) {
        uint64_t expire_time = timer::get_usec() + lease;
        free_blk blk(ptr.off, expire_time);

        pthread_spin_lock(&free_queue_lock);
        free_queue.push(blk);
        pthread_spin_unlock(&free_queue_lock);
    }

    // Execute all expired pending free operations in queue.
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

    bool is_dup(vertex_t *v, uint64_t value) {
        int size = v->ptr.size;
        for (int i = 0; i < size; i++)
            if (edges[v->ptr.off + i].val == value)
                return true;
        return false;
    }

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
                //ASSERT(vertices[slot_id].key != key); // no duplicate key
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

    bool insert_vertex_edge(ikey_t key, uint64_t value, bool &dedup_or_isdup, int tid) {
        uint64_t bucket_id = key.hash() % num_buckets;
        uint64_t lock_id = bucket_id % NUM_LOCKS;
        uint64_t v_ptr = insert_key(key, false);
        vertex_t *v = &vertices[v_ptr];
        pthread_spin_lock(&bucket_locks[lock_id]);
        if (v->ptr.size == 0) {
            uint64_t off = alloc_edges(1, tid);
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

            // a new block is needed
            if (blksz(v->ptr.off) - 1 < need_size) {
                iptr_t old_ptr = v->ptr;

                uint64_t off = alloc_edges(blksz(v->ptr.off) * 2, tid);
                memcpy(&edges[off], &edges[old_ptr.off], e2b(old_ptr.size));
                edges[off + old_ptr.size].val = value;
                // invalidate the old block
                insert_sz(INVALID_EDGES, old_ptr.off);
                v->ptr = iptr_t(need_size, off);

                if (global_enable_caching)
                    add_pending_free(old_ptr);
                else
                    edge_allocator->free(e2b(old_ptr.off));
            } else {
                // update size flag
                insert_sz(need_size, v->ptr.off);
                edges[v->ptr.off + v->ptr.size].val = value;
                v->ptr.size = need_size;
            }

            pthread_spin_unlock(&bucket_locks[lock_id]);
            return false;
        }
    }

    // Allocate space to store edges of given size.
    // Return offset of allocated space.
    inline uint64_t alloc_edges(uint64_t n, int64_t tid = 0) {
        if (global_enable_caching)
            sweep_free(); // collect free space before allocate
        uint64_t sz = e2b(n + 1); // reserve one space for sz
        uint64_t off = b2e(edge_allocator->malloc(sz, tid));
        insert_sz(n, off);
        return off;
    }
#else // !DYNAMIC_GSTORE

    uint64_t last_entry;
    pthread_spinlock_t entry_lock;

    // Allocate space to store edges of given size.
    // Return offset of allocated space.
    uint64_t alloc_edges(uint64_t n, int64_t tid = 0) {
        uint64_t orig;
        pthread_spin_lock(&entry_lock);
        orig = last_entry;
        last_entry += n;
        if (last_entry >= num_entries) {
            logstream(LOG_ERROR) << "out of entry region." << LOG_endl;
            ASSERT(last_entry < num_entries);
        }
        pthread_spin_unlock(&entry_lock);
        return orig;
    }
#endif // end of DYNAMIC_GSTORE

    // Allocate extended buckets
    // @n number of extended buckets to allocate
    // @return start offset of allocated extended buckets
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

    typedef tbb::concurrent_hash_map<sid_t, vector<sid_t>> tbb_hash_map;

    tbb_hash_map pidx_in_map; // predicate-index (IN)
    tbb_hash_map pidx_out_map; // predicate-index (OUT)
    tbb_hash_map tidx_map; // type-index


#ifdef USE_GPU  // enable GPU support
    int num_predicates;  // number of predicates

    // wrapper of atomic counters
    struct triple_cnt_t {
        triple_cnt_t() {
            normal_cnts[IN] = 0ul;
            normal_cnts[OUT] = 0ul;
            index_cnts[IN] = 0ul;
            index_cnts[OUT] = 0ul;
        }

        triple_cnt_t(const triple_cnt_t &cnt) {
            normal_cnts[IN] = cnt.normal_cnts[IN].load();
            normal_cnts[OUT] = cnt.normal_cnts[OUT].load();
            index_cnts[IN] = cnt.index_cnts[IN].load();
            index_cnts[OUT] = cnt.index_cnts[OUT].load();
        }

        atomic<uint64_t> normal_cnts[2];
        atomic<uint64_t> index_cnts[2];
    };

    // multiple engines will access shared_rdf_segment_meta_map
    // key: server id, value: segment metadata of the server
    tbb::concurrent_unordered_map <int, map<segid_t, rdf_segment_meta_t> > shared_rdf_segment_meta_map;
    // metadata of segments (will be used on GPU)
    std::map<segid_t, rdf_segment_meta_t> rdf_segment_meta_map;

    typedef tbb::concurrent_hash_map<ikey_t, vector<triple_t>, ikey_Hasher> tbb_triple_hash_map;
    // triples grouped by (predicate, direction)
    tbb_triple_hash_map triples_map;

    // all predicate IDs
    vector<sid_t> all_predicates;


    void send_segment_meta(TCP_Adaptor *tcp_ad) {
        std::stringstream ss;
        std::string str;
        boost::archive::binary_oarchive oa(ss);
        SyncSegmentMetaMsg msg(rdf_segment_meta_map);

        msg.sender_sid = sid;
        oa << msg;

        // send pred_metas to other servers
        for (int i = 0; i < global_num_servers; ++i) {
            if (i == sid)
                continue;
            tcp_ad->send(i, 0, ss.str());
            logstream(LOG_INFO) << "#" << sid << " sends segment metadata to server " << i << LOG_endl;
        }
    }

    void recv_segment_meta(TCP_Adaptor *tcp_ad) {
        std::string str;
        // receive global_num_servers - 1 messages
        for (int i = 0; i < global_num_servers; ++i) {
            if (i == sid)
                continue;
            std::stringstream ss;
            str = tcp_ad->recv(0);
            ss << str;
            boost::archive::binary_iarchive ia(ss);
            SyncSegmentMetaMsg msg;
            ia >> msg;

            shared_rdf_segment_meta_map.insert(make_pair(msg.sender_sid, msg.data));
            logstream(LOG_INFO) << "#" << sid << " receives segment metadata from server " << msg.sender_sid << LOG_endl;
        }
    }

    void collect_index_info(uint64_t slot_id) {
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

    void insert_tidx_map(tbb_hash_map &tidx_map) {
        sid_t pid = 0;
        uint64_t off;
        for (auto const &e : tidx_map) {
            pid = e.first;

            rdf_segment_meta_t &segment = rdf_segment_meta_map[segid_t(1, pid, IN)];
            if (segment.num_edges == 0) {
                logger(LOG_FATAL, "insert_tidx_map: pid %d is not allocated space. entry_sz: %lu",
                       pid, e.second.size());
                ASSERT(false);
            }

            uint64_t sz = e.second.size();
            uint64_t off = segment.edge_start;
            ASSERT(sz == segment.num_edges);
            logger(LOG_DEBUG, "insert_tidx_map: pid: %lu, sz: %lu", pid, sz);

            ikey_t key = ikey_t(0, pid, IN);
            uint64_t slot_id = insert_key(key);
            iptr_t ptr = iptr_t(sz, off);
            vertices[slot_id].ptr = ptr;

            for (auto const &vid : e.second)
                edges[off++].val = vid;

            ASSERT(off <= segment.edge_start + segment.num_edges);
        }
    }

    void insert_pidx_map(tbb_hash_map &pidx_map, sid_t pid, dir_t d) {
        tbb_hash_map::const_accessor ca;
        bool success = pidx_map.find(ca, pid);
        if (!success)
            return;

        rdf_segment_meta_t &segment = rdf_segment_meta_map[segid_t(1, pid, d)];
        ASSERT(segment.num_edges > 0);
        uint64_t sz = ca->second.size();
        uint64_t off = segment.edge_start;
        ASSERT(sz == segment.num_edges);

        ikey_t key = ikey_t(0, pid, d);
        logger(LOG_DEBUG, "insert_pidx_map[%s]: key: [%lu|%lu|%lu] sz: %lu",
               (d == IN) ? "IN" : "OUT", key.vid, key.pid, key.dir, sz);
        uint64_t slot_id = insert_key(key);
        iptr_t ptr = iptr_t(sz, off);
        vertices[slot_id].ptr = ptr;

        for (auto const &vid : ca->second)
            edges[off++].val = vid;

        ASSERT(off <= segment.edge_start + segment.num_edges);
    }
#endif  // end of USE_GPU

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

    // Get edges of given vertex from dst_sid by RDMA read.
    inline edge_t *rdma_get_edges(int tid, int dst_sid, vertex_t &v) {
        ASSERT(global_use_rdma);

        char *buf = mem->buffer(tid);
        uint64_t r_off = num_slots * sizeof(vertex_t) + v.ptr.off * sizeof(edge_t);

#ifdef DYNAMIC_GSTORE
        // the size of entire blk
        uint64_t r_sz = blksz(v.ptr.off) * sizeof(edge_t);
#else
        // the size of edges
        uint64_t r_sz = v.ptr.size * sizeof(edge_t);
#endif

        uint64_t buf_sz = mem->buffer_size();
        ASSERT(r_sz < buf_sz); // enough space to host the edges

        RDMA &rdma = RDMA::get_rdma();
        rdma.dev->RdmaRead(tid, dst_sid, buf, r_sz, r_off);
        return (edge_t *)buf;
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

    // Get remote edges according to given vid, dir, pid.
    // @sz: size of return edges
    edge_t *get_edges_remote(int tid, sid_t vid, dir_t d, sid_t pid, uint64_t *sz) {
        int dst_sid = wukong::math::hash_mod(vid, global_num_servers);
        ikey_t key = ikey_t(vid, pid, d);
        edge_t *edge_ptr;
        vertex_t v = get_vertex_remote(tid, key);
        if (v.key.is_empty()) {
            *sz = 0;
            return NULL; // not found
        }

        edge_ptr = rdma_get_edges(tid, dst_sid, v);
#ifdef DYNAMIC_GSTORE
        // check the validation of edges
        // if not, invalidate the cache and try again
        while (!edge_is_valid(v, edge_ptr)) {
            rdma_cache.invalidate(key);
            v = get_vertex_remote(tid, key);
            edge_ptr = rdma_get_edges(tid, dst_sid, v);
        }
#endif

        *sz = v.ptr.size;
        return edge_ptr;
    }

    // Get local edges according to given vid, dir, pid.
    // @sz: size of return edges
    edge_t *get_edges_local(int tid, sid_t vid, dir_t d, sid_t pid, uint64_t *sz) {
        ikey_t key = ikey_t(vid, pid, d);
        vertex_t v = get_vertex_local(tid, key);

        if (v.key.is_empty()) {
            *sz = 0;
            return NULL;
        }

        *sz = v.ptr.size;
        return &(edges[v.ptr.off]);
    }

    // get the attribute value from remote
    attr_t get_vertex_attr_remote(int tid, sid_t vid, dir_t d, sid_t pid, bool &has_value) {
        //struct the key
        int dst_sid = wukong::math::hash_mod(vid, global_num_servers);
        ikey_t key = ikey_t(vid, pid, d);
        edge_t *edge_ptr;
        vertex_t v;
        attr_t r;

        //get the vertex from DYNAMIC_GSTORE or normal
#ifdef DYNAMIC_GSTORE
        v = get_vertex_remote(tid, key);
        if (v.key.is_empty()) {
            has_value = false; // not found
            return r;
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
            has_value = false; // not found
            return r;
        }

        edge_ptr = rdma_get_edges(tid, dst_sid, v);
#endif // end of DYNAMIC_GSTORE

        // get the edge
        uint64_t r_off  = num_slots * sizeof(vertex_t) + v.ptr.off * sizeof(edge_t);
        // get the attribute value from its type
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
            logstream(LOG_ERROR) << "Not support value type" << LOG_endl;
            break;
        }

        has_value = true;
        return r;
    }

    // get the attribute value from local
    attr_t get_vertex_attr_local(int tid, sid_t vid, dir_t d, sid_t pid, bool &has_value) {
        // struct the key
        ikey_t key = ikey_t(vid, pid, d);
        // get the vertex
        vertex_t v = get_vertex_local(tid, key);

        attr_t r;
        if (v.key.is_empty()) {
            has_value = false; // not found
            return r;
        }
        // get the edge
        uint64_t off = v.ptr.off;
        // get the attribute with its type
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
            logstream(LOG_ERROR) << "Not support value type" << LOG_endl;
            break;
        }
        has_value = true;
        return r;
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
        num_buckets = wukong::math::hash_prime_u64((num_slots / ASSOCIATIVITY) * MHD_RATIO / 100);
        num_buckets_ext = (num_slots / ASSOCIATIVITY) - num_buckets;

        // entry region
        num_entries = entry_region / sizeof(edge_t);
#ifdef DYNAMIC_GSTORE
#ifdef USE_JEMALLOC
        edge_allocator = new JeMalloc();
#else
        edge_allocator = new BuddyMalloc();
#endif // end of USE_JEMALLOC
        pthread_spin_init(&free_queue_lock, 0);
        lease = SEC(600);
        rdma_cache.set_lease(lease);
#else
        pthread_spin_init(&entry_lock, 0);
#endif // end of DYNAMIC_GSTORE

        // print gstore usage
        logstream(LOG_INFO) << "gstore = ";
        logstream(LOG_INFO) << mem->kvstore_size() << " bytes " << LOG_endl;
        logstream(LOG_INFO) << "  header region: " << num_slots << " slots"
                            << " (main = " << num_buckets << ", indirect = " << num_buckets_ext << ")" << LOG_endl;
        logstream(LOG_INFO) << "  entry region: " << num_entries << " entries" << LOG_endl;

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

#ifdef DYNAMIC_GSTORE
        edge_allocator->init((void *)edges, num_entries * sizeof(edge_t), global_num_engines);
#else
        last_entry = 0;
#endif
    }


#ifdef USE_GPU
    inline void set_num_predicates(sid_t n) {
        num_predicates = n;
    }

    inline int get_num_predicates() const {
        return num_predicates;
    }

    const vector<sid_t>& get_all_predicates() const {
        return all_predicates;
    }

    void sync_metadata(TCP_Adaptor *tcp_ad) {
        send_segment_meta(tcp_ad);
        recv_segment_meta(tcp_ad);
    }

    /**
     * Merge triples with same predicate and direction to a vector
     */
    void init_triples_map(vector<vector<triple_t>> &triple_pso, vector<vector<triple_t>> &triple_pos) {
        #pragma omp parallel for num_threads(global_num_engines)
        for (int tid = 0; tid < global_num_engines; tid++) {
            vector<triple_t> &pso_vec = triple_pso[tid];
            vector<triple_t> &pos_vec = triple_pos[tid];

            int i = 0;
            int current_pid;
            while (!pso_vec.empty() && i < pso_vec.size()) {
                current_pid = pso_vec[i].p;
                tbb_triple_hash_map::accessor a;
                triples_map.insert(a, ikey_t(0, current_pid, OUT));

                for (; pso_vec[i].p == current_pid; i++) {
                    a->second.push_back(pso_vec[i]);
                }
            }

            i = 0;
            while (!pos_vec.empty() && i < pos_vec.size()) {
                current_pid = pos_vec[i].p;
                tbb_triple_hash_map::accessor a;
                triples_map.insert(a, ikey_t(0, current_pid, IN));

                for (; pos_vec[i].p == current_pid; i++) {
                    a->second.push_back(pos_vec[i]);
                }
            }
        }
    }

    void free_triples_map() {
        triples_map.clear();
    }

    ext_bucket_extent_t ext_bucket_extent(uint64_t nbuckets, uint64_t start_off) {
        ext_bucket_extent_t ext;
        ext.num_ext_buckets = nbuckets;
        ext.start = start_off;
    }

    uint64_t main_hdr_off = 0;

    void alloc_buckets_to_segment(rdf_segment_meta_t &seg, segid_t segid, uint64_t total_num_keys) {
        // deduct some buckets from total to prevent overflow
        static uint64_t num_free_buckets = num_buckets - num_predicates * PREDICATE_NSEGS;
        static double total_ratio_ = 0.0;

        // allocate buckets in main-header region to segments
        if (!segid.index) {
            uint64_t nbuckets;
            if (seg.num_keys == 0) {
                nbuckets = 0;
            } else {
                double ratio = static_cast<double>(seg.num_keys) / total_num_keys;
                nbuckets = ratio * num_free_buckets;
                total_ratio_ += ratio;
                logger(LOG_DEBUG, "Seg[%lu|%lu|%lu]: #keys: %lu, nbuckets: %lu, bucket_off: %lu, ratio: %f, total_ratio: %f",
                       segid.index, segid.pid, segid.dir,
                       seg.num_keys, nbuckets, main_hdr_off, ratio, total_ratio_);
            }
            seg.num_buckets = (nbuckets > 0 ? nbuckets : 1);
        } else {
            if (seg.num_keys > 0) {
                seg.num_buckets = 1;
            }
        }
        seg.bucket_start = main_hdr_off;
        main_hdr_off += seg.num_buckets;
        ASSERT(main_hdr_off <= num_buckets);

        // allocate buckets in indirect-header region to segments
        // #buckets : #extended buckets = 1 : 0.15
        if (!segid.index && seg.num_buckets > 0) {
            uint64_t nbuckets = EXT_BUCKET_EXTENT_LEN(seg.num_buckets);
            uint64_t start_off = alloc_ext_buckets(nbuckets);
            seg.add_ext_buckets(ext_bucket_extent_t(nbuckets, start_off));
        }
    }

    // init metadata for each segment
    void init_segment_metas(const vector<vector<triple_t>> &triple_pso,
                            const vector<vector<triple_t>> &triple_pos) {

        map<sid_t, triple_cnt_t> index_cnt_map;  // count children of index vertex
        map<sid_t, triple_cnt_t> normal_cnt_map; // count normal vertices

        // initialization
        for (int i = 0; i <= num_predicates; ++i) {
            index_cnt_map.insert(make_pair(i, triple_cnt_t()));
            normal_cnt_map.insert(make_pair(i, triple_cnt_t()));
            for (int index = 0; index <= 1; index++) {
                for (int dir = 0; dir <= 1; dir++) {
                    segid_t segid = segid_t(index, i, dir);
                    rdf_segment_meta_map.insert(make_pair(segid, rdf_segment_meta_t()));
                }
            }
        }

        #pragma omp parallel for num_threads(global_num_engines)
        for (int tid = 0; tid < global_num_engines; tid++) {
            const vector<triple_t> &pso = triple_pso[tid];
            const vector<triple_t> &pos = triple_pos[tid];

            uint64_t s = 0;
            while (s < pso.size()) {
                uint64_t e = s + 1;

                while ((e < pso.size())
                        && (pso[s].s == pso[e].s)
                        && (pso[s].p == pso[e].p))  {
                    // count #edge of type-idx
                    if (pso[e].p == TYPE_ID && is_tpid(pso[e].o)) {
                        index_cnt_map[ pso[e].o ].index_cnts[ IN ]++;
                    }
                    e++;
                }

                // count #edge of predicate
                normal_cnt_map[ pso[s].p ].normal_cnts[OUT] += (e - s);

                // count #edge of predicate-idx
                index_cnt_map[ pso[s].p ].index_cnts[ IN ]++;

                // count #edge of type-idx
                if (pso[s].p == TYPE_ID && is_tpid(pso[s].o)) {
                    index_cnt_map[ pso[s].o ].index_cnts[ IN ]++;
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

                // count #edge of predicate
                normal_cnt_map[ pos[s].p ].normal_cnts[IN] += (e - s);
                index_cnt_map[ pos[s].p ].index_cnts[ OUT ]++;
                s = e;
            }
        }

        uint64_t total_num_keys = 0;
        // count the total number of keys
        for (int i = 1; i <= num_predicates; ++i) {
            logger(LOG_DEBUG, "pid: %d: normal: #IN: %lu, #OUT: %lu; index: #ALL: %lu, #IN: %lu, #OUT: %lu",
                   i, normal_cnt_map[i].normal_cnts[IN].load(), normal_cnt_map[i].normal_cnts[OUT].load(),
                   (index_cnt_map[i].index_cnts[ IN ].load() + index_cnt_map[i].index_cnts[ OUT ].load()),
                   index_cnt_map[i].index_cnts[ IN ].load(),
                   index_cnt_map[i].index_cnts[ OUT ].load());

            if (normal_cnt_map[i].normal_cnts[IN].load() +  normal_cnt_map[i].normal_cnts[OUT].load() > 0) {
                total_num_keys += (index_cnt_map[i].index_cnts[IN].load() + index_cnt_map[i].index_cnts[OUT].load());
            }
        }

        uint64_t bucket_off = 0;
        uint64_t edge_off = 0;
        for (sid_t pid = 1; pid <= num_predicates; ++pid) {

            rdf_segment_meta_t &seg_normal_out = rdf_segment_meta_map[segid_t(0, pid, OUT)];
            rdf_segment_meta_t &seg_normal_in = rdf_segment_meta_map[segid_t(0, pid, IN)];
            rdf_segment_meta_t &seg_index_out = rdf_segment_meta_map[segid_t(1, pid, OUT)];
            rdf_segment_meta_t &seg_index_in = rdf_segment_meta_map[segid_t(1, pid, IN)];

            seg_normal_out.num_edges = normal_cnt_map[pid].normal_cnts[OUT].load();
            seg_normal_in.num_edges = normal_cnt_map[pid].normal_cnts[IN].load();
            seg_index_out.num_edges = index_cnt_map[pid].index_cnts[OUT].load();
            seg_index_in.num_edges = index_cnt_map[pid].index_cnts[IN].load();

            // collect predicates, excludes type-ids
            if (seg_normal_out.num_edges + seg_normal_in.num_edges > 0) {
                all_predicates.push_back(pid);
            }

            // allocate space for edges in entry-region
            seg_normal_out.edge_start = (seg_normal_out.num_edges > 0) ? alloc_edges(seg_normal_out.num_edges) : 0;
            seg_normal_in.edge_start  = (seg_normal_in.num_edges > 0) ? alloc_edges(seg_normal_in.num_edges) : 0;

            // predicate rdf:type doesn't have index vertices
            if (pid != TYPE_ID) {
                seg_index_out.edge_start = (seg_index_out.num_edges > 0) ? alloc_edges(seg_index_out.num_edges) : 0;
                seg_index_in.edge_start  = (seg_index_in.num_edges > 0) ? alloc_edges(seg_index_in.num_edges) : 0;
            }

            uint64_t normal_nkeys[2] = {seg_index_out.num_edges, seg_index_in.num_edges};
            seg_normal_out.num_keys = (seg_normal_out.num_edges == 0) ? 0 : normal_nkeys[OUT];
            seg_normal_in.num_keys  = (seg_normal_in.num_edges == 0)  ? 0 : normal_nkeys[IN];
            seg_index_out.num_keys  = (seg_index_out.num_edges == 0)  ? 0 : 1;
            seg_index_in.num_keys   = (seg_index_in.num_edges == 0)   ? 0 : 1;

            alloc_buckets_to_segment(seg_normal_out, segid_t(0, pid, OUT), total_num_keys);
            alloc_buckets_to_segment(seg_normal_in, segid_t(0, pid, IN), total_num_keys);
            alloc_buckets_to_segment(seg_index_out, segid_t(1, pid, OUT), total_num_keys);
            alloc_buckets_to_segment(seg_index_in, segid_t(1, pid, IN), total_num_keys);


            logger(LOG_DEBUG, "Predicate[%d]: normal: OUT[#keys: %lu, #buckets: %lu, #edges: %lu] IN[#keys: %lu, #buckets: %lu, #edges: %lu];",
                   pid, seg_normal_out.num_keys, seg_normal_out.num_buckets, seg_normal_out.num_edges,
                   seg_normal_in.num_keys, seg_normal_in.num_buckets, seg_normal_in.num_edges);
            logger(LOG_DEBUG, "index: OUT[#keys: %lu, #buckets: %lu, #edges: %lu], IN[#keys: %lu, #buckets: %lu, #edges: %lu], bucket_off: %lu\n",
                   seg_index_out.num_keys, seg_index_out.num_buckets, seg_index_out.num_edges,
                   seg_index_in.num_keys, seg_index_in.num_buckets, seg_index_in.num_edges, main_hdr_off);

        }

        logger(LOG_DEBUG, "#total_keys: %lu, bucket_off: %lu, #total_entries: %lu", total_num_keys, main_hdr_off, this->last_entry);
    }

    // re-adjust offset of indirect header
    void finalize_segment_metas() {
        for (auto &e : rdf_segment_meta_map) {
            for (int i = 0; i < e.second.ext_list_sz; ++i) {
                auto &ext = e.second.ext_bucket_list[i];
                // TODO: actually we can use ext.off to represent the exact size
                ext.num_ext_buckets = ext.off;
            }
        }
    }

    /**
     * Insert triples beloging to the segment identified by segid to store
     * Notes: This function only insert triples belonging to normal segment
     * @tid
     * @segid
     */
    void insert_triples_to_segment(int tid, segid_t segid) {
        ASSERT(!segid.index);

        auto &segment = rdf_segment_meta_map[segid];
        int index = segid.index;
        sid_t pid = segid.pid;
        int dir = segid.dir;

        if (segment.num_edges == 0) {
            logger(LOG_DEBUG, "Thread(%d): abort! segment(%d|%d|%d) is empty.\n", tid, segid.index, segid.pid, segid.dir);
            return;
        }

        // get OUT edges and IN edges from  map
        tbb_triple_hash_map::accessor a;
        bool has_pso, has_pos;
        bool success = triples_map.find(a, ikey_t(0, pid, (dir_t) dir));

        has_pso = (segid.dir == OUT) ? success : false;
        has_pos = (segid.dir == IN) ? success : false;

        // a segment only contains triples of one direction
        ASSERT((has_pso == true && has_pos == false) || (has_pso == false && has_pos == true));

        uint64_t off = segment.edge_start;
        uint64_t s = 0;
        uint64_t type_triples = 0;

        if (has_pso) {
            vector<triple_t> &pso = a->second;
            while (s < pso.size()) {
                // predicate-based key (subject + predicate)
                uint64_t e = s + 1;
                while ((e < pso.size())
                        && (pso[s].s == pso[e].s)
                        && (pso[s].p == pso[e].p))  { e++; }

                // allocate a vertex and edges
                ikey_t key = ikey_t(pso[s].s, pso[s].p, OUT);

                // insert a vertex
                uint64_t slot_id = insert_key(key);
                iptr_t ptr = iptr_t(e - s, off);
                vertices[slot_id].ptr = ptr;

                // insert edges
                for (uint64_t i = s; i < e; i++)
                    edges[off++].val = pso[i].o;

                collect_index_info(slot_id);
                s = e;
            }
            logger(LOG_DEBUG, "Thread(%d): inserted predicate %d pso(%lu triples).", tid, pid, pso.size());
        }

        ASSERT(off <= segment.edge_start + segment.num_edges);

        if (has_pos) {
            vector<triple_t> &pos = a->second;
            while (type_triples < pos.size() && is_tpid(pos[type_triples].o))
                type_triples++;

            s = type_triples; // skip type triples

            while (s < pos.size()) {
                // predicate-based key (object + predicate)
                uint64_t e = s + 1;
                while ((e < pos.size())
                        && (pos[s].o == pos[e].o)
                        && (pos[s].p == pos[e].p)) { e++; }

                // allocate a vertex and edges
                ikey_t key = ikey_t(pos[s].o, pos[s].p, IN);

                // insert a vertex
                uint64_t slot_id = insert_key(key);
                iptr_t ptr = iptr_t(e - s, off);
                vertices[slot_id].ptr = ptr;

                // insert edges
                for (uint64_t i = s; i < e; i++)
                    edges[off++].val = pos[i].s;

                collect_index_info(slot_id);
                s = e;
            }
            logger(LOG_DEBUG, "Thread(%d): inserted predicate %d pos(%lu triples).", tid, pid, pos.size());
        }

        ASSERT(off <= segment.edge_start + segment.num_edges);

        if (pid == TYPE_ID) {
            // when finish inserting triples of rdf:type, tidx_map will contain all type-idxs
            insert_tidx_map(tidx_map);
        } else {
            if (has_pso)
                insert_pidx_map(pidx_in_map, pid, IN);
            if (has_pos)
                insert_pidx_map(pidx_out_map, pid, OUT);
        }
    }

#endif  // end of USE_GPU

    /// skip all TYPE triples (e.g., <http://www.Department0.University0.edu> rdf:type ub:University)
    /// because Wukong treats all TYPE triples as index vertices. In addition, the triples in triple_pos
    /// has been sorted by the vid of object, and IDs of types are always smaller than normal vertex IDs.
    /// Consequently, all TYPE triples are aggregated at the beggining of triple_pos
    void insert_normal(vector<triple_t> &pso, vector<triple_t> &pos, int tid) {
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
        vector<sid_t> predicates;
#endif // end of VERSATILE

        uint64_t s = 0;
        while (s < pso.size()) {
            // predicate-based key (subject + predicate)
            uint64_t e = s + 1;
            while ((e < pso.size())
                    && (pso[s].s == pso[e].s)
                    && (pso[s].p == pso[e].p))  { e++; }

            // allocate a vertex and edges
            ikey_t key = ikey_t(pso[s].s, pso[s].p, OUT);
            uint64_t off = alloc_edges(e - s, tid);

            // insert a vertex
            uint64_t slot_id = insert_key(key);
            iptr_t ptr = iptr_t(e - s, off);
            vertices[slot_id].ptr = ptr;

            // insert edges
            for (uint64_t i = s; i < e; i++)
                edges[off++].val = pso[i].o;

#ifdef VERSATILE
            // add a new predicate
            predicates.push_back(pso[s].p);

            // insert a special PREDICATE triple (OUT)
            if (e >= pso.size() || pso[s].s != pso[e].s) {
                // allocate a vertex and edges
                ikey_t key = ikey_t(pso[s].s, PREDICATE_ID, OUT);
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
#endif // end of VERSATILE

            s = e;
        }

        s = type_triples; // skip type triples
        while (s < pos.size()) {
            // predicate-based key (object + predicate)
            uint64_t e = s + 1;
            while ((e < pos.size())
                    && (pos[s].o == pos[e].o)
                    && (pos[s].p == pos[e].p)) { e++; }

            // allocate a vertex and edges
            ikey_t key = ikey_t(pos[s].o, pos[s].p, IN);
            uint64_t off = alloc_edges(e - s, tid);

            // insert a vertex
            uint64_t slot_id = insert_key(key);
            iptr_t ptr = iptr_t(e - s, off);
            vertices[slot_id].ptr = ptr;

            // insert edges
            for (uint64_t i = s; i < e; i++)
                edges[off++].val = pos[i].s;

#ifdef VERSATILE
            // add a new predicate
            predicates.push_back(pos[s].p);

            // insert a special PREDICATE triple (OUT)
            if (e >= pos.size() || pos[s].o != pos[e].o) {
                // allocate a vertex and edges
                ikey_t key = ikey_t(pos[s].o, PREDICATE_ID, IN);
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
#endif // end of VERSATILE
            s = e;
        }
    }

    // insert attributes
    void insert_attribute(vector<triple_attr_t> &attrs, int64_t tid) {
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
                logstream(LOG_ERROR) << "Unsupported value type of attribute" << LOG_endl;
            }
        }
    }

    void insert_index() {
        uint64_t t1 = timer::get_usec();

        logstream(LOG_INFO) << " start (parallel) prepare index info " << LOG_endl;


#ifdef DYNAMIC_GSTORE
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
                        ASSERT(false); // (IN) type triples should be skipped
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
        logstream(LOG_DEBUG) << (t2 - t1) / 1000 << " ms for preparing index info (in parallel)" << LOG_endl;

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
        logstream(LOG_DEBUG) << (t3 - t2) / 1000 << " ms for inserting index data into gstore" << LOG_endl;
    }

#ifdef DYNAMIC_GSTORE
    void insert_triple_out(const triple_t &triple, bool check_dup, int tid) {
        bool dedup_or_isdup = check_dup;
        bool nodup = false;
        if (triple.p == TYPE_ID) {
            // for TYPE_ID condition, dedup is always needed
            // for LUBM benchmark, maybe for others,too.
            dedup_or_isdup = true;
            ikey_t key = ikey_t(triple.s, triple.p, OUT);
            // <1> vid's type (7) [need dedup]
            if (insert_vertex_edge(key, triple.o, dedup_or_isdup, tid)) {
#ifdef VERSATILE
                key = ikey_t(triple.s, PREDICATE_ID, OUT);
                // key and its buddy_key should be used to
                // identify the exist of corresponding index
                ikey_t buddy_key = ikey_t(triple.s, PREDICATE_ID, IN);
                // <2> vid's predicate, value is TYPE_ID (*8) [dedup from <1>]
                if (insert_vertex_edge(key, triple.p, nodup, tid) && !check_key_exist(buddy_key)) {
                    key = ikey_t(0, TYPE_ID, IN);
                    // <3> the index to vid (*3) [dedup from <2>]
                    insert_vertex_edge(key, triple.s, nodup, tid);
                }
#endif // end of VERSATILE
            }
            if (!dedup_or_isdup) {
                key = ikey_t(0, triple.o, IN);
                // <4> type-index (2) [if <1>'s result is not dup, this is not dup, too]
                if (insert_vertex_edge(key, triple.s, nodup, tid)) {
#ifdef VERSATILE
                    key = ikey_t(0, TYPE_ID, OUT);
                    // <5> index to this type (*4) [dedup from <4>]
                    insert_vertex_edge(key, triple.o, nodup, tid);
#endif // end of VERSATILE
                }
            }
        } else {
            ikey_t key = ikey_t(triple.s, triple.p, OUT);
            // <6> vid's ngbrs w/ predicate (6) [need dedup]
            if (insert_vertex_edge(key, triple.o, dedup_or_isdup, tid)) {
                key = ikey_t(0, triple.p, IN);
                // key and its buddy_key should be used to
                // identify the exist of corresponding index
                ikey_t buddy_key = ikey_t(0, triple.p, OUT);
                // <7> predicate-index (1) [dedup from <6>]
                if (insert_vertex_edge(key, triple.s, nodup, tid) && !check_key_exist(buddy_key)) {
#ifdef VERSATILE
                    key = ikey_t(0, PREDICATE_ID, OUT);
                    // <8> the index to predicate (*5) [dedup from <7>]
                    insert_vertex_edge(key, triple.p, nodup, tid);
#endif // end of VERSATILE
                }
#ifdef VERSATILE
                key = ikey_t(triple.s, PREDICATE_ID, OUT);
                // key and its buddy_key should be used to
                // identify the exist of corresponding index
                buddy_key = ikey_t(triple.s, PREDICATE_ID, IN);
                // <9> vid's predicate (*8) [dedup from <6>]
                if (insert_vertex_edge(key, triple.p, nodup, tid) && !check_key_exist(buddy_key)) {
                    key = ikey_t(0, TYPE_ID, IN);
                    // <10> the index to vid (*3) [dedup from <9>]
                    insert_vertex_edge(key, triple.s, nodup, tid);
                }
#endif // end of VERSATILE
            }
        }
    }

    void insert_triple_in(const triple_t &triple, bool check_dup, int tid) {
        bool dedup_or_isdup = check_dup;
        bool nodup = false;
        if (triple.p == TYPE_ID) // skipped
            return;
        ikey_t key = ikey_t(triple.o, triple.p, IN);
        // <1> vid's ngbrs w/ predicate (6) [need dedup]
        if (insert_vertex_edge(key, triple.s, dedup_or_isdup, tid)) {
            // key doesn't exist before
            key = ikey_t(0, triple.p, OUT);
            // key and its buddy_key should be used
            // to identify the exist of corresponding index
            ikey_t buddy_key = ikey_t(0, triple.p, IN);
            // <2> predicate-index (1) [dedup from <1>]
            if (insert_vertex_edge(key, triple.o, nodup, tid) && !check_key_exist(buddy_key)) {
#ifdef VERSATILE
                key = ikey_t(0, PREDICATE_ID, OUT);
                // <3> the index to predicate (*5) [dedup from <2>]
                insert_vertex_edge(key, triple.p, nodup, tid);
#endif // end of VERSATILE
            }
#ifdef VERSATILE
            key = ikey_t(triple.o, PREDICATE_ID, IN);
            // key and its buddy_key should be used to
            // identify the exist of corresponding index
            buddy_key = ikey_t(triple.o, PREDICATE_ID, OUT);
            // <4> vid's predicate (*8) [dedup from <1>]
            if (insert_vertex_edge(key, triple.p, nodup, tid) && !check_key_exist(buddy_key)) {
                key = ikey_t(0, TYPE_ID, IN);
                // <5> the index to vid (*3) [dedup from <4>]
                insert_vertex_edge(key, triple.o, nodup, tid);
            }
#endif // end of VERSATILE
        }
    }

#endif // DYNAMIC_GSTORE


    // FIXME: refine parameters with vertex_t
    edge_t *get_edges_global(int tid, sid_t vid, dir_t d, sid_t pid, uint64_t *sz) {
        if (wukong::math::hash_mod(vid, global_num_servers) == sid)
            return get_edges_local(tid, vid, d, pid, sz);
        else
            return get_edges_remote(tid, vid, d, pid, sz);
    }

    edge_t *get_index_edges_local(int tid, sid_t pid, dir_t d, uint64_t *sz) {
        // the vid of index vertex should be 0
        return get_edges_local(tid, 0, d, pid, sz);
    }

    // get vertex attributes global
    // return the attr result
    // if not found has_value will be set to false
    attr_t get_vertex_attr_global(int tid, sid_t vid, dir_t d, sid_t pid, bool &has_value) {
        if (sid == wukong::math::hash_mod(vid, global_num_servers))
            return get_vertex_attr_local(tid, vid, d, pid, has_value);
        else
            return get_vertex_attr_remote(tid, vid, d, pid, has_value);
    }


    uint64_t ivertex_num = 0;
    uint64_t nvertex_num = 0;

    // check on in-dir predicate-index or type-index
    void idx_check_indir(ikey_t key, bool check) {
        if (!check)
            return;
        ivertex_num ++;
        uint64_t vsz = 0;
        // get the vids which refered by index
        edge_t *vres = get_edges_local(0, key.vid, (dir_t)key.dir, key.pid, &vsz);
        for (int i = 0; i < vsz; i++) {
            uint64_t tsz = 0;
            // get the vids's type
            edge_t *tres = get_edges_local(0, vres[i].val, OUT, TYPE_ID, &tsz);
            bool found = false;
            for (int j = 0; j < tsz; j++) {
                if (tres[j].val == key.pid && !found)
                    found = true;
                else if (tres[j].val == key.pid && found) {  //duplicate type
                    logstream(LOG_ERROR) << "In the value part of normal key/value pair "
                                         << "[ " << key.vid << " | TYPE_ID | OUT], "
                                         << "there is DUPLICATE type " << key.pid << LOG_endl;
                }
            }
            // may be it is a predicate_index
            if (tsz != 0 && !found) {
                // check if the key generated by vid and pid exists
                if (get_vertex_local(0, ikey_t(vres[i].val, key.pid, OUT)).key.is_empty()) {
                    logstream(LOG_ERROR) << "if " << key.pid << " is type id, then there is NO type "
                                         << key.pid << " in normal key/value pair ["
                                         << key.vid << " | TYPE_ID | OUT]'s value part" << LOG_endl;
                    logstream(LOG_ERROR) << "And if " << key.pid << " is predicate id, "
                                         << " then there is NO key called "
                                         << "[ " << vres[i].val << " | " << key.pid << " | " << "] exist."
                                         << LOG_endl;
                }
            }
        }

        // VERSATILE is enabled
        ver_idx_check_indir(key, check);
    }

    // check on out-dir predicate-index
    void idx_check_outdir(ikey_t key, bool check) {
        if (!check)
            return;
        ivertex_num ++;
        uint64_t vsz = 0;
        // get the vids which refered by predicate index
        edge_t *vres = get_edges_local(0, key.vid, (dir_t)key.dir, key.pid, &vsz);
        for (int i = 0; i < vsz; i++) {
            // check if the key generated by vid and pid exists
            if (get_vertex_local(0, ikey_t(vres[i].val, key.pid, IN)).key.is_empty()) {
                logstream(LOG_ERROR) << "The key " << " [ " << vres[i].val << " | "
                                     << key.pid << " | " << " IN ] does not exist." << LOG_endl;
            }
        }

        // VERSATILE is enabled
        ver_idx_check_outdir(key, check);
    }

    // check on normal types (7)
    void nt_check(ikey_t key, bool check) {
        if (!check)
            return;
        nvertex_num ++;
        uint64_t tsz = 0;
        // get the vid's all type
        edge_t *tres = get_edges_local(0, key.vid, (dir_t)key.dir, key.pid, &tsz);
        for (int i = 0; i < tsz; i++) {
            uint64_t vsz = 0;
            // get the vids which refered by the type
            edge_t *vres = get_edges_local(0, 0, IN, tres[i].val, &vsz);
            bool found = false;
            for (int j = 0; j < vsz; j++) {
                if (vres[j].val == key.vid && !found)
                    found = true;
                else if (vres[j].val == key.vid && found) { // duplicate vid
                    logstream(LOG_ERROR) << "In the value part of type index "
                                         << "[ 0 | " << tres[i].val << " | IN ], "
                                         << "there is duplicate value " << key.vid << LOG_endl;
                }
            }
            if (!found) { // vid miss
                logstream(LOG_ERROR) << "In the value part of type index "
                                     << "[ 0 | " << tres[i].val << " | IN ], "
                                     << "there is no value " << key.vid << LOG_endl;
            }
        }

        // VERSATILE is enabled
        ver_nt_check(key, check);
    }

    // check vid's ngbrs w/ predicate
    void np_check(ikey_t key, dir_t dir, bool check) {
        if (!check)
            return;
        nvertex_num ++;
        uint64_t vsz = 0;
        // get the vids which refered by the predicated
        edge_t *vres = get_edges_local(0, 0, dir, key.pid, &vsz);
        bool found = false;
        for (int i = 0; i < vsz; i++) {
            if (vres[i].val == key.vid && !found)
                found = true;
            else if (vres[i].val == key.vid && found) { //duplicate vid
                logstream(LOG_ERROR) << "In the value part of predicate index "
                                     << "[ 0 | " << key.pid << " | " << dir << " ], "
                                     << "there is duplicate value " << key.vid << LOG_endl;
                break;
            }
        }

        if (!found) // vid miss
            logstream(LOG_ERROR) << "In the value part of predicate index "
                                 << "[ 0 | " << key.pid << " | " << dir << " ], "
                                 << "there is no value " << key.vid << LOG_endl;
    }

#ifdef VERSATILE
    // check on in-dir predicate-index or type-index
    void ver_idx_check_indir(ikey_t key, bool check) {
        if (!check)
            return;
        uint64_t vsz = 0;
        // get all local types
        edge_t *vres = get_edges_local(0, 0, OUT, TYPE_ID, &vsz);
        bool found = false;
        // check whether the pid exists or duplicate
        for (int i = 0; i < vsz; i++) {
            if (vres[i].val == key.pid && !found)
                found = true;
            else if (vres[i].val == key.pid && found) {
                logstream(LOG_ERROR) << "In the value part of all local types [ 0 | TYPE_ID | OUT ]"
                                     << " there is duplicate value " << key.pid << LOG_endl;
            }
        }
        // pid does not exist in local types, maybe it is predicate
        if (!found) {
            uint64_t psz = 0;
            // get all local predicates
            edge_t *pres = get_edges_local(0, 0, OUT, PREDICATE_ID, &psz);
            bool found = false;
            // check whether the pid exists or duplicate
            for (int i = 0; i < psz; i++) {
                if (pres[i].val == key.pid && !found)
                    found = true;
                else if (pres[i].val == key.pid && found) {
                    logstream(LOG_ERROR) << "In the value part of all local predicates [ 0 | PREDICATE_ID | OUT ]"
                                         << " there is duplicate value " << key.pid << LOG_endl;
                    break;
                }
            }
            if (!found) {
                logstream(LOG_ERROR) << "if " << key.pid << "is predicate, in the value part of all local predicates [ 0 | PREDICATE_ID | OUT ]"
                                     << " there is NO value " << key.pid << LOG_endl;
                logstream(LOG_ERROR) << "if " << key.pid << " is type, in the value part of all local types [ 0 | TYPE_ID | OUT ]"
                                     << " there is NO value " << key.pid << LOG_endl;
            }
            uint64_t vsz = 0;
            // get the vid refered which refered by the type/predicate
            edge_t *vres = get_edges_local(0, 0, IN, key.pid, &vsz);
            if (vsz == 0) {
                logstream(LOG_ERROR) << "if " << key.pid << " is type, in the value part of all local types [ 0 | TYPE_ID | OUT ]"
                                     << " there is NO value " << key.pid << LOG_endl;
                return;
            }
            for (int i = 0; i < vsz; i++) {
                found = false;
                uint64_t sosz = 0;
                // get all local objects/subjects
                edge_t *sores = get_edges_local(0, 0, IN, TYPE_ID, &sosz);
                for (int j = 0; j < sosz; j++) {
                    if (sores[j].val == vres[i].val && !found)
                        found = true;
                    else if (sores[j].val == vres[i].val && found) {
                        logstream(LOG_ERROR) << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                                             << " there is duplicate value " << vres[i].val << LOG_endl;
                        break;
                    }
                }
                if (!found) {
                    logstream(LOG_ERROR) << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                                         << " there is no value " << vres[i].val << LOG_endl;
                }
                found = false;
                uint64_t p2sz = 0;
                // get vid's all predicate
                edge_t *p2res = get_edges_local(0, vres[i].val, OUT, PREDICATE_ID, &p2sz);
                for (int j = 0; j < p2sz; j++) {
                    if (p2res[j].val == key.pid && !found)
                        found = true;
                    else if (p2res[j].val == key.pid && found) {
                        logstream(LOG_ERROR) << "In the value part of " << vres[i].val << "'s all predicates [ "
                                             << vres[i].val << " | PREDICATE_ID | OUT ], there is duplicate value "
                                             << key.pid << LOG_endl;
                        break;
                    }
                }
                if (!found) {
                    logstream(LOG_ERROR) << "In the value part of " << vres[i].val << "'s all predicates [ "
                                         << vres[i].val << " | PREDICATE_ID | OUT ], there is no value "
                                         << key.pid << LOG_endl;
                }
            }
        }
    }

    // check on out-dir predicate-index
    void ver_idx_check_outdir(ikey_t key, bool check) {
        if (!check)
            return;
        uint64_t psz = 0;
        // get all local predicates
        edge_t *pres = get_edges_local(0, 0, OUT, PREDICATE_ID, &psz);
        bool found = false;
        // check whether the pid exists or duplicate
        for (int i = 0; i < psz; i++) {
            if (pres[i].val == key.pid && !found)
                found = true;
            else if (pres[i].val == key.pid && found) {
                logstream(LOG_ERROR) << "In the value part of all local predicates [ 0 | PREDICATE_ID | OUT ]"
                                     << " there is duplicate value " << key.pid << LOG_endl;
                break;
            }
        }
        if (!found) {
            logstream(LOG_ERROR) << "In the value part of all local predicates [ 0 | PREDICATE_ID | OUT ]"
                                 << " there is no value " << key.pid << LOG_endl;
        }
        uint64_t vsz = 0;
        // get the vid refered which refered by the predicate
        edge_t *vres = get_edges_local(0, 0, OUT, key.pid, &vsz);
        for (int i = 0; i < vsz; i++) {
            found = false;
            uint64_t sosz = 0;
            // get all local objects/subjects
            edge_t *sores = get_edges_local(0, 0, IN, TYPE_ID, &sosz);
            for (int j = 0; j < sosz; j++) {
                if (sores[j].val == vres[i].val && !found)
                    found = true;
                else if (sores[j].val == vres[i].val && found) {
                    logstream(LOG_ERROR) << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                                         << " there is duplicate value " << vres[i].val << LOG_endl;
                    break;
                }
            }
            if (!found) {
                logstream(LOG_ERROR) << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                                     << " there is no value " << vres[i].val << LOG_endl;
            }
            found = false;
            uint64_t psz = 0;
            // get vid's all predicate
            edge_t *pres = get_edges_local(0, vres[i].val, IN, PREDICATE_ID, &psz);
            for (int j = 0; j < psz; j++) {
                if (pres[j].val == key.pid && !found)
                    found = true;
                else if (pres[j].val == key.pid && found) {
                    logstream(LOG_ERROR) << "In the value part of " << vres[i].val << "'s all predicates [ "
                                         << vres[i].val << "PREDICATE_ID | IN ], there is duplicate value "
                                         << key.pid << LOG_endl;
                    break;
                }
            }
            if (!found) {
                logstream(LOG_ERROR) << "In the value part of " << vres[i].val << "'s all predicates [ "
                                     << vres[i].val << "PREDICATE_ID | IN ], there is no value "
                                     << key.pid << LOG_endl;
            }
        }
    }

    // check on normal types (7)
    void ver_nt_check(ikey_t key, bool check) {
        if (!check)
            return;
        bool found = false;
        uint64_t psz = 0;
        // get vid' all predicates
        edge_t *pres = get_edges_local(0, key.vid, OUT, PREDICATE_ID, &psz);
        // check if there is TYPE_ID in vid's predicates
        for (int i = 0; i < psz; i++) {
            if (pres[i].val == key.pid && !found)
                found = true;
            else if (pres[i].val == key.pid && found) {
                logstream(LOG_ERROR) << "In the value part of "
                                     << key.vid << "'s all predicates [ "
                                     << key.vid << "PREDICATE_ID | OUT ], there is DUPLICATE value "
                                     << key.pid << LOG_endl;
                break;
            }
        }
        if (!found) {
            logstream(LOG_ERROR) << "In the value part of "
                                 << key.vid << "'s all predicates [ "
                                 << key.vid << "PREDICATE_ID | OUT ], there is NO value "
                                 << key.pid << LOG_endl;
        }
        found = false;
        uint64_t ossz = 0;
        // get all local subjects/objects
        edge_t *osres = get_edges_local(0, 0, IN, key.pid, &ossz);
        for (int i = 0; i < ossz; i++) {
            if (osres[i].val == key.vid && !found)
                found = true;
            else if (osres[i].val == key.vid && found) {
                logstream(LOG_ERROR) << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                                     << " there is DUPLICATE value " << key.vid << LOG_endl;
                break;
            }
        }
        if (!found) {
            logstream(LOG_ERROR) << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                                 << " there is NO value " << key.vid << LOG_endl;
        }
    }
#else // !VERSATILE
    void ver_idx_check_indir(ikey_t key, bool check) { }

    void ver_idx_check_outdir(ikey_t key, bool check) { }

    void ver_nt_check(ikey_t key, bool check) { }
#endif // end of VERSATILE

    void check_on_vertex(ikey_t key, bool index_check, bool normal_check) {
        if (key.vid == 0 && is_tpid(key.pid) && key.dir == IN) // (2)/(1)[IN]
            idx_check_indir(key, index_check);
        else if (key.vid == 0 && is_tpid(key.pid) && key.dir == OUT) // (1)[OUT]
            idx_check_outdir(key, index_check);
        else if (is_vid(key.vid) && key.pid == TYPE_ID && key.dir == OUT) // (7)
            nt_check(key, normal_check);
        else if (is_vid(key.vid) && is_tpid(key.pid) && key.dir == OUT) // (6)[OUT]
            np_check(key, IN, normal_check);
        else if (is_vid(key.vid) && is_tpid(key.pid) && key.dir == IN) // (6)[IN]
            np_check(key, OUT, normal_check);
    }

    int gstore_check(bool index_check, bool normal_check) {
        logstream(LOG_INFO) << "Graph storage intergity check has started on server " << sid << LOG_endl;
        ivertex_num = 0;
        nvertex_num = 0;
        for (uint64_t bucket_id = 0; bucket_id < num_buckets + num_buckets_ext; bucket_id++) {
            uint64_t slot_id = bucket_id * ASSOCIATIVITY;
            for (int i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++)
                if (!vertices[slot_id].key.is_empty())
                    check_on_vertex(vertices[slot_id].key, index_check, normal_check);
        }

        logstream(LOG_INFO) << "Server#" << sid << " has checked "
                            << ivertex_num << " index vertices and "
                            << nvertex_num << " normal vertices." << LOG_endl;
        return 0;
    }

    // prepare data for planner
    void generate_statistic(data_statistic &stat) {
#ifndef VERSATILE
        logstream(LOG_ERROR) << "please turn off generate_statistics in config "
                             << "and use stat file cache instead OR turn on VERSATILE option "
                             << "in CMakefiles to generate_statistic." << LOG_endl;
        exit(-1);
#endif

        unordered_map<ssid_t, int> &tyscount = stat.local_tyscount;
        type_stat &ty_stat = stat.local_tystat;
        // for complex type vertex numbering
        unordered_set<ssid_t> record_set;

        //use index_composition as type of no_type
        auto generate_no_type = [&](ssid_t id) -> ssid_t {
            type_t type;
            uint64_t psize1 = 0;
            unordered_set<int> index_composition;

            edge_t *res1 = get_edges_global(0, id, OUT, PREDICATE_ID, &psize1);
            for (uint64_t k = 0; k < psize1; k++) {
                ssid_t pre = res1[k].val;
                index_composition.insert(pre);
            }

            uint64_t psize2 = 0;
            edge_t *res2 = get_edges_global(0, id, IN, PREDICATE_ID, &psize2);
            for (uint64_t k = 0; k < psize2; k++) {
                ssid_t pre = res2[k].val;
                index_composition.insert(-pre);
            }

            type.set_index_composition(index_composition);
            // TODO: there should be no following situation according to comments
            // on gstore layout, but actually it happends 25 times and will not affect
            // the correctness of optimizer
            // if(index_composition.size() == 0){
            //     cout << "empty index, may be type" << endl;
            // }
            return stat.get_simple_type(type);
        };

        //use type_composition as type of no_type
        auto generate_multi_type = [&](edge_t *res, uint64_t type_sz) -> ssid_t {
            type_t type;
            unordered_set<int> type_composition;
            for (int i = 0; i < type_sz; i ++)
                type_composition.insert(res[i].val);

            type.set_type_composition(type_composition);
            return stat.get_simple_type(type);
        };

        // return success or not, because one id can only be recorded once
        auto insert_no_type_count = [&](ssid_t id, ssid_t type) -> bool{
            if (record_set.count(id) > 0) {
                return false;
            } else{
                record_set.insert(id);

                if (tyscount.find(type) == tyscount.end())
                    tyscount[type] = 1;
                else
                    tyscount[type]++;
                return true;
            }
        };

        for (uint64_t bucket_id = 0; bucket_id < num_buckets + num_buckets_ext; bucket_id++) {
            uint64_t slot_id = bucket_id * ASSOCIATIVITY;
            for (int i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++) {
                // skip empty slot
                if (vertices[slot_id].key.is_empty()) continue;

                sid_t vid = vertices[slot_id].key.vid;
                sid_t pid = vertices[slot_id].key.pid;

                uint64_t sz = vertices[slot_id].ptr.size;
                uint64_t off = vertices[slot_id].ptr.off;
                if (vid == PREDICATE_ID || pid == PREDICATE_ID)
                    continue; // skip for index vertex

                if (vertices[slot_id].key.dir == IN) {
                    // for type derivation
                    // get types of values found by key (Subjects)
                    vector<ssid_t> res_type;
                    for (uint64_t k = 0; k < sz; k++) {
                        ssid_t sbid = edges[off + k].val;
                        uint64_t type_sz = 0;
                        edge_t *res = get_edges_global(0, sbid, OUT, TYPE_ID, &type_sz);
                        if (type_sz > 1) {
                            ssid_t type = generate_multi_type(res, type_sz);
                            res_type.push_back(type); //10 for 10240, 19 for 2560, 23 for 40, 2 for 640
                        } else if (type_sz == 0) {
                            //cout << "no type: " << sbid << endl;
                            ssid_t type = generate_no_type(sbid);
                            res_type.push_back(type);
                        } else if (type_sz == 1) {
                            res_type.push_back(res[0].val);
                        } else {
                            assert(false);
                        }
                    }

                    // type for objects
                    // get type of vid (Object)
                    uint64_t type_sz = 0;
                    edge_t *res = get_edges_local(0, vid, OUT, TYPE_ID, &type_sz);
                    ssid_t type;
                    if (type_sz > 1) {
                        type = generate_multi_type(res, type_sz);
                    } else {
                        if (type_sz == 0) {
                            //cout << "no type: " << vid << endl;
                            type = generate_no_type(vid);
                            insert_no_type_count(vid, type);
                        } else {
                            type = res[0].val;
                        }
                    }

                    ty_stat.insert_otype(pid, type, 1);
                    for (int j = 0; j < res_type.size(); j++)
                        ty_stat.insert_finetype(pid, type, res_type[j], 1);
                } else {
                    // no_type only need to be counted in one direction (using OUT)
                    // get types of values found by key (Objects)
                    vector<ssid_t> res_type;
                    for (uint64_t k = 0; k < sz; k++) {
                        ssid_t obid = edges[off + k].val;
                        uint64_t type_sz = 0;
                        edge_t *res = get_edges_global(0, obid, OUT, TYPE_ID, &type_sz);

                        if (type_sz > 1) {
                            ssid_t type = generate_multi_type(res, type_sz);
                            res_type.push_back(type);
                        } else if (type_sz == 0) {
                            // in this situation, obid may be some TYPE
                            if (pid != 1) {
                                logstream(LOG_DEBUG) << "[DEBUG] no type: " << obid << LOG_endl;
                                ssid_t type = generate_no_type(obid);
                                res_type.push_back(type);
                            }
                        } else if (type_sz == 1) {
                            res_type.push_back(res[0].val);
                        } else {
                            assert(false);
                        }
                    }

                    // type for subjects
                    // get type of vid (Subject)
                    uint64_t type_sz = 0;
                    edge_t *res = get_edges_local(0, vid, OUT, TYPE_ID, &type_sz);
                    ssid_t type;
                    if (type_sz > 1) {
                        type = generate_multi_type(res, type_sz);
                    } else {
                        if (type_sz == 0) {
                            // cout << "no type: " << vid << endl;
                            type = generate_no_type(vid);
                            insert_no_type_count(vid, type);
                        } else {
                            type = res[0].val;
                        }
                    }

                    ty_stat.insert_stype(pid, type, 1);
                    for (int j = 0; j < res_type.size(); j++)
                        ty_stat.insert_finetype(type, pid, res_type[j], 1);

                    // count type predicate
                    if (pid == TYPE_ID) {
                        // multi-type
                        if (sz > 1) {
                            type_t complex_type;
                            unordered_set<int> type_composition;
                            for (int i = 0; i < sz; i ++)
                                type_composition.insert(edges[off + i].val);

                            complex_type.set_type_composition(type_composition);
                            ssid_t type_number = stat.get_simple_type(complex_type);

                            if (tyscount.find(type_number) == tyscount.end())
                                tyscount[type_number] = 1;
                            else
                                tyscount[type_number]++;
                        } else if (sz == 1) { // single type
                            sid_t obid = edges[off].val;

                            if (tyscount.find(obid) == tyscount.end())
                                tyscount[obid] = 1;
                            else
                                tyscount[obid]++;
                        }
                    }
                }
            }
        }

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
#ifdef DYNAMIC_GSTORE
        edge_allocator->print_memory_usage();
#else
        logstream(LOG_INFO) << "\tused: " << 100.0 * last_entry / num_entries
                            << " % (" << last_entry << " entries)" << LOG_endl;
#endif

        // uint64_t sz = 0;
        // get_edges_local(0, 0, IN, TYPE_ID, &sz);
        // logstream(LOG_INFO) << "#vertices: " << sz << LOG_endl;
        // get_edges_local(0, 0, OUT, TYPE_ID, &sz);
        // logstream(LOG_INFO) << "#predicates: " << sz << LOG_endl;
    }
};
