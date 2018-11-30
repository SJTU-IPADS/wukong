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

// store
#include "gstore.hpp"

#include "mm/malloc_interface.hpp"
#include "mm/jemalloc.hpp"
#include "mm/buddy_malloc.hpp"

using namespace std;

class DynamicGStore : public GStore {
private:
    // block deferred to be freed
    struct free_blk {
        uint64_t off;
        uint64_t expire_time;
        free_blk(uint64_t off, uint64_t expire_time): off(off), expire_time(expire_time) { }
    };

    // manage the memory of edges(val)
    MAInterface *edge_allocator;
    /// Defer blk's free operation for dynamic cache.
    /// Pend the free operation when blk is to be collected by add_pending_free()
    /// When allocating a new blk by alloc_edges(), check if pending free's lease expires
    /// and collect free space by sweep_free().
    uint64_t lease;

    /// A size flag is put into the tail of edges (in the entry region) for dynamic cache.
    /// NOTE: the (remote) edges accessed by (local) RDMA cache are valid
    ///       if and only if the size flag of edges is consistent with the size within the pointer.
    static const sid_t INVALID_EDGES = 1 << NBITS_SIZE; // flag indicates invalidate edges

    queue<free_blk> free_queue;
    pthread_spinlock_t free_queue_lock;

    // Convert given byte units to edge units.
    inline uint64_t b2e(uint64_t sz) { return sz / sizeof(edge_t); }
    // Convert given edge uints to byte units.
    inline uint64_t e2b(uint64_t sz) { return sz * sizeof(edge_t); }
    // Return exact block size of given size in edge unit.
    inline uint64_t blksz(uint64_t sz) { return b2e(edge_allocator->sz_to_blksz(e2b(sz))); }
    /* Insert size flag size flag in edges.
     * @flag: size flag to insert
     * @sz: actual size of edges
     * @off: offset of edges
    */
    inline void insert_sz(sid_t flag, uint64_t sz, uint64_t off) {
        uint64_t blk_sz = blksz(sz + 1);   // reserve one space for flag
        edges[off + blk_sz - 1].val = flag;
    }

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

    bool insert_vertex_edge(ikey_t key, sid_t value, bool &dedup_or_isdup, int tid) {
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
            if (blksz(v->ptr.size + 1) - 1 < need_size) {
                iptr_t old_ptr = v->ptr;

                uint64_t off = alloc_edges(need_size, tid);
                memcpy(&edges[off], &edges[old_ptr.off], e2b(old_ptr.size));
                edges[off + old_ptr.size].val = value;
                // invalidate the old block
                insert_sz(INVALID_EDGES, old_ptr.size, old_ptr.off);
                v->ptr = iptr_t(need_size, off);

                if (global_enable_caching)
                    add_pending_free(old_ptr);
                else
                    edge_allocator->free(e2b(old_ptr.off));
            } else {
                // update size flag
                insert_sz(need_size, need_size, v->ptr.off);
                edges[v->ptr.off + v->ptr.size].val = value;
                v->ptr.size = need_size;
            }

            pthread_spin_unlock(&bucket_locks[lock_id]);
            return false;
        }
    }

    uint64_t bucket_local(ikey_t key) {
        uint64_t bucket_id;
        bucket_id = key.hash() % num_buckets;
        return bucket_id;
    }

    uint64_t bucket_remote(ikey_t key, int dst_sid) {
        return bucket_local(key);
    }

    // Allocate space to store edges of given size.
    // @return offset of allocated space.
    uint64_t alloc_edges(uint64_t n, int64_t tid = 0) {
        if (global_enable_caching)
            sweep_free(); // collect free space before allocate
        uint64_t sz = e2b(n + 1); // reserve one space for sz
        uint64_t off = b2e(edge_allocator->malloc(sz, tid));
        insert_sz(n, n, off);
        return off;
    }

    /* Check the validation of given edge according to given vertex.
     * The edge is valid only when the size flag of edge is consistent with the size within the vertex.
     */
    bool edge_is_valid(vertex_t &v, edge_t *edge_ptr) {
        if (!global_enable_caching)
            return true;
        uint64_t blk_sz = blksz(v.ptr.size + 1);  // reserve one space for flag
        return (edge_ptr[blk_sz - 1].val == v.ptr.size);
    }

    // Get edges of given vertex from dst_sid by RDMA read.
    edge_t *rdma_get_edges(int tid, int dst_sid, vertex_t &v) {
        ASSERT(global_use_rdma);
        char *buf = mem->buffer(tid);
        uint64_t r_off = num_slots * sizeof(vertex_t) + v.ptr.off * sizeof(edge_t);
        // the size of entire blk
        uint64_t r_sz = blksz(v.ptr.size + 1) * sizeof(edge_t);
        uint64_t buf_sz = mem->buffer_size();
        ASSERT(r_sz < buf_sz); // enough space to host the edges
        RDMA &rdma = RDMA::get_rdma();
        rdma.dev->RdmaRead(tid, dst_sid, buf, r_sz, r_off);
        return (edge_t *)buf;
    }


public:
    DynamicGStore(int sid, Mem *mem): GStore(sid, mem) {
#ifdef USE_JEMALLOC
        edge_allocator = new JeMalloc();
#else
        edge_allocator = new BuddyMalloc();
#endif // end of USE_JEMALLOC
        pthread_spin_init(&free_queue_lock, 0);
        lease = SEC(600);
        rdma_cache.set_lease(lease);
    }

    ~DynamicGStore() {}

    void refresh() {
        #pragma omp parallel for num_threads(global_num_engines)
        for (uint64_t i = 0; i < num_slots; i++) {
            vertices[i].key = ikey_t();
            vertices[i].ptr = iptr_t();
        }
        last_ext = 0;
        edge_allocator->init((void *)edges, num_entries * sizeof(edge_t), global_num_engines);
    }

    void print_mem_usage() {
        GStore::print_mem_usage();
        edge_allocator->print_memory_usage();
    }

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
};
