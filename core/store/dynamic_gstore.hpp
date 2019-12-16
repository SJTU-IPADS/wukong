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

    /**
     * global dynamic reserve factor
     * when creating new segment during dynamic loading,
     * #buckets = (#buckets-remain * global_dyn_res_factor / 100) / #new-segments
     */
    int global_dyn_res_factor = 50;

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
        uint64_t bucket_id = bucket_local(key);
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
        uint64_t bucket_id = bucket_local(key);
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

                if (Global::enable_caching)
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

    // Allocate space to store edges of given size.
    // @return offset of allocated space.
    uint64_t alloc_edges(uint64_t n, int tid = 0, rdf_seg_meta_t *seg = NULL) {
        if (Global::enable_caching)
            sweep_free(); // collect free space before allocate

        uint64_t sz = e2b(n + 1); // reserve one space for sz
        uint64_t off = b2e(edge_allocator->malloc(sz, tid));
        insert_sz(n, n, off);
        return off;
    }

    // dynamic store doesn't group edges into segments
    uint64_t alloc_edges_to_seg(uint64_t num_edges) { return 0; }

    /* Check the validation of given edge according to given vertex.
     * The edge is valid only when the size flag of edge is consistent with the size within the vertex.
     */
    bool edge_is_valid(vertex_t &v, edge_t *edge_ptr) {
        if (!Global::enable_caching)
            return true;

        uint64_t blk_sz = blksz(v.ptr.size + 1);  // reserve one space for flag
        return (edge_ptr[blk_sz - 1].val == v.ptr.size);
    }

    uint64_t get_edge_sz(const vertex_t &v) { return blksz(v.ptr.size + 1) * sizeof(edge_t); }

    void insert_idx(const tbb_hash_map &pidx_map, const tbb_hash_map &tidx_map,
                    dir_t d, int tid = 0) {
        tbb_hash_map::const_accessor ca;
        rdf_seg_meta_t &segment = rdf_seg_meta_map[segid_t(1, PREDICATE_ID, d)];
        // it is possible that num_edges = 0 if loading an empty dataset
        // ASSERT(segment.num_edges > 0);

        for (int i = 0; i < all_local_preds.size(); i++) {
            sid_t pid = all_local_preds[i];
            bool success = pidx_map.find(ca, pid);
            if (!success)
                continue;

            uint64_t sz = ca->second.size();
            ASSERT(sz <= segment.num_edges);

            ikey_t key = ikey_t(0, pid, d);
            uint64_t off = alloc_edges(sz, tid);
            logger(LOG_DEBUG, "insert_pidx[%s]: key: [%lu|%lu|%lu] sz: %lu",
                   (d == IN) ? "IN" : "OUT", key.vid, key.pid, key.dir, sz);
            uint64_t slot_id = insert_key(key);
            iptr_t ptr = iptr_t(sz, off);
            vertices[slot_id].ptr = ptr;

            for (auto const &vid : ca->second)
                edges[off++].val = vid;
        }
        // type index
        if (d == IN) {
            for (auto const &e : tidx_map) {
                sid_t pid = e.first;
                uint64_t sz = e.second.size();
                ASSERT(sz <= segment.num_edges);
                logger(LOG_DEBUG, "insert_tidx: pid: %lu, sz: %lu", pid, sz);

                ikey_t key = ikey_t(0, pid, IN);
                uint64_t off = alloc_edges(sz, tid);
                uint64_t slot_id = insert_key(key);
                iptr_t ptr = iptr_t(sz, off);
                vertices[slot_id].ptr = ptr;

                for (auto const &vid : e.second)
                    edges[off++].val = vid;
            }
        }
#ifdef VERSATILE
        if (d == IN) {
            // all local entities, key: [0 | TYPE_ID | IN]
            uint64_t off = alloc_edges(v_set.size(), tid);
            insert_idx_set(v_set, off, TYPE_ID, IN);
            tbb_unordered_set().swap(v_set);
        } else {
            uint64_t off = alloc_edges(t_set.size(), tid);
            // all local types, key: [0 | TYPE_ID | OUT]
            insert_idx_set(t_set, off, TYPE_ID, OUT);
            tbb_unordered_set().swap(t_set);
            off = alloc_edges(p_set.size(), tid);
            // all local predicates, key: [0 | PREDICATE_ID | OUT]
            insert_idx_set(p_set, off, PREDICATE_ID, OUT);
            tbb_unordered_set().swap(p_set);
        }
#endif // VERSATILE
    }

#ifdef VERSATILE
    void alloc_vp_edges(dir_t d) {
        return;
    }

    // insert vid's preds into gstore
    void insert_vp(int tid, const vector<triple_t> &pso, const vector<triple_t> &pos) {
        vector<sid_t> preds;

        uint64_t s = 0;
        while (s < pso.size()) {
            // predicate-based key (subject + predicate)
            uint64_t e = s + 1;
            while ((e < pso.size())
                    && (pso[s].s == pso[e].s)
                    && (pso[s].p == pso[e].p))  { e++; }

            preds.push_back(pso[s].p);

            // insert a vp key-value pair (OUT)
            if (e >= pso.size() || pso[s].s != pso[e].s) {
                // allocate a vertex and edges
                ikey_t key = ikey_t(pso[s].s, PREDICATE_ID, OUT);
                uint64_t sz = preds.size();
                uint64_t off = alloc_edges(sz, tid);

                // insert a vertex
                uint64_t slot_id = insert_key(key);
                iptr_t ptr = iptr_t(sz, off);
                vertices[slot_id].ptr = ptr;

                // insert edges
                for (auto const &p : preds)
                    edges[off++].val = p;

                preds.clear();
            }
            s = e;
        }

        // treat type triples as index vertices
        uint64_t type_triples = 0;
        while (type_triples < pos.size() && is_tpid(pos[type_triples].o))
            type_triples++;

        s = type_triples; // skip type triples
        while (s < pos.size()) {
            // predicate-based key (object + predicate)
            uint64_t e = s + 1;
            while ((e < pos.size())
                    && (pos[s].o == pos[e].o)
                    && (pos[s].p == pos[e].p)) { e++; }

            // add a new predicate
            preds.push_back(pos[s].p);

            // insert a vp key-value pair (IN)
            if (e >= pos.size() || pos[s].o != pos[e].o) {
                // allocate a vertex and edges
                ikey_t key = ikey_t(pos[s].o, PREDICATE_ID, IN);
                uint64_t sz = preds.size();
                uint64_t off = alloc_edges(sz, tid);

                // insert a vertex
                uint64_t slot_id = insert_key(key);
                iptr_t ptr = iptr_t(sz, off);
                vertices[slot_id].ptr = ptr;

                // insert edges
                for (auto const &p : preds)
                    edges[off++].val = p;

                preds.clear();
            }
            s = e;
        }
    }
#endif // VERSATILE


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
        #pragma omp parallel for num_threads(Global::num_engines)
        for (uint64_t i = 0; i < num_slots; i++) {
            vertices[i].key = ikey_t();
            vertices[i].ptr = iptr_t();
        }

        last_ext = 0;
        // Since tid of engines is not from 0, allocator should init num_threads.
        edge_allocator->init((void *)edges, num_entries * sizeof(edge_t), Global::num_threads);
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

    /**
     * Used during loading data dynamically
     */
    void create_new_seg(sid_t pid, int num_new_preds) {
        // step 1: insert 2 segments into map
        rdf_seg_meta_map.insert(make_pair(segid_t(0, pid, IN), rdf_seg_meta_t()));
        rdf_seg_meta_map.insert(make_pair(segid_t(0, pid, OUT), rdf_seg_meta_t()));
        auto &in_seg = rdf_seg_meta_map[segid_t(0, pid, IN)];
        auto &out_seg = rdf_seg_meta_map[segid_t(0, pid, OUT)];

        // step 2: allocate space for the segments
        in_seg.bucket_start = main_hdr_off;
        in_seg.num_buckets = ((num_buckets - main_hdr_off) * global_dyn_res_factor / 100) / (num_new_preds * PREDICATE_NSEGS);
        main_hdr_off += in_seg.num_buckets;
        ASSERT(main_hdr_off <= num_buckets);
        out_seg.bucket_start = main_hdr_off;
        out_seg.num_buckets = ((num_buckets - main_hdr_off) * global_dyn_res_factor / 100) / (num_new_preds * PREDICATE_NSEGS);
        main_hdr_off += out_seg.num_buckets;
        ASSERT(main_hdr_off <= num_buckets);

        // allocate buckets in indirect-header region to segments
        // #buckets : #extended buckets = 1 : 0.15
        if (in_seg.num_buckets > 0) {
            uint64_t start_off = alloc_ext_buckets(EXT_BUCKET_EXTENT_LEN);
            in_seg.add_ext_buckets(ext_bucket_extent_t(EXT_BUCKET_EXTENT_LEN, start_off));
        }
        if (out_seg.num_buckets > 0) {
            uint64_t start_off = alloc_ext_buckets(EXT_BUCKET_EXTENT_LEN);
            out_seg.add_ext_buckets(ext_bucket_extent_t(EXT_BUCKET_EXTENT_LEN, start_off));
        }
        // step 2: init metadata
        num_normal_preds += 1;
        num_segments += PREDICATE_NSEGS;
    }

    void init(vector<vector<triple_t>> &triple_pso,
              vector<vector<triple_t>> &triple_pos,
              vector<vector<triple_attr_t>> &triple_sav) {
        num_segments = num_normal_preds * PREDICATE_NSEGS + INDEX_NSEGS + num_attr_preds;
        uint64_t start, end;
        start = timer::get_usec();
        init_seg_metas(triple_pso, triple_pos, triple_sav);
        end = timer::get_usec();
        logstream(LOG_DEBUG) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                             << "for initializing predicate segment statistics." << LOG_endl;

#ifdef VERSATILE
        start = timer::get_usec();
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int tid = 0; tid < Global::num_engines; tid++) {
            insert_vp(tid, triple_pso[tid], triple_pos[tid]);
            // Re-sort triples array by pso to accelarate normal triples insert. 
            sort(triple_pso[tid].begin(), triple_pso[tid].end(), triple_sort_by_pso());
            sort(triple_pos[tid].begin(), triple_pos[tid].end(), triple_sort_by_pos());
        }
        end = timer::get_usec();
        logstream(LOG_DEBUG) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                             << "for inserting vid's predicates." << LOG_endl;
#endif // VERSATILE

        start = timer::get_usec();
        logstream(LOG_DEBUG) << "#" << sid << ": all_local_preds: "
                             << all_local_preds.size() << LOG_endl;
        triple_map_t out_triple_map = init_triple_map(triple_pso);
        triple_map_t in_triple_map = init_triple_map(triple_pos);
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < all_local_preds.size(); i++) {
            int localtid = omp_get_thread_num();
            sid_t pid = all_local_preds[i];
            insert_triples(localtid, segid_t(0, pid, OUT), out_triple_map, triple_pso);
            insert_triples(localtid, segid_t(0, pid, IN), in_triple_map, triple_pos);
        }

        triple_map_t attr_map = init_triple_map(triple_sav);
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int localtid = 0; localtid < Global::num_engines; localtid++) {
            int idx = 0;
            for (auto it = attr_set.begin(); it != attr_set.end(); it++, idx++) {
                if (idx % Global::num_engines == localtid)  // partition attr_set
                    insert_attr(localtid, segid_t(0, *it, OUT), attr_map, triple_sav);
            }
        }

        // insert type-index edges in parallel
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < 2; i++) {
            if (i == 0) insert_idx(pidx_in_map, tidx_map, IN, i);
            else insert_idx(pidx_out_map, tidx_map, OUT, i);
        }

        end = timer::get_usec();
        logstream(LOG_DEBUG) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                             << "for inserting triples as segments into gstore" << LOG_endl;

        edge_allocator->merge_freelists();
        finalize_seg_metas();
        finalize_init();
        // synchronize segment metadata among servers
        sync_metadata();
    }
};
