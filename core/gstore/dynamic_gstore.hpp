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

#include "mm/malloc_interface.hpp"
#include "mm/jemalloc.hpp"
#include "mm/buddy_malloc.hpp"
#include "gstore.hpp"

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
    /* Check the validation of given edge according to given vertex.
     * The edge is valid only when the size flag of edge is consistent with the size within the vertex.
     */
    inline bool edge_is_valid(vertex_t &v, edge_t *edge_ptr) {
        if (!global_enable_caching)
            return true;
        uint64_t blk_sz = blksz(v.ptr.size + 1);  // reserve one space for flag
        return (edge_ptr[blk_sz - 1].val == v.ptr.size);
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

    // Allocate space to store edges of given size.
    // Return offset of allocated space.
    uint64_t alloc_edges(uint64_t n, int64_t tid = 0) {
        if (global_enable_caching)
            sweep_free(); // collect free space before allocate
        uint64_t sz = e2b(n + 1); // reserve one space for sz
        uint64_t off = b2e(edge_allocator->malloc(sz, tid));
        insert_sz(n, n, off);
        return off;
    }

    uint64_t bucket_local(ikey_t key) {
        uint64_t bucket_id;
        bucket_id = key.hash() % num_buckets;
        return bucket_id;
    }

    uint64_t bucket_remote(ikey_t key, int dst_sid) {
        uint64_t bucket_id;
        bucket_id = key.hash() % num_buckets;
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
            vertices[slot_id].key.vid = alloc_ext_buckets(1);
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
    void insert_attr(vector<triple_attr_t> &attrs, int64_t tid) {
        variant_type get_type;
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
        edge_allocator->merge_freelists();

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

    edge_t *get_edges_remote(int tid, sid_t vid, sid_t pid, dir_t d, uint64_t &sz, int &type = *(int *)NULL) {
        ikey_t key = ikey_t(vid, pid, d);
        vertex_t v = get_vertex_remote(tid, key);
        if (v.key.is_empty()) {
            sz = 0;
            return NULL; // not found
        }
        // remote edges
        int dst_sid = wukong::math::hash_mod(vid, global_num_servers);
        edge_t *edge_ptr = rdma_get_edges(tid, dst_sid, v);
        // check the validation of remote edges
        while (!edge_is_valid(v, edge_ptr)) {
            // invalidate local cache and try again
            rdma_cache.invalidate(key);
            v = get_vertex_remote(tid, key);
            edge_ptr = rdma_get_edges(tid, dst_sid, v);
        }
        sz = v.ptr.size;
        if (&type != NULL)
            type = v.ptr.type;
        return edge_ptr;
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

    void init(vector<vector<triple_t>> &triple_pso, vector<vector<triple_t>> &triple_pos, vector<vector<triple_attr_t>> &triple_sav) {
        uint64_t start, end;
        start = timer::get_usec();
        #pragma omp parallel for num_threads(global_num_engines)
        for (int t = 0; t < global_num_engines; t++) {
            insert_normal(triple_pso[t], triple_pos[t], t);
            // release memory
            vector<triple_t>().swap(triple_pso[t]);
            vector<triple_t>().swap(triple_pos[t]);
        }
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting normal data into gstore" << LOG_endl;

        start = timer::get_usec();
        #pragma omp parallel for num_threads(global_num_engines)
        for (int t = 0; t < global_num_engines; t++) {
            insert_attr(triple_sav[t], t);

            // release memory
            vector<triple_attr_t>().swap(triple_sav[t]);
        }
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting attributes into gstore" << LOG_endl;

        start = timer::get_usec();
        insert_index();
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting index data into gstore" << LOG_endl;
    }

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
