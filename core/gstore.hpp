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
 *      http://ipads.se.sjtu.edu.cn/projects/wukong.html
 *
 */

#pragma once

#include <stdint.h> //uint64_t
#include <vector>
#include <iostream>
#include <pthread.h>
#include <boost/unordered_set.hpp>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_set.h>

#include "config.hpp"
#include "rdma_resource.hpp"
#include "graph_basic_types.hpp"

#include "mymath.hpp"
#include "timer.hpp"
#include "unit.hpp"

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

    static const int KEY_RATIO = 6;     // key_ratio : 1 = direct : indirect in (key) header
    static const int ASSOCIATIVITY = 8; // the associativity of slots in each bucket

    uint64_t sid;
    RdmaResource *rdma;

    vertex_t *vertices;
    edge_t *edges;


    // the size of slot is sizeof(vertex_t)
    // the size of entry is sizeof(edge_t)
    uint64_t num_slots;       // 1 bucket = ASSOCIATIVITY slots
    uint64_t num_buckets;     // main-header region (pre-allocated hash-table)
    uint64_t num_buckets_ext; // indirect-header region (dynamical allocation)
    uint64_t num_entries;     // entry region (dynamical allocation)

    // allocated
    uint64_t last_ext;
    uint64_t last_entry;


    RDMA_Cache rdma_cache;

    pthread_spinlock_t entry_lock;
    pthread_spinlock_t bucket_ext_lock;
    pthread_spinlock_t bucket_locks[NUM_LOCKS]; // lock virtualization (see paper: vLokc CGO'13)

    // cluster chaining hash-table (see paper: DrTM SOSP'15)
    uint64_t insert_key(ikey_t key) {
        uint64_t bucket_id = key.hash() % num_buckets;
        uint64_t slot_id = bucket_id * ASSOCIATIVITY;
        uint64_t lock_id = bucket_id % NUM_LOCKS;

        bool found = false;
        pthread_spin_lock(&bucket_locks[lock_id]);
        while (slot_id < num_slots) {
            // the last slot of each bucket is always reserved for pointer to indirect header
            /// TODO: add type info to slot and resue the last slot to store key
            for (uint64_t i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++) {
                //assert(vertices[slot_id].key != key); // no duplicate key
                if (vertices[slot_id].key == key) {
                    key.print();
                    vertices[slot_id].key.print();
                    assert(false);
                }

                // insert to an empty slot
                if (vertices[slot_id].key == ikey_t()) {
                    vertices[slot_id].key = key;
                    goto done;
                }
            }

            // move to the last slot of bucket and check whether a bucket_ext is used
            if (vertices[++slot_id].key != ikey_t()) {
                slot_id = vertices[slot_id].key.vid * ASSOCIATIVITY;
                continue; // continue and jump to next bucket
            }


            // allocate and link a new indirect header
            pthread_spin_lock(&bucket_ext_lock);
            assert(last_ext < num_buckets_ext);
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

    uint64_t sync_fetch_and_alloc_edges(uint64_t n) {
        uint64_t orig;
        pthread_spin_lock(&entry_lock);
        orig = last_entry;
        last_entry += n;
        assert(last_entry < num_entries);
        pthread_spin_unlock(&entry_lock);
        return orig;
    }

    vertex_t get_vertex_local(ikey_t key) {
        uint64_t bucket_id = key.hash() % num_buckets;
        while (true) {
            for (uint64_t i = 0; i < ASSOCIATIVITY; i++) {
                uint64_t slot_id = bucket_id * ASSOCIATIVITY + i;
                if (i < ASSOCIATIVITY - 1) {
                    //data part
                    if (vertices[slot_id].key == key) {
                        //we found it
                        return vertices[slot_id];
                    }
                } else {
                    if (vertices[slot_id].key != ikey_t()) {
                        //next pointer
                        bucket_id = vertices[slot_id].key.vid;
                        //break from for loop, will go to next bucket
                        break;
                    } else {
                        return vertex_t();
                    }
                }
            }
        }
    }

    vertex_t get_vertex_remote(int tid, ikey_t key) {
        int dst_sid = mymath::hash_mod(key.vid, global_num_servers);
        uint64_t bucket_id = key.hash() % num_buckets;
        vertex_t ret;

        if (rdma_cache.lookup(key, ret))
            return ret;

        char *buf = rdma->get_buffer(tid);
        while (true) {
            uint64_t off = bucket_id * ASSOCIATIVITY * sizeof(vertex_t);
            uint64_t sz = ASSOCIATIVITY * sizeof(vertex_t);
            rdma->RdmaRead(tid, dst_sid, buf, sz, off);
            vertex_t *ptr = (vertex_t *)buf;
            for (uint64_t i = 0; i < ASSOCIATIVITY; i++) {
                if (i < ASSOCIATIVITY - 1) {
                    if (ptr[i].key == key) {
                        //we found it
                        rdma_cache.insert(ptr[i]);
                        return ptr[i];
                    }
                } else {
                    if (ptr[i].key != ikey_t()) {
                        //next pointer
                        bucket_id = ptr[i].key.vid;
                        //break from for loop, will go to next bucket
                        break;
                    } else {
                        return vertex_t();
                    }
                }
            }
        }
    }

    // TODO: define as public, should be refined
    typedef tbb::concurrent_hash_map<uint64_t, vector< uint64_t>> tbb_hash_map;
    typedef tbb::concurrent_unordered_set<uint64_t> tbb_unordered_set;


    tbb_hash_map src_predicate_table; // predicate-index (IN)
    tbb_hash_map dst_predicate_table; // predicate-index (OUT)
    tbb_hash_map tidx_map; // type-index

#ifdef VAR_PREDICATE
    tbb_unordered_set p_set; // all of predicates
    tbb_unordered_set v_set; // all of vertices (subjects and objects)
#endif

    void insert_hash_map(tbb_hash_map &c, uint64_t key, uint64_t value) {
        tbb_hash_map::accessor a;
        c.insert(a, key);
        a->second.push_back(value);
    }

public:
    // Encoding Rule
    // subject/object (vid) >= 2^17, 2^17 > predicate/type (p/tid) > 2^1,
    // TYPE_ID = 1, PREDICATE_ID = 0, OUT = 1, IN = 0
    //
    // NORMAL key/value pair
    //  key = [vid |    predicate | IN/OUT]  value = [vid0, vid1, ..]  i.e., vid's ngbrs w/ predicate
    //  key = [vid |      TYPE_ID |    OUT]  value = [tid0, tid1, ..]  i.e., vid's all types
    //  key = [vid | PREDICATE_ID | IN/OUT]  value = [pid0, pid1, ..]  i.e., vid's all predicates
    // INDEX key/value pair
    //  key = [  0 |          pid | IN/OUT]  value = [vid0, vid1, ..]  i.e., predicate-index
    //  key = [  0 |          tid |     IN]  value = [vid0, vid1, ..]  i.e., type-index
    //  key = [  0 | PREDICATE_ID |    OUT]  value = [vid0, vid1, ..]  i.e., all objects/subjects
    // Empty key
    //  key = [  0 |            0 |      0]  value = [vid0, vid1, ..]  i.e., init


    // GStore: key (main-header and indirect-header region) | value (entry region)
    // The key (head region) is a cluster chaining hash-table (with associativity)
    // The value (entry region) is a varying-size array
    GStore(RdmaResource *rdma, uint64_t sid): rdma(rdma), sid(sid) {
        num_slots = global_num_keys_million * 1000 * 1000;
        num_buckets = (num_slots / ASSOCIATIVITY) / (KEY_RATIO + 1) * KEY_RATIO;
        num_buckets_ext = (num_slots / ASSOCIATIVITY) / (KEY_RATIO + 1);

        vertices = (vertex_t *)(rdma->get_kvs());
        edges = (edge_t *)(rdma->get_kvs() + num_slots * sizeof(vertex_t));

        if (rdma->get_kvs_size() <= num_slots * sizeof(vertex_t)) {
            cout << "ERROR: " << global_memstore_size_gb
                 << "GB memory store is not enough to store hash table with "
                 << global_num_keys_million << "M keys" << std::endl;
            assert(false);
        }

        num_entries = (rdma->get_kvs_size() - num_slots * sizeof(vertex_t)) / sizeof(edge_t);
        last_entry = 0;

        pthread_spin_init(&entry_lock, 0);
        pthread_spin_init(&bucket_ext_lock, 0);
        for (int i = 0; i < NUM_LOCKS; i++)
            pthread_spin_init(&bucket_locks[i], 0);
    }

    void init() {
        // initiate keys
        #pragma omp parallel for num_threads(global_num_engines)
        for (uint64_t i = 0; i < num_slots; i++)
            vertices[i].key = ikey_t();
    }

    // skip all TYPE triples (e.g., <http://www.Department0.University0.edu> rdf:type ub:University)
    // because Wukong treats all TYPE triples as index vertices. In addition, the triples in triple_ops
    // has been sorted by the vid of object, and IDs of types are always smaller than normal vertex IDs.
    // Consequently, all TYPE triples are aggregated at the beggining of triple_ops
    void atomic_batch_insert(vector<triple_t> &spo, vector<triple_t> &ops) {
        uint64_t type_triples = 0;
        while (type_triples < ops.size() && is_idx(ops[type_triples].o))
            type_triples++;

        // the number of separate combinations of subject/object and predicate
        uint64_t accum_predicate = 0;

        // allocate edges in entry region for triples
        uint64_t off = sync_fetch_and_alloc_edges(spo.size() + ops.size() - type_triples);

        uint64_t s = 0;
        while (s < spo.size()) {
            // predicate-based key (subject + predicate)
            uint64_t e = s + 1;
            while ((e < spo.size())
                    && (spo[s].s == spo[e].s)
                    && (spo[s].p == spo[e].p))  { e++; }

            accum_predicate++;

            // insert vertex
            ikey_t key = ikey_t(spo[s].s, OUT, spo[s].p);
            uint64_t vertex_ptr = insert_key(key);
            iptr_t ptr = iptr_t(e - s, off);
            vertices[vertex_ptr].ptr = ptr;

            // insert edges
            for (uint64_t i = s; i < e; i++)
                edges[off++].val = spo[i].o;

            s = e;
        }

        s = type_triples;
        while (s < ops.size()) {
            // predicate-based key (object + predicate)
            uint64_t e = s + 1;
            while ((e < ops.size())
                    && (ops[s].o == ops[e].o)
                    && (ops[s].p == ops[e].p)) { e++; }

            accum_predicate++;

            // insert vertex
            ikey_t key = ikey_t(ops[s].o, IN, ops[s].p);
            uint64_t slot_id = insert_key(key);
            iptr_t ptr = iptr_t(e - s, off);
            vertices[slot_id].ptr = ptr;

            // insert edges
            for (uint64_t i = s; i < e; i++)
                edges[off++].val = ops[i].s;

            s = e;
        }

#ifdef VAR_PREDICATE
        // The following code is used to support a rare case where the predicate is unknown
        // (e.g., <http://www.Department0.University0.edu> ?P ?O). Each normal vertex should
        // add two key/value pairs with a reserved ID (i.e., PREDICATE_ID) as the predicate
        // to store the IN and OUT lists of its predicates.
        // e.g., key=(vid, PREDICATE_ID, IN/OUT), val=(predicate0, predicate1, ...)
        //
        // NOTE, it is disabled by default in order to save memory.

        // allocate edges in entry region for special PREDICATE triples
        off = sync_fetch_and_alloc_edges(accum_predicate);

        s = 0;
        while (s < spo.size()) {
            // insert vertex
            ikey_t key = ikey_t(spo[s].s, OUT, PREDICATE_ID);
            uint64_t slot_id = insert_key(key);

            // insert edges
            uint64_t e = s, sz = 0;
            do {
                uint64_t m = e;
                edges[off++].val = spo[e++].p; // insert a new predicate
                sz++;

                // skip the triples with the same subject and predicate
                while ((e < spo.size())
                        && (spo[s].s == spo[e].s)
                        && (spo[m].p == spo[e].p)) { e++; }
            } while (e < spo.size() && spo[s].s == spo[e].s);

            // link to edges
            iptr_t ptr = iptr_t(sz, off - sz);
            vertices[slot_id].ptr = ptr;

            s = e;
        }

        s = type_triples;
        while (s < ops.size()) {
            // insert vertex
            ikey_t key = ikey_t(ops[s].o, IN, PREDICATE_ID);
            uint64_t slot_id = insert_key(key);

            // insert edges
            uint64_t e = s, sz = 0;
            do {
                uint64_t m = e;
                edges[off++].val = ops[e++].p; // insert a new predicate
                sz++;

                // skip the triples with the same object and predicate
                while ((e < ops.size())
                        && (ops[s].o == ops[e].o)
                        && (ops[m].p == ops[e].p)) { e++; }
            } while (e < ops.size() && ops[s].o == ops[e].o);

            // link to edges
            iptr_t ptr = iptr_t(sz, off - sz);
            vertices[slot_id].ptr = ptr;

            s = e;
        }
#endif
    }

    // NORMAL key/value pair
    //   key = [vid |    predicate | IN/OUT]  value = [vid0, vid1, ..]  i.e., vid's ngbrs w/ predicate
    //   key = [vid |      TYPE_ID |    OUT]  value = [tid0, tid1, ..]  i.e., vid's all types
    //   key = [vid | PREDICATE_ID | IN/OUT]  value = [pid0, pid1, ..]  i.e., vid's all predicates
    // INDEX key/value pair
    //   key = [  0 |          pid | IN/OUT]  value = [vid0, vid1, ..]  i.e., predicate-index
    //   key = [  0 |          tid |     IN]  value = [vid0, vid1, ..]  i.e., type-index
    //   key = [  0 |      TYPE_ID |    OUT]  value = [vid0, vid1, ..]  i.e., all objects/subjects
    //   key = [  0 |      TYPE_ID |    OUT]  value = [vid0, vid1, ..]  i.e., all predicates
    // Empty key
    //   key = [  0 |            0 |      0]  value = [vid0, vid1, ..]  i.e., init

    void init_index_table(void) {
        uint64_t t1 = timer::get_usec();

        #pragma omp parallel for num_threads(global_num_engines)
        for (int bucket_id = 0; bucket_id < num_buckets + num_buckets_ext; bucket_id++) {
            uint64_t slot_id = bucket_id * ASSOCIATIVITY;
            for (int i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++) {
                // empty slot, skip it
                if (vertices[slot_id].key == ikey_t()) continue;

                uint64_t vid = vertices[slot_id].key.vid;
                uint64_t pid = vertices[slot_id].key.pid;
                uint64_t sz = vertices[slot_id].ptr.size;
                uint64_t off = vertices[slot_id].ptr.off;

                if (vertices[slot_id].key.dir == IN) {
                    if (pid == PREDICATE_ID) {
#ifdef VAR_PREDICATE
                        v_set.insert(vid);
                        for (uint64_t e = 0; e < sz; e++)
                            p_set.insert(edges[off + e].val);
#endif
                    } else if (pid == TYPE_ID) {
                        // the (IN) type triples has been skipped
                        assert(false);
                    } else {
                        // predicate-index (OUT) vid
                        insert_hash_map(dst_predicate_table, pid, vid);
                    }
                } else {
                    if (pid == PREDICATE_ID) {
#ifdef VAR_PREDICATE
                        v_set.insert(vid);
                        for (uint64_t e = 0; e < sz; e++)
                            p_set.insert(edges[off + e].val);
#endif
                    } else if (pid == TYPE_ID) {
                        // type-index (IN) vid
                        for (uint64_t e = 0; e < sz; e++)
                            insert_hash_map(tidx_map, edges[off + e].val, vid);
                    } else {
                        // predicate-index (IN) vid
                        insert_hash_map(src_predicate_table, pid, vid);
                    }
                }
            }
        }

        uint64_t t2 = timer::get_usec();
        cout << (t2 - t1) / 1000 << " ms for parallel generate tbb_table" << endl;

        // add type-index
        for (tbb_hash_map::iterator i = tidx_map.begin();
                i != tidx_map.end(); ++i) {
            uint64_t sz = i->second.size();
            uint64_t off = sync_fetch_and_alloc_edges(sz);
            uint64_t tid = i->first;

            // insert vertex
            ikey_t key = ikey_t(tid, IN, 0);
            uint64_t slot_id = insert_key(key);
            iptr_t ptr = iptr_t(sz, off);
            vertices[slot_id].ptr = ptr;

            // insert edges
            for (uint64_t k = 0; k < sz; k++)
                edges[off++].val = i->second[k];
        }

        // add predicate-index
        for (tbb_hash_map::iterator i = src_predicate_table.begin();
                i != src_predicate_table.end(); ++i) {
            uint64_t curr_edge_ptr = sync_fetch_and_alloc_edges(i->second.size());
            ikey_t key = ikey_t(i->first, IN, 0);
            uint64_t vertex_ptr = insert_key(key);
            iptr_t ptr = iptr_t(i->second.size(), curr_edge_ptr);
            vertices[vertex_ptr].ptr = ptr;
            for (uint64_t k = 0; k < i->second.size(); k++) {
                edges[curr_edge_ptr].val = i->second[k];
                curr_edge_ptr++;
            }
        }

        for (tbb_hash_map::iterator i = dst_predicate_table.begin();
                i != dst_predicate_table.end(); ++i) {
            uint64_t curr_edge_ptr = sync_fetch_and_alloc_edges(i->second.size());
            ikey_t key = ikey_t(i->first, OUT, 0);
            uint64_t vertex_ptr = insert_key(key);
            iptr_t ptr = iptr_t(i->second.size(), curr_edge_ptr);
            vertices[vertex_ptr].ptr = ptr;
            for (uint64_t k = 0; k < i->second.size(); k++) {
                edges[curr_edge_ptr].val = i->second[k];
                curr_edge_ptr++;
            }
        }

        tbb_hash_map().swap(src_predicate_table);
        tbb_hash_map().swap(dst_predicate_table);
        tbb_hash_map().swap(tidx_map);

#ifdef VAR_PREDICATE
        tbb_unordered_set().swap(p_set);
        tbb_unordered_set().swap(v_set);
#endif

        uint64_t t3 = timer::get_usec();
        cout << (t3 - t2) / 1000
             << " ms for sequence insert tbb_table to gstore"
             << endl;
    }

    edge_t *get_edges_global(int tid, uint64_t vid, int d, int predicate, int *size) {
        int dst_sid = mymath::hash_mod(vid, global_num_servers);
        if (dst_sid == sid)
            return get_edges_local(tid, vid, d, predicate, size);

        ikey_t key = ikey_t(vid, d, predicate);
        vertex_t v = get_vertex_remote(tid, key);

        if (v.key == ikey_t()) {
            *size = 0;
            return NULL;
        }

        char *buf = rdma->get_buffer(tid);
        uint64_t off  = num_slots * sizeof(vertex_t) + v.ptr.off * sizeof(edge_t);
        uint64_t sz = v.ptr.size * sizeof(edge_t);
        rdma->RdmaRead(tid, dst_sid, buf, sz, off);
        edge_t *result_ptr = (edge_t *)buf;
        *size = v.ptr.size;
        return result_ptr;
    }

    edge_t *get_edges_local(int tid, uint64_t vid, int d, int predicate, int *size) {
        assert(mymath::hash_mod(vid, global_num_servers) == sid || is_idx(vid));

        ikey_t key = ikey_t(vid, d, predicate);
        vertex_t v = get_vertex_local(key);
        if (v.key == ikey_t()) {
            *size = 0;
            return NULL;
        }

        *size = v.ptr.size;
        uint64_t ptr = v.ptr.off;
        return &(edges[ptr]);
    }

    edge_t *get_index_edges_local(int tid, uint64_t index_id, int d, int *size) {
        // predicate is not important, so we set it 0
        return get_edges_local(tid, index_id, d, 0, size);
    }

    // analysis and debuging
    void print_memory_usage() {
        uint64_t used_header_slot = 0;
        for (int x = 0; x < num_buckets + num_buckets_ext; x++) {
            for (int y = 0; y < ASSOCIATIVITY - 1; y++) {
                uint64_t i = x * ASSOCIATIVITY + y;
                if (vertices[i].key == ikey_t())
                    continue; // skip the empty slot
                used_header_slot++;
            }
        }

        cout << "gstore direct_header = "
             << B2MiB(num_buckets * ASSOCIATIVITY * sizeof(vertex_t))
             << " MB "
             << endl;
        cout << "\t\treal_data = "
             << B2MiB(used_header_slot * sizeof(vertex_t))
             << " MB " << endl;
        cout << "\t\tnext_ptr = "
             << B2MiB(num_buckets * sizeof(vertex_t))
             << " MB " << endl;
        cout << "\t\tempty_slot = "
             << B2MiB((num_buckets * ASSOCIATIVITY - num_buckets - used_header_slot) * sizeof(vertex_t))
             << " MB " << endl;

        uint64_t used_indirect_slot = 0;
        uint64_t used_indirect_bucket = 0;
        for (int x = num_buckets; x < num_buckets + num_buckets_ext; x++) {
            bool all_empty = true;
            for (int y = 0; y < ASSOCIATIVITY - 1; y++) {
                uint64_t i = x * ASSOCIATIVITY + y;
                if (vertices[i].key == ikey_t())
                    continue; // skip the empty slot
                all_empty = false;
                used_indirect_slot++;
            }

            if (!all_empty)
                used_indirect_bucket++;
        }

        cout << "gstore indirect_header = "
             << B2MiB(num_buckets_ext * ASSOCIATIVITY * sizeof(vertex_t))
             << " MB "
             << endl;
        cout << "\t\tnot_empty_data = "
             << B2MiB(used_indirect_bucket * ASSOCIATIVITY * sizeof(vertex_t))
             << " MB "
             << endl;
        cout << "\t\treal_data = "
             << B2MiB(used_indirect_slot * sizeof(vertex_t))
             << " MB "
             << endl;

        cout << "gstore uses "
             << last_ext
             << " / "
             << num_buckets_ext
             << " num_buckets_ext"
             << endl;

        cout << "gstore uses "
             << B2MiB(num_slots * sizeof(vertex_t))
             << " MB for vertex data"
             << endl;

        cout << "gstore edge_data = "
             << B2MiB(last_entry * sizeof(edge_t))
             << " / "
             << B2MiB(num_entries * sizeof(edge_t))
             << " MB "
             << endl;
    }
};
