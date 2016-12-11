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

#include "config.hpp"
#include "rdma_resource.hpp"
#include "graph_basic_types.hpp"

#include "mymath.hpp"
#include "timer.hpp"
#include "unit.hpp"

class GStore {
    class rdma_cache {
        struct cache_item {
            pthread_spinlock_t lock;
            vertex v;
            cache_item() {
                pthread_spin_init(&lock, 0);
            }
        };

        static const int num_cache = 100000;
        cache_item array[num_cache];

    public:
        bool lookup(local_key key, vertex& ret) {
            if (!global_use_loc_cache) {
                return false;
            }
            int idx = key.hash() % num_cache;
            bool found = false;
            pthread_spin_lock(&(array[idx].lock));
            if (array[idx].v.key == key) {
                ret = array[idx].v;
                found = true;
            }
            pthread_spin_unlock(&(array[idx].lock));
            return found;
        }

        void insert(vertex& v) {
            if (!global_use_loc_cache) {
                return ;
            }
            int idx = v.key.hash() % num_cache;
            pthread_spin_lock(&(array[idx].lock));
            array[idx].v = v;
            pthread_spin_unlock(&(array[idx].lock));
        }
    };

    static const int num_locks = 1024;
    static const int indirect_ratio = 7;  // 1/indirect_ratio of buckets are used as indirect buckets
    static const int cluster_size = 8;    // each bucket has cluster_size slots
    pthread_spinlock_t allocation_lock;
    pthread_spinlock_t fine_grain_locks[num_locks];

    rdma_cache rdmacache;

    vertex *vertex_addr;
    edge *edge_addr;
    RdmaResource *rdma;

    uint64_t slot_num;
    uint64_t m_num;
    uint64_t m_id;

    uint64_t header_num;
    uint64_t indirect_num;
    uint64_t max_edge_ptr;
    uint64_t new_edge_ptr;

    uint64_t used_indirect_num; // used to print memory usage
    uint64_t type_index_edge_num; // used to print memory usage
    uint64_t predicate_index_edge_num; // used to print memory usage

    uint64_t insertKey(local_key key) {
        uint64_t vertex_ptr;
        uint64_t bucket_id = key.hash() % header_num;
        uint64_t lock_id = bucket_id % num_locks;
        uint64_t slot_id = 0;
        bool found = false;

        pthread_spin_lock(&fine_grain_locks[lock_id]);
        //last slot is used as next pointer
        while (!found) {
            for (uint64_t i = 0; i < cluster_size - 1; i++) {
                slot_id = bucket_id * cluster_size + i;
                if (vertex_addr[slot_id].key == key) {
                    cout << "inserting duplicate key" << endl;
                    key.print();
                    assert(false);
                }
                if (vertex_addr[slot_id].key == local_key()) {
                    vertex_addr[slot_id].key = key;
                    found = true;
                    break;
                }
            }
            if (found) {
                break;
            } else {
                slot_id = bucket_id * cluster_size + cluster_size - 1;
                if (vertex_addr[slot_id].key != local_key()) {
                    bucket_id = vertex_addr[slot_id].key.vid;
                    //continue and jump to next bucket
                    continue;
                } else {
                    pthread_spin_lock(&allocation_lock);
                    if (used_indirect_num >= indirect_num) {
                        assert(false);
                    }
                    vertex_addr[slot_id].key.vid = header_num + used_indirect_num;
                    used_indirect_num++;
                    pthread_spin_unlock(&allocation_lock);
                    bucket_id = vertex_addr[slot_id].key.vid;
                    slot_id = bucket_id * cluster_size + 0;
                    vertex_addr[slot_id].key = key;
                    //break the while loop since we successfully insert
                    break;
                }
            }
        }
        pthread_spin_unlock(&fine_grain_locks[lock_id]);
        assert(vertex_addr[slot_id].key == key);
        return slot_id;
    }

    uint64_t atomic_alloc_edges(uint64_t num_edge) {
        uint64_t curr_edge_ptr;
        pthread_spin_lock(&allocation_lock);
        curr_edge_ptr = new_edge_ptr;
        new_edge_ptr += num_edge;
        pthread_spin_unlock(&allocation_lock);
        if (new_edge_ptr >= max_edge_ptr) {
            cout << "atomic_alloc_edges out of memory !!!! " << endl;
            exit(-1);
        }
        return curr_edge_ptr;
    }

    vertex get_vertex_local(local_key key) {
        uint64_t bucket_id = key.hash() % header_num;
        while (true) {
            for (uint64_t i = 0; i < cluster_size; i++) {
                uint64_t slot_id = bucket_id * cluster_size + i;
                if (i < cluster_size - 1) {
                    //data part
                    if (vertex_addr[slot_id].key == key) {
                        //we found it
                        return vertex_addr[slot_id];
                    }
                } else {
                    if (vertex_addr[slot_id].key != local_key()) {
                        //next pointer
                        bucket_id = vertex_addr[slot_id].key.vid;
                        //break from for loop, will go to next bucket
                        break;
                    } else {
                        return vertex();
                    }
                }
            }
        }
    }

    vertex get_vertex_remote(int tid, local_key key) {
        char *local_buffer = rdma->GetMsgAddr(tid);
        uint64_t bucket_id = key.hash() % header_num;
        vertex ret;

        if (rdmacache.lookup(key, ret))
            return ret;

        while (true) {
            uint64_t start_addr = sizeof(vertex) * bucket_id * cluster_size;
            uint64_t read_length = sizeof(vertex) * cluster_size;
            rdma->RdmaRead(tid, mymath::hash_mod(key.vid, m_num),
                           (char *)local_buffer, read_length, start_addr);
            vertex* ptr = (vertex*)local_buffer;
            for (uint64_t i = 0; i < cluster_size; i++) {
                if (i < cluster_size - 1) {
                    if (ptr[i].key == key) {
                        //we found it
                        rdmacache.insert(ptr[i]);
                        return ptr[i];
                    }
                } else {
                    if (ptr[i].key != local_key()) {
                        //next pointer
                        bucket_id = ptr[i].key.vid;
                        //break from for loop, will go to next bucket
                        break;
                    } else {
                        return vertex();
                    }
                }
            }
        }
    }

public:
    GStore() {
        pthread_spin_init(&allocation_lock, 0);
        for (int i = 0; i < num_locks; i++)
            pthread_spin_init(&fine_grain_locks[i], 0);
    }

    void init(RdmaResource* _rdma, uint64_t machine_num, uint64_t machine_id) {
        rdma = _rdma;
        m_num = machine_num;
        m_id = machine_id;
        slot_num = 1000000 * global_hash_header_million;
        header_num = (slot_num / cluster_size)
                     / indirect_ratio
                     * (indirect_ratio - 1);
        indirect_num = (slot_num / cluster_size) / indirect_ratio;

        vertex_addr = (vertex*)(rdma->get_buffer());
        edge_addr   = (edge*)(rdma->get_buffer() + slot_num * sizeof(vertex));

        if (rdma->get_memorystore_size() <= slot_num * sizeof(vertex)) {
            std::cout << "No enough memory to store edge" << std::endl;
            exit(-1);
        }
        max_edge_ptr = (rdma->get_memorystore_size()
                        - slot_num * sizeof(vertex)) / sizeof(edge);
        new_edge_ptr = 0;

        #pragma omp parallel for num_threads(20)
        for (uint64_t i = 0; i < slot_num; i++) {
            vertex_addr[i].key = local_key();
        }

        // if(global_use_loc_cache){
        //  assert(false);
        // }
    }

    void atomic_batch_insert(vector<edge_triple> &spo, vector<edge_triple> &ops) {
        uint64_t accum_predicate = 0;
        uint64_t nedges_to_skip = 0;
        while (nedges_to_skip < ops.size()) {
            if (is_idx(ops[nedges_to_skip].o))
                nedges_to_skip++;
            else
                break;
        }

        uint64_t curr_edge_ptr = atomic_alloc_edges(spo.size()
                                 + ops.size() - nedges_to_skip);
        uint64_t s = 0;
        while (s < spo.size()) {
            uint64_t e = s + 1;
            while ((e < spo.size())
                    && (spo[s].s == spo[e].s)
                    && (spo[s].p == spo[e].p))  { e++; }

            accum_predicate++;
            local_key key = local_key(spo[s].s, OUT, spo[s].p);
            uint64_t vertex_ptr = insertKey(key);
            local_val val = local_val(e - s, curr_edge_ptr);
            vertex_addr[vertex_ptr].val = val;
            for (uint64_t i = s; i < e; i++) {
                edge_addr[curr_edge_ptr].val = spo[i].o;
                curr_edge_ptr++;
            }
            s = e;
        }

        s = nedges_to_skip;
        while (s < ops.size()) {
            uint64_t e = s + 1;
            while ((e < ops.size())
                    && (ops[s].o == ops[e].o)
                    && (ops[s].p == ops[e].p)) { e++; }

            accum_predicate++;
            local_key key = local_key(ops[s].o, IN, ops[s].p);
            uint64_t vertex_ptr = insertKey(key);
            local_val val = local_val(e - s, curr_edge_ptr);
            vertex_addr[vertex_ptr].val = val;
            for (uint64_t i = s; i < e; i++) {
                edge_addr[curr_edge_ptr].val = ops[i].s;
                curr_edge_ptr++;
            }
            s = e;
        }

// The following code is used to support a rare case where the predicate is unknown.
// We disable it to save memory by default.
// Each normal vertex should add a key/value pair with a reserved ID (i.e., __PREDICATE__)
// to store the list of predicates
#if 0
        curr_edge_ptr = atomic_alloc_edges(accum_predicate);
        s = 0;
        while (s < spo.size()) {
            // __PREDICATE__
            local_key key = local_key(spo[s].s, OUT, 0);
            local_val val = local_val(0, curr_edge_ptr);
            uint64_t vertex_ptr = insertKey(key);
            uint64_t e = s;
            while (e < spo.size() && vec_spo[s].s == spo[e].s) {
                if (e == s || spo[e].p != spo[e - 1].p) {
                    edge_addr[curr_edge_ptr].val = spo[e].p;
                    curr_edge_ptr++;
                    val.size = val.size + 1;
                }
                e++;
            }
            vertex_addr[vertex_ptr].val = val;
            s = e;
        }

        s = nedges_to_skip;
        while (s < ops.size()) {
            local_key key = local_key(ops[s].o, IN, 0);
            local_val val = local_val(0, curr_edge_ptr);
            uint64_t vertex_ptr = insertKey(key);
            uint64_t e = s;
            while (e < ops.size() && ops[s].o == ops[e].o) {
                if (e == s || ops[e].p != ops[e - 1].p) {
                    edge_addr[curr_edge_ptr].val = ops[e].p;
                    curr_edge_ptr++;
                    val.size = val.size + 1;
                }
                e++;
            }
            vertex_addr[vertex_ptr].val = val;
            s = e;
        }
#endif
    }

    edge *get_edges_global(int tid, uint64_t id, int direction, int predicate, int *size) {
        if ( mymath::hash_mod(id, m_num) == m_id)
            return get_edges_local(tid, id, direction, predicate, size);

        local_key key = local_key(id, direction, predicate);
        vertex v = get_vertex_remote(tid, key);

        if (v.key == local_key()) {
            *size = 0;
            return NULL;
        }

        char *local_buffer = rdma->GetMsgAddr(tid);
        uint64_t start_addr  = sizeof(vertex) * slot_num + sizeof(edge) * (v.val.ptr);
        uint64_t read_length = sizeof(edge) * v.val.size;
        rdma->RdmaRead(tid, mymath::hash_mod(id, m_num),
                       (char *)local_buffer, read_length, start_addr);
        edge *result_ptr = (edge *)local_buffer;
        *size = v.val.size;
        return result_ptr;
    }

    edge *get_edges_local(int tid, uint64_t id, int direction, int predicate, int *size) {
        assert(mymath::hash_mod(id, m_num) == m_id || is_idx(id));

        local_key key = local_key(id, direction, predicate);
        vertex v = get_vertex_local(key);
        if (v.key == local_key()) {
            *size = 0;
            return NULL;
        }

        *size = v.val.size;
        uint64_t ptr = v.val.ptr;
        return &(edge_addr[ptr]);
    }

    edge *get_index_edges_local(int tid, uint64_t index_id, int d, int *size) {
        // predicate is not important, so we set it 0
        return get_edges_local(tid, index_id, d, 0, size);
    }

    // TODO: define as public, should be refined
    typedef tbb::concurrent_hash_map<uint64_t, vector< uint64_t>> tbb_vector_table;
    tbb_vector_table src_predicate_table;
    tbb_vector_table dst_predicate_table;
    tbb_vector_table type_table;

    void insert_vector(tbb_vector_table &table, uint64_t index_id, uint64_t value_id) {
        tbb_vector_table::accessor a;
        table.insert(a, index_id);
        a->second.push_back(value_id);
    }

    void init_index_table(void) {
        uint64_t t1 = timer::get_usec();

        #pragma omp parallel for num_threads(8)
        for (int x = 0; x < header_num + indirect_num; x++) {
            for (int y = 0; y < cluster_size - 1; y++) {
                uint64_t i = x * cluster_size + y;
                if (vertex_addr[i].key == local_key()) {
                    //empty slot, skip it
                    continue;
                }
                uint64_t vid = vertex_addr[i].key.vid;
                uint64_t p = vertex_addr[i].key.pid;
                if (vertex_addr[i].key.dir == IN) {
                    if (p == TYPE_ID) {
                        //it means vid is a type vertex
                        //we just skip it
                        cout << "[error] type vertices are not skipped" << endl;
                        assert(false);
                        continue;
                    } else {
                        //this edge is in-direction, so vid is the dst of predicate
                        insert_vector(dst_predicate_table, p, vid);
                    }
                } else {
                    if (p == TYPE_ID) {
                        uint64_t degree = vertex_addr[i].val.size;
                        uint64_t edge_ptr = vertex_addr[i].val.ptr;
                        for (uint64_t j = 0; j < degree; j++) {
                            //src may belongs to multiple types
                            insert_vector(type_table, edge_addr[edge_ptr + j].val, vid);
                        }
                    } else {
                        insert_vector(src_predicate_table, p, vid);
                    }
                }
            }
        }
        uint64_t t2 = timer::get_usec();

        for (tbb_vector_table::iterator i = type_table.begin();
                i != type_table.end(); ++i) {
            uint64_t curr_edge_ptr = atomic_alloc_edges(i->second.size());
            local_key key = local_key(i->first, IN, 0);
            uint64_t vertex_ptr = insertKey(key);
            local_val val = local_val(i->second.size(), curr_edge_ptr);
            vertex_addr[vertex_ptr].val = val;
            for (uint64_t k = 0; k < i->second.size(); k++) {
                edge_addr[curr_edge_ptr].val = i->second[k];
                curr_edge_ptr++;
                type_index_edge_num++;
            }
        }

        for (tbb_vector_table::iterator i = src_predicate_table.begin();
                i != src_predicate_table.end(); ++i) {
            uint64_t curr_edge_ptr = atomic_alloc_edges(i->second.size());
            local_key key = local_key(i->first, IN, 0);
            uint64_t vertex_ptr = insertKey(key);
            local_val val = local_val(i->second.size(), curr_edge_ptr);
            vertex_addr[vertex_ptr].val = val;
            for (uint64_t k = 0; k < i->second.size(); k++) {
                edge_addr[curr_edge_ptr].val = i->second[k];
                curr_edge_ptr++;
                predicate_index_edge_num++;
            }
        }

        for (tbb_vector_table::iterator i = dst_predicate_table.begin();
                i != dst_predicate_table.end(); ++i) {
            uint64_t curr_edge_ptr = atomic_alloc_edges(i->second.size());
            local_key key = local_key(i->first, OUT, 0);
            uint64_t vertex_ptr = insertKey(key);
            local_val val = local_val(i->second.size(), curr_edge_ptr);
            vertex_addr[vertex_ptr].val = val;
            for (uint64_t k = 0; k < i->second.size(); k++) {
                edge_addr[curr_edge_ptr].val = i->second[k];
                curr_edge_ptr++;
                predicate_index_edge_num++;
            }
        }

        tbb_vector_table().swap(src_predicate_table);
        tbb_vector_table().swap(dst_predicate_table);
        uint64_t t3 = timer::get_usec();
        cout << (t2 - t1) / 1000
             << " ms for parallel generate tbb_table "
             << endl;
        cout << (t3 - t2) / 1000
             << " ms for sequence insert tbb_table to gstore"
             << endl;
    }

    void print_memory_usage() {
        uint64_t used_header_slot = 0;
        for (int x = 0; x < header_num + indirect_num; x++) {
            for (int y = 0; y < cluster_size - 1; y++) {
                uint64_t i = x * cluster_size + y;
                if (vertex_addr[i].key == local_key())
                    continue; // skip the empty slot
                used_header_slot++;
            }
        }

        cout << "gstore direct_header = "
             << B2MiB(header_num * cluster_size * sizeof(vertex))
             << " MB "
             << endl;
        cout << "\t\treal_data = "
             << B2MiB(used_header_slot * sizeof(vertex))
             << " MB " << endl;
        cout << "\t\tnext_ptr = "
             << B2MiB(header_num * sizeof(vertex))
             << " MB " << endl;
        cout << "\t\tempty_slot = "
             << B2MiB((header_num * cluster_size - header_num - used_header_slot) * sizeof(vertex))
             << " MB " << endl;

        uint64_t used_indirect_slot = 0;
        uint64_t used_indirect_bucket = 0;
        for (int x = header_num; x < header_num + indirect_num; x++) {
            bool all_empty = true;
            for (int y = 0; y < cluster_size - 1; y++) {
                uint64_t i = x * cluster_size + y;
                if (vertex_addr[i].key == local_key())
                    continue; // skip the empty slot
                all_empty = false;
                used_indirect_slot++;
            }

            if (!all_empty)
                used_indirect_bucket++;
        }

        cout << "gstore indirect_header= "
             << B2MiB(indirect_num * cluster_size * sizeof(vertex))
             << " MB "
             << endl;
        cout << "\t\tnot_empty_data= "
             << B2MiB(used_indirect_bucket * cluster_size * sizeof(vertex))
             << " MB "
             << endl;
        cout << "\t\treal_data= "
             << B2MiB(used_indirect_slot * sizeof(vertex))
             << " MB "
             << endl;

        cout << "gstore uses "
             << used_indirect_num
             << " / "
             << indirect_num
             << " indirect_num"
             << endl;
        cout << "gstore uses "
             << B2MiB(slot_num * sizeof(vertex))
             << " MB for vertex data"
             << endl;

        cout << "gstore edge_data= "
             << B2MiB(new_edge_ptr * sizeof(edge))
             << "/"
             << B2MiB(max_edge_ptr * sizeof(edge))
             << " MB "
             << endl;
        cout << "\t\tfor type_index= "
             << B2MiB(type_index_edge_num * sizeof(edge))
             << "/"
             << B2MiB(max_edge_ptr * sizeof(edge))
             << " MB "
             << endl;
        cout << "\t\tfor predicate_index= "
             << B2MiB(predicate_index_edge_num * sizeof(edge))
             << "/"
             << B2MiB(max_edge_ptr * sizeof(edge))
             << " MB "
             << endl;
        cout << "\t\tfor normal_vertex= "
             << B2MiB((new_edge_ptr - predicate_index_edge_num - type_index_edge_num) * sizeof(edge))
             << "/"
             << B2MiB(max_edge_ptr * sizeof(edge))
             << " MB " << endl;
    }
};
