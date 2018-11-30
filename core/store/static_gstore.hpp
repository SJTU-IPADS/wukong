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

#include "gstore.hpp"


using namespace std;

class StaticGStore : public GStore {
private:
    // used to alloc edges
    uint64_t last_entry;
    pthread_spinlock_t entry_lock;

    // get bucket_id according to key
    uint64_t bucket_local(ikey_t key) {
        uint64_t bucket_id;
        auto &seg = rdf_segment_meta_map[segid_t(key)];
        ASSERT(seg.num_buckets > 0);
        bucket_id = seg.bucket_start + key.hash() % seg.num_buckets;
        return bucket_id;
    }

    uint64_t bucket_remote(ikey_t key, int dst_sid) {
        uint64_t bucket_id;
        auto &remote_meta_map = shared_rdf_segment_meta_map[dst_sid];
        auto &seg = remote_meta_map[segid_t(key)];
        ASSERT(seg.num_buckets > 0);
        bucket_id = seg.bucket_start + key.hash() % seg.num_buckets;
        return bucket_id;
    }

    // Allocate space to store edges of given size. Return offset of allocated space.
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

    /// edge is always valid
    bool edge_is_valid(vertex_t &v, edge_t *edge_ptr) { return true; }

    // Get edges of given vertex from dst_sid by RDMA read.
    edge_t *rdma_get_edges(int tid, int dst_sid, vertex_t &v) {
        ASSERT(global_use_rdma);

        char *buf = mem->buffer(tid);
        uint64_t r_off = num_slots * sizeof(vertex_t) + v.ptr.off * sizeof(edge_t);
        // the size of edges
        uint64_t r_sz = v.ptr.size * sizeof(edge_t);
        uint64_t buf_sz = mem->buffer_size();
        ASSERT(r_sz < buf_sz); // enough space to host the edges

        RDMA &rdma = RDMA::get_rdma();
        rdma.dev->RdmaRead(tid, dst_sid, buf, r_sz, r_off);
        return (edge_t *)buf;
    }

public:
    StaticGStore(int sid, Mem *mem): GStore(sid, mem) {
        pthread_spin_init(&entry_lock, 0);
    }

    ~StaticGStore() {}

    void refresh() {
        #pragma omp parallel for num_threads(global_num_engines)
        for (uint64_t i = 0; i < num_slots; i++) {
            vertices[i].key = ikey_t();
            vertices[i].ptr = iptr_t();
        }
        last_ext = 0;
        last_entry = 0;
    }

    void print_mem_usage() {
        GStore::print_mem_usage();
        logstream(LOG_INFO) << "\tused: " << 100.0 * last_entry / num_entries
                            << " % (" << last_entry << " entries)" << LOG_endl;
    }

};
