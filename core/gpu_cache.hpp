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

#include <list>
#include <vector>
#include <map>
#include <unordered_map>
#include <cuda_runtime.h>
#include "rdf_meta.hpp"
#include "gstore.hpp"
#include "unit.hpp"

using namespace std;

/*
 * Manage GPU kvstore(cache)
 */
class GPUCache {
private:
    // number of segments
    int seg_num;
    /* the rdf data is divided into segments.
     * each predicate has 4 segments: in/out normal/index
     * GPUCache load data in segments
     */
    const map<segid_t, rdf_segment_meta_t> rdf_metas;

    /* capacity of slots/buckets/values on GPU cache
     * 1 bucket contains ASSOCIATIVITY slots
     */
    uint64_t cap_gpu_slots;
    uint64_t cap_gpu_buckets;
    uint64_t cap_gpu_entries;

    // number of buckets per gpu key block
    uint64_t num_buckets_per_block;
    // number of entries per gpu value block
    uint64_t num_entries_per_block;

    // capacity of key/value blocks
    uint64_t cap_gpu_key_blocks;
    uint64_t cap_gpu_value_blocks;

    // store free key block ids
    std::list<int> free_key_blocks;
    // store free value block ids
    std::list<int> free_value_blocks;


    /* map size: seg_num
     * vector size: number of key blocks this segment needs
     * value records the block ids this segment is using for storing main and indirect header
     */
    unordered_map<segid_t, vector<int>> vertex_allocation;
    // number of key blocks per segement needs
    unordered_map<segid_t, int> num_key_blocks_seg_need;
    // number of key blocks per segement using now
    unordered_map<segid_t, int> num_key_blocks_seg_using;
    /* map size: seg_num
     * vector size: number of value blocks this segment needs
     * value records the block ids this segment is using for storing edges(value)
     */
    unordered_map<segid_t, vector<int>> edge_allocation;
    // number of value blocks per segement needs
    unordered_map<segid_t, int> num_value_blocks_seg_need;
    // number of value blocks per segement using now
    unordered_map<segid_t, int> num_value_blocks_seg_using;

    // pointing to key area of gpu mem kvstore
    vertex_t *d_vertex_addr;
    // pointing to value area of gpu mem kvstore
    edge_t *d_edge_addr;
    // pointing to key area of cpu kvstore
    vertex_t *vertex_addr;
    // pointing to value area of cpu kvstore
    edge_t *edge_addr;

    const int BLOCK_ID_ERR = -1;
public:
    GPUCache(vertex_t *d_v_a, edge_t *d_e_a, vertex_t *v_a, edge_t *e_a, const map<segid_t, rdf_segment_meta_t> &rdf_metas): d_vertex_addr(d_v_a), d_edge_addr(d_e_a), vertex_addr(v_a), edge_addr(e_a), rdf_metas(rdf_metas) {
        // step 1: calculate capacities
        seg_num = rdf_metas.size();

        cap_gpu_slots = global_gpu_num_keys_million * 1000 * 1000;
        cap_gpu_buckets = cap_gpu_slots / GStore::ASSOCIATIVITY;
        cap_gpu_entries = (GiB2B(global_gpu_kvstore_size_gb) - cap_gpu_slots * sizeof(vertex_t)) / sizeof(edge_t);

        num_buckets_per_block = MiB2B(global_gpu_key_block_size_mb) / (sizeof(vertex_t) * GStore::ASSOCIATIVITY);
        num_entries_per_block = MiB2B(global_gpu_value_block_size_mb) / sizeof(edge_t);

        cap_gpu_key_blocks = cap_gpu_buckets / num_buckets_per_block;
        cap_gpu_value_blocks = cap_gpu_entries / num_entries_per_block;

        // step 2: init free_key/value blocks
        for (int i = 0; i < cap_gpu_key_blocks; i++) {
            free_key_blocks.push_back(i);
        }
        for (int i = 0; i < cap_gpu_value_blocks; i++) {
            free_value_blocks.push_back(i);
        }

        // step 3: init vertex/edge allocations
        for (auto it = rdf_metas.begin(); it != rdf_metas.end(); it++) {
            int key_blocks_need = ceil((double)it->second.get_total_num_buckets() / num_buckets_per_block);
            vertex_allocation.insert(std::make_pair(it->first, vector<int>(key_blocks_need, BLOCK_ID_ERR)));
            num_key_blocks_seg_need.insert(std::make_pair(it->first, key_blocks_need));
            num_key_blocks_seg_using.insert(std::make_pair(it->first, 0));

            int value_blocks_need = ceil((double)it->second.num_edges / num_entries_per_block);
            edge_allocation.insert(std::make_pair(it->first, vector<int>(value_blocks_need, BLOCK_ID_ERR)));
            num_value_blocks_seg_need.insert(std::make_pair(it->first, value_blocks_need));
            num_value_blocks_seg_using.insert(std::make_pair(it->first, 0));
        }
    } // end of constructor
};
