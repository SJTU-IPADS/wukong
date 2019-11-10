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

#ifdef USE_GPU

#include <list>
#include <vector>
#include <map>
#include <algorithm>
#include <cuda_runtime.h>

#include "store/gstore.hpp"
#include "store/meta.hpp"

// utils
#include "unit.hpp"
#include "gpu.hpp"
#include "global.hpp"

using namespace std;

/*
 * Manage GPU kvstore (cache)
 */
class GPUCache {
private:
    /* the rdf data is divided into segments.
     * each predicate has 4 segments: in/out normal/index
     * GPUCache load data in segments
     */
    map<segid_t, rdf_seg_meta_t> rdf_metas;

    // number of buckets per gpu key block
    uint64_t nbuckets_kblk;
    // number of entries per gpu value block
    uint64_t nentries_vblk;

    // number of key blocks and value blocks
    uint64_t num_key_blks;
    uint64_t num_value_blks;

    // store free key block ids
    std::list<int> free_key_blocks;
    // store free value block ids
    std::list<int> free_value_blocks;

    /* map size: #segments
     * vector size: number of key blocks this segment needs
     * value records the block ids this segment is using for storing main and indirect header
     */
    map<segid_t, vector<int>> vertex_allocation;
    // number of key blocks per segement needs
    map<segid_t, int> num_key_blocks_seg_need;
    // number of key blocks per segement using now
    map<segid_t, int> num_key_blocks_seg_using;
    /* map size: seg_num
     * vector size: number of value blocks this segment needs
     * value records the block ids this segment is using for storing edges(value)
     */
    map<segid_t, vector<int>> edge_allocation;
    // number of value blocks per segement needs
    map<segid_t, int> num_value_blocks_seg_need;
    // number of value blocks per segement using now
    map<segid_t, int> num_value_blocks_seg_using;

    // segments that are in key/value cache
    std::list<segid_t> segs_in_key_cache;
    std::list<segid_t> segs_in_value_cache;

    // gpu mem
    GPUMem *gmem;
    // pointing to key area of gpu mem kvstore
    vertex_t *vertex_gaddr;
    // pointing to value area of gpu mem kvstore
    edge_t *edge_gaddr;
    // pointing to key area of cpu kvstore
    vertex_t *vertex_addr;
    // pointing to value area of cpu kvstore
    edge_t *edge_addr;

    const int BLOCK_ID_ERR = -1;

    void evict_key_blocks(const vector<segid_t> &conflicts, segid_t seg_to_load,
                          segid_t seg_in_pattern, int num_need_blocks) {
        // step 1: traverse segments in key cache, evict segments that are not in conflicts or not to load/using
        for (auto it = segs_in_key_cache.begin(); it != segs_in_key_cache.end(); ) {
            segid_t seg = *it;
            if (seg == seg_to_load || seg == seg_in_pattern) {
                it++;
                continue;
            }
            if (find(conflicts.begin(), conflicts.end(), seg) != conflicts.end()) {
                it++;
                continue;
            }

            for (int i = 0; i < num_key_blocks_seg_need[seg]; i++) {
                int block_id = vertex_allocation[seg][i];
                if (block_id != BLOCK_ID_ERR) {
                    vertex_allocation[seg][i] = BLOCK_ID_ERR;
                    num_key_blocks_seg_using[seg]--;
                    if (num_key_blocks_seg_using[seg] == 0) {
                        it = segs_in_key_cache.erase(it);
                    }
                    free_key_blocks.push_back(block_id);

                    if (free_key_blocks.size() >= num_need_blocks) {
                        return;
                    }
                }
            }
        } // end of traverse segs_in_key_cache

        // step 2: worst case. we have to evict segments that in conflicts
        for (auto rit = conflicts.rbegin(); rit != conflicts.rend(); rit++) {
            segid_t seg = *rit;
            // skip segments not in cache or to load or using now
            if (num_key_blocks_seg_using[seg] == 0 || seg == seg_to_load || seg == seg_in_pattern) {
                continue;
            }

            for (int i = 0; i < num_key_blocks_seg_need[seg]; i++) {
                int block_id = vertex_allocation[seg][i];
                if (block_id != BLOCK_ID_ERR) {
                    vertex_allocation[seg][i] = BLOCK_ID_ERR;
                    num_key_blocks_seg_using[seg]--;
                    if (num_key_blocks_seg_using[seg] == 0) {
                        for (auto it = segs_in_key_cache.begin(); it != segs_in_key_cache.end(); it++) {
                            if (*it == seg) {
                                segs_in_key_cache.erase(it);
                                break;
                            }
                        }
                    }
                    free_key_blocks.push_back(block_id);

                    if (free_key_blocks.size() >= num_need_blocks) {
                        return;
                    }
                }
            }
        } // end of worst case

        logstream(LOG_WARNING) << "GPU Cache: evict_key_blocks() cannot provide enough free key blocks."
                               << LOG_endl;
    } // end of evict_key_blocks

    void evict_value_blocks(const vector<segid_t> &conflicts,
                            segid_t seg_to_load, segid_t seg_in_pattern,
                            int num_need_blocks) {
        // step 1: traverse segments in value cache, evict segments that are not in conflicts or not to load/using
        for (auto it = segs_in_value_cache.begin(); it != segs_in_value_cache.end(); ) {
            segid_t seg = *it;
            if (seg == seg_to_load || seg == seg_in_pattern) {
                it++;
                continue;
            }
            if (find(conflicts.begin(), conflicts.end(), seg) != conflicts.end()) {
                it++;
                continue;
            }

            for (int i = 0; i < num_value_blocks_seg_need[seg]; i++) {
                int block_id = edge_allocation[seg][i];
                if (block_id != BLOCK_ID_ERR) {
                    edge_allocation[seg][i] = BLOCK_ID_ERR;
                    num_value_blocks_seg_using[seg]--;
                    if (num_value_blocks_seg_using[seg] == 0) {
                        it = segs_in_value_cache.erase(it);
                    }
                    free_value_blocks.push_back(block_id);

                    if (free_value_blocks.size() >= num_need_blocks) {
                        return;
                    }
                }
            }
        } // end of traverse segs_in_value_cache

        // step 2: worst case. we have to evict segments that in conflicts
        for (auto rit = conflicts.rbegin(); rit != conflicts.rend(); rit++) {
            segid_t seg = *rit;
            // skip segments not in cache or to load or using now
            if (num_value_blocks_seg_using[seg] == 0 || seg == seg_to_load || seg == seg_in_pattern) {
                continue;
            }

            for (int i = 0; i < num_value_blocks_seg_need[seg]; i++) {
                int block_id = edge_allocation[seg][i];
                if (block_id != BLOCK_ID_ERR) {
                    edge_allocation[seg][i] = BLOCK_ID_ERR;
                    num_value_blocks_seg_using[seg]--;
                    if (num_value_blocks_seg_using[seg] == 0) {
                        for (auto it = segs_in_value_cache.begin(); it != segs_in_value_cache.end(); it++) {
                            if (*it == seg) {
                                segs_in_value_cache.erase(it);
                                break;
                            }
                        }
                    }
                    free_value_blocks.push_back(block_id);

                    if (free_value_blocks.size() >= num_need_blocks) {
                        return;
                    }
                }
            }
        } // end of worst case

        logstream(LOG_WARNING) << "GPU Cache: evict_value_blocks() could not provide enough free value blocks." << LOG_endl;
    } // end of evict_value_blocks

    /* load one key block of a segment to a free block
     * seg: the segment
     * seg_block: the block index of the segment
     * block_id: the free block on GPU cache
     * vertex blocks layout: main | main | ... | main (+ indirect) | indirect | ... | indirect
     */
    void load_vertex_block(segid_t seg, int seg_block_idx, int block_id, cudaStream_t stream) {
        // step 1: calculate direct size
        int end_main_block_idx = ceil((double)(rdf_metas[seg].num_buckets) / nbuckets_kblk) - 1;
        int end_indirect_block_idx = ceil((double)(rdf_metas[seg].get_total_num_buckets()) / nbuckets_kblk) - 1;

        uint64_t main_size = 0;
        uint64_t indirect_size = 0;
        uint64_t indirect_start = 0;

        if (seg_block_idx == end_main_block_idx) {
            // the loading block is the last block contains main headers, may contain indirect headers
            main_size = rdf_metas[seg].num_buckets % nbuckets_kblk;
            indirect_start = main_size;
        } else if (seg_block_idx < end_main_block_idx) {
            // the loading block contains main header (not tail)
            main_size = nbuckets_kblk;
        } else {
            // the loading block contains indirect headers
            main_size = 0;
        }
        // step 2: calculate indirect size
        if (seg_block_idx < end_main_block_idx) {
            indirect_size = 0;
        } else if (seg_block_idx == end_indirect_block_idx) {
            indirect_size = rdf_metas[seg].get_total_num_buckets() % nbuckets_kblk - indirect_start;
        } else if (seg_block_idx < end_indirect_block_idx) {
            indirect_size = nbuckets_kblk - indirect_start;
        }
        // step 3: load direct
        if (main_size != 0) {
            CUDA_ASSERT(cudaMemcpyAsync(
                            vertex_gaddr + block_id * nbuckets_kblk * GStore::ASSOCIATIVITY,
                            vertex_addr + (rdf_metas[seg].bucket_start
                                           + seg_block_idx * nbuckets_kblk) * GStore::ASSOCIATIVITY,
                            sizeof(vertex_t) * main_size * GStore::ASSOCIATIVITY,
                            cudaMemcpyHostToDevice,
                            stream));
        }
        // step 4: load indirect
        if (indirect_size != 0) {
            // remain number of buckets to load
            uint64_t remain = indirect_size;

            // step 4.1: locate start ext bucket index
            uint64_t start_bucket_idx = 0;
            if (seg_block_idx != end_main_block_idx) {
                start_bucket_idx = seg_block_idx * nbuckets_kblk - rdf_metas[seg].num_buckets;
            }
            // step 4.2 traverse the ext_bucket_list and load
            uint64_t passed_buckets = 0;

            for (int i = 0; i < rdf_metas[seg].get_ext_bucket_list_size(); i++) {
                ext_bucket_extent_t ext = rdf_metas[seg].ext_bucket_list[i];
                /* load from this ext
                 * inside_off: the offset inside the ext
                 * inside_load: number of buckets to be loaded from this ext
                 */
                if ((passed_buckets + ext.num_ext_buckets) > start_bucket_idx) {
                    uint64_t inside_off = start_bucket_idx - passed_buckets;
                    uint64_t inside_load = ext.num_ext_buckets - inside_off;
                    if (inside_load > remain) inside_load = remain;
                    uint64_t dst_off = (block_id * nbuckets_kblk + indirect_start + indirect_size - remain)
                                       * GStore::ASSOCIATIVITY;
                    uint64_t src_off = (ext.start + inside_off) * GStore::ASSOCIATIVITY;
                    CUDA_ASSERT(cudaMemcpyAsync(vertex_gaddr + dst_off, vertex_addr + src_off,
                                                sizeof(vertex_t) * inside_load * GStore::ASSOCIATIVITY,
                                                cudaMemcpyHostToDevice, stream));
                    remain -= inside_load;
                    start_bucket_idx += inside_load;
                    // load complete
                    if (remain == 0) {
                        break;
                    }
                }
                passed_buckets += ext.num_ext_buckets;
            }
        }
    } // end of load_vertex_block

    /* load one value block of a segment to a free block
     * seg: the segment
     * seg_block: the block index of the segment
     * block_id: the free block on GPU cache
     */
    void load_edge_block(segid_t seg, int seg_block_idx, int block_id, cudaStream_t stream) {
        // the number of entries in this seg block
        uint64_t data_size = 0;
        if (seg_block_idx == (num_value_blocks_seg_need[seg] - 1))
            data_size = rdf_metas[seg].num_edges % nentries_vblk;
        else
            data_size = nentries_vblk;
        CUDA_ASSERT(cudaMemcpyAsync(edge_gaddr + block_id * nentries_vblk,
                                    edge_addr + rdf_metas[seg].edge_start + seg_block_idx * nentries_vblk,
                                    sizeof(edge_t) * data_size,
                                    cudaMemcpyHostToDevice,
                                    stream));
    } // end of load_edge_block

    void _load_segment(segid_t seg_to_load, segid_t seg_in_use,
                       const vector<segid_t> &conflicts, cudaStream_t stream_id, bool preload) {
        // step 1.1: evict key blocks
        int num_need_key_blocks = num_key_blocks_seg_need[seg_to_load]
                                  - num_key_blocks_seg_using[seg_to_load];

#ifdef GPU_DEBUG
        logstream(LOG_EMPH) << "load_segment: segment: " << seg_to_load.to_string()
                            << ", #need_key_blks: " << num_need_key_blocks << LOG_endl;
#endif

        if (free_key_blocks.size() < num_need_key_blocks) {
            logstream(LOG_WARNING) << "load_segment: evict_key_blocks" << LOG_endl;
            evict_key_blocks(conflicts, seg_to_load, seg_in_use, num_need_key_blocks);
        }
        // step 1.2: load key blocks
        for (int i = 0; i < num_key_blocks_seg_need[seg_to_load]; i++) {
            // skip the blocks that are already loaded
            if (vertex_allocation[seg_to_load][i] != BLOCK_ID_ERR)
                continue;

            if (free_key_blocks.empty()) {
                // abort preload
                if (preload) {
                    logstream(LOG_WARNING) << "GPU Cache: No enough free key blocks. "
                                           << "Preload is not complete. segment_to_load: "
                                           << seg_to_load.to_string() << ", seg_in_use: "
                                           << seg_in_use.to_string() << LOG_endl;
                    break;
                }
                // crash if it is not preload
                logstream(LOG_WARNING) << "GPU Cache: No enough free key blocks. "
                                       << "segment_to_load: " << seg_to_load.to_string()
                                       << ", seg_in_use: " << seg_in_use.to_string() << LOG_endl;
                ASSERT(false);
            }
            // load one block
            int block_id = free_key_blocks.front();
            free_key_blocks.pop_front();
            load_vertex_block(seg_to_load, i, block_id, stream_id);
            vertex_allocation[seg_to_load][i] = block_id;
            if (num_key_blocks_seg_using[seg_to_load] == 0) {
                segs_in_key_cache.push_back(seg_to_load);
            }
            num_key_blocks_seg_using[seg_to_load]++;
        }

        //step 2.1: evict value blocks
        int num_need_value_blocks = num_value_blocks_seg_need[seg_to_load]
                                    - num_value_blocks_seg_using[seg_to_load];
        if (free_value_blocks.size() < num_need_value_blocks) {
            evict_value_blocks(conflicts, seg_to_load, seg_in_use, num_need_value_blocks);
        }
        // step 2.2: load value blocks
        for (int i = 0; i < num_value_blocks_seg_need[seg_to_load]; i++) {
            // skip the blocks that are already loaded
            if (edge_allocation[seg_to_load][i] != BLOCK_ID_ERR)
                continue;

            if (free_value_blocks.empty()) {
                // abort preload
                if (preload) {
                    logstream(LOG_WARNING) << "GPU Cache: No enough free value blocks. "
                                           << "Preload is not complete. segment_to_load: "
                                           << seg_to_load.to_string() << ", seg_in_use: "
                                           << seg_in_use.to_string() << LOG_endl;
                    break;
                }
                // crash if it is not preload
                logstream(LOG_WARNING) << "GPU Cache: No enough free value blocks. "
                                       << "segment_to_load: " << seg_to_load.to_string()
                                       << ", seg_in_use: " << seg_in_use.to_string() << LOG_endl;
                ASSERT(false);
            }
            // load one block
            int block_id = free_value_blocks.front();
            free_value_blocks.pop_front();
            load_edge_block(seg_to_load, i, block_id, stream_id);
            edge_allocation[seg_to_load][i] = block_id;
            if (num_value_blocks_seg_using[seg_to_load] == 0)
                segs_in_value_cache.push_back(seg_to_load);

            num_value_blocks_seg_using[seg_to_load]++;
        }
    } // end of _load_segment

public:
    GPUCache(GPUMem * gmem, vertex_t *v_a, edge_t *e_a, const map<segid_t, rdf_seg_meta_t> &rdf_metas):
        gmem(gmem), vertex_addr(v_a), edge_addr(e_a), rdf_metas(rdf_metas) {

        // step 1: calculate #slots, #buckets, #entries
        uint64_t num_slots = (GiB2B(Global::gpu_kvcache_size_gb) * GStore::HD_RATIO) / (100 * sizeof(vertex_t));
        uint64_t num_buckets = num_slots / GStore::ASSOCIATIVITY;
        uint64_t num_entries = (GiB2B(Global::gpu_kvcache_size_gb) - num_slots * sizeof(vertex_t)) / sizeof(edge_t);

        nbuckets_kblk = MiB2B(Global::gpu_key_blk_size_mb) / (sizeof(vertex_t) * GStore::ASSOCIATIVITY);
        nentries_vblk = MiB2B(Global::gpu_value_blk_size_mb) / sizeof(edge_t);

        num_key_blks = num_buckets / nbuckets_kblk;
        num_value_blks = num_entries / nentries_vblk;

        vertex_gaddr = (vertex_t *)gmem->kvcache();
        edge_gaddr = (edge_t *)(gmem->kvcache() + num_slots * sizeof(vertex_t));

        logstream(LOG_INFO) << "GPU_Cache: #key_blocks: " << num_key_blks
                            << ", #value_blocks: " << num_value_blks
                            << ", #buckets_per_block: " << nbuckets_kblk
                            << ", #edges_per_block: " << nentries_vblk << LOG_endl;

        // step 2: init free_key/value blocks
        for (int i = 0; i < num_key_blks; i++) {
            free_key_blocks.push_back(i);
        }
        for (int i = 0; i < num_value_blks; i++) {
            free_value_blocks.push_back(i);
        }

        // step 3: init vertex/edge allocations
        for (auto it = rdf_metas.begin(); it != rdf_metas.end(); it++) {
            int key_blocks_need = it->second.num_key_blks;
            vertex_allocation.insert(std::make_pair(it->first, vector<int>(key_blocks_need, BLOCK_ID_ERR)));
            num_key_blocks_seg_need.insert(std::make_pair(it->first, key_blocks_need));
            num_key_blocks_seg_using.insert(std::make_pair(it->first, 0));

            int value_blocks_need = it->second.num_value_blks;
            edge_allocation.insert(std::make_pair(it->first, vector<int>(value_blocks_need, BLOCK_ID_ERR)));
            num_value_blocks_seg_need.insert(std::make_pair(it->first, value_blocks_need));
            num_value_blocks_seg_using.insert(std::make_pair(it->first, 0));
        }
    } // end of constructor

    // check whether a segment is in cache
    bool seg_in_cache(segid_t seg) {
        if (num_key_blocks_seg_using[seg] == num_key_blocks_seg_need[seg]
                && num_value_blocks_seg_using[seg] == num_value_blocks_seg_need[seg])
            return true;
        return false;
    }

    rdf_seg_meta_t get_segment_meta(segid_t seg) {
        auto it = rdf_metas.find(seg);
        if (it != rdf_metas.end())
            return rdf_metas[seg];

        ASSERT_MSG(false, "segment not found");
    }

    // return the bucket offset of each key block in a segment
    vector<uint64_t> get_vertex_mapping(segid_t seg) {
        ASSERT(num_key_blocks_seg_using[seg] == num_key_blocks_seg_need[seg]);
        ASSERT(vertex_allocation[seg].size() == num_key_blocks_seg_need[seg]);

        vector<uint64_t> headers;
        for (auto block_id : vertex_allocation[seg])
            headers.push_back(block_id * nbuckets_kblk);

        return headers;
    }

    // return the entry offset of each value block in a segment
    vector<uint64_t> get_edge_mapping(segid_t seg) {
        ASSERT(num_value_blocks_seg_using[seg] == num_value_blocks_seg_need[seg]);
        ASSERT(edge_allocation[seg].size() == num_value_blocks_seg_need[seg]);
        vector<uint64_t> headers;
        for (auto block_id : edge_allocation[seg]) {
            headers.push_back(block_id * nentries_vblk);
        }
        return headers;
    }

    uint64_t get_num_key_blks() { return num_key_blks; }

    uint64_t get_num_value_blks() { return num_value_blks; }

    uint64_t get_nbuckets_kblk() { return nbuckets_kblk; }

    uint64_t get_nentries_vblk() { return nentries_vblk; }

    vertex_t *get_vertex_gaddr() { return vertex_gaddr; }

    edge_t *get_edge_gaddr() { return edge_gaddr; }

    void reset() {
        for (auto it = rdf_metas.begin(); it != rdf_metas.end(); it++) {
            segid_t seg = it->first;
            for (int i = 0; i < num_key_blocks_seg_need[seg]; i++) {
                int block_id = vertex_allocation[seg][i];
                if (block_id != BLOCK_ID_ERR) {
                    vertex_allocation[seg][i] = BLOCK_ID_ERR;
                    num_key_blocks_seg_using[seg]--;
                    free_key_blocks.push_back(block_id);
                }
            }

            for (int i = 0; i < num_value_blocks_seg_need[seg]; i++) {
                int block_id = edge_allocation[seg][i];
                if (block_id != BLOCK_ID_ERR) {
                    edge_allocation[seg][i] = BLOCK_ID_ERR;
                    num_value_blocks_seg_using[seg]--;
                    free_value_blocks.push_back(block_id);
                }
            }
        }
        segs_in_key_cache.clear();
        segs_in_value_cache.clear();

        ASSERT(free_key_blocks.size() == num_key_blks);
        ASSERT(free_value_blocks.size() == num_value_blks);
    } // end of reset

    void load_segment(segid_t seg_to_load,
                      const vector<segid_t> &conflicts, cudaStream_t stream_id) {
        _load_segment(seg_to_load, seg_to_load, conflicts, stream_id, false);
    }

    void prefetch_segment(segid_t seg_to_load, segid_t seg_in_use,
                          const vector<segid_t> &conflicts, cudaStream_t stream_id) {
        _load_segment(seg_to_load, seg_in_use, conflicts, stream_id, true);
    }


};
#endif
