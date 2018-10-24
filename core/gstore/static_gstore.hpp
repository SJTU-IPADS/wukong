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

#ifdef USE_GPU
#include <tbb/concurrent_unordered_map.h>
#include "rdf_meta.hpp"
#include "comm/tcp_adaptor.hpp"
#endif // USE_GPU
#include "gstore.hpp"

using namespace std;

class StaticGStore : public GStore {
    private:
    uint64_t last_entry;
    pthread_spinlock_t entry_lock;

    // get bucket_id according to key
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

    // insert key to a slot
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

    // Get remote edges according to given vid, dir, pid.
    // @sz: size of return edges
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
        sz = v.ptr.size;
        if (&type != NULL)
            type = v.ptr.type;
        return edge_ptr;
    }

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

        uint64_t bucket_off = 0, edge_off = 0;
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

    // re-adjust attributes of segments
    void finalize_segment_metas() {
        uint64_t nbuckets_per_blk = MiB2B(global_gpu_key_blk_size_mb) / (sizeof(vertex_t) * GStore::ASSOCIATIVITY);
        uint64_t nentries_per_blk = MiB2B(global_gpu_value_blk_size_mb) / sizeof(edge_t);

        // set the number of cache blocks needed by each segment
        for (auto &e : rdf_segment_meta_map) {
            e.second.num_key_blks = ceil(((double) e.second.get_total_num_buckets()) / nbuckets_per_blk);
            e.second.num_value_blks = ceil(((double) e.second.num_edges) / nentries_per_blk);
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

    public:
    StaticGStore(int sid, Mem *mem): GStore(sid, mem) {
        pthread_spin_init(&entry_lock, 0);
    }

    ~StaticGStore() {}

    void init(vector<vector<triple_t>> &triple_pso, vector<vector<triple_t>> &triple_pos, vector<vector<triple_attr_t>> &triple_sav) {
        uint64_t start, end;
        #ifdef USE_GPU
            start = timer::get_usec();
            // merge triple_pso and triple_pos into a map
            init_triples_map(triple_pso, triple_pos);
            end = timer::get_usec();
            logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                                << "for merging triple_pso and triple_pos." << LOG_endl;

            start = timer::get_usec();
            init_segment_metas(triple_pso, triple_pos);
            end = timer::get_usec();
            logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                                << "for initializing predicate segment statistics." << LOG_endl;

            start = timer::get_usec();
            auto& predicates = get_all_predicates();
            logstream(LOG_DEBUG) << "#" << sid << ": all_predicates: " << predicates.size() << LOG_endl;
            #pragma omp parallel for num_threads(global_num_engines)
            for (int i = 0; i < predicates.size(); i++) {
                int localtid = omp_get_thread_num();
                sid_t pid = predicates[i];
                insert_triples_to_segment(localtid, segid_t(0, pid, OUT));
                insert_triples_to_segment(localtid, segid_t(0, pid, IN));
            }
            end = timer::get_usec();
            logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                                << "for inserting triples as segments into gstore" << LOG_endl;

            finalize_segment_metas();
            free_triples_map();

            // synchronize segment metadata among servers
            extern TCP_Adaptor *con_adaptor;
            sync_metadata(con_adaptor);
        #else   // !USE_GPU
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
        #endif  // end of USE_GPU
    }

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

    #ifdef USE_GPU
    inline const std::map<segid_t, rdf_segment_meta_t> &get_rdf_segment_metas() { return rdf_segment_meta_map; }

    inline void set_num_predicates(sid_t n) { num_predicates = n; }

    inline int get_num_predicates() const { return num_predicates; }

    const vector<sid_t>& get_all_predicates() const { return all_predicates; }
    #endif // USE_GPU

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
};
