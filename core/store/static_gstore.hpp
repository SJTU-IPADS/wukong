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

#include <tbb/concurrent_unordered_map.h>
#include <unordered_set>
#include <thread>
#include "store/meta.hpp"
#include "comm/tcp_adaptor.hpp"
#include "gstore.hpp"


using namespace std;

class StaticGStore : public GStore {
private:
    struct cnt_t {
        cnt_t() {
            in = 0ul;
            out = 0ul;
        }

        cnt_t(const cnt_t &cnt) {
            in = cnt.in.load();
            out = cnt.out.load();
        }

        atomic<uint64_t> in, out;
    };

    // all local normal predicate IDs
    vector<sid_t> all_local_preds;

    /**
     * metadata of local segments (will be used on GPU)
     * #total: 2*|normal pred| + 1 + 2 + 1 + 1 + 1 + 1 + 2
     * description              key                         segid                   num
     * 1. normal segments       [vid|pid|IN/OUT]            [0|pid|IN/OUT]          2*|normal pred|
     * 2. vid's all types       [vid|TYPE_ID|OUT]           [0|TYPE_ID|OUT]         1
     * 3. predicate index OUT   [0|pid|OUT]                 [1|PREDICATE_ID|OUT]    1
     * 4. predicate index IN
     *    type index            [0|pid/typeid|IN]           [0|PREDICATE_ID|IN]     1
     * 5*. vid's all predicates [vid|PREDICATE_ID|IN/OUT]   [0|PREDICATE_ID|IN/OUT] 2
     */
    // multiple engines will access shared_rdf_segment_meta_map
    // key: server id, value: segment metadata of the server
    tbb::concurrent_unordered_map <int, map<segid_t, rdf_segment_meta_t> > shared_rdf_segment_meta_map;
    std::map<segid_t, rdf_segment_meta_t> rdf_segment_meta_map;

    typedef tbb::concurrent_unordered_set<sid_t> tbb_unordered_set;
    tbb_unordered_set attr_set;
    tbb::concurrent_unordered_map<sid_t, int> attr_type_map;
#ifdef VERSATILE
    /**
     * local sets will be freed after inserting into edges
     * description                key
     * 1*. all local entities     [0|TYPE_ID|IN]
     * 2*. all local types        [0|TYPE_ID|OUT]
     * 3*. all local predicates   [0|PREDICATE_ID|OUT]
     */
    tbb_unordered_set t_set;    // all local types
    tbb_unordered_set v_set;    // all local entities(subjects & objects)
    tbb_unordered_set p_set;    // all local predicates
    tbb::concurrent_unordered_map<sid_t, std::unordered_set<sid_t> > out_preds;  // vid's out predicates
    tbb::concurrent_unordered_map<sid_t, std::unordered_set<sid_t> > in_preds;   // vid's in predicates

    void insert_idx_set(tbb_unordered_set &set, uint64_t &off, sid_t pid, dir_t d) {
        uint64_t sz = set.size();

        ikey_t key = ikey_t(0, pid, d);
        uint64_t slot_id = insert_key(key);
        iptr_t ptr = iptr_t(sz, off);
        vertices[slot_id].ptr = ptr;

        for (auto const &e : set)
            edges[off++].val = e;
    }

    void insert_preds(sid_t vid, const unordered_set<sid_t> &preds, dir_t d) {
        uint64_t sz = preds.size();
        auto &seg = rdf_segment_meta_map[segid_t(0, PREDICATE_ID, d)];
        uint64_t off = seg.edge_start + seg.edge_off;
        seg.edge_off += sz;

        ikey_t key = ikey_t(vid, PREDICATE_ID, d);
        uint64_t slot_id = insert_key(key);
        iptr_t ptr = iptr_t(sz, off);
        vertices[slot_id].ptr = ptr;

        for (auto const &e : preds)
            edges[off++].val = e;
    }
#endif // VERSATILE

    typedef tbb::concurrent_hash_map<ikey_t, vector<triple_t>, ikey_Hasher> tbb_triple_hash_map;
    // triples grouped by (predicate, direction)
    tbb_triple_hash_map triples_map;

    typedef tbb::concurrent_hash_map<ikey_t, vector<triple_attr_t>, ikey_Hasher> tbb_triple_attr_hash_map;
    // triples grouped by (predicate, direction)
    tbb_triple_attr_hash_map attr_triples_map;

    uint64_t last_entry;
    pthread_spinlock_t entry_lock;

    uint64_t main_hdr_off = 0;

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
            rdf_segment_meta_t &seg = rdf_segment_meta_map[segid_t(key)];
            uint64_t ext_bucket_id = seg.get_ext_bucket();
            if (ext_bucket_id == 0) {
                uint64_t nbuckets = EXT_BUCKET_EXTENT_LEN(seg.num_buckets);
                uint64_t start_off = alloc_ext_buckets(nbuckets);
                seg.add_ext_buckets(ext_bucket_extent_t(nbuckets, start_off));
                ext_bucket_id = seg.get_ext_bucket();
            }
            vertices[slot_id].key.vid = ext_bucket_id;

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
            logstream(LOG_INFO) << "#" << sid
                                << " receives segment metadata from server " << msg.sender_sid
                                << LOG_endl;
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

    void insert_idx(tbb_hash_map *pidx_map, tbb_hash_map *tidx_map, dir_t d) {
        tbb_hash_map::const_accessor ca;
        rdf_segment_meta_t &segment = rdf_segment_meta_map[segid_t(1, PREDICATE_ID, d)];
        ASSERT(segment.num_edges > 0);

        uint64_t off = segment.edge_start;

        for (int i = 0; i < all_local_preds.size(); i++) {
            sid_t pid = all_local_preds[i];
            bool success = pidx_map->find(ca, pid);
            if (!success)
                continue;

            uint64_t sz = ca->second.size();
            ASSERT(sz <= segment.num_edges);

            ikey_t key = ikey_t(0, pid, d);
            logger(LOG_DEBUG, "insert_pidx[%s]: key: [%lu|%lu|%lu] sz: %lu",
                (d == IN) ? "IN" : "OUT", key.vid, key.pid, key.dir, sz);
            uint64_t slot_id = insert_key(key);
            iptr_t ptr = iptr_t(sz, off);
            vertices[slot_id].ptr = ptr;

            for (auto const &vid : ca->second)
                edges[off++].val = vid;

            ASSERT(off <= segment.edge_start + segment.num_edges);
        }
        // type index
        if (d == IN) {
            for (auto const &e : *tidx_map) {
                sid_t pid = e.first;
                uint64_t sz = e.second.size();
                ASSERT(sz <= segment.num_edges);
                logger(LOG_DEBUG, "insert_tidx: pid: %lu, sz: %lu", pid, sz);

                ikey_t key = ikey_t(0, pid, IN);
                uint64_t slot_id = insert_key(key);
                iptr_t ptr = iptr_t(sz, off);
                vertices[slot_id].ptr = ptr;

                for (auto const &vid : e.second)
                    edges[off++].val = vid;

                ASSERT(off <= segment.edge_start + segment.num_edges);
            }
        }
#ifdef VERSATILE
        if (d == IN) {
            // all local entities, key: [0 | TYPE_ID | IN]
            insert_idx_set(v_set, off, TYPE_ID, IN);
            tbb_unordered_set().swap(v_set);
        } else {
            // all local types, key: [0 | TYPE_ID | OUT]
            insert_idx_set(t_set, off, TYPE_ID, OUT);
            tbb_unordered_set().swap(t_set);
            // all local predicates, key: [0 | PREDICATE_ID | OUT]
            insert_idx_set(p_set, off, PREDICATE_ID, OUT);
            tbb_unordered_set().swap(p_set);
        }
#endif // VERSATILE
    }

    void sync_metadata(TCP_Adaptor *tcp_ad) {
        send_segment_meta(tcp_ad);
        recv_segment_meta(tcp_ad);
    }

    void alloc_buckets_to_segment(rdf_segment_meta_t &seg, segid_t segid, uint64_t total_num_keys) {
        // deduct some buckets from total to prevent overflow
        static uint64_t num_free_buckets = num_buckets
                                           - num_normal_preds * PREDICATE_NSEGS
                                           - INDEX_NSEGS
                                           - num_attr_preds;
        static double total_ratio_ = 0.0;

        // allocate buckets in main-header region to segments
        uint64_t nbuckets;
        if (seg.num_keys == 0) {
            nbuckets = 0;
        } else {
            double ratio = static_cast<double>(seg.num_keys) / total_num_keys;
            nbuckets = ratio * num_free_buckets;
            total_ratio_ += ratio;
            logger(LOG_DEBUG, "Seg[%lu|%lu|%lu]: "
                   "#keys: %lu, nbuckets: %lu, bucket_off: %lu, "
                   "ratio: %f, total_ratio: %f",
                   segid.index, segid.pid, segid.dir,
                   seg.num_keys, nbuckets, main_hdr_off,
                   ratio, total_ratio_);
        }
        seg.num_buckets = (nbuckets > 0 ? nbuckets : 1);

        seg.bucket_start = main_hdr_off;
        main_hdr_off += seg.num_buckets;
        ASSERT(main_hdr_off <= num_buckets);

        // allocate buckets in indirect-header region to segments
        // #buckets : #extended buckets = 1 : 0.15
        if (seg.num_buckets > 0) {
            uint64_t nbuckets = EXT_BUCKET_EXTENT_LEN(seg.num_buckets);
            uint64_t start_off = alloc_ext_buckets(nbuckets);
            seg.add_ext_buckets(ext_bucket_extent_t(nbuckets, start_off));
        }
    }

    // re-adjust attributes of segments
    void finalize_segment_metas() {
        uint64_t nbuckets_per_blk = MiB2B(global_key_blk_size_mb) / (sizeof(vertex_t) * ASSOCIATIVITY);
        uint64_t nentries_per_blk = MiB2B(global_val_blk_size_mb) / sizeof(edge_t);

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
            logger(LOG_DEBUG, "Thread(%d): abort! segment(%d|%d|%d) is empty.\n",
                   tid, segid.index, segid.pid, segid.dir);
            return;
        }

        // get OUT edges and IN edges from triples map
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
            logger(LOG_DEBUG, "Thread(%d): inserted predicate %d pso(%lu triples).",
                   tid, pid, pso.size());
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
            logger(LOG_DEBUG, "Thread(%d): inserted predicate %d pos(%lu triples).",
                   tid, pid, pos.size());
        }

        ASSERT(off <= segment.edge_start + segment.num_edges);
    }

    // insert attributes
    void insert_attr_to_segment(segid_t segid) {
        auto &segment = rdf_segment_meta_map[segid];
        sid_t aid = segid.pid;
        int dir = segid.dir;

        if (segment.num_edges == 0) {
            logger(LOG_DEBUG, "Segment(%d|%d|%d) is empty.\n",
                   segid.index, segid.pid, segid.dir);
            return;
        }
        // get OUT edges and IN edges from triples map
        tbb_triple_attr_hash_map::accessor a;
        bool success = attr_triples_map.find(a, ikey_t(0, aid, (dir_t) dir));
        ASSERT(success);
        vector<triple_attr_t> &asv = a->second;
        variant_type get_type;
        uint64_t off = segment.edge_start;
        int type = attr_type_map[aid];
        uint64_t sz = (get_sizeof(type) - 1) / sizeof(edge_t) + 1;   // get the ceil size;
        uint64_t s = 0;
        for (auto &attr : asv) {
            // allocate a vertex and edges
            ikey_t key = ikey_t(attr.s, attr.a, OUT);

            // insert a vertex
            uint64_t slot_id = insert_key(key);
            iptr_t ptr = iptr_t(sz, off, type);
            vertices[slot_id].ptr = ptr;

            // insert edges
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
            off += sz;
        }

        ASSERT(off <= segment.edge_start + segment.num_edges);
    }

    /**
     * Merge triples with same predicate and direction to a vector
     */
    void init_triples_map(vector<vector<triple_t>> &triple_pso,
                          vector<vector<triple_t>> &triple_pos,
                          vector<vector<triple_attr_t>> &triple_sav) {
        #pragma omp parallel for num_threads(global_num_engines)
        for (int tid = 0; tid < global_num_engines; tid++) {
            vector<triple_t> &pso_vec = triple_pso[tid];
            vector<triple_t> &pos_vec = triple_pos[tid];
            vector<triple_attr_t> &sav_vec = triple_sav[tid];

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

            i = 0;
            while (!sav_vec.empty() && i < sav_vec.size()) {
                current_pid = sav_vec[i].a;
                tbb_triple_attr_hash_map::accessor a;
                attr_triples_map.insert(a, ikey_t(0, current_pid, OUT));

                for (; sav_vec[i].a == current_pid; i++) {
                    a->second.push_back(sav_vec[i]);
                }
            }
        }
    }

    void finalize_init() {
        tbb_triple_hash_map().swap(triples_map);
        tbb_triple_attr_hash_map().swap(attr_triples_map);
        tbb_unordered_set().swap(attr_set);
        tbb::concurrent_unordered_map<sid_t, int>().swap(attr_type_map);
    }

    // init metadata for each segment
    void init_segment_metas(const vector<vector<triple_t>> &triple_pso,
                            const vector<vector<triple_t>> &triple_pos,
                            const vector<vector<triple_attr_t>> &triple_sav) {
        /**
         * count(|pred| means number of local predicates):
         * 1. normal vertices [vid|pid|IN/OUT], key: pid, cnt_t: in&out, #item: |pred|
         * 2. vid's all types [vid|TYPE_ID(1)|OUT], key: TYPE_ID(1), cnt_t: out, #item: contained above
         * 3*. vid's all predicates [vid|PREDICATE_ID(0)|IN/OUT], key: PREDICATE_ID(0), cnt_t: in&out, #item: 1
         * 4^. attr triples [vid|pid|out], key: pid, cnt_t: out, #item: |attrpred|
         */
        map<sid_t, cnt_t> normal_cnt_map;

        /**
         * count(|pred| means number of local predicates):
         * 1. predicate index [0|pid|IN/OUT], key: pid, cnt_t: in&out, #item: |pred|
         * 2. type index [0|typeid|IN], key: typeid, cnt_t: in, #item: |type|
         */
        map<sid_t, cnt_t> index_cnt_map;

        // initialization
        for (int i = 0; i <= get_num_preds(); ++i) {
            index_cnt_map.insert(make_pair(i, cnt_t()));
            normal_cnt_map.insert(make_pair(i, cnt_t()));
            for (int dir = 0; dir <= 1; dir++)
                rdf_segment_meta_map.insert(make_pair(segid_t(0, i, dir), rdf_segment_meta_t()));
        }
        // init index segment
        rdf_segment_meta_map.insert(make_pair(segid_t(1, PREDICATE_ID, IN), rdf_segment_meta_t()));
        rdf_segment_meta_map.insert(make_pair(segid_t(1, PREDICATE_ID, OUT), rdf_segment_meta_t()));

        #pragma omp parallel for num_threads(global_num_engines)
        for (int tid = 0; tid < global_num_engines; tid++) {
            const vector<triple_t> &pso = triple_pso[tid];
            const vector<triple_t> &pos = triple_pos[tid];
            const vector<triple_attr_t> &sav = triple_sav[tid];

            uint64_t s = 0;
            while (s < pso.size()) {
                uint64_t e = s + 1;

                while ((e < pso.size())
                        && (pso[s].s == pso[e].s)
                        && (pso[s].p == pso[e].p))  {
                    // count #edge of type-idx
                    if (pso[e].p == TYPE_ID && is_tpid(pso[e].o)) {
#ifdef VERSATILE
                        t_set.insert(pso[e].o);
#endif
                        index_cnt_map[ pso[e].o ].in++;
                    }
                    e++;
                }

#ifdef VERSATILE
                v_set.insert(pso[s].s);
                p_set.insert(pso[s].p);
                out_preds[pso[s].s].insert(pso[s].p);
                // count vid's all predicates OUT (number of value in the segment)
                normal_cnt_map[PREDICATE_ID].out++;
#endif

                // count #edge of predicate
                normal_cnt_map[ pso[s].p ].out += (e - s);

                // count #edge of predicate-idx
                index_cnt_map[ pso[s].p ].in++;

                // count #edge of type-idx
                if (pso[s].p == TYPE_ID && is_tpid(pso[s].o)) {
#ifdef VERSATILE
                    t_set.insert(pso[s].o);
#endif
                    index_cnt_map[ pso[s].o ].in++;
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
#ifdef VERSATILE
                v_set.insert(pos[s].o);
                p_set.insert(pos[s].p);
                in_preds[pos[s].o].insert(pos[s].p);
                // count vid's all predicates IN (number of value in the segment)
                normal_cnt_map[PREDICATE_ID].in++;
#endif

                // count #edge of predicate
                normal_cnt_map[ pos[s].p ].in += (e - s);
                index_cnt_map[ pos[s].p ].out++;
                s = e;
            }

            s = 0;
            while (s < sav.size()) {
                uint64_t e = s + 1;

                while ((e < sav.size()) && (sav[s].s == sav[e].s) && (sav[s].a == sav[e].a))
                    e++;

                // count #edge of predicate
                normal_cnt_map[ sav[s].a ].out += (e - s);
                attr_set.insert(sav[s].a);
                attr_type_map.insert(std::make_pair(sav[s].a, boost::apply_visitor(variant_type(), sav[s].v)));

                s = e;
            }
        }

        // count the total number of keys
        uint64_t total_num_keys = 0, num_typeid = 0;
#ifdef VERSATILE
        logger(LOG_DEBUG, "pid: %d: normal: #IN: %lu, #OUT: %lu", PREDICATE_ID,
               normal_cnt_map[PREDICATE_ID].in.load(), normal_cnt_map[PREDICATE_ID].out.load());
        total_num_keys += out_preds.size() + in_preds.size();
#endif
        for (int i = 1; i <= get_num_preds(); ++i) {
            logger(LOG_DEBUG, "pid: %d: normal: #IN: %lu, #OUT: %lu; index: #ALL: %lu, #IN: %lu, #OUT: %lu",
                   i, normal_cnt_map[i].in.load(), normal_cnt_map[i].out.load(),
                   (index_cnt_map[i].in.load() + index_cnt_map[i].out.load()),
                   index_cnt_map[i].in.load(), index_cnt_map[i].out.load());
            /**
             * this if-sentence checks if the i is a typeid
             * if the sentence is true, it means i is a normal predicate
             * index_cnt_map[i] stores #edges of pid's predicate index (IN, OUT)
             * whose sum equals to #keys of the predicate
             */
            if (attr_set.find(i) != attr_set.end()) {
                total_num_keys += normal_cnt_map[i].out.load();
            } else if (normal_cnt_map[i].in.load() + normal_cnt_map[i].out.load() > 0) {
                all_local_preds.push_back(i);
                total_num_keys += (index_cnt_map[i].in.load()
                                   + index_cnt_map[i].out.load());
            } else if (index_cnt_map[i].in.load() > 0) {
                num_typeid++;
            }
        }
        /**
         * #predicate index = #normal predicate * 2
         * #type index = #typeid
         */
        total_num_keys += all_local_preds.size() * 2 + num_typeid;

        uint64_t bucket_off = 0, edge_off = 0;

        rdf_segment_meta_t &idx_out_seg = rdf_segment_meta_map[segid_t(1, PREDICATE_ID, OUT)];
        rdf_segment_meta_t &idx_in_seg = rdf_segment_meta_map[segid_t(1, PREDICATE_ID, IN)];

#ifdef VERSATILE
        // vid's all predicates OUT
        rdf_segment_meta_t &pred_out_seg = rdf_segment_meta_map[segid_t(0, PREDICATE_ID, OUT)];
        pred_out_seg.num_edges = normal_cnt_map[PREDICATE_ID].out.load();
        pred_out_seg.edge_start = (pred_out_seg.num_edges > 0) ? alloc_edges(pred_out_seg.num_edges) : 0;
        pred_out_seg.num_keys = out_preds.size();
        alloc_buckets_to_segment(pred_out_seg, segid_t(0, PREDICATE_ID, OUT), total_num_keys);
        // vid's all predicates IN
        rdf_segment_meta_t &pred_in_seg = rdf_segment_meta_map[segid_t(0, PREDICATE_ID, IN)];
        pred_in_seg.num_edges = normal_cnt_map[PREDICATE_ID].in.load();
        pred_in_seg.edge_start = (pred_in_seg.num_edges > 0) ? alloc_edges(pred_in_seg.num_edges) : 0;
        pred_in_seg.num_keys = in_preds.size();
        alloc_buckets_to_segment(pred_in_seg, segid_t(0, PREDICATE_ID, IN), total_num_keys);

        // all local entities
        idx_in_seg.num_edges += v_set.size();
        idx_in_seg.num_keys += 1;
        // all local types
        idx_out_seg.num_edges += t_set.size();
        idx_out_seg.num_keys += 1;
        // all local predicates
        idx_out_seg.num_edges += p_set.size();
        idx_out_seg.num_keys += 1;

        logstream(LOG_DEBUG) <<  "s_set: " << pred_out_seg.num_keys << ", o_set: " << pred_in_seg.num_keys
            << ", v_set: " << v_set.size() << ", p_set: " << p_set.size() << ", t_set: " << t_set.size() << LOG_endl;
#endif

        for (sid_t pid = 1; pid <= get_num_preds(); ++pid) {
            rdf_segment_meta_t &out_seg = rdf_segment_meta_map[segid_t(0, pid, OUT)];
            rdf_segment_meta_t &in_seg = rdf_segment_meta_map[segid_t(0, pid, IN)];

            out_seg.num_edges = normal_cnt_map[pid].out.load();
            in_seg.num_edges = normal_cnt_map[pid].in.load();

            idx_out_seg.num_edges += index_cnt_map[pid].out.load();
            idx_in_seg.num_edges += index_cnt_map[pid].in.load();

            if (attr_set.find(pid) != attr_set.end()) {
                // attribute segment
                out_seg.num_keys = out_seg.num_edges;
                in_seg.num_keys = 0;
                // calculate the number of edge_t needed to store 1 value
                uint64_t sz = (attr_type_map[pid] - 1) / sizeof(edge_t) + 1;   // get the ceil size;
                out_seg.num_edges = out_seg.num_edges * sz;
            } else {
                // normal pred segment
                uint64_t normal_nkeys[2] = {index_cnt_map[pid].out, index_cnt_map[pid].in};
                out_seg.num_keys = (out_seg.num_edges == 0) ? 0 : normal_nkeys[OUT];
                in_seg.num_keys  = (in_seg.num_edges == 0)  ? 0 : normal_nkeys[IN];
            }

            // allocate space for edges in entry-region
            out_seg.edge_start = (out_seg.num_edges > 0) ?
                                        alloc_edges(out_seg.num_edges) : 0;
            in_seg.edge_start  = (in_seg.num_edges > 0) ?
                                        alloc_edges(in_seg.num_edges) : 0;

            alloc_buckets_to_segment(out_seg, segid_t(0, pid, OUT), total_num_keys);
            alloc_buckets_to_segment(in_seg, segid_t(0, pid, IN), total_num_keys);

            logger(LOG_DEBUG, "Predicate[%d]: normal: OUT[#keys: %lu, #buckets: %lu, #edges: %lu] "
                   "IN[#keys: %lu, #buckets: %lu, #edges: %lu];",
                   pid, out_seg.num_keys, out_seg.num_buckets, out_seg.num_edges,
                   in_seg.num_keys, in_seg.num_buckets, in_seg.num_edges);

        }

        idx_out_seg.edge_start = (idx_out_seg.num_edges > 0) ? alloc_edges(idx_out_seg.num_edges) : 0;
        idx_out_seg.num_keys = all_local_preds.size();
        alloc_buckets_to_segment(idx_out_seg, segid_t(1, PREDICATE_ID, OUT), total_num_keys);

        idx_in_seg.edge_start = (idx_in_seg.num_edges > 0) ? alloc_edges(idx_in_seg.num_edges) : 0;
        idx_in_seg.num_keys = all_local_preds.size() + num_typeid;
        alloc_buckets_to_segment(idx_in_seg, segid_t(1, PREDICATE_ID, IN), total_num_keys);

        logger(LOG_DEBUG, "index: OUT[#keys: %lu, #buckets: %lu, #edges: %lu], "
                   "IN[#keys: %lu, #buckets: %lu, #edges: %lu], bucket_off: %lu\n",
                   idx_out_seg.num_keys, idx_out_seg.num_buckets, idx_out_seg.num_edges,
                   idx_in_seg.num_keys, idx_in_seg.num_buckets, idx_in_seg.num_edges, main_hdr_off);

        logger(LOG_DEBUG, "#total_keys: %lu, bucket_off: %lu, #total_entries: %lu",
               total_num_keys, main_hdr_off, this->last_entry);
    }

public:
    StaticGStore(int sid, Mem *mem): GStore(sid, mem) {
        pthread_spin_init(&entry_lock, 0);
    }

    ~StaticGStore() {}

    void init(vector<vector<triple_t>> &triple_pso,
              vector<vector<triple_t>> &triple_pos,
              vector<vector<triple_attr_t>> &triple_sav) {
        uint64_t start, end;
        start = timer::get_usec();
        // merge triple_pso and triple_pos into a map
        init_triples_map(triple_pso, triple_pos, triple_sav);
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for merging triple_pso, triple_pos and triple_sav." << LOG_endl;

        start = timer::get_usec();
        init_segment_metas(triple_pso, triple_pos, triple_sav);
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for initializing predicate segment statistics." << LOG_endl;

        start = timer::get_usec();
        logstream(LOG_DEBUG) << "#" << sid << ": all_local_preds: " << all_local_preds.size() << LOG_endl;
        #pragma omp parallel for num_threads(global_num_engines)
        for (int i = 0; i < all_local_preds.size(); i++) {
            int localtid = omp_get_thread_num();
            sid_t pid = all_local_preds[i];
            insert_triples_to_segment(localtid, segid_t(0, pid, OUT));
            insert_triples_to_segment(localtid, segid_t(0, pid, IN));
        }

        vector<sid_t> aids;
        for (auto iter = attr_set.begin(); iter != attr_set.end(); iter++)
            aids.push_back(*iter);
        #pragma omp parallel for num_threads(global_num_engines)
        for (int i = 0; i < aids.size(); i++) {
            int localtid = omp_get_thread_num();
            insert_attr_to_segment(segid_t(0, aids[i], OUT));
        }
        vector<sid_t>().swap(aids);

#ifdef VERSATILE
        #pragma omp parallel for num_threads(2)
        for (int i = 0; i < 2; i++) {
            if (i == 0) {
                for (auto &item : in_preds) {
                    insert_preds(item.first, item.second, IN);
                    std::unordered_set<sid_t>().swap(item.second);
                }
            } else {
                for (auto &item : out_preds) {
                    insert_preds(item.first, item.second, OUT);
                    std::unordered_set<sid_t>().swap(item.second);
                }
            }
        }
        tbb::concurrent_unordered_map<sid_t, std::unordered_set<sid_t> >().swap(in_preds);
        tbb::concurrent_unordered_map<sid_t, std::unordered_set<sid_t> >().swap(out_preds);
#endif // VERSATILE
        #pragma omp parallel for num_threads(2)
        for (int i = 0; i < 2; i++) {
            if (i == 0)
                insert_idx(&pidx_in_map, &tidx_map, IN);
            else
                insert_idx(&pidx_out_map, nullptr, OUT);
        }
        end = timer::get_usec();
        logstream(LOG_INFO) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                            << "for inserting triples as segments into gstore" << LOG_endl;

        finalize_segment_metas();
        finalize_init();

        // synchronize segment metadata among servers
        extern TCP_Adaptor *con_adaptor;
        sync_metadata(con_adaptor);
    }

    inline const std::map<segid_t, rdf_segment_meta_t> &get_rdf_segment_metas() { return rdf_segment_meta_map; }

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
