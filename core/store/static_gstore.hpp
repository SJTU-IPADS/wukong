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

    uint64_t alloc_edges_to_segment(uint64_t num_edges) {
        return (num_edges > 0) ? alloc_edges(num_edges) : 0;
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

        logger(LOG_DEBUG, "Seg[%lu|%lu|%lu]: "
               "#edges: %lu, edge_start: %lu, off: %lu, ",
               segid.index, segid.pid, segid.dir,
               segment.num_edges, segment.edge_start, off);
        ASSERT(off <= segment.edge_start + segment.num_edges);
    }

    // insert attributes
    void insert_attr_to_segment(int tid, segid_t segid) {
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
        uint64_t off = segment.edge_start;
        int type = attr_type_map[aid];
        uint64_t sz = (get_sizeof(type) - 1) / sizeof(edge_t) + 1;   // get the ceil size;

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
     * insert {predicate index OUT, t_set*, p_set*}
     * or {predicate index IN, type index, v_set*}
     */
    void insert_idx(const tbb_hash_map &pidx_map, const tbb_hash_map &tidx_map, dir_t d) {
        tbb_hash_map::const_accessor ca;
        rdf_segment_meta_t &segment = rdf_segment_meta_map[segid_t(1, PREDICATE_ID, d)];
        // it is possible that num_edges = 0 if loading an empty dataset
        // ASSERT(segment.num_edges > 0);

        uint64_t off = segment.edge_start;

        for (int i = 0; i < all_local_preds.size(); i++) {
            sid_t pid = all_local_preds[i];
            bool success = pidx_map.find(ca, pid);
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
            for (auto const &e : tidx_map) {
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

#ifdef VERSATILE
    // insert vid's preds into gstore
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
