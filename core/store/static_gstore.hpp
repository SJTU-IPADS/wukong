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

    // Allocate space in given segment.
    // NOTICE! This function is not thread-safe.
    uint64_t alloc_edges(uint64_t n, int tid = 0, rdf_seg_meta_t *seg = NULL) {
        ASSERT(seg != NULL);
        // Init offset if first visited.
        if (seg->edge_off == 0)
            seg->edge_off = seg->edge_start;
        uint64_t orig = seg->edge_off;
        seg->edge_off += n;
        ASSERT(seg->edge_off <= seg->edge_start + seg->num_edges);
        return orig;
    }

    // Allocate space to store edges of given size. Return offset of allocated space.
    uint64_t alloc_edges_to_seg(uint64_t num_edges) {
        if (num_edges == 0)
            return 0;
        uint64_t orig;
        pthread_spin_lock(&entry_lock);
        orig = last_entry;
        last_entry += num_edges;
        if (last_entry >= num_entries) {
            logstream(LOG_ERROR) << "out of entry region." << LOG_endl;
            ASSERT(last_entry < num_entries);
        }
        pthread_spin_unlock(&entry_lock);
        return orig;
    }

    /// edge is always valid
    bool edge_is_valid(vertex_t &v, edge_t *edge_ptr) { return true; }

    uint64_t get_edge_sz(const vertex_t &v) { return v.ptr.size * sizeof(edge_t); }

    /**
     * insert {predicate index OUT, t_set*, p_set*}
     * or {predicate index IN, type index, v_set*}
     */
    void insert_idx(const tbb_hash_map &pidx_map, const tbb_hash_map &tidx_map,
                    dir_t d, int tid = 0) {
        tbb_hash_map::const_accessor ca;
        rdf_seg_meta_t &segment = rdf_seg_meta_map[segid_t(1, PREDICATE_ID, d)];
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
    void alloc_vp_edges(dir_t d) {
        rdf_seg_meta_t &seg = rdf_seg_meta_map[segid_t(0, PREDICATE_ID, d)];
        uint64_t off = 0, sz = 0;
        for (auto iter = vp_meta[d].begin(); iter != vp_meta[d].end(); iter++) {
            sz = iter->second;
            iter->second = off;
            off += sz;
        }
        ASSERT(off == seg.num_edges);
    }

    // insert vid's preds into gstore
    void insert_vp(int tid, const vector<triple_t> &pso, const vector<triple_t> &pos) {
        vector<sid_t> preds;
        uint64_t s = 0;
        auto const &out_seg = rdf_seg_meta_map[segid_t(0, PREDICATE_ID, OUT)];
        auto const &in_seg = rdf_seg_meta_map[segid_t(0, PREDICATE_ID, IN)];

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
                tbb::concurrent_hash_map<sid_t, uint64_t>::accessor a;
                vp_meta[OUT].find(a, pso[s].s);
                uint64_t off = out_seg.edge_start + a->second;
                a.release();

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
                tbb::concurrent_hash_map<sid_t, uint64_t>::accessor a;
                vp_meta[IN].find(a, pos[s].o);
                uint64_t off = in_seg.edge_start + a->second;
                a.release();

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
    StaticGStore(int sid, Mem *mem): GStore(sid, mem) {
        pthread_spin_init(&entry_lock, 0);
    }

    ~StaticGStore() {}

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
        logstream(LOG_DEBUG) << "#" << sid << ": all_local_preds: " << all_local_preds.size() << LOG_endl;
        // insert normal triples
        auto out_triple_map = init_triple_map(triple_pso);
        auto in_triple_map = init_triple_map(triple_pos);
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < all_local_preds.size(); i++) {
            int localtid = omp_get_thread_num();
            sid_t pid = all_local_preds[i];
            insert_triples(localtid, segid_t(0, pid, OUT), out_triple_map, triple_pso);
            insert_triples(localtid, segid_t(0, pid, IN), in_triple_map, triple_pos);
        }

        vector<sid_t> aids;
        for (auto iter = attr_set.begin(); iter != attr_set.end(); iter++)
            aids.push_back(*iter);

        auto attr_map = init_triple_map(triple_sav);
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < aids.size(); i++) {
            int localtid = omp_get_thread_num();
            insert_attr(localtid, segid_t(0, aids[i], OUT), attr_map, triple_sav);
        }

        // insert type-index edges in parallel
        #pragma omp parallel for num_threads(Global::num_engines)
        for (int i = 0; i < 2; i++) {
            if (i == 0) insert_idx(pidx_in_map, tidx_map, IN);
            else insert_idx(pidx_out_map, tidx_map, OUT);
        }

        end = timer::get_usec();
        logstream(LOG_DEBUG) << "#" << sid << ": " << (end - start) / 1000 << "ms "
                             << "for inserting triples as segments into gstore" << LOG_endl;

        finalize_seg_metas();
        finalize_init();

        // synchronize segment metadata among servers
        sync_metadata();
    }

    void refresh() {
        #pragma omp parallel for num_threads(Global::num_engines)
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
                            << " % (last edge position: " << last_entry << ")" << LOG_endl;
    }

};
