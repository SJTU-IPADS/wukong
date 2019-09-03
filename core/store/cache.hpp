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

#include "global.hpp"
#include "store/vertex.hpp"

// utils
#include "unit.hpp"
#include "atomic.hpp"
#include "logger2.hpp"
#include "timer.hpp"

using namespace std;

/**
 * An RDMA-frienldy cache for distributed key-value store,
 * which caches the key (location) to skip one RDMA READ for retrieving one remote key-value pair.
 */
class RDMA_Cache {
    static const int NUM_BUCKETS = 1 << 20;
    static const int ASSOCIATIVITY = 8;  /// associativity of items in a bucket

    struct item_t {
        vertex_t v;  /// the key (location) of the key-value pair

        uint64_t expire_time;  /// expire time (DYNAMIC_GSTORE=ON)
        uint32_t cnt;          /// access count

        /// The version is used to detect reader-writer conflict.
        /// version == 0, when an insertion occurs.
        /// Init value is set to 1.
        /// version always increases after an insertion.
        uint32_t version;

        item_t() : expire_time(0), cnt(0), version(1) { }
    };

    struct bucket_t {
        item_t items[ASSOCIATIVITY]; /// associativity
    };

    bucket_t *hashtable; /// a hash-based 1-to-1 mapping cache

    uint64_t lease;  /// the period of cache invalidation (DYNAMIC_GSTORE=ON)

public:
    RDMA_Cache() {
        hashtable = new bucket_t[NUM_BUCKETS];
        lease = SEC(120);

        size_t mem_size = sizeof(bucket_t) * NUM_BUCKETS;
        logstream(LOG_INFO) << "allocate " << B2MiB(mem_size) << "MB RDMA cache" << LOG_endl;
    }

    /**
     * Lookup a vertex in cache according to the given key.
     * @param key the key to be looked up.
     * @param ret a reference vertex_t, store found vertex.
     * @return Found or not
     */
    bool lookup(ikey_t key, vertex_t &ret) {
        if (!Global::enable_caching)
            return false;

        int idx = key.hash() % NUM_BUCKETS;
        item_t *items = hashtable[idx].items;
        bool found = false;
        uint32_t ver;

        /// Lookup vertex in item list.
        for (int i = 0; i < ASSOCIATIVITY; i++) {
            if (items[i].v.key == key) {
                while (true) {
                    ver = items[i].version;
                    /// Re-check key since key may be replaced.
                    if (items[i].v.key == key) {
#ifdef DYNAMIC_GSTORE
                        if (timer::get_usec() < items[i].expire_time)
#endif
                        {
                            ret = items[i].v;
                            items[i].cnt++;
                            found = true;
                        }

                        asm volatile("" ::: "memory"); // barrier

                        // invalidation check
                        if (ver != 0 && items[i].version == ver)
                            return found;
                    } else
                        return false;
                }
            }
        }
        return false;
    } // end of lookup

    /**
     * Insert a vertex into cache.
     * @param v the item to be inserted.
     */
    void insert(vertex_t &v) {
        if (!Global::enable_caching)
            return;

        int idx = v.key.hash() % NUM_BUCKETS;
        item_t *items = hashtable[idx].items;

        uint64_t min_cnt;
        int pos = -1;  // position to insert v

        while (true) {
            for (int i = 0; i < ASSOCIATIVITY; i++) {
                if (items[i].v.key == v.key || items[i].v.key.is_empty()) {
                    pos = i;
                    break;
                }
            }
            // no free cache line
            // pick one to replace (LFU)
            if (pos == -1) {
                min_cnt = 1ul << 63;
                for (int i = 0; i < ASSOCIATIVITY; i++) {
                    if (items[i].cnt < min_cnt) {
                        min_cnt = items[i].cnt;
                        pos = i;
                        if (min_cnt == 0)
                            break;
                    }
                }
            }

            volatile uint32_t old_ver = items[pos].version;
            if (old_ver != 0) {
                uint32_t ret_ver = wukong::atomic::compare_and_swap(&items[pos].version, old_ver, 0);
                if (ret_ver == old_ver) {
#ifdef DYNAMIC_GSTORE
                    // Do not reset visit cnt for the same vertex
                    items[pos].cnt = (items[pos].v.key == v.key) ? items[pos].cnt : 0;
                    items[pos].v = v;
                    items[pos].expire_time = timer::get_usec() + lease;
#else
                    items[pos].cnt = 0;
                    items[pos].v = v;
#endif
                    asm volatile("" ::: "memory");
                    ret_ver = wukong::atomic::compare_and_swap(&items[pos].version, 0, old_ver + 1);
                    assert(ret_ver == 0);
                    return;
                }
            }
        }
    } // end of insert

    /**
     * Set lease term.
     * @param _lease the length of lease.
     */
    void set_lease(uint64_t _lease) { lease = _lease; }

    /**
     * Invalidate cache item of the given key.
     * Only work when the corresponding vertex exists.
     * @param key an ikey_t argument.
     */
    void invalidate(ikey_t key) {
        if (!Global::enable_caching)
            return;

        int idx = key.hash() % NUM_BUCKETS;
        item_t *items = hashtable[idx].items;
        for (int i = 0; i < ASSOCIATIVITY; i++) {
            if (items[i].v.key == key) {

                /// Version is not checked and set here
                /// since inconsistent expire time does not cause staleness.
                /// The only possible overhead is
                /// an extra update of vertex and expire time.
                items[i].expire_time = timer::get_usec();
                return;
            }
        }
    }
};
