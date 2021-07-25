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

#include <pthread.h>
#include <stdint.h>  // uint64_t
#include <atomic>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>

#include <boost/unordered_set.hpp>

#include <tbb/concurrent_hash_map.h>       // NOLINT
#include <tbb/concurrent_unordered_set.h>  // NOLINT
#include <tbb/concurrent_vector.h>         // NOLINT

#include "core/common/global.hpp"
#include "core/common/mem.hpp"
#include "core/common/rdma.hpp"
#include "core/common/type.hpp"

#include "core/store/rdma_cache.hpp"
#include "core/store/segment_meta.hpp"
#include "core/store/vertex.hpp"

#include "utils/math.hpp"
#include "utils/timer.hpp"
#include "utils/unit.hpp"

namespace wukong {

/* Memory region used by KV */
struct KVMem {
    char *kvs;              // kv region in Mem
    uint64_t kvs_sz;        // kv region size
    char *rrbuf;            // RDMA-read region in Mem
    uint64_t rrbuf_sz;      // RDMA-read region size per thread
};

/**
 * @brief A general RDMA-based KV store
 * 
 * KVStore: key (main-header and indirect-header region) | value (entry region)
 * head region is a cluster chaining hash-table (with associativity)
 * entry region is a varying-size array
 * 
 * @tparam KeyType 
 * @tparam PtrType 
 * @tparam ValueType 
 */
template <class KeyType, class PtrType, class ValueType>
class KVStore {
    friend class GChecker;
    friend class RDFGraph;
    friend class SegmentRDFGraph;

public:
    // the number of locks to protect buckets
    static const int NUM_LOCKS = 1024;
    // the associativity of slots in each bucket
    static const int ASSOCIATIVITY = 8;

    /** 
     * Memory Usage (estimation):
     *   header region: |vertex| = 128-bit; 
     *   #verts = (#S + #O) * AVG(#P) ～= #T
     * 
     *   entry region:  |edge| = 32-bit; 
     *   #edges = #T * 2 + (#S + #O) * AVG(#P) ～= #T * 3
     *
     *   (+VERSATILE)
     *   #verts += #S + #O
     *   #edges += (#S + #O) * AVG(#P) ~= #T
     */
    // main-header / (main-header + indirect-header)
    static const int MHD_RATIO = 80;
    // header * 100 / (header + entry)
    static const int HD_RATIO = (128 * 100 / (128 + 3 * std::numeric_limits<sid_t>::digits));

protected:
    // slot = [key|pointer]
    struct slot_t {
        KeyType key;
        PtrType ptr;
    };

    int sid;

    KVMem kv_mem;

    slot_t* slots;
    ValueType* values;

    uint64_t num_slots;        // 1 bucket = ASSOCIATIVITY * slots
    uint64_t num_buckets;      // main-header region (static)
    uint64_t num_buckets_ext;  // indirect-header region (dynamical)
    uint64_t last_ext;         // allocation offset of ext bucket

    // lock virtualization (see paper: vLock CGO'13)
    pthread_spinlock_t bucket_locks[NUM_LOCKS];
    pthread_spinlock_t bucket_ext_lock;

    uint64_t num_entries;  // value region

    /**
     * Cache remote vertex(location) of the given key, eleminating one RDMA read.
     * This only works when RDMA enabled.
     */
    RDMA_Cache<KeyType, slot_t> rdma_cache;

    // allocate space to store values of given size. Return offset of allocated space.
    virtual uint64_t alloc_entries(uint64_t num_values, int tid = 0) = 0;

    // check the validation of given values according to given key.
    virtual bool value_is_valid(slot_t& slot, ValueType* value_ptr) = 0;

    // get the values size of given key
    virtual uint64_t get_value_sz(const slot_t& slot) = 0;

    /**
     * @brief insert a given key to store, 
     * 
     * The values of this key are allocated before this function call
     * 
     * @param key given key
     * @param ptr value pointer
     * @return uint64_t slot id
     */
    uint64_t insert_key(KeyType key, PtrType ptr) {
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
                if (this->slots[slot_id].key == key) {
                    logstream(LOG_ERROR) << "Conflict at slot["
                                         << slot_id << "] of bucket["
                                         << bucket_id << "], "
                                         << key.to_string() << ", "
                                         << this->slots[slot_id].key.to_string()
                                         << LOG_endl;
                    ASSERT(false);
                }

                // insert to an empty slot
                if (this->slots[slot_id].key.is_empty()) {
                    this->slots[slot_id].key = key;
                    this->slots[slot_id].ptr = ptr;
                    goto done;
                }
            }

            // whether the bucket_ext (indirect-header region) is used
            if (!this->slots[slot_id].key.is_empty()) {
                slot_id = this->slots[slot_id].key.vid * ASSOCIATIVITY;
                continue;  // continue and jump to next bucket
            }

            // allocate and link a new indirect header
            this->slots[slot_id].key.vid = alloc_ext_buckets(1);

            // move to a new bucket_ext
            slot_id = this->slots[slot_id].key.vid * ASSOCIATIVITY;
            // insert to the first slot
            this->slots[slot_id].key = key;
            this->slots[slot_id].ptr = ptr;
            goto done;
        }
    done:
        pthread_spin_unlock(&bucket_locks[lock_id]);
        ASSERT(slot_id < num_slots);
        return slot_id;
    }

    /**
     * @brief search slot id for given key
     * 
     * if not found, return the new slot id to insert
     * 
     * @param key given key
     * @return uint64_t slot id
     */
    uint64_t search_key(KeyType key) {
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
                if (this->slots[slot_id].key == key) {
                    goto done;
                }

                // end of search in this bucket
                if (this->slots[slot_id].key.is_empty()) {
                    goto done;
                }
            }

            // whether the bucket_ext (indirect-header region) is used
            if (!this->slots[slot_id].key.is_empty()) {
                slot_id = this->slots[slot_id].key.vid * ASSOCIATIVITY;
                continue;  // continue and jump to next bucket
            }

            // allocate and link a new indirect header
            this->slots[slot_id].key.vid = alloc_ext_buckets(1);
            // move to a new bucket_ext
            slot_id = this->slots[slot_id].key.vid * ASSOCIATIVITY;
            goto done;
        }
    done:
        pthread_spin_unlock(&bucket_locks[lock_id]);
        ASSERT(slot_id < num_slots);
        return slot_id;
    }

    /**
     * @brief calculate the bucket of given key
     * 
     * @param key given key
     * @param seg segment-based search
     * @return uint64_t bucket id
     */
    uint64_t bucket_local(KeyType key, rdf_seg_meta_t* seg = nullptr) {
        uint64_t bucket_id;
        if (seg != nullptr) {
            bucket_id = seg->bucket_start + key.hash() % seg->num_buckets;
        } else {
            bucket_id = key.hash() % this->num_buckets;
        }
        return bucket_id;
    }

    /**
     * @brief calculate the bucket of given key
     * 
     * NOTICE: we assume every machine has the same memory layout
     * 
     * @param key 
     * @param dst_sid target server
     * @param seg segment-based search
     * @return uint64_t bucket id
     */
    uint64_t bucket_remote(KeyType key, int dst_sid, rdf_seg_meta_t* seg = nullptr) {
        return bucket_local(key, seg);
    }

    /**
     * @brief Get the local slot of given key
     * 
     * @param tid caller thread id
     * @param key given key
     * @param seg segment-based search
     * @return slot_t 
     */
    slot_t get_slot_local(int tid, KeyType key, rdf_seg_meta_t* seg = nullptr) {
        uint64_t bucket_id = bucket_local(key, seg);
        while (true) {
            for (int i = 0; i < ASSOCIATIVITY; i++) {
                uint64_t slot_id = bucket_id * ASSOCIATIVITY + i;
                if (i < ASSOCIATIVITY - 1) {
                    if (this->slots[slot_id].key == key) {
                        // we found it
                        return this->slots[slot_id];
                    }
                } else {
                    if (this->slots[slot_id].key.is_empty())
                        // not found, return empty slot
                        return slot_t();

                    // move to next bucket
                    bucket_id = this->slots[slot_id].key.vid;
                    // break for-loop
                    break;
                }
            }
        }
    }

    /**
     * @brief Get remote slot of given key
     * 
     * This func will fail if RDMA is disabled.
     * 
     * @param tid caller thread id
     * @param dst_sid target server
     * @param key given key
     * @param seg segment-based search
     * @return slot_t 
     */
    slot_t get_slot_remote(int tid, int dst_sid, KeyType key, rdf_seg_meta_t* seg = nullptr) {
        uint64_t bucket_id = bucket_remote(key, dst_sid, seg);
        slot_t slot;

        // NOTICE: wukong doesn't support to 
        // directly get remote key/value without RDMA
        ASSERT(Global::use_rdma);

        // check cache
        if (rdma_cache.lookup(key, slot))
            return slot;

        // get remote bucket by RDMA-read
        char* buf = kv_mem.rrbuf + kv_mem.rrbuf_sz * tid;
        uint64_t buf_sz = kv_mem.rrbuf_sz;
        while (true) {
            uint64_t off = bucket_id * ASSOCIATIVITY * sizeof(slot_t);
            uint64_t sz = ASSOCIATIVITY * sizeof(slot_t);
            ASSERT(sz < buf_sz);  // enough space to host the slot

            RDMA& rdma = RDMA::get_rdma();
            rdma.dev->RdmaRead(tid, dst_sid, buf, sz, off);
            slot_t* slots = reinterpret_cast<slot_t*>(buf);
            for (int i = 0; i < ASSOCIATIVITY; i++) {
                if (i < ASSOCIATIVITY - 1) {
                    if (slots[i].key == key) {
                        rdma_cache.insert(slots[i]);
                        return slots[i];  // found
                    }
                } else {
                    if (slots[i].key.is_empty())
                        return slot_t();  // not found

                    // move to next bucket
                    bucket_id = slots[i].key.vid;
                    // break for-loop
                    break;
                }
            }
        }
    }

    // Get values of given key from dst_sid by RDMA read.
    ValueType* rdma_get_values(int tid, int dst_sid, slot_t& slot) {
        ASSERT(Global::use_rdma);

        char* buf = kv_mem.rrbuf + kv_mem.rrbuf_sz * tid;
        uint64_t r_off = num_slots * sizeof(slot_t) + slot.ptr.off * sizeof(ValueType);
        // the size of values
        uint64_t r_sz = get_value_sz(slot);
        uint64_t buf_sz = kv_mem.rrbuf_sz;
        ASSERT(r_sz < buf_sz);  // enough space to host the values

        RDMA& rdma = RDMA::get_rdma();
        rdma.dev->RdmaRead(tid, dst_sid, buf, r_sz, r_off);
        return reinterpret_cast<ValueType*>(buf);
    }

    /**
     * @brief Get local values according to given key
     * 
     * @param tid caller thread id
     * @param key given key
     * @param sz for return (value size)
     * @param seg for segment-based search
     * @return ValueType* 
     */
    ValueType* get_values_local(int tid, KeyType key, uint64_t& sz, rdf_seg_meta_t* seg = nullptr) {
        slot_t slot = get_slot_local(tid, key, seg);

        if (slot.key.is_empty()) {
            sz = 0;
            return nullptr;  // not found
        }

        // local values
        ValueType* value_ptr = &(this->values[slot.ptr.off]);

        sz = slot.ptr.size;
        return value_ptr;
    }

    /**
     * @brief Get remote values according to given key.
     * 
     * @param tid caller thread id
     * @param dst_sid target server id
     * @param key given key
     * @param sz for return (value size)
     * @param seg for segment-based search
     * @return ValueType* 
     */
    ValueType* get_values_remote(int tid, int dst_sid, KeyType key, uint64_t& sz, rdf_seg_meta_t* seg = nullptr) {
        slot_t slot = get_slot_remote(tid, dst_sid, key, seg);

        if (slot.key.is_empty()) {
            sz = 0;
            return nullptr;  // not found
        }

        // remote values
        ValueType* value_ptr = rdma_get_values(tid, dst_sid, slot);
        // check cache validation
        while (!value_is_valid(slot, value_ptr)) {
            // invalidate cache and try again
            rdma_cache.invalidate(key);
            slot = get_slot_remote(tid, dst_sid, key, seg);
            value_ptr = rdma_get_values(tid, dst_sid, slot);
        }

        sz = slot.ptr.size;
        return value_ptr;
    }

    /**
     * @brief Allocate extended buckets
     * 
     * @param n number of extended buckets to allocate
     * @return uint64_t start offset of allocated extended buckets
     */
    uint64_t alloc_ext_buckets(uint64_t n) {
        uint64_t orig;
        pthread_spin_lock(&bucket_ext_lock);
        orig = this->last_ext;
        this->last_ext += n;
        if (this->last_ext >= this->num_buckets_ext) {
            logstream(LOG_ERROR) << "out of indirect-header region." << LOG_endl;
            ASSERT(false);
        }
        pthread_spin_unlock(&bucket_ext_lock);
        return this->num_buckets + orig;
    }

public:
    /**
     * @brief Construct a new KVStore object
     * 
     * @param sid server id
     * @param mem main memory
     */
    // KVStore(int sid, Mem* mem) : sid(sid), mem(mem) {
    KVStore(int sid, KVMem kv_mem) : 
            sid(sid), kv_mem(kv_mem) {
        uint64_t header_region = kv_mem.kvs_sz * HD_RATIO / 100;
        uint64_t entry_region = kv_mem.kvs_sz - header_region;

        // header region
        this->num_slots = header_region / sizeof(slot_t);
        this->num_buckets = wukong::math::hash_prime_u64((num_slots / ASSOCIATIVITY) * MHD_RATIO / 100);
        this->num_buckets_ext = (num_slots / ASSOCIATIVITY) - num_buckets;
        // entry region
        this->num_entries = entry_region / sizeof(ValueType);

        this->slots = reinterpret_cast<slot_t*>(kv_mem.kvs);
        this->values = reinterpret_cast<ValueType*>(kv_mem.kvs + this->num_slots * sizeof(slot_t));

        pthread_spin_init(&this->bucket_ext_lock, 0);
        for (int i = 0; i < NUM_LOCKS; i++) {
            pthread_spin_init(&this->bucket_locks[i], 0);
        }

        // clean kv store
        this->refresh();

        // print kvstore usage
        logstream(LOG_INFO) << "[KV] kvstore = ";
        logstream(LOG_INFO) << kv_mem.kvs_sz << " bytes " << LOG_endl;
        logstream(LOG_INFO) << "  header region: " << this->num_slots << " slots"
                            << " (main = " << this->num_buckets
                            << ", indirect = " << this->num_buckets_ext << ")" << LOG_endl;
        logstream(LOG_INFO) << "  entry region: " << this->num_entries << " entries" << LOG_endl;
    }

    virtual ~KVStore() {}

    /**
     * @brief clean kv store
     */
    virtual void refresh() {
        #pragma omp parallel for num_threads(Global::num_engines)
        for (uint64_t i = 0; i < num_slots; i++) {
            this->slots[i].key = KeyType();
            this->slots[i].ptr = PtrType();
        }
        this->last_ext = 0;
    }


    inline void* get_slot_addr() { return reinterpret_cast<void*>(this->slots); }
    inline void* get_value_addr() { return reinterpret_cast<void*>(this->values); }

    /**
     * @brief Get the values for given key
     * 
     * @param tid caller thread id
     * @param dst_sid target server id
     * @param key given key
     * @param sz value size(return value)
     * @param seg for segment-based 
     * @return ValueType* value address
     */
    ValueType* get_values(int tid, int dst_sid, KeyType key, uint64_t& sz, rdf_seg_meta_t* seg = nullptr) {
        if (dst_sid == this->sid)
            return get_values_local(tid, key, sz, seg);
        else
            return get_values_remote(tid, dst_sid, key, sz, seg);
    }

    /**
     * @brief Check if the given key exists
     * 
     * @param key 
     * @return true the key exist
     */
    bool check_key_exist(KeyType key) {
        uint64_t bucket_id = bucket_local(key);
        uint64_t slot_id = bucket_id * ASSOCIATIVITY;
        uint64_t lock_id = bucket_id % NUM_LOCKS;

        pthread_spin_lock(&bucket_locks[lock_id]);
        while (slot_id < num_slots) {
            // the last slot of each bucket is always reserved for pointer to indirect header
            /// TODO: add type info to slot and reuse the last slot to store key
            /// TODO: key.vid is reused to store the bucket_id of indirect header rather than ptr.off,
            ///       since the is_empty() is not robust.
            for (int i = 0; i < ASSOCIATIVITY - 1; i++, slot_id++) {
                if (this->slots[slot_id].key == key) {
                    pthread_spin_unlock(&bucket_locks[lock_id]);
                    return true;
                }

                // insert to an empty slot
                if (this->slots[slot_id].key.is_empty()) {
                    pthread_spin_unlock(&bucket_locks[lock_id]);
                    return false;
                }
            }
            // whether the bucket_ext (indirect-header region) is used
            if (!this->slots[slot_id].key.is_empty()) {
                slot_id = this->slots[slot_id].key.vid * ASSOCIATIVITY;
                continue;  // continue and jump to next bucket
            }
            pthread_spin_unlock(&bucket_locks[lock_id]);
            return false;
        }
    }

    /**
     * @brief insert a new key-value pair (dynamically)
     * 
     * @param key 
     * @param value 
     * @param dedup_or_isdup param:if dedup; return value:is duplicate 
     * @param tid caller
     * @return true insert succeed
     * @return false insert failed
     */
    virtual bool insert_key_value(KeyType key, ValueType value, bool& dedup_or_isdup, int tid) = 0;

    /**
     * @brief print the memory usage of KV
     */
    virtual void print_mem_usage() {
        uint64_t used_slots = 0;
        uint64_t used_entries = 0;
        for (uint64_t x = 0; x < this->num_buckets; x++) {
            uint64_t slot_id = x * ASSOCIATIVITY;
            for (int y = 0; y < ASSOCIATIVITY - 1; y++, slot_id++) {
                if (this->slots[slot_id].key.is_empty())
                    continue;
                used_slots++;
                used_entries += this->slots[slot_id].ptr.size;
            }
        }

        logstream(LOG_INFO) << "[KV] main header: " << B2MiB(this->num_buckets * ASSOCIATIVITY * sizeof(slot_t))
                            << " MB (" << this->num_buckets * ASSOCIATIVITY << " slots)" << LOG_endl;
        logstream(LOG_INFO) << "\tused: " << 100.0 * used_slots / (this->num_buckets * ASSOCIATIVITY)
                            << " % (" << used_slots << " slots)" << LOG_endl;
        logstream(LOG_INFO) << "\tchain: " << 100.0 * this->num_buckets / (this->num_buckets * ASSOCIATIVITY)
                            << " % (" << this->num_buckets << " slots)" << LOG_endl;

        used_slots = 0;
        for (uint64_t x = this->num_buckets; x < this->num_buckets + this->last_ext; x++) {
            uint64_t slot_id = x * ASSOCIATIVITY;
            for (int y = 0; y < ASSOCIATIVITY - 1; y++, slot_id++) {
                if (this->slots[slot_id].key.is_empty())
                    continue;
                used_slots++;
                used_entries += this->slots[slot_id].ptr.size;
            }
        }

        logstream(LOG_INFO) << "[KV] indirect header: " << B2MiB(this->num_buckets_ext * ASSOCIATIVITY * sizeof(slot_t))
                            << " MB (" << this->num_buckets_ext * ASSOCIATIVITY << " slots)" << LOG_endl;
        logstream(LOG_INFO) << "\talloced: " << 100.0 * this->last_ext / this->num_buckets_ext
                            << " % (" << this->last_ext << " buckets)" << LOG_endl;
        logstream(LOG_INFO) << "\tused: " << 100.0 * used_slots / (num_buckets_ext * ASSOCIATIVITY)
                            << " % (" << used_slots << " slots)" << LOG_endl;

        logstream(LOG_INFO) << "[KV] entry: " << B2MiB(this->num_entries * sizeof(ValueType))
                            << " MB (" << num_entries << " entries)" << LOG_endl;
        logstream(LOG_INFO) << "\tused entries: " << B2MiB(used_entries * sizeof(ValueType))
                            << " MB (" << used_entries << " entries)" << LOG_endl;
    }
};

}  // end of namespace wukong
