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

#include <queue>

#include "core/store/kvstore.hpp"

#include "core/store/mm/buddy_malloc.hpp"
#include "core/store/mm/jemalloc.hpp"
#include "core/store/mm/malloc_interface.hpp"

namespace wukong {

/**
 * @brief Dynamic KVStore
 * 
 * used by DynamicRDFGraph
 * 
 * @tparam KeyType 
 * @tparam PtrType 
 * @tparam ValueType 
 */
template <class KeyType, class PtrType, class ValueType>
class DynamicKVStore : public KVStore<KeyType, PtrType, ValueType> {
protected:
    // block deferred to be freed
    struct free_blk {
        uint64_t off;
        uint64_t expire_time;
        free_blk(uint64_t off, uint64_t expire_time) : off(off), expire_time(expire_time) {}
    };

    using slot_t = typename KVStore<KeyType, PtrType, ValueType>::slot_t;

    // manage the memory of value
    MAInterface* value_allocator;

    /**
     * Defer blk's free operation for dynamic cache.
     * Pend the free operation when blk is to be collected by add_pending_free()
     * When allocating a new blk by alloc_entries(), check if pending free's lease expires
     * and collect free space by sweep_free().
     */
    uint64_t lease;

    /**
     * A size flag is put into the tail of edges (in the entry region) for dynamic cache.
     * NOTE: the (remote) edges accessed by (local) RDMA cache are valid
     *       if and only if the size flag of edges is consistent with the size within the pointer.
     */
    static const sid_t INVALID_EDGES = 1 << NBITS_SIZE;  // flag indicates invalidate edges

    /**
     * global dynamic reserve factor
     * when creating new segment during dynamic loading,
     * #buckets = (#buckets-remain * global_dyn_res_factor / 100) / #new-segments
     */
    static const int global_dyn_res_factor = 50;

    std::queue<free_blk> free_queue;
    pthread_spinlock_t free_queue_lock;

    // Convert given byte units to edge units.
    inline uint64_t b2e(uint64_t sz) { return sz / sizeof(edge_t); }
    // Convert given edge uints to byte units.
    inline uint64_t e2b(uint64_t sz) { return sz * sizeof(edge_t); }
    // Return exact block size of given size in edge unit.
    inline uint64_t blksz(uint64_t sz) { return b2e(value_allocator->sz_to_blksz(e2b(sz))); }
    /* Insert size flag in values.
     * @flag: size flag to insert
     * @sz: actual size of values
     * @off: offset of values
    */
    inline void insert_sz(sid_t flag, uint64_t sz, uint64_t off) {
        uint64_t blk_sz = blksz(sz + 1);  // reserve one space for flag
        this->values[off + blk_sz - 1] = flag;
    }

    // Pend the free operation of given block.
    inline void add_pending_free(PtrType ptr) {
        uint64_t expire_time = timer::get_usec() + lease;
        free_blk blk(ptr.off, expire_time);

        pthread_spin_lock(&free_queue_lock);
        free_queue.push(blk);
        pthread_spin_unlock(&free_queue_lock);
    }

    // Execute all expired pending free operations in queue.
    inline void sweep_free() {
        pthread_spin_lock(&free_queue_lock);
        while (!free_queue.empty()) {
            free_blk blk = free_queue.front();
            if (timer::get_usec() < blk.expire_time)
                break;
            value_allocator->free(e2b(blk.off));
            free_queue.pop();
        }
        pthread_spin_unlock(&free_queue_lock);
    }

    bool is_dup(slot_t* slot, ValueType value) {
        int size = slot->ptr.size;
        for (int i = 0; i < size; i++)
            if (this->values[slot->ptr.off + i] == value)
                return true;
        return false;
    }

    // Allocate space in given segment.
    // NOTICE! This function is not thread-safe.
    uint64_t alloc_entries(uint64_t num_values, int tid = 0) override {
        if (Global::enable_caching)
            sweep_free();  // collect free space before allocate

        uint64_t sz = e2b(num_values + 1);  // reserve one space for sz
        uint64_t off = b2e(value_allocator->malloc(sz, tid));
        insert_sz(num_values, num_values, off);
        return off;
    }

    /// edge is always valid
    bool value_is_valid(slot_t& slot, ValueType* value_ptr) override {
        if (!Global::enable_caching)
            return true;

        uint64_t blk_sz = blksz(slot.ptr.size + 1);  // reserve one space for flag
        return (value_ptr[blk_sz - 1] == slot.ptr.size);
    }

    uint64_t get_value_sz(const slot_t& slot) override { return blksz(slot.ptr.size + 1) * sizeof(ValueType); }

public:
    DynamicKVStore(int sid, Mem* mem) : KVStore<KeyType, PtrType, ValueType>(sid, mem) {
#ifdef USE_JEMALLOC
        value_allocator = new JeMalloc();
#else
        value_allocator = new BuddyMalloc();
#endif  // end of USE_JEMALLOC
        pthread_spin_init(&free_queue_lock, 0);
        lease = SEC(600);
        this->rdma_cache.set_lease(lease);
    }

    ~DynamicKVStore() {}

    bool insert_key_value(KeyType key, ValueType value, bool& dedup_or_isdup, int tid) override {
        uint64_t bucket_id = this->bucket_local(key);
        uint64_t lock_id = bucket_id % this->NUM_LOCKS;
        uint64_t slot_id = this->search_key(key);
        slot_t* slot = &this->slots[slot_id];
        pthread_spin_lock(&this->bucket_locks[lock_id]);
        if (slot->ptr.size == 0) {
            uint64_t off = this->alloc_entries(1, tid);
            this->values[off] = value;
            this->slots[slot_id].ptr = PtrType(1, off);
            pthread_spin_unlock(&this->bucket_locks[lock_id]);
            dedup_or_isdup = false;
            return true;
        } else {
            if (dedup_or_isdup && is_dup(slot, value)) {
                pthread_spin_unlock(&this->bucket_locks[lock_id]);
                return false;
            }
            dedup_or_isdup = false;
            uint64_t need_size = slot->ptr.size + 1;

            // a new block is needed
            if (blksz(slot->ptr.size + 1) - 1 < need_size) {
                PtrType old_ptr = slot->ptr;

                uint64_t off = this->alloc_entries(need_size, tid);
                memcpy(&this->values[off], &this->values[old_ptr.off], e2b(old_ptr.size));
                this->values[off + old_ptr.size] = value;
                // invalidate the old block
                insert_sz(INVALID_EDGES, old_ptr.size, old_ptr.off);
                slot->ptr = PtrType(need_size, off);

                if (Global::enable_caching) {
                    add_pending_free(old_ptr);
                } else {
                    add_pending_free(old_ptr);
                    ///FIXME: there is a bug about free here, but I can't fix it, so I replace free with add_pending_free
                    //edge_allocator->free(e2b(old_ptr.off)); 
                }
            } else {
                // update size flag
                insert_sz(need_size, need_size, slot->ptr.off);
                this->values[slot->ptr.off + slot->ptr.size] = value;
                slot->ptr.size = need_size;
            }

            pthread_spin_unlock(&this->bucket_locks[lock_id]);
            return false;
        }
    }

    void refresh() override {
        KVStore<KeyType, PtrType, ValueType>::refresh();
        // Since tid of engines is not from 0, allocator should init num_threads.
        value_allocator->init(reinterpret_cast<void*>(this->values), this->num_entries * sizeof(ValueType), Global::num_threads);
    }

    void print_mem_usage() override {
        KVStore<KeyType, PtrType, ValueType>::print_mem_usage();
        value_allocator->merge_freelists();
        value_allocator->print_memory_usage();
    }
};

}  // end of namespace wukong
