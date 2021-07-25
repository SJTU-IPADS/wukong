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

#include "core/store/kvstore.hpp"

#include "core/store/mm/buddy_malloc.hpp"
#include "core/store/mm/jemalloc.hpp"
#include "core/store/mm/malloc_interface.hpp"

namespace wukong {

/**
 * @brief Static KVStore
 * 
 * used by StaticRDFGraph and SegmentRDFGraph
 * 
 * @tparam KeyType 
 * @tparam PtrType 
 * @tparam ValueType 
 */
template <class KeyType, class PtrType, class ValueType>
class StaticKVStore : public KVStore<KeyType, PtrType, ValueType> {
protected:
    // allocation offset of value entry
    uint64_t last_entry;
    // lock to protect entry allocation
    pthread_spinlock_t entry_lock;

    using slot_t = typename KVStore<KeyType, PtrType, ValueType>::slot_t;

    uint64_t alloc_entries(uint64_t num_values, int tid = 0) override {
        uint64_t orig;
        pthread_spin_lock(&entry_lock);
        orig = last_entry;
        last_entry += num_values;
        if (last_entry >= this->num_entries) {
            logstream(LOG_ERROR) << "out of entry region." << LOG_endl;
            ASSERT(last_entry < this->num_entries);
        }
        pthread_spin_unlock(&entry_lock);
        return orig;
    }

    /// edge is always valid
    bool value_is_valid(slot_t& slot, ValueType* value_ptr) override { return true; }

    uint64_t get_value_sz(const slot_t& slot) override { return slot.ptr.size * sizeof(ValueType); }

public:
    StaticKVStore(int sid, KVMem kv_mem) : KVStore<KeyType, PtrType, ValueType>(sid, kv_mem) {
        pthread_spin_init(&entry_lock, 0);

        // clear KV
        this->last_entry = 0;
    }

    ~StaticKVStore() {}

    bool insert_key_value(KeyType key, ValueType value, bool& dedup_or_isdup, int tid) override {
        // static kvstore doesn't support inserting kv-pair dynamically
        ASSERT(false);
    }

    void refresh() override {
        KVStore<KeyType, PtrType, ValueType>::refresh();
        this->last_entry = 0;
    }

    void print_mem_usage() {
        KVStore<KeyType, PtrType, ValueType>::print_mem_usage();
        logstream(LOG_INFO) << "\tused: " << 100.0 * this->last_entry / this->num_entries
                            << " % (last edge position: " << this->last_entry << ")" << LOG_endl;
    }
};

}  // end of namespace wukong
