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

#include <iomanip>
#include "mm/malloc_interface.hpp"

class BuddyMalloc : public MAInterface {
private:

    // block size >= 2^level_low_bound units
    static const uint64_t level_low_bound = 4;

    // the dividing line which we can use multithread small malloc
    // block size >= 2^level_low_bound units && block size < 2^level_dividing_line
    static const uint64_t level_dividing_line = 22;

    // block size <= 2^level_up_bound units
    static const uint64_t level_up_bound = 32;

    char *start_ptr;

    char *top_of_heap;

    pthread_spinlock_t malloc_free_lock_large;

    pthread_spinlock_t malloc_free_lock_small;

    pthread_spinlock_t counter_lock;

    // prev_free_idx and next_free_idx are for bidirection link-list
    struct header {
        uint64_t in_use: 1;
        uint64_t level: 6;
        uint64_t prev_free_idx: 48;
        uint64_t next_free_idx: 48;
        uint64_t align: 25;
    } __attribute__((packed));

    // the thread num used to insert_normal
    uint64_t nthread_parallel_load;

    // Byte num of header
    uint64_t size_per_header;

    //total size of the memory
    uint64_t heap_size;

    //total size has been sbrked
    uint64_t malloc_size;

    // free_list[i]->next points to the first free block with size 2^i bytes
    // the freelist for block size >= 2^level_dividing_line
    header *large_free_list[level_up_bound - level_dividing_line + 1];

    // the freelist for block size < 2^level_dividing_line
    header *small_free_list[level_dividing_line - level_low_bound + 1];

    // small freelists for multithread insert_normal, every thread has one freelist,
    // merge when insert normal finished
    header **tmp_small_free_list;

    // a statistic for print_memory_usage()
    uint64_t usage_counter[level_up_bound + 1];

    bool dynamic_load = false;

    //get the certain index of large freelist
    inline uint64_t level_to_index_large(uint64_t level) {
        assert(level >= level_dividing_line && level <= level_up_bound);
        return level - level_dividing_line;
    }

    //get the certain index of small freelist
    inline uint64_t level_to_index_small(uint64_t level, int64_t tid) {
        assert(level <= level_dividing_line && level >= level_low_bound);
        return (level - level_low_bound) + (level_dividing_line - level_low_bound + 1) * tid;

    }

    inline uint64_t level_to_index_small(uint64_t level) {
        assert(level <= level_dividing_line && level >= level_low_bound);
        return level - level_low_bound;
    }

    //return ptr from index
    inline void *idx_to_ptr(uint64_t idx) {
        return start_ptr + idx;
    }
    //return index ftom ptr
    inline uint64_t ptr_to_idx(char *ptr) {
        return (uint64_t)((char*)ptr - start_ptr);
    }
    //return value index by header index
    inline uint64_t get_value_idx(uint64_t header_idx) {
        return header_idx + size_per_header;
    }
    //return header index by value index
    inline uint64_t get_header_idx(uint64_t value_idx) {
        return value_idx - size_per_header;
    }

    // return whether there exit free block for need level or not
    inline bool is_empty_large(uint64_t level) {
        return large_free_list[level_to_index_large(level)]->next_free_idx == ptr_to_idx((char*) large_free_list[level_to_index_large(level)]);
    }

    // return whether there exit free block for need level or not
    inline bool is_empty_small(uint64_t level, int64_t tid) {
        if (dynamic_load)
            return small_free_list[level_to_index_small(level)]->next_free_idx == ptr_to_idx((char*)small_free_list[level_to_index_small(level)]);
        else
            return tmp_small_free_list[level_to_index_small(level, tid)]->next_free_idx == ptr_to_idx((char*) tmp_small_free_list[level_to_index_small(level, tid)]);
    }

    // truncate level
    inline uint64_t truncate_level(uint64_t level) {
        if (level < level_low_bound)
            return level_low_bound;
        // should be less than level_up_bound
        if (level > level_up_bound) {
            logstream(LOG_ERROR) << "need level: " << level << " level_up_bound: " << level_up_bound << LOG_endl;
            ASSERT(false);
        }
        return level;
    }

    // convert a size into level. useful for malloc
    inline uint64_t size_to_level(uint64_t size) {
        uint64_t level = 0, tmp = 1;
        size += size_per_header;
        while (true) {
            tmp <<= 1;
            level++;
            if (tmp >= size)
                break;
        }
        return truncate_level(level);
    }

    // methods useful for malloc and free, mark a large block as free block
    inline void mark_free_large(header *start, uint64_t level) {
        start->in_use = 0;
        start->level = level;
        header *free_header = large_free_list[level_to_index_large(level)];

        // add to free list
        // free_list[level] <--> start <--> free_list[level]->next_free_idx
        header *prev = free_header;
        header *next = (header*)idx_to_ptr(free_header->next_free_idx);

        uint64_t prev_idx = ptr_to_idx((char*)free_header);
        uint64_t this_idx = ptr_to_idx((char*) start);
        uint64_t next_idx = free_header->next_free_idx;
        // maintain bi-direction link list
        prev->next_free_idx = this_idx;
        next->prev_free_idx = this_idx;
        start->prev_free_idx = prev_idx;
        start->next_free_idx = next_idx;
    }

    // methods useful for malloc and free, mark a small block as free block
    inline void mark_free_small(header *start, uint64_t level, int64_t tid) {
        start->in_use = 0;
        start->level = level;

        header *free_header;
        // system initialization stage or not
        if (dynamic_load)
            free_header = small_free_list[level_to_index_small(level)];
        else
            free_header = tmp_small_free_list[level_to_index_small(level, tid)];


        // add to free list
        // free_list[level] <--> start <--> free_list[level]->next_free_idx
        header *prev = free_header;
        header *next = (header*)idx_to_ptr(free_header->next_free_idx);

        uint64_t prev_idx = ptr_to_idx((char*)free_header);
        uint64_t this_idx = ptr_to_idx((char*) start);
        uint64_t next_idx = free_header->next_free_idx;
        // maintain bi-direction link list
        prev->next_free_idx = this_idx;
        next->prev_free_idx = this_idx;
        start->prev_free_idx = prev_idx;
        start->next_free_idx = next_idx;
    }

    //mark a free block as used
    inline void mark_used(header *start, uint64_t level) {
        start->in_use = 1;
        start->level = level;

        // remove from free list
        // start->prev_free_idx <--> start <--> start->next_free_idx
        header *prev = (header*)idx_to_ptr(start->prev_free_idx);
        header *next = (header*)idx_to_ptr(start->next_free_idx);


        // maintain bi-direction link list
        prev->next_free_idx = start->next_free_idx;
        next->prev_free_idx = start->prev_free_idx;
    }

    //Since the address of the buddy should just be size away from the block,
    //either add or subtract that depending on which buddy it is. So we XOR
    //the address by the size, same thing as:
    //buddy_idx +/- (1 << level)
    uint64_t get_free_buddy(uint64_t idx, uint64_t level) {
        uint64_t buddy_idx = idx ^ (1LL << level);
        header *buddy_ptr = (header*)idx_to_ptr(buddy_idx);
        //buddy should be at the same level and free
        if ((buddy_ptr->level == level) && (!buddy_ptr->in_use))
            return buddy_idx;
        else
            return UINT64_MAX;;
    }

    // get a free index whose level >= need level, for small malloc
    uint64_t get_free_idx_large(uint64_t need_level, uint64_t& free_level) {
        uint64_t free_idx = UINT64_MAX;
        assert(need_level >= level_dividing_line);
        for (free_level = need_level; free_level <= level_up_bound; free_level++) {
            if (is_empty_large(free_level) && free_level != level_up_bound)
                continue;
            if (is_empty_large(free_level) && free_level == level_up_bound) {
                //no suitable free block, sbrk to get a 1 << level_up_bound size block
                header *new_heap = (header*)large_sbrk();
                if (!new_heap)
                    return free_idx;
                //put into freelist
                mark_free_large(new_heap, free_level);
            }
            free_idx = large_free_list[level_to_index_large(free_level)]->next_free_idx;
            return free_idx;
        }
    }

    // get a free index whose level >= need level, for small malloc
    uint64_t get_free_idx_small(uint64_t need_level, uint64_t& free_level, int64_t tid) {
        uint64_t free_idx = UINT64_MAX;
        assert(need_level < level_dividing_line);
        for (free_level = need_level; free_level <= level_dividing_line; free_level++) {
            if (is_empty_small(free_level, tid) && free_level != level_dividing_line)
                continue;
            if (is_empty_small(free_level, tid) && free_level == level_dividing_line) {
                //no suitable free block, sbrk to get a 1 << level_dividing_line size block
                //from large malloc
                header *new_heap = (header*)small_sbrk();
                if (!new_heap)
                    return free_idx;
                //put into freelist
                mark_free_small(new_heap, free_level, tid);
            }
            if (dynamic_load)
                free_idx = small_free_list[level_to_index_small(free_level)]->next_free_idx;
            else
                free_idx = tmp_small_free_list[level_to_index_small(free_level, tid)]->next_free_idx;
            return free_idx;
        }
    }

    //get block from heap
    char *large_sbrk() {
        char *ret_val;
        uint64_t sbrk_size = 1LL << level_up_bound;
        if (top_of_heap == NULL)
            return NULL;
        ret_val = top_of_heap;
        if ((malloc_size + sbrk_size) > heap_size) {
            // heap size not enough
            logstream(LOG_ERROR) << "out of memory, can not sbrk any more" << LOG_endl;
            ASSERT(false);
        }
        top_of_heap += sbrk_size;
        malloc_size += sbrk_size;
        return ret_val;
    }

    //get block form large malloc
    char *small_sbrk() {
        uint64_t new_heap = large_malloc(level_dividing_line);
        return (char*)idx_to_ptr(new_heap) - size_per_header;
    }

    // return value: an index of starting unit
    uint64_t large_malloc(uint64_t need_level) {
        uint64_t free_level;
        uint64_t free_idx = UINT64_MAX;
        pthread_spin_lock(&malloc_free_lock_large);
        // find the smallest available block
        free_idx = get_free_idx_large(need_level, free_level);
        if (free_idx == UINT64_MAX) {
            // no block big enough
            logstream(LOG_ERROR) << "malloc_buddysystem: memory is full" << LOG_endl;
            print_memory_usage();
            assert(false);
        }
        // split larger block
        for (uint64_t i = free_level - 1; i >= need_level; i--)
            mark_free_large((header*)idx_to_ptr(free_idx + (1LL << i)), i);

        mark_used((header*)idx_to_ptr(free_idx), need_level);

        pthread_spin_unlock(&malloc_free_lock_large);
        return get_value_idx(free_idx);
    }

    uint64_t small_malloc(uint64_t need_level, int64_t tid) {
        uint64_t free_level;
        uint64_t free_idx = UINT64_MAX;
        // find the smallest available block
        if (dynamic_load)
            pthread_spin_lock(&malloc_free_lock_small);

        free_idx = get_free_idx_small(need_level, free_level, tid);
        if (free_idx == UINT64_MAX) {
            // no block big enough
            logstream(LOG_ERROR) << "malloc_buddysystem: memory is full" << LOG_endl;
            print_memory_usage();
            assert(false);
        }
        // split larger block
        for (uint64_t i = free_level - 1; i >= need_level; i--)
            mark_free_small((header*)idx_to_ptr(free_idx + (1LL << i)), i, tid);

        mark_used((header*)idx_to_ptr(free_idx), need_level);

        if (dynamic_load)
            pthread_spin_unlock(&malloc_free_lock_small);

        return get_value_idx(free_idx);

    }

    void large_free(uint64_t free_header_idx) {
        uint64_t buddy_header_idx = 0;
        header *buddy_header_ptr = NULL;

        header *free_header_ptr = (header*) idx_to_ptr(free_header_idx);
        uint64_t cur_level = free_header_ptr->level;
        assert(cur_level >= level_dividing_line);

        pthread_spin_lock(&malloc_free_lock_large);
        //find the buddy and merge it to be larger blocks
        while (cur_level < level_up_bound && ((buddy_header_idx = get_free_buddy(free_header_idx, cur_level)) != UINT64_MAX)) {
            buddy_header_ptr = (header*)idx_to_ptr(buddy_header_idx);
            mark_used(buddy_header_ptr, cur_level);
            if (buddy_header_idx < free_header_idx) {
                free_header_idx = buddy_header_idx;
                free_header_ptr = buddy_header_ptr;
            }
            //we also need to check whether the larger block has a buddy
            cur_level++;
            free_header_ptr->level = cur_level;
        }
        mark_free_large(free_header_ptr, cur_level);

        pthread_spin_unlock(&malloc_free_lock_large);
    }

    void small_free(uint64_t free_header_idx) {
        uint64_t buddy_header_idx = 0;
        header *buddy_header_ptr = NULL;

        header *free_header_ptr = (header*) idx_to_ptr(free_header_idx);
        uint64_t cur_level = free_header_ptr->level;
        pthread_spin_lock(&malloc_free_lock_small);

        //find the buddy and merge it to be larger blocks
        while (cur_level < level_dividing_line && (buddy_header_idx = get_free_buddy(free_header_idx, cur_level)) != UINT64_MAX) {
            buddy_header_ptr = (header*)idx_to_ptr(buddy_header_idx);
            mark_used(buddy_header_ptr, cur_level);
            if (buddy_header_idx < free_header_idx) {
                free_header_idx = buddy_header_idx;
                free_header_ptr = buddy_header_ptr;
            }
            //we should also check whether the larger block has a buddy
            cur_level++;
            free_header_ptr->level = cur_level;
        }
        if (cur_level == level_dividing_line)
            // merge into a 1 << level_dividing_line size block, give it to large_free
            large_free(free_header_idx);
        else
            mark_free_small(free_header_ptr, cur_level, -1);
        pthread_spin_unlock(&malloc_free_lock_small);
    }

public:

    void init(void *start, uint64_t size, uint64_t n) {
        // the smallest memory size to use this memory management system
        ASSERT(size >= 1LL << level_up_bound);

        malloc_size = 0;
        heap_size = size;
        nthread_parallel_load = n;

        memset(usage_counter, 0, sizeof(usage_counter));

        size_per_header = sizeof(header);
        start_ptr = (char*) start;
        top_of_heap = (char*) start;

        pthread_spin_init(&malloc_free_lock_large, 0);
        pthread_spin_init(&malloc_free_lock_small, 0);
        pthread_spin_init(&counter_lock, 0);

        //the freelist for small_malloc, all the threads share one freelist
        for (uint64_t i = level_low_bound; i <= level_dividing_line; i++) {
            uint64_t idx = (i - level_low_bound + 1) * size_per_header;

            small_free_list[level_to_index_small(i)] = (header*) idx_to_ptr(idx);
            small_free_list[level_to_index_small(i)]->prev_free_idx = small_free_list[level_to_index_small(i)]->next_free_idx = idx;
        }

        //the freelist for large_malloc, all the threads share one freelist
        for (uint64_t i = level_dividing_line; i <= level_up_bound; i++) {
            uint64_t idx = (i - level_low_bound + 2) * size_per_header;

            large_free_list[level_to_index_large(i)] = (header*) idx_to_ptr(idx);
            large_free_list[level_to_index_large(i)]->prev_free_idx = large_free_list[level_to_index_large(i)]->next_free_idx = idx;
        }

        // malloc memory for all freelists
        uint64_t small_free_list_idx = large_malloc(level_dividing_line);

        // the freelist for small malloc in system initialization, every thread has one
        tmp_small_free_list = new header*[nthread_parallel_load * (level_dividing_line - level_low_bound + 1)];

        uint64_t idx_cnt = (level_up_bound - level_low_bound + 10) * size_per_header;
        for (uint64_t j = 0; j < nthread_parallel_load ; j++) {
            for (uint64_t i = level_low_bound; i <= level_dividing_line; i++) {
                tmp_small_free_list[level_to_index_small(i, j)] = (header*) idx_to_ptr(idx_cnt);
                tmp_small_free_list[level_to_index_small(i, j)]->prev_free_idx = tmp_small_free_list[level_to_index_small(i, j)]->next_free_idx = idx_cnt;
                idx_cnt += size_per_header;
            }
        }
    }

    // return value: an index of starting unit
    uint64_t malloc(uint64_t size, int64_t tid) {
        uint64_t need_level = size_to_level(size);

        pthread_spin_lock(&counter_lock);
        usage_counter[need_level]++;
        pthread_spin_unlock(&counter_lock);

        if (need_level >= level_dividing_line) {
            return large_malloc(need_level);
        } else {
            return small_malloc(need_level, tid);
        }
    }

    void free(uint64_t free_idx) {
        uint64_t free_header_idx = get_header_idx(free_idx);
        header *free_header_ptr = (header*) idx_to_ptr(free_header_idx);

        pthread_spin_lock(&counter_lock);
        usage_counter[free_header_ptr->level]--;
        pthread_spin_unlock(&counter_lock);

        if (free_header_ptr->level >= level_dividing_line) {
            return large_free(free_header_idx);
        } else {
            return small_free(free_header_idx);
        }
    }

    uint64_t sz_to_blksz (uint64_t size) {
        return ((1 << size_to_level(size)) - size_per_header);
    }

    //merge all threads' small freelists into one
    void merge_freelists() {
        for (int i = level_low_bound; i <= level_dividing_line; i++) {
            header *this_ptr = small_free_list[level_to_index_small(i)];

            uint64_t this_idx = ptr_to_idx((char*)this_ptr);
            for (int j = 0; j < nthread_parallel_load; j++) {
                if (is_empty_small(i, j))
                    continue;
                //link the end of this thread's freelist with the start of next thread's free list
                uint64_t next_idx = tmp_small_free_list[level_to_index_small(i, j)]->next_free_idx;
                header *next_ptr = (header*)idx_to_ptr(next_idx);
                this_ptr->next_free_idx = next_idx;
                next_ptr->prev_free_idx = this_idx;
                this_ptr = next_ptr;
                //get to the end of free list
                while (this_ptr->next_free_idx != ptr_to_idx((char*)tmp_small_free_list[level_to_index_small(i, j)]))
                    this_ptr = (header*)idx_to_ptr(this_ptr->next_free_idx);
                //anyway, the end's next_free_idx should be free_list[i]
                this_ptr->next_free_idx = ptr_to_idx((char*) small_free_list[level_to_index_small(i)]);
                this_idx = ptr_to_idx((char*)this_ptr);
            }
        }
        //free the array tmp_small_free_list
        delete[]tmp_small_free_list;
        dynamic_load = true;
    }

    void print_memory_usage() {
        logstream(LOG_INFO) << "graph_storage edge memory status:" << LOG_endl;
        uint64_t size_count = 0;

        for (int i = level_low_bound; i <= level_up_bound; i++) {
            logstream(LOG_INFO) << "level" << setw(2) << i << ": " << setw(10) << usage_counter[i] << "|\t";
            if ((i - level_low_bound + 1) % 4 == 0) logstream(LOG_INFO) << LOG_endl;
            size_count += (1LL << i) * usage_counter[i];
        }

        logstream(LOG_INFO) << "Size count: " << size_count << LOG_endl << LOG_endl;
    }
};
