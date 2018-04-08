#pragma once

#include <iostream>
#include <iomanip>
#include <config.hpp>

// NOTICE: any implentation of this interface should be *tread-safe*
class Malloc_Interface {
public:
    //init the memory area which start from start and have size bytes
    virtual void init(void *start, uint64_t size, uint64_t n) = 0;

    // return value: (the ptr which can write value - start)
    virtual uint64_t malloc(uint64_t size, int64_t tid = -1) = 0;

    //the idx is exact the value return by alloc
    virtual void free(uint64_t idx) = 0;

    //merge the tmp freelists used to multithread insert_normal to the freelist
    virtual void merge_freelists() = 0;

    virtual void print_memory_usage() = 0;

    virtual uint64_t sz_to_blksz(uint64_t sz) = 0;
};

class Buddy_Malloc : public Malloc_Interface {
private:
    // block size >= 2^level_low_bound units
    static const uint64_t level_low_bound = 4;
    // block size <= 2^level_up_bound units
    static const uint64_t level_up_bound = 26;

    static const uint64_t sbrk_size = 1LL << level_up_bound;

    char *start_ptr;

    char *top_of_heap;

    // lock for thread-safe
    pthread_spinlock_t sbrk_lock;

    pthread_spinlock_t malloc_free_lock;

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

    uint64_t size_per_header;
    //total size of the memory
    uint64_t heap_size;
    //total size has been sbrked
    uint64_t malloc_size;
    //idx of tmp_free_list
    uint64_t tmp_free_list_idx;

    // free_list[i]->next points to the first free block with size 2^i bytes
    header *free_list[level_up_bound + 1];
    // freelists for multithread insert_normal, every thread has a freelist
    header **tmp_free_list;
    //for print_memory_usage()
    uint64_t usage_counter[level_up_bound + 1];

    inline void *idx_to_ptr(uint64_t idx) {
        return start_ptr + idx;
    }

    inline uint64_t ptr_to_idx(char *ptr) {
        return (uint64_t)((char*)ptr - start_ptr);
    }

    inline uint64_t get_value_idx(uint64_t header_idx) {
        return header_idx + size_per_header;
    }

    inline uint64_t get_header_idx(uint64_t value_idx) {
        return value_idx - size_per_header;
    }

    // return whether there exit free block for need level or not
    inline bool is_empty(uint64_t level, int64_t tid) {
        if (tid < 0)
            return free_list[level]->next_free_idx == ptr_to_idx((char*) free_list[level]);
        else
            return tmp_free_list[level + tid * (level_up_bound + 1)]->next_free_idx == ptr_to_idx((char*) tmp_free_list[level + tid * (level_up_bound + 1)]);
    }

    inline uint64_t block_size(uint64_t idx) {
        header *h = (header*)idx_to_ptr(get_header_idx(idx));
        return (1 << (h->level)) - size_per_header;
    }

    inline void copy(uint64_t dst_idx, uint64_t src_idx, uint64_t size) {
        memcpy(idx_to_ptr(dst_idx), idx_to_ptr(src_idx), size);
    }

    // convert a size into level. useful for malloc
    inline uint64_t truncate_level(uint64_t level) {
        if (level < level_low_bound)
            return level_low_bound;
        if (level > level_up_bound) { // should be less than level_up_bound
            logstream(LOG_ERROR) << "need level: " << level << " level_up_bound: " << level_up_bound << LOG_endl;
            ASSERT(false);
        }
        return level;
    }

    inline uint64_t size_to_level(uint64_t size) {
        uint64_t level = 0, tmp = 1;
        size += size_per_header;
        while (true) {
            tmp <<= 1;
            level++;
            if (tmp >= size) break;
        }
        return truncate_level(level);
    }

    // methods useful for malloc and free
    inline void mark_free(header *start, uint64_t level , int64_t tid) {
        start->in_use = 0;
        start->level = level;
        header *free_header;
        if (tid < 0)
            free_header = free_list[level];
        else
            free_header = tmp_free_list[level + tid * (level_up_bound + 1)];

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
            return 0;
    }

    int64_t get_free_idx(uint64_t need_level, uint64_t& free_level, int64_t tid) {
        int64_t free_idx = -1;
        for (free_level = need_level; free_level <= level_up_bound; free_level++) {
            if (is_empty(free_level, tid) && free_level != level_up_bound)
                continue;
            if (is_empty(free_level, tid) && free_level == level_up_bound) {
                header *new_heap = (header*)sbrk(sbrk_size);
                if (!new_heap)
                    return free_idx;
                mark_free(new_heap, free_level, tid);
            }
            if (tid < 0)
                free_idx = free_list[free_level]->next_free_idx;
            else
                free_idx = tmp_free_list[free_level + (level_up_bound + 1) * tid]->next_free_idx;
            return free_idx;
        }
    }

    char *sbrk(uint64_t n) {
        char *ret_val;
        if (top_of_heap == NULL)
            return NULL;
        pthread_spin_lock(&sbrk_lock);
        ret_val = top_of_heap;
        if ((malloc_size + n) > heap_size) {
            logstream(LOG_ERROR) << "out of memory, can not sbrk any more" << LOG_endl;
            ASSERT(false);
        }
        top_of_heap += n;
        malloc_size += n;
        pthread_spin_unlock(&sbrk_lock);
        return ret_val;
    }

public:

    void init(void *start, uint64_t size, uint64_t n) {
        //ASSERT(level_low_bound >= 4);
        //ASSERT(level_up_bound <= 26);
        ASSERT(size / (n + 1) >= 1LL << level_up_bound);

        malloc_size = 0;
        heap_size = size;
        nthread_parallel_load = n;
        memset(usage_counter, 0, sizeof(usage_counter));
        // malloc enough space for header
        size_per_header = sizeof(header);
        start_ptr = (char*) start;
        top_of_heap = (char*) start;
        pthread_spin_init(&sbrk_lock, 0);
        pthread_spin_init(&malloc_free_lock, 0);

        for (int i = level_low_bound; i <= level_up_bound; i++) {
            uint64_t idx = (i - level_low_bound + 1) * size_per_header;

            free_list[i] = (header*) idx_to_ptr(idx);
            free_list[i]->prev_free_idx = free_list[i]->next_free_idx = idx;
        }
        // malloc memory for freelist
        uint64_t free_list_idx = malloc((level_up_bound - level_low_bound + 1) * size_per_header);

        // new the tmp_freelists for every thread and malloc memory for them
        tmp_free_list = new header*[nthread_parallel_load * (level_up_bound + 2)];
        tmp_free_list_idx = malloc ((level_up_bound - level_low_bound + 1) * size_per_header * nthread_parallel_load);

        uint64_t idx_cnt = tmp_free_list_idx;
        for (int i = level_low_bound; i <= level_up_bound; i++) {
            for (int j = 0; j < nthread_parallel_load ; j++) {
                tmp_free_list[i + (level_up_bound + 1) * j] = (header*) idx_to_ptr(idx_cnt);
                tmp_free_list[i + (level_up_bound + 1) * j]->prev_free_idx = tmp_free_list[i + (level_up_bound + 1) * j]->next_free_idx = idx_cnt;
                idx_cnt += size_per_header;
            }
        }
    }

    uint64_t sz_to_blksz (uint64_t size) {
        return ((1 << size_to_level(size)) - size_per_header);
    }
    // return value: an index of starting unit
    uint64_t malloc(uint64_t size, int64_t tid = -1) {
        //pthread_spin_lock(&debug_lock);
        uint64_t need_level = size_to_level(size);
        uint64_t free_level;
        int64_t free_idx = -1;
        if (tid < 0)
            pthread_spin_lock(&malloc_free_lock);
        // find the smallest available block
        free_idx = get_free_idx(need_level, free_level, tid);

        if (free_idx == -1) {
            // no block big enough
            logstream(LOG_ERROR) << "malloc_buddysystem: memory is full" << LOG_endl;
            print_memory_usage();
            ASSERT(false);
            //return -1;
        }
        // split larger block
        for (uint64_t i = free_level - 1; i >= need_level; i--)
            mark_free((header*)idx_to_ptr(free_idx + (1LL << i)), i, tid);

        mark_used((header*)idx_to_ptr(free_idx), need_level);
        if (tid < 0)
            pthread_spin_unlock(&malloc_free_lock);

        usage_counter[need_level]++;
        return get_value_idx(free_idx);
    }

    void free(uint64_t free_idx) {
        uint64_t free_header_idx = get_header_idx(free_idx);
        header *free_header_ptr = (header*) idx_to_ptr(free_header_idx);
        uint64_t cur_level = free_header_ptr->level;
        uint64_t buddy_header_idx = 0;
        header *buddy_header_ptr = NULL;
        pthread_spin_lock(&malloc_free_lock);
        //find the buddy and merge it to be larger blocks
        while (cur_level < level_up_bound && (buddy_header_idx = get_free_buddy(free_header_idx, cur_level)) > 0) {
            buddy_header_ptr = (header*)idx_to_ptr(buddy_header_idx);
            mark_used(buddy_header_ptr, cur_level);
            if (buddy_header_idx < free_header_idx) {
                free_header_idx = buddy_header_idx;
                free_header_ptr = buddy_header_ptr;
            }
            cur_level++; //we should also check whether the larger block has a buddy
        }
        mark_free(free_header_ptr, cur_level, -1);
        pthread_spin_unlock(&malloc_free_lock);
    }

    //merge all threads' freelists into one
    void merge_freelists() {
        for (int i = level_low_bound; i <= level_up_bound; i++) {
            header *this_ptr = free_list[i];
            //get to the end of free list
            while (this_ptr->next_free_idx != ptr_to_idx((char*) free_list[i]))
                this_ptr = (header*)idx_to_ptr(this_ptr->next_free_idx);
            uint64_t this_idx = ptr_to_idx((char*)this_ptr);
            for (int j = 0; j < nthread_parallel_load; j++) {
                if (is_empty(i, j))
                    continue;
                //merge the end of this freelist with the start of next free list
                uint64_t next_idx = tmp_free_list[i + (level_up_bound + 1) * j]->next_free_idx;
                header *next_ptr = (header*)idx_to_ptr(next_idx);
                this_ptr->next_free_idx = next_idx;
                next_ptr->prev_free_idx = this_idx;
                this_ptr = next_ptr;
                //get to the end of free list
                while (this_ptr->next_free_idx != ptr_to_idx((char*)tmp_free_list[i + (level_up_bound + 1) * j]))
                    this_ptr = (header*)idx_to_ptr(this_ptr->next_free_idx);
                //anyway, the end's next_free_idx should be free_list[i]
                this_ptr->next_free_idx = ptr_to_idx((char*) free_list[i]);
                this_idx = ptr_to_idx((char*)this_ptr);
            }
        }
        //free all that concerned with tmp_free_list
        free(tmp_free_list_idx);
        delete[]tmp_free_list;
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




