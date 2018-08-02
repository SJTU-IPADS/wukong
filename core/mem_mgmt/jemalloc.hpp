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
 
#ifdef USE_JEMALLOC
#pragma once

#include "mem_mgmt/malloc_interface.hpp"
#include <jemalloc/jemalloc.h>

class JeMalloc :public Malloc_Interface{
private:

    vector<unsigned> arena_inds;
    vector<unsigned> tcache_inds;
    uint64_t memsize;
    extent_hooks_t *new_hooks;
public:
    static char *start_ptr;
    static char *end_ptr;
    static char *top_of_heap;
    static pthread_spinlock_t jelock;

    // custom extent hooks which comprises function pointers
    extent_hooks_t hooks = {
        JeMalloc::extent_alloc_hook,
        JeMalloc::extent_dalloc_hook,
        JeMalloc::extent_destroy_hook,
        JeMalloc::extent_commit_hook,
        JeMalloc::extent_decommit_hook,
        JeMalloc::extent_purge_lazy_hook,
        JeMalloc::extent_purge_forced_hook,
        JeMalloc::extent_split_hook,
        JeMalloc::extent_merge_hook
    };

    //hook functions which manage extent lifetime
    static void *extent_alloc_hook(extent_hooks_t *extent_hooks, void *new_addr, size_t size, size_t alignment, bool *zero, bool *commit, unsigned arena_ind){
        //logstream(LOG_INFO) << __func__ << "(extent_hooks = " << extent_hooks << " , new_addr = " << new_addr << ", size = " << size <<  " , alignment = " << alignment << " , *zero = " << *zero << " , *commit = " << *commit << " , arena_ind = " << arena_ind << ")" << LOG_endl;

        pthread_spin_lock(&jelock);
        char *ret = (char*)top_of_heap;
        //align the return address
        if((uintptr_t)ret % alignment != 0) {
            ret = ret + (alignment- (uintptr_t)ret % alignment);
        }
        if ((char*)ret + size >= (char*)end_ptr) {
            logstream(LOG_ERROR) << "Out of memory, can not allocate any extent." << LOG_endl;
            ASSERT(false);
            pthread_spin_unlock(&jelock);
            return NULL;
        }
        top_of_heap = ret + size;
        pthread_spin_unlock(&jelock);
        if (*zero) //extent should be zeroed
            memset(ret, size, 0);

        if((uintptr_t)ret % alignment != 0) {
            logstream(LOG_ERROR) << "Alignment error." << LOG_endl;
        }
        return ret;
    }

    static bool extent_dalloc_hook(extent_hooks_t *extent_hooks, void *addr, size_t size, bool committed, unsigned arena_ind) {
        //logstream(LOG_INFO) << __func__ << "(extent_hooks = " << extent_hooks << " , addr = " << addr << ", size = " << size <<  " , committed = " << committed << " , arena_ind = " << arena_ind << ")" << LOG_endl;

        // opt out from dalloc
        return true;
    }

    static void extent_destroy_hook(extent_hooks_t *extent_hooks, void *addr, size_t size, bool committed, unsigned arena_ind) {
        //logstream(LOG_INFO) << __func__ << "(extent_hooks = " << extent_hooks << " , addr = " << addr << ", size = " << size <<  " , committed = " << committed << " , arena_ind = " << arena_ind << ")" << LOG_endl;

        return;
    }

    static bool extent_commit_hook(extent_hooks_t *extent_hooks, void *addr, size_t size, size_t offset, size_t length, unsigned arena_ind) {
        //logstream(LOG_INFO) << __func__ << "(extent_hooks = " << extent_hooks << " , addr = " << addr << ", size = " << size <<  " , offset = " << offset << " , length = " << length << " , arena_ind = " << arena_ind << ")" << LOG_endl;

        // commit should always succeed
        return false;
    }

    static bool extent_decommit_hook(extent_hooks_t *extent_hooks, void *addr, size_t size, size_t offset, size_t length, unsigned arena_ind) {
        //logstream(LOG_INFO) << __func__ << "(extent_hooks = " << extent_hooks << " , addr = " << addr << ", size = " << size <<  " , offset = " << offset << " , length = " << length << " , arena_ind = " << arena_ind << ")" << LOG_endl;

        // decommit should always succeed
        return false;
    }

    static bool extent_purge_lazy_hook(extent_hooks_t *extent_hooks, void *addr, size_t size, size_t offset, size_t length, unsigned arena_ind) {
        //logstream(LOG_INFO) << __func__ << "(extent_hooks = " << extent_hooks << " , addr = " << addr << ", size = " << size <<  " , offset = " << offset << " , length = " << length << " , arena_ind = " << arena_ind << ")" << LOG_endl;

        // opt out
        return true;
    }

    static bool extent_purge_forced_hook(extent_hooks_t *extent_hooks, void *addr, size_t size, size_t offset, size_t length, unsigned arena_ind) {
        //logstream(LOG_INFO) << __func__ << "(extent_hooks = " << extent_hooks << " , addr = " << addr << ", size = " << size <<  " , offset = " << offset << " , length = " << length << " , arena_ind = " << arena_ind << ")" << LOG_endl;

        // opt out
        return true;
    }

    static bool extent_split_hook(extent_hooks_t *extent_hooks, void *addr, size_t size, size_t size_a, size_t size_b, bool committed, unsigned arena_ind) {
        //logstream(LOG_INFO) << __func__ << "(extent_hooks = " << extent_hooks << " , addr = " << addr << ", size = " << size <<  " , size_a = " << size_a << " , size_b = " << size_b << " , committed = " << committed << " , arena_ind = " << arena_ind << ")" << LOG_endl;

        // split should always succeed
        return false;
    }

    static bool extent_merge_hook(extent_hooks_t *extent_hooks, void *addr_a, size_t size_a, void *addr_b, size_t size_b, bool committed, unsigned arena_ind) {
        //logstream(LOG_INFO) << __func__ << "(extent_hooks = " << extent_hooks << " , addr_a = " << addr_a << ", size_a = " << size_a <<  " , addr_b = " << addr_b << " , size_b = " << size_b << " , committed = " << committed << " , arena_ind = " << arena_ind << ")" << LOG_endl;

        // merge should always succeed
        return false;
    }

    void init(void *start, uint64_t size, uint64_t n) {
        start_ptr = top_of_heap = (char*)start;
        memsize = size;
        end_ptr = start_ptr + memsize;
        pthread_spin_init(&jelock, 0);
        size_t hooks_len = sizeof(extent_hooks_t *);
        size_t  sz = sizeof(unsigned);
        arena_inds.resize(n);
        tcache_inds.resize(n);
        new_hooks = &hooks;
        //create new arena for each engine and install custom extent hooks on it
        for(int i = 0; i < n; i++) {
            mallctl("arenas.create", (void *)&arena_inds[i], &sz, (void *)&new_hooks, sizeof(extent_hooks_t *));
        }
        //create thread-specific cache for each engine
        for (int i = 0; i < n; i++) {
            mallctl("tcache.create", (void *)&tcache_inds[i], &sz, NULL,0);
        }
    }

    uint64_t malloc(uint64_t size, int64_t tid) {
        // malloc from engine's own arena and tcache
        void *ptr = mallocx(size, MALLOCX_ARENA(arena_inds[tid])| MALLOCX_TCACHE(tcache_inds[tid]));
        if ((char*)ptr < start_ptr || (char*)ptr + size > end_ptr) {
            logstream(LOG_ERROR) << "memory range false" << LOG_endl;
            ASSERT(false);
        }
        uint64_t ret = (uint64_t)((char*)ptr - start_ptr);
        return ret;
    }

    void free(uint64_t idx) {
        void* ptr = (void*)(start_ptr + idx);
        //make the memory be available for future allocations
        dallocx(ptr, 0);
        return;
    }

    size_t block_size(uint64_t idx) {
        void* ptr = (void*)(start_ptr + idx);
        // get the real size of the allocation at ptr.
        return sallocx(ptr, 0);
    }

    //to be suited with buddy malloc
    void merge_freelists() {
        return;
    }

    void print_memory_usage() {
        uint64_t allocated_size = ((char*) top_of_heap - (char*)start_ptr) / (1024 * 1024);
        logstream(LOG_INFO) << "graph_storage edge memory status:" << LOG_endl;
        logstream(LOG_INFO) << "allocated " << allocated_size << " MB" << LOG_endl;
        return;
    }
};

//It is needed to provide a definition
//for static member variables
char* JeMalloc::start_ptr = NULL;
char* JeMalloc::end_ptr = NULL;
char* JeMalloc::top_of_heap = NULL;
pthread_spinlock_t JeMalloc::jelock;

#endif
