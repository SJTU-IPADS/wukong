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

#ifdef USE_JEMALLOC
#include <jemalloc/jemalloc.h>
/*
 * To use jemalloc API distinctly from glibc API, 
 * jemalloc API has prefix(je) configured in deps/deps.sh.
 * e.g. API mallctl is named as jemallctl.
 * */

#include "mm/malloc_interface.hpp"

class JeMalloc : public MAInterface {
private:
    vector<unsigned> arena_inds;
    vector<unsigned> tcache_inds;
    uint64_t memsize;
    int pfactor;

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

    // hook functions which manage extent lifetime
    static void *extent_alloc_hook(extent_hooks_t *extent_hooks, void *new_addr, size_t size,
                                   size_t alignment, bool *zero, bool *commit, unsigned arena_ind) {
        logstream(LOG_DEBUG) << "(extent_hooks = " << extent_hooks
                             << " , new_addr = " << new_addr << " , size = " << size
                             << " , alignment = " << alignment
                             << " , *zero = " << *zero << " , *commit = " << *commit
                             << " , arena_ind = " << arena_ind << ")" << LOG_endl;

        pthread_spin_lock(&jelock);
        char *ret = top_of_heap;

        // align the return address
        if ((uintptr_t)ret % alignment != 0)
            ret += (alignment - (uintptr_t)ret % alignment);

        if (ret + size >= end_ptr) {
            ASSERT_MSG(false, "Out of memory, cannot allocate any extent.");
            pthread_spin_unlock(&jelock);
            return NULL;
        }

        top_of_heap = ret + size;
        pthread_spin_unlock(&jelock);

        if (*zero) // extent should be zeroed
            memset(ret, 0, size);

        if ((uintptr_t)ret % alignment != 0)
            logstream(LOG_ERROR) << "Alignment error." << LOG_endl;

        return ret;
    }

    static bool extent_dalloc_hook(extent_hooks_t *extent_hooks, void *addr, size_t size,
                                   bool committed, unsigned arena_ind) {
        return true; // opt out
    }

    static void extent_destroy_hook(extent_hooks_t *extent_hooks, void *addr, size_t size,
                                    bool committed, unsigned arena_ind) {
        return;
    }

    static bool extent_commit_hook(extent_hooks_t *extent_hooks, void *addr, size_t size,
                                   size_t offset, size_t length, unsigned arena_ind) {
        return false; // commit should always succeed
    }

    static bool extent_decommit_hook(extent_hooks_t *extent_hooks, void *addr, size_t size,
                                     size_t offset, size_t length, unsigned arena_ind) {
        return false; // decommit should always succeed
    }

    static bool extent_purge_lazy_hook(extent_hooks_t *extent_hooks, void *addr, size_t size,
                                       size_t offset, size_t length, unsigned arena_ind) {
        return true; // opt out
    }

    static bool extent_purge_forced_hook(extent_hooks_t *extent_hooks, void *addr, size_t size,
                                         size_t offset, size_t length, unsigned arena_ind) {
        return true; // opt out
    }

    static bool extent_split_hook(extent_hooks_t *extent_hooks, void *addr, size_t size,
                                  size_t size_a, size_t size_b, bool committed, unsigned arena_ind) {
        return false; // split should always succeed
    }

    static bool extent_merge_hook(extent_hooks_t *extent_hooks, void *addr_a, size_t size_a,
                                  void *addr_b, size_t size_b, bool committed, unsigned arena_ind) {
        return false; // merge should always succeed
    }

    void init(void *start, uint64_t size, uint64_t p) {
        start_ptr = top_of_heap = (char *)start;
        memsize = size;
        end_ptr = start_ptr + memsize;
        pfactor = p;

        pthread_spin_init(&jelock, 0);

        size_t hooks_len = sizeof(extent_hooks_t *);
        size_t sz = sizeof(unsigned);
        arena_inds.resize(pfactor);
        tcache_inds.resize(pfactor);
        new_hooks = &hooks;

        // create new arena for each engine and install custom extent hooks on it
        for (int i = 0; i < pfactor; i++)
            jemallctl("arenas.create", (void *)&arena_inds[i], &sz,
                    (void *)&new_hooks, sizeof(extent_hooks_t *));

        // create thread-specific cache for each engine
        for (int i = 0; i < pfactor; i++)
            jemallctl("tcache.create", (void *)&tcache_inds[i], &sz, NULL, 0);
    }

    uint64_t malloc(uint64_t size, int64_t tid) {
        ASSERT_MSG((tid >= 0) && (tid < pfactor), "Exceed the parallel factor.");

        // malloc from engine's own arena and tcache
        void *ptr = jemallocx(size, MALLOCX_ARENA(arena_inds[tid]) | MALLOCX_TCACHE(tcache_inds[tid]));
        if ((char *)ptr < start_ptr || (char *)ptr + size >= end_ptr) {
            logstream(LOG_ERROR) << "Out of memory range" << LOG_endl;
            ASSERT(false);
        }

        return (uint64_t)((char *)ptr - start_ptr); // offset
    }

    void free(uint64_t offset) {
        ASSERT_MSG(start_ptr + offset < end_ptr, "Out of memory range");
        jedallocx((void *)(start_ptr + offset), 0);
        return;
    }

    uint64_t sz_to_blksz(uint64_t size) { return (uint64_t)jenallocx(size, 0); }

    //to be suited with buddy malloc
    void merge_freelists() { return; }

    void print_memory_usage() {
        uint64_t size = ((uintptr_t)top_of_heap - (uintptr_t)start_ptr) / (1024 * 1024);

        logstream(LOG_INFO) << "graph_storage edge memory status:" << LOG_endl;
        logstream(LOG_INFO) << "allocated " << size << " MB" << LOG_endl;
        return;
    }
};

// needed to provide a definition for static member variables
char *JeMalloc::start_ptr = NULL;
char *JeMalloc::end_ptr = NULL;
char *JeMalloc::top_of_heap = NULL;
pthread_spinlock_t JeMalloc::jelock;

#endif
