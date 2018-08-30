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

#include <iostream>

// NOTE: any implentation of this interface should be *thread-safe*
class MAInterface {
public:

    /**
     * init the memory management system with the memory region which starts
     * from address 'start' and contains 'size' bytes, n is the number of
     * threads which would use the memory management system concurrently
     * Note that all the memory blocks it manages should in that memory region.
     */
    virtual void init(void *start, uint64_t size, uint64_t n) = 0;

    /**
     * malloc a 'size' bytes memory block from the memory management system,and
     * returns the offset to the 'start' which indicate the base address of the
     * allocation. tid is the specific id of current thread which calls this interface.
     * Note that the tid should range from 0 to n-1.
     */
    virtual uint64_t malloc(uint64_t size, int64_t tid) = 0;

    /**
     * free will cause the memory referenced by 'idx' (the offset return by malloc)
     * to be available for future allocations
     */
    virtual void free(uint64_t idx) = 0;

    /**
     * this interface is only used by buddy_malloc, you can just return while
     * implementing other memory mangement systems
     */
    virtual void merge_freelists() = 0;

    /**
     * used for dynamic cache, returns the real size of the allocation
     * that would result from the equivalent malloc(size) function call
     */
    virtual uint64_t sz_to_blksz(uint64_t size) = 0;

    /**
     * print out current memory usage status
     */
    virtual void print_memory_usage() = 0;
};
