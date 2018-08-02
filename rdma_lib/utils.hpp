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

#ifndef RDMA_IO_UTIL
#define RDMA_IO_UTIL

#include <stdio.h>
#include <byteswap.h>
#include <string.h>
#include <errno.h>

// time utilites
#include <chrono>
#include <ctime>

#include <iostream>


#define _VERBOSE 1
#define DEBUG(t,a)  if(_VERBOSE != 0)           \
        if(_VERBOSE == 1 && t == 0){a;}         \
        else if (_VERBOSE == 2){a;}


#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define forceinline inline __attribute__((always_inline))
#define _unused(x) ((void)(x))  /* Make production build happy */


#if __BYTE_ORDER == __LITTLE_ENDIAN
inline uint64_t htonll (uint64_t x) {
    return bswap_64 (x);
}

inline uint64_t ntohll (uint64_t x) {
    return bswap_64 (x);
}
#elif __BYTE_ORDER == __BIG_ENDIAN

inline uint64_t htonll (uint64_t x) {
    return x;
}

inline uint64_t ntohll (uint64_t x) {
    return x;
}
#else
#error __BYTE_ORDER is neither __LITTLE_ENDIAN nor __BIG_ENDIAN
#endif


//Conditional Exit
#define CE(cc, error_msg) if(cc){fprintf(stderr,"Get error msg %s,%s\n",error_msg,strerror(errno));assert(false);}
#define CE_1(cc, error_msg, error_arg) if(cc){fprintf(stderr,error_msg,error_arg);exit(-1);}
#define CE_2(cc, error_msg, error_arg1, error_arg2) if(cc){fprintf(stderr,error_msg,error_arg1,error_arg2);exit(-1);}
#define CE_3(cc, error_msg, error_arg1, error_arg2, error_arg3) if(cc){fprintf(stderr,error_msg,error_arg1,error_arg2,error_arg3);exit(-1);}



#define MOD_ADD(n, N) do{n = (n+1) % N;}while(0)//compare with HRD_MOD_ADD! which is better?


inline uint64_t
rdtsc(void)
{
    uint32_t hi, lo;
    __asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)lo) | (((uint64_t)hi) << 32);
}

namespace rdmaio {
namespace util {

class Timer {
    std::clock_t start_;
    std::clock_t end_;
public:
    Timer() {
        start_ = std::clock();
    }

    void end() { end_ = std::clock();}

    void reset() { start_ = std::clock(); end_ = start_; }

    double elapsed_sec() {
        return ( (double) (end_ - start_) / CLOCKS_PER_SEC);
    }
};
}

inline uint64_t ip_checksum(void* vdata, size_t length)
{
    // Cast the data pointer to one that can be indexed.
    char* data = (char*)vdata;

    // Initialise the accumulator.
    uint64_t acc = 0xffff;

    // Handle any partial block at the start of the data.
    unsigned int offset = ((uintptr_t)data) & 3;
    if (offset) {
        size_t count = 4 - offset;
        if (count > length) count = length;
        uint32_t word = 0;
        memcpy(offset + (char*)&word, data, count);
        acc += ntohl(word);
        data += count;
        length -= count;
    }

    // Handle any complete 32-bit blocks.
    char* data_end = data + (length & ~3);
    while (data != data_end) {
        uint32_t word;
        memcpy(&word, data, 4);
        acc += ntohl(word);
        data += 4;
    }
    length &= 3;

    // Handle any partial block at the end of the data.
    if (length) {
        uint32_t word = 0;
        memcpy(&word, data, length);
        acc += ntohl(word);
    }

    // Handle deferred carries.
    acc = (acc & 0xffffffff) + (acc >> 32);
    while (acc >> 16) {
        acc = (acc & 0xffff) + (acc >> 16);
    }

    // If the data began at an odd byte address
    // then reverse the byte order to compensate.
    if (offset & 1) {
        acc = ((acc & 0xff00) >> 8) | ((acc & 0x00ff) << 8);
    }

    // Return the checksum in network byte order.
    return htons(~acc);
}

}; // namespace rdmaio

#endif
