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
#include <sstream>
#include <string>

// definitions of "__host__" and "__device__"
#ifdef USE_GPU
#include <cuda_runtime.h>
#endif

// utils
#include "utils/math.hpp"

namespace wukong {

// IN/OUT
enum { NBITS_DIR = 1 };
// equal to the size of t/pid
enum { NBITS_IDX = 17 };
// 0: index vertex, ID: normal vertex
enum { NBITS_VID = (64 - NBITS_IDX - NBITS_DIR) };

// reserve two special index IDs (predicate and type)
enum { PREDICATE_ID = 0,
       TYPE_ID = 1 };

static inline bool is_tpid(ssid_t id) { return (id > 1) && (id < (1 << NBITS_IDX)); }

static inline bool is_vid(ssid_t id) { return id >= (1 << NBITS_IDX); }

#define PARTITION(vid) wukong::math::hash_mod(vid, Global::num_servers)

/**
 * predicate-base key/value store
 * key: vid | t/pid | direction
 * value: v/t/pid list
 */
struct ikey_t {
    uint64_t dir : NBITS_DIR;  // direction
    uint64_t pid : NBITS_IDX;  // predicate
    uint64_t vid : NBITS_VID;  // vertex

    // clang-format off
#ifdef USE_GPU
    __host__ __device__
#endif
    ikey_t() : vid(0), pid(0), dir(0) {}

#ifdef USE_GPU
    __host__ __device__
#endif
    ikey_t(uint64_t v, uint64_t p, uint64_t d) : vid(v), pid(p), dir(d) {
        assert((vid == v) && (dir == d) && (pid == p));  // no key truncate
    }

#ifdef USE_GPU
    __host__ __device__
#endif
    bool operator==(const ikey_t& key) const {
        if ((vid == key.vid) && (pid == key.pid) && (dir == key.dir))
            return true;
        return false;
    }
    // clang-format on

    bool operator!=(const ikey_t& key) const { return !(operator==(key)); }

    bool is_empty() { return ((vid == 0) && (pid == 0) && (dir == 0)); }

    void print_key() { std::cout << "[" << vid << "|" << pid << "|" << dir << "]" << std::endl; }

    std::string to_string() {
        std::ostringstream ss;
        ss << "[" << vid << "|" << pid << "|" << dir << "]";
        return ss.str();
    }

    uint64_t hash() const {
        uint64_t r = 0;
        r += vid;
        r <<= NBITS_IDX;
        r += pid;
        r <<= NBITS_DIR;
        r += dir;
        // the standard hash is too slow
        // (i.e., std::hash<uint64_t>()(r))
        return wukong::math::hash_u64(r);
    }
};

struct ikey_Hasher {
    static size_t hash(const ikey_t& k) {
        return k.hash();
    }

    static bool equal(const ikey_t& x, const ikey_t& y) {
        return x.operator==(y);
    }
};

// 64-bit internal pointer
//   NBITS_SIZE: the max number of edges (edge_t) for a single vertex (256M)
//   NBITS_PTR: the max number of edges (edge_t) for the entire gstore (16GB)
//   NBITS_TYPE: the type of edge, used for attribute triple, sid(0), int(1), float(2), double(4)
enum { NBITS_SIZE = 28 };
enum { NBITS_PTR = 34 };
enum { NBITS_TYPE = 2 };

struct iptr_t {
    uint64_t size : NBITS_SIZE;
    uint64_t off : NBITS_PTR;
    uint64_t type : NBITS_TYPE;

    // clang-format off
#ifdef USE_GPU
    __host__ __device__
#endif
    iptr_t() : size(0), off(0), type(0) {}

    // the default type is sid(type = 0)
#ifdef USE_GPU
    __host__ __device__
#endif
    iptr_t(uint64_t s, uint64_t o, uint64_t t = 0) : size(s), off(o), type(t) {
        // no truncated
        assert((size == s) && (off == o) && (type == t));
    }

#ifdef USE_GPU
    __host__ __device__
#endif
    bool operator==(const iptr_t& ptr) {
        if ((size == ptr.size) && (off == ptr.off) && (type == ptr.type))
            return true;
        return false;
    }

#ifdef USE_GPU
    __host__ __device__
#endif
    bool operator!=(const iptr_t& ptr) {
        return !(operator==(ptr));
    }
    // clang-format on
};

// 128-bit vertex (key)
struct vertex_t {
    ikey_t key;  // 64-bit: vertex | predicate | direction
    iptr_t ptr;  // 64-bit: size | offset
};

// 32-bit edge (value)
struct edge_t {
    int val;  // vertex ID

#ifdef TRDF_MODE
    int64_t ts;  // start timestamp
    int64_t te;  // end timestamp

    bool valid(int64_t _ts, int64_t _te) const {
        bool rev = ((_ts <= _te) && (ts <= te) && (_ts <= te && _te >= ts)) || (_ts == 0L && _te == 0L);
        return rev;
    }

    // clang-format off
#if USE_GPU
    __host__ __device__
#endif
    edge_t(sid_t id, int64_t ts, int64_t te): val(id), ts(ts), te(te) {}
    // clang-format on
#endif

    edge_t() {}

    // clang-format off
#ifdef USE_GPU
    __host__ __device__
#endif
    explicit edge_t(sid_t id) : val(id) {}
    // clang-format on

    edge_t(const edge_t& edge) : val(edge.val) {
    #ifdef TRDF_MODE
        ts = edge.ts;
        te = edge.te;
    #endif
    }

    edge_t(edge_t&& edge) : val(edge.val) {
    #ifdef TRDF_MODE
        ts = edge.ts;
        te = edge.te;
    #endif
    }

    edge_t& operator=(sid_t id) {
        this->val = id;
        return *this;
    }

    edge_t& operator=(const edge_t& e) {
        if (this != &e) {
            val = e.val;
        #ifdef TRDF_MODE
            ts = e.ts;
            te = e.te;
        #endif
        }
        return *this;
    }

    edge_t& operator=(const edge_t&& e) {
        if (this != &e) {
            val = e.val;
        #ifdef TRDF_MODE
            ts = e.ts;
            te = e.te;
        #endif
        }
        return *this;
    }

    bool operator==(const edge_t& e) {
    #ifdef TRDF_MODE
        return this->val == e.val && this->ts == e.ts && this->te == e.te;
    #else
        return this->val == e.val;
    #endif
    }

    bool operator==(const sid_t& id) {
        return this->val == id;
    }
};

#ifdef TRDF_MODE
struct time_edge_t {
    int val;  // vertex ID
    int64_t ts;  // start timestamp
    int64_t te;  // end timestamp

    // clang-format off
#ifdef USE_GPU
    __host__ __device__
#endif
    time_edge_t(sid_t id, int64_t ts, int64_t te): val(id), ts(ts), te(te) {}
    // clang-format on

    bool valid(int64_t _ts, int64_t _te) const {
        bool rev = ((_ts <= _te) && (ts <= te) && (_ts <= te && _te >= ts)) || (_ts == 0L && _te == 0L);
        return rev;
    }

    time_edge_t(const time_edge_t& edge) : val(edge.val), ts(edge.ts), te(edge.te) {}

    time_edge_t(time_edge_t&& edge) : val(edge.val), ts(edge.ts), te(edge.te) {}

    time_edge_t& operator=(time_edge_t& e) {
        if (this != &e) {
            val = e.val;
            ts = e.ts;
            te = e.te;
        }
        return *this;
    }

    time_edge_t& operator=(time_edge_t&& e) {
        if (this != &e) {
            val = e.val;
            ts = e.ts;
            te = e.te;
        }
        return *this;
    }

    bool operator==(const time_edge_t& e) {
        return this->val == e.val && this->ts == e.ts && this->te == e.te;
    }
};
#endif

}  // namespace wukong
