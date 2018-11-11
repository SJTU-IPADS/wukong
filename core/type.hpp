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

#include <stdint.h>
#include "variant.hpp"

#ifdef DTYPE_64BIT

typedef uint64_t sid_t;  // data type for string-id
typedef int64_t ssid_t;  // signed string id
#define BLANK_ID UINT64_MAX

#else

typedef uint32_t sid_t;  // data type for string-id
typedef int32_t ssid_t;  // signed string id
#define BLANK_ID UINT32_MAX

#endif

struct triple_t {
    sid_t s; // subject
    sid_t p; // predicate
    sid_t o; // object

    triple_t(): s(0), p(0), o(0) { }

    triple_t(sid_t _s, sid_t _p, sid_t _o): s(_s), p(_p), o(_o) { }
};

struct triple_attr_t {
    sid_t s;  // subject
    sid_t a;  // attribute
    attr_t v; // value

    triple_attr_t():  s(0), a(0), v(0) { }

    triple_attr_t(sid_t _s, sid_t _a, attr_t _v): s(_s), a(_a), v(_v) { }
};

struct triple_sort_by_spo {
    inline bool operator()(const triple_t &t1, const triple_t &t2) {
        if (t1.s < t2.s)
            return true;
        else if (t1.s == t2.s)
            if (t1.p < t2.p)
                return true;
            else if (t1.p == t2.p && t1.o < t2.o)
                return true;
        return false;
    }
};

struct triple_sort_by_ops {
    inline bool operator()(const triple_t &t1, const triple_t &t2) {
        if (t1.o < t2.o)
            return true;
        else if (t1.o == t2.o)
            if (t1.p < t2.p)
                return true;
            else if ((t1.p == t2.p) && (t1.s < t2.s))
                return true;
        return false;
    }
};

struct triple_sort_by_pso {
    inline bool operator()(const triple_t &t1, const triple_t &t2) {
        if (t1.p < t2.p)
            return true;
        else if (t1.p == t2.p)
            if (t1.s < t2.s)
                return true;
            else if (t1.s == t2.s && t1.o < t2.o)
                return true;
        return false;
    }
};

struct triple_sort_by_pos {
    inline bool operator()(const triple_t &t1, const triple_t &t2) {
        if (t1.p < t2.p)
            return true;
        else if (t1.p == t2.p)
            if (t1.o < t2.o)
                return true;
            else if (t1.o == t2.o && t1.s < t2.s)
                return true;
        return false;
    }
};

struct triple_sort_by_asv {
    inline bool operator()(const triple_attr_t &t1, const triple_attr_t &t2) {
        if (t1.a < t2.a)
            return true;
        else if (t1.a == t2.a)
            if (t1.s < t2.s)
                return true;
            else if (t1.s == t2.s && t1.v < t2.v)
                return true;
        return false;
    }
};

enum dir_t { IN = 0, OUT, CORUN }; // direction: IN=0, OUT=1, and optimization hints
