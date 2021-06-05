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

#include <boost/variant.hpp>

namespace wukong {

#ifdef DTYPE_64BIT

using sid_t = uint64_t;  // data type for string-id
using ssid_t = int64_t;  // signed string id
#define BLANK_ID UINT64_MAX

#else

using sid_t = uint32_t;  // data type for string-id
using ssid_t = int32_t;  // signed string id
#define BLANK_ID UINT32_MAX
#define TIMESTAMP_MAX INT64_MAX
#define TIMESTAMP_MIN INT64_MIN

#endif

//attr_t unions int, double, float
using attr_t = boost::variant<int, double, float>;

enum dir_t { IN = 0, OUT, CORUN }; // direction: IN=0, OUT=1, and optimization hints
enum data_type { SID_t = 0, INT_t, FLOAT_t, DOUBLE_t, TIME_t };

struct triple_t {
    sid_t s; // subject
    sid_t p; // predicate
    sid_t o; // object

#ifdef TRDF_MODE
    int64_t ts; // start timestamp
    int64_t te; // end timestamp

    triple_t(sid_t _s, sid_t _p, sid_t _o, int64_t _ts, int64_t _te): s(_s), p(_p), o(_o), ts(_ts), te(_te) { }
#endif

    triple_t(): s(0), p(0), o(0) { }

    triple_t(sid_t _s, sid_t _p, sid_t _o): s(_s), p(_p), o(_o) { }

    triple_t(const triple_t& triple) {
        s = triple.s;
        p = triple.p;
        o = triple.o;
    #ifdef TRDF_MODE
        ts = triple.ts;
        te = triple.te;
    #endif
    }

    triple_t(const triple_t&& triple) {
        s = triple.s;
        p = triple.p;
        o = triple.o;
    #ifdef TRDF_MODE
        ts = triple.ts;
        te = triple.te;
    #endif
    }

    triple_t& operator=(const triple_t& triple) {
        s = triple.s;
        p = triple.p;
        o = triple.o;
    #ifdef TRDF_MODE
        ts = triple.ts;
        te = triple.te;
    #endif
        return *this;
    }

    bool operator==(const triple_t& triple) const {
        if ((s == triple.s) 
            && (p == triple.p) 
            && (o == triple.o)
        #ifdef TRDF_MODE
            && (ts == triple.ts) 
            && (te == triple.te)
        #endif
            ) {
            return true;
        }
        return false;
    }
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

        #ifdef TRDF_MODE
            else if (t1.p == t2.p)
                if(t1.o < t2.o)
                    return true;
                else if(t1.o == t2.o)
                    if(t1.ts < t2.ts)
                        return true;
                    else if(t1.ts == t2.ts && t1.te < t2.te)
                        return true;
        #else
            else if (t1.p == t2.p && t1.o < t2.o)
                return true;
        #endif
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
        #ifdef TRDF_MODE
            else if (t1.p == t2.p)
                if(t1.s < t2.s)
                    return true;
                else if(t1.s == t2.s)
                    if(t1.ts < t2.ts)
                        return true;
                    else if(t1.ts == t2.ts && t1.te < t2.te)
                        return true;
        #else
            else if ((t1.p == t2.p) && (t1.s < t2.s))
                return true;
        #endif
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
        #ifdef TRDF_MODE
            else if (t1.s == t2.s)
                if(t1.o < t2.o)
                    return true;
                else if(t1.o == t2.o)
                    if(t1.ts < t2.ts)
                        return true;
                    else if(t1.ts == t2.ts && t1.te < t2.te)
                        return true;
        #else
            else if (t1.s == t2.s && t1.o < t2.o)
                return true;
        #endif
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
        #ifdef TRDF_MODE
            else if (t1.o == t2.o)
                if(t1.s < t2.s)
                    return true;
                else if(t1.s == t2.s)
                    if(t1.ts < t2.ts)
                        return true;
                    else if(t1.ts == t2.ts && t1.te < t2.te)
                        return true;
        #else
            else if (t1.o == t2.o && t1.s < t2.s)
                return true;
        #endif
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

// get the variant type
class variant_type : public boost::static_visitor<int> {
public:
    int operator ()(int i) const { return INT_t; }
    int operator ()(float f) const { return FLOAT_t; }
    int operator ()(double d) const { return DOUBLE_t; }
};

// get the size of variant type
static inline size_t get_sizeof(int type) {
    switch (type) {
    case INT_t: return sizeof(int);
    case FLOAT_t: return sizeof(float);
    case DOUBLE_t: return sizeof(double);
    default: return 0;
    }
}

} // namespace wukong