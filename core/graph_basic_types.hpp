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
 *      http://ipads.se.sjtu.edu.cn/projects/wukong.html
 *
 */

#pragma once

#include <functional>
#include <iostream>

#include "config.hpp"
#include "mymath.hpp"

struct triple_t {
	uint64_t s; // subject
	uint64_t p; // predicate
	uint64_t o; // object

	triple_t(): s(0), p(0), o(0) { }

	triple_t(uint64_t _s, uint64_t _p, uint64_t _o): s(_s), p(_p), o(_o) { }
};

struct edge_sort_by_spo {
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

struct edge_sort_by_ops {
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

// 64-bit internal key
enum { NBITS_DIR = 1 };
enum { NBITS_IDX = 17 }; // equal to the size of t/pid
enum { NBITS_VID = (64 - NBITS_IDX - NBITS_DIR) }; // 0: index vertex, ID: normal vertex

enum { PREDICATE_ID = 0, TYPE_ID = 1 }; // reserve two special index IDs
enum dir_t { IN, OUT, CORUN }; // direction: IN=0, OUT=1, and optimization hints

static inline bool is_idx(int id) { return (id > 0) && (id < (1 << NBITS_IDX)); }

/**
 * predicate-base key/value store
 * key: vid | t/pid | direction
 * value: v/t/pid list
 */
struct ikey_t {
uint64_t dir : NBITS_DIR; // direction
uint64_t pid : NBITS_IDX; // predicate
uint64_t vid : NBITS_VID; // vertex

	ikey_t(): dir(0), pid(0), vid(0) { }

	ikey_t(uint64_t v, uint64_t d, uint64_t p): vid(v), dir(d), pid(p) {
		if ((vid != v) || (dir != d) || (pid != p)) {
			cout << "WARNING: key truncated! "
			     << "[" << v << "|" << p << "|" << d << "]"
			     << " => "
			     << "[" << vid << "|" << pid << "|" << dir << "]"
			     << endl;
		}
	}

	bool operator == (const ikey_t &_key) {
		if ((dir == _key.dir) && (pid == _key.pid) && (vid == _key.vid))
			return true;
		return false;
	}

	bool operator != (const ikey_t &_key) {
		return !(operator == (_key));
	}

	void print() {
		cout << "[" << vid << "|" << pid << "|" << dir << "]" << endl;
	}

	uint64_t hash() {
		uint64_t r = 0;
		r += vid;
		r <<= NBITS_IDX;
		r += pid;
		r <<= NBITS_DIR;
		r += dir;
		return mymath::hash_u64(r);  // the standard hash is too slow (i.e., std::hash<uint64_t>()(r))
	}
};

// 64-bit internal pointer
enum { NBITS_SIZE = 28 };
enum { NBITS_PTR = 36 };

struct iptr_t {
uint64_t size: NBITS_SIZE;
uint64_t ptr: NBITS_PTR;

	iptr_t(): size(0), ptr(0) { }

	iptr_t(uint64_t s, uint64_t p): size(s), ptr(p) {
		if ((size != s) || (ptr != p)) {
			cout << "WARNING: key truncated! "
			     << "[" << p << "|" << s << "]"
			     << " => "
			     << "[" << ptr << "|" << size << "]"
			     << endl;
		}
	}

	bool operator == (const iptr_t &_val) {
		if ((size == _val.size) && (ptr == _val.ptr))
			return true;
		return false;
	}

	bool operator != (const iptr_t &_val) {
		return !(operator == (_val));
	}
};

// 128-bit vertex (key)
struct vertex {
	ikey_t key; // 64-bit: vertex | predicate | direction
	iptr_t val; // 64-bit: size | offset
};

// edge (value)
struct edge {
	// uint64_t val;
	unsigned int val;
};

