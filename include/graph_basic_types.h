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

#include "utils.h"
#include <functional>
#include <iostream>
#include "global_cfg.h"

struct edge_triple {
	uint64_t s, p, o;

	edge_triple(uint64_t _s, uint64_t _p, uint64_t _o): s(_s), p(_p), o(_o) {}

	edge_triple(): s(-1), p(-1), o(-1) {}
};

struct edge_sort_by_spo {
	inline bool operator() (const edge_triple &s1, const edge_triple &s2) {
		if (s1.s < s2.s) {
			return true;
		} else if (s1.s == s2.s) {
			if (s1.p < s2.p) {
				return true;
			} else if (s1.p == s2.p && s1.o < s2.o) {
				return true;
			}
		}

		return false;
	}
};

struct edge_sort_by_ops {
	inline bool operator() (const edge_triple &s1, const edge_triple &s2) {
		if (s1.o < s2.o) {
			return true;
		} else if (s1.o == s2.o) {
			if (s1.p < s2.p) {
				return true;
			} else if ((s1.p == s2.p) && (s1.s < s2.s)) {
				return true;
			}
		}

		return false;
	}
};

// The ID space of predicate/type ID in [0, 2^NBITS_PID)
enum { NBITS_DIR = 1 };  // direction: 0=in, 1=out
enum { NBITS_PID = 17 }; // equal to the size of t/pid
enum { NBITS_VID = (64 - NBITS_PID - NBITS_DIR) }; // 0: index vertex, ID: normal vertex

//const int nbit_predict = 17;
//const int nbit_id = 63 - nbit_predict;
static inline bool is_pid(int id) { return id < (1 << NBITS_PID); }

/**
 * Predicate-base Key/value Store
 * Key: vid | t/pid | direction
 * Val: v/t/pid list
 */
struct local_key {
uint64_t dir : NBITS_DIR;
uint64_t pid : NBITS_PID;
uint64_t vid : NBITS_VID;

	//local_key(): dir(0), pid(0), vid(0) {}

	local_key(): dir(0), pid(0), vid(0) { }

	local_key(uint64_t i, uint64_t d, uint64_t p): vid(i), dir(d), pid(p) {
		if ((vid != i) || (dir != d) || (pid != p)) {
			cout << "WARNING: key truncated! "
			     << "[" << i << "|" << p << "|" << d << "]"
			     << " => "
			     << "[" << vid << "|" << pid << "|" << dir << "]"
			     << endl;
		}
	}

	bool operator == (const local_key &_key) {
		if ((dir == _key.dir) && (pid == _key.pid) && (vid == _key.vid))
			return true;
		return false;
	}

	bool operator != (const local_key &_key) {
		return !(operator == (_key));
	}

	void print() {
		cout << "[" << vid << "|" << pid << "|" << dir << "]" << endl;
	}

	uint64_t hash() {
		uint64_t r = 0;
		r += vid;
		r <<= NBITS_PID;
		r += pid;
		r <<= NBITS_DIR;
		r += dir;
		//return std::hash<uint64_t>()(r);
		return mymath::hash(r);
	}
};


enum { NBITS_SIZE = 28 };
enum { NBITS_PTR = 36 };

struct local_val {
uint64_t size: NBITS_SIZE;
uint64_t ptr: NBITS_PTR;

	//local_val(): size(0), ptr(0) {}

	local_val(): size(0), ptr(0) { }

	local_val(uint64_t s, uint64_t p): size(s), ptr(p) {
		if ((size != s) || (ptr != p)) {
			cout << "WARNING: key truncated! "
			     << "[" << p << "|" << s << "]"
			     << " => "
			     << "[" << ptr << "|" << size << "]"
			     << endl;
		}
	}

	bool operator == (const local_val &_val) {
		if ((size == _val.size) && (ptr == _val.ptr))
			return true;
		return false;
	}

	bool operator != (const local_val &_val) {
		return !(operator == (_val));
	}
};

struct vertex {
	local_key key;
	local_val val;
};

struct edge {
	//	uint64_t val;
	unsigned int val;
};

enum direction {
	IN,
	OUT,
	JOIN
};
