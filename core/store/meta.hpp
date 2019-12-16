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

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>
#include <map>
#include <sstream>

#include "type.hpp"
#include "vertex.hpp"

using namespace std;
using namespace boost::archive;

#ifdef USE_GPU
#define EXT_BUCKET_LIST_CAPACITY 1
#define EXT_BUCKET_EXTENT_LEN(num_buckets) (num_buckets * 15 / 100 + 1)
#else
#define EXT_BUCKET_EXTENT_LEN 256
#endif
#define PREDICATE_NSEGS 2
#ifdef VERSATILE
#define INDEX_NSEGS 4   // index(2) + vid's all preds(2)
#else // VERSATILE
#define INDEX_NSEGS 2   // index(2)
#endif // VERSATILE
/**
 * A contiguous space in the indirect-header region
 */
struct ext_bucket_extent_t {
    uint64_t num_ext_buckets;   // capacity
    uint64_t off;       // current offset
    uint64_t start;     // start offset in the indirect header region

    ext_bucket_extent_t() : num_ext_buckets(0), off(0), start(0) {}
    ext_bucket_extent_t(uint64_t nbuckets, uint64_t start_off)
        : num_ext_buckets(nbuckets), off(0), start(start_off) { }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & num_ext_buckets;
        ar & off;
        ar & start;
    }
};

/**
 * Metadata of a segment
 *
 * Each predicate has four segments:
 * 1) Normal-OUT 2) Normal-IN 3) Index-OUT 4) Index-IN.
 * Normal segments are for normal vertices, while index segments for index vertices.
 * And the IN/OUT indicates the direction of triples in the segment.
 */
struct rdf_seg_meta_t {
    uint64_t num_keys = 0;      // #keys of the segment
    uint64_t num_buckets = 0;   // allocated main headers (hash space)
    uint64_t bucket_start = 0;  // start offset of main-header region of gstore
    uint64_t num_edges = 0;     // #edges of the segment
    uint64_t edge_start = 0;    // start offset in the entry region of gstore
    uint64_t edge_off = 0;      // current available offset in the entry region, only used by static gstore

    int num_key_blks = 0;       // #key-blocks needed in gcache
    int num_value_blks = 0;     // #value-blocks needed in gcache

#ifdef USE_GPU
    ext_bucket_extent_t ext_bucket_list[EXT_BUCKET_LIST_CAPACITY];
    size_t ext_bucket_list_sz = 0;

    rdf_seg_meta_t() {
        memset(&ext_bucket_list, 0, sizeof(ext_bucket_list));
    }

    size_t get_ext_bucket_list_size() const { return ext_bucket_list_sz; }

    void add_ext_buckets(const ext_bucket_extent_t &ext) {
        assert(ext_bucket_list_sz < EXT_BUCKET_LIST_CAPACITY);
        ext_bucket_list[ext_bucket_list_sz++] = ext;
    }
#else
    vector<ext_bucket_extent_t> ext_bucket_list;

    size_t get_ext_bucket_list_size() const { return ext_bucket_list.size(); }

    void add_ext_buckets(const ext_bucket_extent_t &ext) {
        ext_bucket_list.push_back(ext);
    }

#endif

    uint64_t get_ext_bucket() {
        for (int i = 0; i < get_ext_bucket_list_size(); ++i) {
            ext_bucket_extent_t &ext = ext_bucket_list[i];
            if (ext.off < ext.num_ext_buckets) {
                return ext.start + ext.off++;
            }
        }
        return 0;
    }

    inline uint64_t get_total_num_buckets() const {
        uint64_t total = num_buckets;
        for (int i = 0; i < get_ext_bucket_list_size(); ++i) {
            total += ext_bucket_list[i].num_ext_buckets;
        }
        return total;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & num_buckets;
        ar & bucket_start;
        ar & ext_bucket_list;
#ifdef USE_GPU
        ar & ext_bucket_list_sz;
#endif
        ar & num_edges;
        ar & edge_start;
    }
};

/**
 * Identifier of a segment
 */
struct segid_t {
    int index;  // normal or index segment
    dir_t dir;  // direction of triples in the segment
    sid_t pid;  // predicate id
    segid_t(): index(0), pid(0), dir(IN) { }
    segid_t(int idx, sid_t p, dir_t d) : index(idx), pid(p), dir(d) { }
    segid_t(const ikey_t &key) {
        dir = (dir_t)key.dir;
        // index
        if (key.vid == 0) {
            index = 1;
            pid = PREDICATE_ID;
        } else {
            index = 0;
            pid = key.pid;
        }
    }

    uint64_t hash() const {
        uint64_t r = 0;
        if (index != 0) {
            r = dir;
        } else {
            r = ((pid + 1) << 1) + dir;
        }
        return wukong::math::hash_u64(r); // the standard hash is too slow (i.e., std::hash<uint64_t>()(r))
    }


    bool operator == (const segid_t &s) const {
        return (index == s.index && dir == s.dir && pid == s.pid);
    }

    bool operator < (const segid_t& segid) const {
        if (pid < segid.pid)
            return true;
        else if (pid == segid.pid) {
            if (index < segid.index)
                return true;
            else if (index == segid.index && dir < segid.dir)
                return true;
        }
        return false;
    }

    string to_string() {
        ostringstream ss;
        ss << "[" << index << "|" << pid << "|" << dir << "]";
        return ss.str();
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & index;
        ar & dir;
        ar & pid;
    }
};

/**
 * Metadata synchronization message
 *
 * When run on multiple servers, we need to sync the metadata of segments
 * with each server. Because the layout of kvstore is different in each server.
 * Then light queries running in in-place mode can work correctly.
 */
class SyncSegmentMetaMsg {
public:
    int sender_sid;
    map<segid_t, rdf_seg_meta_t> data;

    SyncSegmentMetaMsg() { }

    SyncSegmentMetaMsg(map<segid_t, rdf_seg_meta_t> data) {
        this->data = data;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & sender_sid;
        ar & data;
    }
};
