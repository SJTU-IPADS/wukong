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

#ifdef USE_GPU
#pragma once

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>
#include <map>
#include <sstream>
using namespace std;
using namespace boost::archive;

#define EXT_LIST_MAX_LEN 4
#define EXT_BUCKET_EXTENT_LEN(num_buckets) (num_buckets * 15 / 100 + 1)
#define PREDICATE_NSEGS 4

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
struct rdf_segment_meta_t {
    uint64_t num_keys;      // #key of the segment
    uint64_t num_buckets;   // allocated main headers (hash space)
    uint64_t bucket_start;  // start offset of main-header region
    ext_bucket_extent_t ext_bucket_list[EXT_LIST_MAX_LEN];
    int ext_list_sz;        // the length of ext_bucket_list
    uint64_t num_edges;     // #edge of the segment
    uint64_t edge_start;    // start offset in the entry region

    rdf_segment_meta_t() : num_keys(0), num_buckets(0), bucket_start(0),
        ext_list_sz(0), edge_start(0), num_edges(0) {
        memset(&ext_bucket_list, 0, sizeof(ext_bucket_list));
    }

    uint64_t get_ext_bucket() {
        for (int i = 0; i < ext_list_sz; ++i) {
            ext_bucket_extent_t &ext = ext_bucket_list[i];
            if (ext.off < ext.num_ext_buckets) {
                return ext.start + ext.off++;
            }
        }

        return 0;
    }

    void add_ext_buckets(const ext_bucket_extent_t &ext) {
        ext_bucket_list[ext_list_sz++] = ext;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & num_buckets;
        ar & bucket_start;
        ar & ext_bucket_list;
        ar & ext_list_sz;
        ar & num_edges;
        ar & edge_start;
    }
};

/**
 * Identifier of a segment
 */
struct segid_t {
    int index;  // normal or index segment
    int dir;  // direction of triples in the segment
    sid_t pid;  // predicate id
    segid_t(): index(0), dir(0), pid(0) { }
    segid_t(int idx, sid_t p, int d) : index(idx), pid(p), dir(d) { }

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

    string stringify() {
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
    map<segid_t, rdf_segment_meta_t> data;

    SyncSegmentMetaMsg() { }

    SyncSegmentMetaMsg(map<segid_t, rdf_segment_meta_t> data) {
        this->data = data;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & sender_sid;
        ar & data;
    }
};

#endif  // USE_GPU
