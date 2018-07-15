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
#include <boost/serialization/vector.hpp>
#include <vector>
using namespace std;
using namespace boost::archive;

// Metadata of a predicate segment
struct rdf_segment_meta_t {
    uint64_t num_buckets;  // allocated main headers (hash space)
    uint64_t bucket_start;  // start offset of main-header region
    uint64_t bucket_end;
    uint64_t ext_bucket_start; // start offset of indirect-header region
    uint64_t ext_bucket_end;
    uint64_t edge_start;    // start offset of edge region
    uint64_t edge_end;

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & num_buckets;
        ar & bucket_start;
        ar & bucket_end;
        ar & ext_bucket_start;
        ar & ext_bucket_end;
        ar & edge_start;
        ar & edge_end;
    }
};

class SyncSegmentMetaMsg {
public:
    int sender_sid;
    vector<rdf_segment_meta_t> data;

    SyncSegmentMetaMsg() { }

    SyncSegmentMetaMsg(vector<rdf_segment_meta_t> data) {
        this->data = data;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & sender_sid;
        ar & data;
    }
};

