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

#include <iostream>
#include <boost/serialization/unordered_map.hpp>

#include "utils.h"

using namespace std;

class Logger {
private:
    struct req_stats {
        int query_type;
        uint64_t start_time;
        uint64_t end_time;
        template <typename Archive>
        void serialize(Archive &ar, const unsigned int version) {
            ar & query_type;
            ar & start_time;
            ar & end_time;
        }
    };

    uint64_t init_time;
    unordered_map<int, req_stats> stats_map;

public:
    void init() {
        init_time = timer::get_usec();
    }

    void start_record(int reqid, int query_type) {
        stats_map[reqid].query_type = query_type;
        stats_map[reqid].start_time = timer::get_usec() - init_time;
    }

    void end_record(int reqid) {
        stats_map[reqid].end_time = timer::get_usec() - init_time;
    }

    void finish() {

    }

    void merge(Logger &other) {
        for (auto iter : other.stats_map) {
            stats_map[iter.first] = iter.second;
        }
    }

    void print() {
        vector<int> throughput_vec;
        int print_interval = 200; //100ms
        cout << "Throughput (Kops)" << endl;
        for (auto iter : stats_map) {
            int idx = iter.second.start_time / (1000 * print_interval);
            if (throughput_vec.size() <= idx) {
                throughput_vec.resize(idx + 1);
            }
            throughput_vec[idx]++;
        }
        for (int i = 0; i < throughput_vec.size(); i++) {
            cout << throughput_vec[i] * 1.0 / print_interval << " ";
            if (i % 5 == 4 || i == throughput_vec.size() - 1) {
                cout << endl;
            }
        }

        ///
        cout << "CDF graph" << endl;
        vector<int> cdf_data;
        int cdf_pirnt_rate = 100;
        for (auto iter : stats_map) {
            cdf_data.push_back(iter.second.end_time - iter.second.start_time);
        }
        sort(cdf_data.begin(), cdf_data.end());
        int count = 0;
        for (int j = 0; j < cdf_data.size(); j++) {
            if ((j + 1) % (cdf_data.size() / cdf_pirnt_rate) == 0 ) {
                count++;
                if (count != cdf_pirnt_rate) {
                    cout << cdf_data[j] << "\t";
                } else {
                    cout << cdf_data[cdf_data.size() - 1] << "\t";
                }
                if (count % 5 == 0) {
                    cout << endl;
                }
            }
        }
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & stats_map;
    }
};
