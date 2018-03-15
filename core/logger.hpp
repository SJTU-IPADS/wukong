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

#include <iostream>
#include <boost/serialization/unordered_map.hpp>

#include "timer.hpp"
#include "unit.hpp"

using namespace std;

class Logger {
private:
    struct req_stats {
        int query_type;
        uint64_t start_time = 0ull;
        uint64_t end_time = 0ull;

        template <typename Archive>
        void serialize(Archive &ar, const unsigned int version) {
            ar & query_type;
            ar & start_time;
            ar & end_time;
        }
    };

    uint64_t init_time = 0ull, done_time = 0ull;

    uint64_t last_time = 0ull, last_separator = 0ull;
    uint64_t last_cnt = 0ull;

    uint64_t interval = MSEC(500);  // 50msec

    uint64_t thpt_time = 0ull;
    uint64_t cnt = 0ull;
    float thpt = 0.0;

    int nquery_types = 0;
    bool is_finish = false;
    bool is_aggregated = false;

    // key: query_type, value: latency of query
    // ordered by query_type
    std::map<int, vector<uint64_t>> total_latency_map;

    unordered_map<int, req_stats> stats_map; // CDF

public:
    void init() {
        init_time = timer::get_usec();
        last_time = last_separator = timer::get_usec();
    }

    void init(int nquery_types) {
        is_finish = false;
        is_aggregated = false;
        this->nquery_types = nquery_types;
        done_time = 0ull;
        last_cnt = 0ull;
        thpt_time = 0ull;
        cnt = 0ull;
        thpt = 0.0;
        init_time = timer::get_usec();
        last_time = 0ull;
        last_separator = 0ull;
        stats_map.clear();
        total_latency_map.clear();
        for (int i = 0; i < nquery_types; ++i) {
            total_latency_map[i] = std::vector<uint64_t>();
        }
    }

    void finish() {
        if (!is_finish) {
            done_time = timer::get_usec();
            is_finish = true;
        }
    }

    // for single query
    void print_latency(int round = 1) {
        cout << "(average) latency: " << ((done_time - init_time) / round) << " usec" << endl;
    }

    void set_interval(uint64_t update) { interval = update; }

    // print the throughput of a fixed interval
    void print_timely_thpt(uint64_t cur_cnt, int sid, int tid) {
        // for brevity, only print the timely thpt of a single proxy.
        if (!(sid == 0 && tid == 0)) return;

        uint64_t now = timer::get_usec();
        // periodically print timely throughput
        if ((now - last_time) > interval) {
            float cur_thpt = 1000000.0 * (cur_cnt - last_cnt) / (now - last_time);
            cout << "Throughput: " << cur_thpt / 1000.0 << "K queries/sec" << endl;
            last_time = now;
            last_cnt = cur_cnt;
        }

        // print separators per second
        if (now - last_separator > SEC(1)) {
            cout << "[" << (now - init_time) / SEC(1) << "sec]" << endl;
            last_separator = now;
        }
    }

    void start_thpt(uint64_t start) {
        printf(">>>> Start throughput evaluation!\n");
        thpt_time = timer::get_usec();
        cnt = start;
    }

    void end_thpt(uint64_t end) {
        thpt = 1000000.0 * (end - cnt) / (timer::get_usec() - thpt_time);
        printf("<<<< End throughput evaluation!\n");
    }

    void print_thpt() {
        cout << "Throughput: " << thpt / 1000.0 << "K queries/sec" << endl;
    }

    void start_record(int reqid, int type) {
        stats_map[reqid].query_type = type;
        stats_map[reqid].start_time = timer::get_usec() - init_time;
    }

    void end_record(int reqid) {
        stats_map[reqid].end_time = timer::get_usec() - init_time;
    }

    void aggregate() {
        for (auto s : stats_map) {
            total_latency_map[s.second.query_type].push_back(s.second.end_time - s.second.start_time);
        }
        // sort
        for (int i = 0; i < nquery_types; ++i) {
            vector<uint64_t> &lats = total_latency_map[i];
            if (!lats.empty())
                sort(lats.begin(), lats.end());
        }

        is_aggregated = true;
    }

    void print_cdf() {
#if 0
        // print range throughput with certain interval
        vector<int> thpts;
        int print_interval = 200 * 1000; // 200ms

        for (auto s : stats_map) {
            int i = s.second.start_time / print_interval;
            if (thpts.size() <= i)
                thpts.resize(i + 1);
            thpts[i]++;
        }

        cout << "Range Throughput (K queries/sec)" << endl;
        for (int i = 0; i < thpts.size(); i++)
            cout << "[" << (print_interval * i) / 1000 << "ms ~ "
                 << print_interval * (i + 1) / 1000 << "ms)\t"
                 << (float)thpts[i] / (print_interval / 1000) << endl;
#endif
        assert(is_finish);
        assert(is_aggregated);

        vector<double> cdf_rates = {0.01};

        for (int i = 1; i < 20; ++i) {
            cdf_rates.push_back(0.05 * i);
        }
        for (int i = 1; i <= 5; ++i) {
            cdf_rates.push_back(0.95 + i * 0.01);
        }
        assert(cdf_rates.size() == 25);

        cout << "Per-query CDF graph" << endl;
        int cnt, query_type;//, query_cnt = 0;
        map<int, vector<uint64_t>> cdf_res;

        // 从total_latency_map中选出25个点，作为最终的CDF图数据
        for (auto e : total_latency_map) {
            query_type = e.first;
            vector<uint64_t> &lats = e.second;
            // assert(lats.size() > cdf_rates.size());
            if (lats.empty())
                continue;

            cnt = 0;
            cout << "Query: " << query_type + 1 << ", size: " << lats.size() << endl;
            // result of CDF figure
            cdf_res[query_type] = std::vector<uint64_t>();
            // 利用cdf_rates从lats中筛选出对应下标的点
            for (auto rate : cdf_rates) {
                int idx = lats.size() * rate;
                if (idx >= lats.size()) idx = lats.size() - 1;
                cout << lats[idx] << "\t";
                cdf_res[query_type].push_back(lats[idx]);
                cnt++;
                if (cnt % 5 == 0)   cout << endl;
            }
            assert(cdf_res[query_type].size() == 25);
            cout << endl;
        }

        cout << "CDF Res: " << endl;
        cout << "P ";
        for (int i = 1; i <= nquery_types; ++i) {
            cout << "Q" << i << " ";
        }
        cout << endl;

        // print cdf data
        int row, p;
        for (row = 1; row <= 25; ++row) {
            if (row == 1)
                cout << row << " ";
            else if (row <= 20)
                cout << 5 * (row - 1) << " ";
            else
                cout << 95 + (row - 20) << " ";

            for (int i = 0; i < nquery_types; ++i) {
                cout << cdf_res[i][row - 1] << " ";

            }
            cout << endl;
        }
    }

    void merge(Logger & other) {
        for (auto s : other.stats_map)
            stats_map[s.first] = s.second;
        thpt += other.thpt;
    }

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & stats_map;
        ar & thpt;
    }
};
