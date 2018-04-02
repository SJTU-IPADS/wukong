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
        is_aggregated = false;
        this->nquery_types = nquery_types;
        done_time = 0ull;
        last_cnt = 0ull;
        thpt_time = 0ull;
        cnt = 0ull;
        thpt = 0.0;
        init_time = timer::get_usec();
        last_time = last_separator = timer::get_usec();
        stats_map.clear();
        total_latency_map.clear();
        for (int i = 0; i < nquery_types; ++i) {
            total_latency_map[i] = std::vector<uint64_t>();
        }
    }

    void finish() {
        done_time = timer::get_usec();
    }

    // for single query
    void print_latency(int round = 1) {
        logstream(LOG_INFO) << "(average) latency: " << ((done_time - init_time) / round) << " usec" << LOG_endl;
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
            logstream(LOG_INFO) << "Throughput: " << cur_thpt / 1000.0 << "K queries/sec" << LOG_endl;
            last_time = now;
            last_cnt = cur_cnt;
        }

        // print separators per second
        if (now - last_separator > SEC(1)) {
            logstream(LOG_INFO) << "[" << (now - init_time) / SEC(1) << "sec]" << LOG_endl;
            last_separator = now;
        }
    }

    void start_thpt(uint64_t start) {
        thpt_time = timer::get_usec();
        cnt = start;
    }

    void end_thpt(uint64_t end) {
        thpt = 1000000.0 * (end - cnt) / (timer::get_usec() - thpt_time);
    }

    void print_thpt() {
        logstream(LOG_INFO) << "Throughput: " << thpt / 1000.0 << "K queries/sec" << LOG_endl;
    }

    void start_record(int reqid, int type) {
        stats_map[reqid].query_type = type;
        stats_map[reqid].start_time = timer::get_usec() - init_time;
    }

    void end_record(int reqid) {
        stats_map[reqid].end_time = timer::get_usec() - init_time;
    }

    // calculate each query's cdf, then sort respectively
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

        logstream(LOG_INFO) << "Range Throughput (K queries/sec)" << LOG_endl;
        for (int i = 0; i < thpts.size(); i++)
            logstream(LOG_INFO) << "[" << (print_interval * i) / 1000 << "ms ~ "
                 << print_interval * (i + 1) / 1000 << "ms)\t"
                 << (float)thpts[i] / (print_interval / 1000) << LOG_endl;
#endif
        assert(is_aggregated);
        vector<double> cdf_rates = {0.01};

        for (int i = 1; i < 20; ++i) {
            cdf_rates.push_back(0.05 * i);
        }
        for (int i = 1; i <= 5; ++i) {
            cdf_rates.push_back(0.95 + i * 0.01);
        }
        assert(cdf_rates.size() == 25);

        logstream(LOG_INFO) << "Per-query CDF graph" << LOG_endl;
        int cnt, query_type;//, query_cnt = 0;
        map<int, vector<uint64_t>> cdf_res;

        // select 25 points from total_latency_map
        for (auto e : total_latency_map) {
            query_type = e.first;
            vector<uint64_t> &lats = e.second;

            // assert(lats.size() > cdf_rates.size());
            if (lats.empty())
                continue;

            cnt = 0;
            // result of CDF figure
            cdf_res[query_type] = std::vector<uint64_t>();
            // select points from lats corresponding to cdf_rates
            for (auto rate : cdf_rates) {
                int idx = lats.size() * rate;
                if (idx >= lats.size()) idx = lats.size() - 1;
                cdf_res[query_type].push_back(lats[idx]);
                cnt++;
            }
            assert(cdf_res[query_type].size() == 25);
        }

        logstream(LOG_INFO) << "CDF Res: " << LOG_endl;
        logstream(LOG_INFO) << "P";
        for (int i = 1; i <= nquery_types; ++i) {
            logstream(LOG_INFO) << "\t" << "Q" << i;
        }
        logstream(LOG_INFO) << LOG_endl;

        // print cdf data
        int row, p;
        for (row = 1; row <= 25; ++row) {
            if (row == 1)
                logstream(LOG_INFO) << row << "\t";
            else if (row <= 20)
                logstream(LOG_INFO) << 5 * (row - 1) << "\t";
            else
                logstream(LOG_INFO) << 95 + (row - 20) << "\t";

            for (int i = 0; i < nquery_types; ++i) {
                logstream(LOG_INFO) << cdf_res[i][row - 1] << "\t";
            }
            logstream(LOG_INFO) << LOG_endl;
        }
    }

    void merge(Logger & other) {
        if (nquery_types < other.nquery_types) nquery_types = other.nquery_types;
        for (auto s : other.stats_map)
            stats_map[s.first] = s.second;
        thpt += other.thpt;
    }

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & nquery_types;
        ar & stats_map;
        ar & thpt;
    }
};
