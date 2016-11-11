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

#include "graph_basic_types.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

#include <vector>

using namespace std;
using namespace boost::archive;

enum var_type {
    known_var,
    unknown_var,
    const_var
};

// defined as constexpr due to switch-case
constexpr int var_pair(int t1, int t2) { return ((t1 << 4) | t2); }

struct request_template {
    vector<int64_t> cmd_chains;

    // no serialize
    vector<string> ptypes_str;  // strings of pattern-types
    vector<int64_t> ptypes_pos;  // place-holders of pattern-types
    vector<vector<int64_t> *> ptypes_grp; // a group of IDs matching pattern-types
};

struct request_or_reply {
    int first_target; // no serialize

    int id;
    int parent_id;
    int step;
    int col_num;
    bool silent;
    int silent_row_num;
    int64_t local_var;
    vector<int64_t> cmd_chains; // N * (subject, predicat, direction, object)
    vector<int64_t> result_table;

    int mt_total_thread;
    int mt_current_thread;

    request_or_reply() {
        first_target = -1;
        parent_id = -1;
        id = -1;
        step = 0;
        col_num = 0;
        silent = false;
        silent_row_num = 0;
        local_var = 0;
        mt_total_thread = 1;
        mt_current_thread = 0;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & id;
        ar & parent_id;
        ar & step;
        ar & col_num;
        ar & silent;
        ar & silent_row_num;
        ar & local_var;
        ar & cmd_chains;
        ar & result_table;
        ar & mt_total_thread;
        ar & mt_current_thread;
    }

    void clear_data() { result_table.clear(); }

    bool is_finished() { return (step * 4 >= cmd_chains.size()); }

    bool is_request() { return (id == -1); }

    bool use_index_vertex() {
        if ((cmd_chains[0] >= 0l) && (cmd_chains[0] < (1l << NBITS_PID)))
            return true;
        return false;
        // return cmd_chains[2]==pindex_in ||
        //  cmd_chains[2]==pindex_out ||
        //  cmd_chains[2]==tindex_in ;
    }

    var_type variable_type(int64_t vid) {
        if (vid >= 0)
            return const_var;

        if ((- vid) > column_num())
            return unknown_var;
        else
            return known_var;
    };

    int64_t var2column(int64_t vid) {
        assert(vid < 0); // pattern variable
        return ((- vid) - 1);
    }

    void set_column_num(int n) { col_num = n; }

    int column_num() { return col_num; };

    int row_num() {
        if (col_num == 0)
            return 0;
        return result_table.size() / col_num;
    }

    int64_t get_row_column(int r, int c) {
        return result_table[col_num * r + c];
    }

    void append_row_to(int r, vector<int64_t> &updated_result_table) {
        for (int c = 0; c < column_num(); c++)
            updated_result_table.push_back(get_row_column(r, c));
    };
};
