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

#include <vector>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/unordered_set.hpp>

#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_set.h>

#include "core/common/type.hpp"

#include "utils/logger2.hpp"

namespace wukong {

#define MINIMUM_COUNT_THRESHOLD 0.01 //count below that this value will be abandoned
#define COST_THRESHOLD 1000

enum class model_t { ALL = 0, L2U, K2L, K2U, K2K, T2U };

struct plan
{
    double cost;       // min cost
    double result_num; // intermediate results

    std::vector<ssid_t> orders; // best orders
};

struct type_t {
    bool data_type;   //true for type_composition, false for index_composition
    std::unordered_set<int> composition;

    void set_type_composition(std::unordered_set<int> c) {
        data_type = true;
        this->composition = c;
    }

    void set_index_composition(std::unordered_set<int> c) {
        data_type = false;
        this->composition = c;
    };

    bool operator == (const type_t &other) const {
        if (data_type != other.data_type) return false;
        return this->composition == other.composition;
    }

    bool equal(const type_t &other) const {
        if (data_type != other.data_type) return false;
        return this->composition == other.composition;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & data_type;
        ar & composition;
    }
};

struct type_t_hasher {
    size_t operator()( const type_t& type ) const {
        return hash(type);
    }

    // for tbb hashcompare
    size_t hash( const type_t& type ) const {
        size_t res = 17;
        for (auto it = type.composition.cbegin(); it != type.composition.cend(); ++it)
            res += *it + 17;
        return res;
    }

    // for tbb hashcompare
    bool equal(const type_t& type1, const type_t& type2) const {
        return type1.equal(type2);
    }
};

struct ty_count {
    ssid_t ty;
    int count;
    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & ty;
        ar & count;
    }
};

struct type_stat {
    std::unordered_map<ssid_t, std::vector<ty_count>> pstype;
    std::unordered_map<ssid_t, std::vector<ty_count>> potype;
    std::unordered_map<std::pair<ssid_t, ssid_t>, std::vector<ty_count>, boost::hash<std::pair<int, int>>> fine_type;

    // pair<subject, predicate> means subject predicate -> ?
    // pair<predicate, object> means ? predicate -> object
    int get_pstype_count(ssid_t predicate, ssid_t type) {
        std::vector<ty_count> &types = pstype[predicate];
        for (size_t i = 0; i < types.size(); i++)
            if (types[i].ty == type)
                return types[i].count;
        return 0;
    }

    int get_potype_count(ssid_t predicate, ssid_t type) {
        std::vector<ty_count> &types = potype[predicate];
        for (size_t i = 0; i < types.size(); i++)
            if (types[i].ty == type)
                return types[i].count;
        return 0;
    }

    int insert_stype(ssid_t predicate, ssid_t type, int count) {
        std::vector<ty_count> &types = pstype[predicate];
        for (size_t i = 0; i < types.size(); i++) {
            if (types[i].ty == type) {
                types[i].count += count;
                return 0;
            }
        }

        ty_count newty;
        newty.ty = type;
        newty.count = count;
        types.push_back(newty);
        return 1;
    }

    int insert_otype(ssid_t predicate, ssid_t type, int count) {
        std::vector<ty_count> &types = potype[predicate];
        for (size_t i = 0; i < types.size(); i++) {
            if (types[i].ty == type) {
                types[i].count += count;
                return 0;
            }
        }

        ty_count newty;
        newty.ty = type;
        newty.count = count;
        types.push_back(newty);
        return 1;
    }

    int insert_finetype(ssid_t first, ssid_t second, ssid_t type, int count) {
        std::vector<ty_count> &types = fine_type[std::make_pair(first, second)];
        for (size_t i = 0; i < types.size(); i++) {
            if (types[i].ty == type) {
                types[i].count += count;
                return 0;
            }
        }

        ty_count newty;
        newty.ty = type;
        newty.count = count;
        types.push_back(newty);
        return 1;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & pstype;
        ar & potype;
        ar & fine_type;
    }
};

using tbb_set = tbb::concurrent_unordered_set<ssid_t>;
using tbb_map = tbb::concurrent_hash_map<type_t, ssid_t, type_t_hasher>;

class TypeTable
{
    int col_num = 0;

public:
    std::vector<double> tytable;
    // tytable store all the type info during planning
    // struct like
    /* |  0  |  -1  |  -2  |  -3  |
       |count|type..|type..|type..|
       |.....|......|......|......|
    */
    TypeTable() {}
    TypeTable(const TypeTable &table) {
        tytable.assign(table.tytable.begin(), table.tytable.end());
        col_num = table.col_num;

    }
    void set_col_num(int n) { col_num = n; }
    int get_col_num() { return col_num; };
    int get_row_num()
    {
        if (col_num == 0)
            return 0;
        return tytable.size() / col_num;
    }
    double get_row_col(int r, int c)
    {
        return tytable[col_num * r + c];
    }
    void set_row_col(int r, int c, double val)
    {
        tytable[col_num * r + c] = val;
    }
    void append_row_to(int r, std::vector<double> &updated_result_table)
    {
        for (int c = 0; c < col_num; c++)
            updated_result_table.push_back(get_row_col(r, c));
    }
    void append_newv_row_to(int r, std::vector<double> &updated_result_table, double val)
    {
        updated_result_table.push_back(val);
        for (int c = 1; c < col_num; c++)
            updated_result_table.push_back(get_row_col(r, c));
    }
    void print()
    {
        int c = 0;
        int r = 0;
        logstream(LOG_INFO) <<"-- type table --" LOG_endl;
        for(int i = 0;i < tytable.size();i++){
            if(c==col_num){
                logstream(LOG_INFO) << LOG_endl;
                c = 0;
                r++;
                if(r > 5){
                    logstream(LOG_INFO) << "................" << LOG_endl;
                    break;
                }
            }
            logstream(LOG_INFO) << tytable[i] << "\t";
            c++;
        }
        logstream(LOG_INFO) << LOG_endl;
    }

    void merge(int col) {
        // std::cout << "merge: " << col << std::endl;
        // long start = timer::get_usec();
        std::vector<double> tytable_new;
        int row_num = get_row_num();

        std::unordered_map<int, int> hash2col;
        // put type_table to tytable
        for (int i = 0; i < row_num; i++) {
            // test if tytable.contains type_table.row(i)
            size_t hash_key = 17;
            for(int j = 1; j < col_num; j++){
                if(j==col)
                    continue;
                hash_key += get_row_col(i, j) + 17;
            }
            if(hash2col.find(hash_key)!=hash2col.end()){
                int row = hash2col[hash_key];
                tytable_new[row * col_num] += get_row_col(i, 0);
            }else{
                int row = tytable_new.size()/col_num;
                hash2col[hash_key] = row;
                append_row_to(i, tytable_new);
                tytable_new[row*col_num+col] = 0;
            }
        }

        tytable.swap(tytable_new);
        // long end = timer::get_usec();
        // std::cout << "using time: " << ( end -start ) << std::endl;
    }
};

class CostResult
{   
    public:
    double add_cost, all_cost;
    double prune_ratio;
    double init_bind, prune_bind, explore_bind, match_bind, result_bind;
    
    TypeTable typetable;
    std::unordered_map<ssid_t, int> var2col;
    std::unordered_map<ssid_t, std::unordered_map<ssid_t, double>> var2map;
    std::vector<ssid_t> path;
    unsigned int pt_bits;
    model_t current_model;

    bool small_prune;

    CostResult () {}
    CostResult(int i) {
        add_cost = 0; match_bind = 0;
        init_bind = 0; prune_bind = 0;
        explore_bind = 0; result_bind = 0;
        all_cost = 0;
        prune_ratio = 1;
        current_model = model_t::L2U;
        if(i >= 0){
            pt_bits = (0 | (1 << i));
        }else{
            pt_bits = 0;
        }
        small_prune = false;
    }

    void push_path(ssid_t a1, ssid_t a2, ssid_t a3, ssid_t a4){
        path.push_back(a1);
        path.push_back(a2);
        path.push_back(a3);
        path.push_back(a4);
    }

    void init(CostResult &result) {
        typetable = TypeTable(result.typetable);
        var2col = result.var2col;
        path.assign(result.path.begin(), result.path.end());
        all_cost = result.all_cost;
        pt_bits = (result.pt_bits | pt_bits);
        small_prune = result.small_prune;
    }

    void print(){
        logstream(LOG_DEBUG)
            << "init: " << init_bind << " prune: " << prune_bind
            << "  explore: " << explore_bind << "  result: " << result_bind
            << "  add cost: " << add_cost << "  all cost: " << all_cost
            << LOG_endl;
    }

    bool is_known(ssid_t o){
        return (o < 0) &&
               ((var2col.find(o) != var2col.end()) && var2col[o] > 0);
    }

    bool is_dup(ssid_t p, ssid_t o1) { return (path[0] == p) && (path[3] == o1); }

    void update_result(std::vector<double> &updated_result_table) {
        all_cost += add_cost;
        typetable.tytable.swap(updated_result_table);
        updated_result_table.clear();
        typetable.set_col_num(var2col.size() + 1);
    }

    bool operator < (const CostResult & cr) const{
        if(cr.current_model == current_model){
            return add_cost > cr.add_cost;
        } else if (cr.current_model == model_t::K2L) {
            return true;
        } else if (current_model == model_t::K2L){
            return false;
        } else {
            return add_cost > cr.add_cost;
        }
    }
};

} // namespace wukong
