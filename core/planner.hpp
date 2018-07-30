#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string.hpp>
#include <math.h>

#include "data_statistic.hpp"
#include "mymath.hpp"
#include "timer.hpp"

using namespace std;

#define COST_THRESHOLD 1000
#define AA 4
#define BB 2.25
#define CC 1

// for more fine-grained cost model
#define AA_full 0.404
#define AA_early 0.33
#define BB_ifor 0.006
#define CC_const_known 0.123
#define CC_unknown 0.139
#define CACHE_ratio 0 // 0 or 0.537 or 1/1.86

struct plan {
    double cost;           // min cost
    vector<ssid_t> orders; // best orders
    double result_num;     // intermediate results
};

class Type_table {
    int col_num = 0;
    int row_num = 0;
public:
    vector<double> tytable;
    // tytable store all the type info during planning
    // struct like
    /* |  0  |  -1  |  -2  |  -3  |
       |count|type..|type..|type..|
       |.....|......|......|......|
    */
    Type_table() { }
    void set_col_num(int n) { col_num = n; }
    int get_col_num() { return col_num; };
    int get_row_num() {
        if (col_num == 0) return 0;
        return tytable.size() / col_num;
    }
    double get_row_col(int r, int c) {
        return tytable[col_num * r + c];
    }
    void set_row_col(int r, int c, double val) {
        tytable[col_num * r + c] = val;
    }
    void append_row_to(int r, vector<double> &updated_result_table) {
        for (int c = 0; c < col_num; c++)
            updated_result_table.push_back(get_row_col(r, c));
    }
    void append_newv_row_to(int r, vector<double> &updated_result_table, double val) {
        updated_result_table.push_back(val);
        for (int c = 1; c < col_num; c++)
            updated_result_table.push_back(get_row_col(r, c));
    }

};

class Planner {
    // members
    data_statistic *statistic ;
    vector<ssid_t> triples;
    double min_cost;
    vector<ssid_t> path;
    bool is_empty;            // help identify empty queries

    // for dfs
    vector<ssid_t> min_path;
    int _chains_size_div_4 ;

    // remove the attr pattern query before doing the planner and transfer pattern to cmd_chains
    void transfer_to_cmd_chains(vector<SPARQLQuery::Pattern> &p, vector<ssid_t> &attr_pattern, vector<int>& attr_pred_chains, vector<ssid_t> &temp_cmd_chains) {
        for (int i = 0; i < p.size(); i++) {
            SPARQLQuery::Pattern pattern = p[i];
            if (pattern.pred_type == 0) {
                temp_cmd_chains.push_back(pattern.subject);
                temp_cmd_chains.push_back(pattern.predicate);
                temp_cmd_chains.push_back((ssid_t)pattern.direction);
                temp_cmd_chains.push_back(pattern.object);
            }
            else {
                attr_pattern.push_back(pattern.subject);
                attr_pattern.push_back(pattern.predicate);
                attr_pattern.push_back((ssid_t)pattern.direction);
                attr_pattern.push_back(pattern.object);

                attr_pred_chains.push_back(pattern.pred_type);
            }
        }
    }

    // for type-centric method
    Type_table type_table;
    unordered_map<ssid_t, int> var2col;  // convert
    unordered_map<ssid_t, int> var2ptindex;  // find the first appearance for var
    DGraph *graph;    

    // get the type of constant using get_edges_global
    ssid_t get_type(ssid_t constant){
        uint64_t type_sz = 0;
        edge_t *res = graph->get_edges_global(0, constant, OUT, TYPE_ID, &type_sz);
        if(type_sz == 1)    return res[0].val;
        else if(type_sz > 1){
            type_t type;
            unordered_set<int> type_composition;
            for(int i = 0;i < type_sz; i ++){
                type_composition.insert(res[i].val);
            }
            type.set_type_composition(type_composition);
            return statistic->global_type2int[type];
        }
        else{
            type_t type;
            uint64_t psize1 = 0;
            unordered_set<int> index_composition;
            edge_t *res1 = graph->get_edges_global(0, constant, OUT, PREDICATE_ID, &psize1);
            for (uint64_t k = 0; k < psize1; k++) {
                ssid_t pre = res1[k].val;
                index_composition.insert(pre);
            }
            uint64_t psize2 = 0;
            edge_t *res2 = graph->get_edges_global(0, constant, IN, PREDICATE_ID, &psize2);
            for (uint64_t k = 0; k < psize2; k++) {
                ssid_t pre = res2[k].val;
                index_composition.insert(-pre);
            }
            type.set_index_composition(index_composition);
            return statistic->global_type2int[type];
        }
    }

    // type-centric dfs enumeration
    bool plan_enum(unsigned int pt_bits, double cost, double pre_results) {
        if (is_empty) return false;
        if (pt_bits == ( 1 << _chains_size_div_4 ) - 1) {
            //cout << "estimated cost : " << cost << endl;
            //finalpath ++;
            bool ctn = true;
            if (min_cost == std::numeric_limits<double>::max()
                    && cost < COST_THRESHOLD
                    && (path[0] >= (1 << NBITS_IDX)) ) {
                ctn = false;  // small query
                //cout << "small query and use heuristic.\n";
            }
            //cout << "this path cost: " << cost << "-------------------------------------------------------------------" << endl;
            if (cost < min_cost) {
                min_cost = cost;
                min_path = path;
            }
            return ctn;
        }
        for (int pt_pick = 0; pt_pick < _chains_size_div_4; pt_pick++) {
            if ( pt_bits & ( 1 << pt_pick ) )
                continue ;
            int i = 4 * pt_pick;
            double add_cost = 0;
            double prune_ratio = 1;
            double correprune_boost_results = 0;
            double condprune_results = 0;
            vector<double> updated_result_table;
            ssid_t o1 = triples[i];
            ssid_t p = triples[i + 1];
            ssid_t d = triples[i + 2];
            ssid_t o2 = triples[i + 3];
            if (path.size() == 0) {
                if (o1 < 0 && o2 < 0) {
                    //count00++;
                    // use index vertex, find subject first
                    path.push_back(p); path.push_back(0); path.push_back(IN); path.push_back(o1);
                    //cout << "pick : " << p << " " << "0" << " " << IN << " " << o1 << "-------------------------------------" << endl;

                    // initialize
                    add_cost = 0;
                    correprune_boost_results = 0;
                    condprune_results = 0;

                    // selectivity and cost estimation
                    // find all types, push into result table
                    vector<ty_count> tycountv = statistic->global_tystat.pstype[p];
                    for (size_t k = 0; k < tycountv.size(); k++) {
                        ssid_t vtype = tycountv[k].ty;
                        double vcount = double(tycountv[k].count) / global_num_servers;
                        updated_result_table.push_back(vcount);
                        updated_result_table.push_back(vtype);
                    }

                    // calculate cost
                    for (size_t j = 0; j < updated_result_table.size(); j += 2) {
                        correprune_boost_results += updated_result_table[j];
                    }
                    condprune_results = correprune_boost_results;
                    if (condprune_results == 0){
                        is_empty = true;
                        return false;
                    }
                    add_cost = CC_const_known*condprune_results;
                    //cout << "results: " << condprune_results << endl;
                    //cout << "add cost: " << add_cost << endl;
                    double new_cost = cost + add_cost;
                    if (new_cost > min_cost) {
                        path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                        updated_result_table.clear();
                        //cout << "this path dropped " << "-------------------------------------------------------------------" << endl;
                        continue;
                    }

                    // store to type table
                    var2col[o1] = 1;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(2);

                    // next iteration
                    bool ctn = plan_enum(pt_bits, new_cost, condprune_results); // next level
                    //cout << "back : " << p << " " << "0" << " " << IN << " " << o1 << "-------------------------------------" << endl;
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                    var2col[o1] = -1;
                    //cout << "change var2col[o1] to: " << var2col[o1] << endl;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(0);
                    updated_result_table.clear();

                    // different direction, find object first
                    path.push_back(p); path.push_back(0); path.push_back(OUT); path.push_back(o2);
                    //cout << "pick : " << p << " " << "0" << " " << OUT << " " << o2 << "-------------------------------------" << endl;

                    // initialize
                    add_cost = 0;
                    correprune_boost_results = 0;
                    condprune_results = 0;

                    // selectivity and cost estimation
                    // find all types, push into result table
                    tycountv = statistic->global_tystat.potype[p];
                    for (size_t k = 0; k < tycountv.size(); k++) {
                        ssid_t vtype = tycountv[k].ty;
                        double vcount = double(tycountv[k].count) / global_num_servers;
                        updated_result_table.push_back(vcount);
                        updated_result_table.push_back(vtype);
                    }

                    // calculate cost
                    for (size_t j = 0; j < updated_result_table.size(); j += 2) {
                        correprune_boost_results += updated_result_table[j];
                    }
                    condprune_results = correprune_boost_results;
                    if (condprune_results == 0){
                        is_empty = true;
                        return false;
                    }
                    add_cost = CC_const_known*condprune_results;
                    //cout << "results: " << condprune_results << endl;
                    //cout << "add cost: " << add_cost << endl;
                    new_cost = cost + add_cost;
                    if (new_cost > min_cost) {
                        path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                        updated_result_table.clear();
                        //cout << "this path dropped " << "-------------------------------------------------------------------" << endl;
                        continue;
                    }

                    // store to type table
                    var2col[o2] = 1;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(2);

                    // next iteration
                    ctn = plan_enum(pt_bits, new_cost, condprune_results);
                    //cout << "back : " << p << " " << "0" << " " << OUT << " " << o2 << "-------------------------------------" << endl;
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                    var2col[o2] = -1;
                    //cout << "change var2col[o2] to: " << var2col[o2] << endl;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(0);
                    updated_result_table.clear();
                }
                if (o1 > 0) {
                    //count01++;
                    path.push_back(o1); path.push_back(p); path.push_back(d); path.push_back(o2);
                    //cout << "pick : " << o1 << " " << p << " " << d << " " << o2 << "-------------------------------------" << endl;

                    // initialize
                    add_cost = 0;
                    correprune_boost_results = 0;
                    condprune_results = 0;

                    // selectivity and cost estimation
                    // find all types, push into result table

                    // use o1 get_global_edges
                    ssid_t o1type = get_type(o1);

                    int tycount = statistic->global_tystat.get_pstype_count(p, o1type);
                    vector<ty_count> tycountv = statistic->global_tystat.fine_type[make_pair(o1type, p)];
                    for (size_t k = 0; k < tycountv.size(); k++) {
                        ssid_t vtype = tycountv[k].ty;
                        double vcount = double(tycountv[k].count) / tycount;
                        updated_result_table.push_back(vcount);
                        updated_result_table.push_back(vtype);
                    }

                    // calculate cost
                    for (size_t j = 0; j < updated_result_table.size(); j += 2) {
                        correprune_boost_results += updated_result_table[j];
                    }
                    condprune_results = correprune_boost_results;
                    if (condprune_results == 0){
                        is_empty = true;
                        return false;
                    }
                    add_cost = CC_const_known*condprune_results;
                    //cout << "results: " << condprune_results << endl;
                    //cout << "add cost: " << add_cost << endl;
                    double new_cost = cost + add_cost;
                    if (new_cost > min_cost) {
                        path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                        updated_result_table.clear();
                        //cout << "this path dropped " << "-------------------------------------------------------------------" << endl;
                        continue;
                    }

                    // store to type table
                    var2col[o2] = 1;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(2);

                    // next iteration
                    bool ctn = plan_enum(pt_bits | (1 << pt_pick), new_cost, condprune_results);
                    //cout << "back : " << o1 << " " << p << " " << d << " " << o2 << "-------------------------------------" << endl;
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                    var2col[o2] = -1;
                    //cout << "change var2col[o2] to: " << var2col[o2] << endl;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(0);
                    updated_result_table.clear();

                } else if (o2 > 0) {
                    //count01++;
                    path.push_back(o2); path.push_back(p); path.push_back(IN); path.push_back(o1);
                    //cout << "pick : " << o2 << " " << p << " " << IN << " " << o1 << "-------------------------------------" << endl;

                    // initialize
                    add_cost = 0;
                    correprune_boost_results = 0;
                    condprune_results = 0;


                    // selectivity and cost estimation
                    // find all types, push into result table
                    // for type triples
                    if (p == TYPE_ID) {
                        // start from ty-index vertex
                        ssid_t vtype = o2; 
                        if(statistic->global_single2complex.find(vtype) == statistic->global_single2complex.end()){
                            double vcount = double(statistic->global_tyscount[o2]) / global_num_servers;
                            updated_result_table.push_back(vcount);
                            updated_result_table.push_back(vtype);
                        }
                        else{
                            // single type o2 may not exist in muititype situation
                            if(statistic->global_tyscount.find(vtype) != statistic->global_tyscount.end()){
                                double vcount = double(statistic->global_tyscount[o2]) / global_num_servers;
                                updated_result_table.push_back(vcount);
                                updated_result_table.push_back(vtype);
                            }
                            // single type o2 may be contained in complex type
                            unordered_set<ssid_t> type_set = statistic->global_single2complex[vtype];
                            for(auto iter = type_set.cbegin(); iter != type_set.cend(); ++iter){
                                double vcount = double(statistic->global_tyscount[*iter]) / global_num_servers;
                                updated_result_table.push_back(vcount);
                                updated_result_table.push_back(*iter);
                            }
                        }
                    } else { // normal triples
                        // use o2 get_global_edges
                        ssid_t o2type = get_type(o2);

                        int tycount = statistic->global_tystat.get_potype_count(p, o2type);
                        vector<ty_count> tycountv = statistic->global_tystat.fine_type[make_pair(p, o2type)];
                        for (size_t k = 0; k < tycountv.size(); k++) {
                            ssid_t vtype = tycountv[k].ty;
                            double vcount = double(tycountv[k].count) / tycount;
                            updated_result_table.push_back(vcount);
                            updated_result_table.push_back(vtype);
                        }
                    }

                    // calculate cost
                    for (size_t j = 0; j < updated_result_table.size(); j += 2) {
                        correprune_boost_results += updated_result_table[j];
                    }
                    condprune_results = correprune_boost_results;
                    if (condprune_results == 0){
                        is_empty = true;
                        return false;
                    }
                    add_cost = CC_const_known*condprune_results;
                    //cout << "results: " << condprune_results << endl;
                    //cout << "add cost: " << add_cost << endl;
                    double new_cost = cost + add_cost;
                    if (new_cost > min_cost) {
                        path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                        updated_result_table.clear();
                        //cout << "this path dropped " << "-------------------------------------------------------------------" << endl;
                        continue;
                    }

                    // store to type table
                    var2col[o1] = 1;
                    //cout << o1 << "is pushed\n";
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(2);

                    // next iteration and backtrack
                    bool ctn = plan_enum(pt_bits | (1 << pt_pick), new_cost, condprune_results);
                    //cout << "back : " << o2 << " " << p << " " << IN << " " << o1 << "-------------------------------------" << endl;
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                    var2col[o1] = -1;
                    //cout << "change var2col[o1]" << o1 << " to: " << var2col[o1] << endl;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(0);
                    updated_result_table.clear();
                }
            } else {
                // var o1 in the result table
                if (o1 < 0 && (var2col.find(o1) != var2col.end()) && var2col[o1] > 0) {
                    //count02++;
                    //cout << "var2col[o1]:" << var2col[o1] << endl;
                    if (var2col[o1] > type_table.get_col_num()) assert(false);
                    //cout << "var2col[o2]: " << var2col[o2] << endl;
                    //cout << "o1: " << o1 << " var2col[o1]: " << var2col[o1] << endl;
                    path.push_back(o1); path.push_back(p); path.push_back(d); path.push_back(o2);
                    //cout << "pick : " << o1 << " " << p << " " << d << " " << o2 << "-------------------------------------" << endl;

                    // initialize
                    add_cost = 0;
                    correprune_boost_results = 0;
                    condprune_results = 0;
                    double early_ret = 0;
                    double final_ret = 0;


                    // selectivity and cost estimation
                    // find all types, push into result table
                    // prune based on correlation and constant
                    int row_num = type_table.get_row_num();
                    int var_col = var2col[o1];
                    int prune_flag = (o2 > 0) || ((var2col.find(o2) != var2col.end()) && var2col[o2] > 0);
                    int dup_flag = (path[0] == p) && (path[3] == o1);
                    for (size_t i = 0; i < row_num; i++) {
                        ssid_t pre_tyid = type_table.get_row_col(i, var_col);
                        double pre_count = type_table.get_row_col(i, 0);
                        // handle type predicate first
                        if (p == TYPE_ID && o2 > 0) {
                            correprune_boost_results += pre_count;
                            early_ret += pre_count;
                            // if pre_tyid do not belong to o2, prune it
                            if(pre_tyid >= 0){
                                if (pre_tyid != o2) continue;
                            }
                            else{
                                if(statistic->global_single2complex.find(o2) != statistic->global_single2complex.end()){
                                    unordered_set<ssid_t> type_set = statistic->global_single2complex[o2];
                                    if(type_set.count(pre_tyid) == 0) continue;
                                }
                                else continue;
                            }
                            type_table.append_row_to(i, updated_result_table);
                            condprune_results += pre_count;
                            continue;
                        }
                        int tycount = statistic->global_tyscount[pre_tyid];
                        if (dup_flag) tycount = statistic->global_tystat.get_pstype_count(p, pre_tyid);
                        prune_ratio = double(statistic->global_tystat.get_pstype_count(p, pre_tyid)) / tycount; // for cost model
                        double afterprune = pre_count * prune_ratio;
                        early_ret += afterprune;
                        final_ret += pre_count - afterprune;
                        vector<ty_count> tycountv = statistic->global_tystat.fine_type[make_pair(pre_tyid, p)];
                        int match_flag = 0; // for constant & var pruning
                        for (size_t k = 0; k < tycountv.size(); k++) {
                            ssid_t vtype = tycountv[k].ty;
                            double vcount = double(tycountv[k].count) / tycount * pre_count;
                            correprune_boost_results += vcount;
                            if (o2 > 0) { 
                                // for constant pruning
                                ssid_t o2type = get_type(o2);
                                if (vtype == o2type) match_flag = 1;
                            }
                            else if (var2col.find(o2) != var2col.end() && var2col[o2] > 0) { 
                                // for variable pruning
                                ssid_t pretype = type_table.get_row_col(i, var2col[o2]);
                                if (vtype == pretype) {
                                    int type_num = statistic->global_tyscount[pretype];
                                    vcount = vcount / type_num;
                                    type_table.append_newv_row_to(i, updated_result_table, vcount);
                                    condprune_results += vcount;
                                }
                            } else {
                                // normal case
                                type_table.append_newv_row_to(i, updated_result_table, vcount); 
                                updated_result_table.push_back(vtype);
                                condprune_results += vcount;
                            }
                        }
                        if (prune_flag && match_flag) {
                            type_table.append_row_to(i, updated_result_table);
                            condprune_results += pre_count;
                        }
                    }

                    if (condprune_results == 0){
                        is_empty = true;
                        return false;
                    }
                    // calculate cost
                    add_cost = AA_full*final_ret + AA_early*early_ret
                              + BB_ifor * correprune_boost_results
                              + (prune_flag ? CC_const_known : CC_unknown) * condprune_results;
                    //cout << "results: " << condprune_results << endl;
                    //cout << "add cost: " << add_cost << endl;
                    double new_cost = cost + add_cost;
                    if (new_cost > min_cost) {
                        path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                        updated_result_table.clear();
                        //cout << "this path dropped " << "-------------------------------------------------------------------" << endl;
                        continue;
                    }

                    // store to type table
                    type_table.tytable.swap(updated_result_table);
                    if (!prune_flag) {
                        var2col[o2] = type_table.get_col_num();
                        //cout << o2 << "is pushed\n";
                        //cout << "in ! prune flag o2: " << o2 << " var2col[o2]: " << var2col[o2] << endl;
                        type_table.set_col_num(type_table.get_col_num() + 1);
                    }

                    // next iteration
                    bool ctn = plan_enum(pt_bits | (1 << pt_pick), new_cost, condprune_results);
                    //cout << "back : " << o1 << " " << p << " " << d << " " << o2 << "-------------------------------------" << endl;
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                    type_table.tytable.swap(updated_result_table);
                    if (!prune_flag) {
                        //cout << "out !prune_flag\n";
                        var2col[o2] = -1;
                        //cout << "change var2col[o2] " << o2 << " to: " << var2col[o2] << endl;
                        type_table.set_col_num(type_table.get_col_num() - 1);
                    }
                    updated_result_table.clear();
                }
                // var o2 in the result table
                if (o2 < 0 && (var2col.find(o2) != var2col.end()) && var2col[o2] > 0) {
                  //count02++;
                    if (var2col[o2] > type_table.get_col_num()) assert(false);
                    //cout << "o2: " << o2 << " var2col[o2]: " << var2col[o2] << endl;
                    path.push_back(o2); path.push_back(p); path.push_back(IN); path.push_back(o1);
                    //cout << "pick : " << o2 << " " << p << " " << IN << " " << o1 << "-------------------------------------" << endl;

                    // initialize
                    add_cost = 0;
                    correprune_boost_results = 0;   //get edges number
                    condprune_results = 0;          //push back number
                    double early_ret = 0;
                    double final_ret = 0;


                    // selectivity and cost estimation
                    // find all types, push into result table
                    // prune based on correlation and constant
                    int row_num = type_table.get_row_num();
                    int var_col = var2col[o2];
                    int prune_flag = (o1 > 0) || ((var2col.find(o1) != var2col.end()) && var2col[o1] > 0);
                    int dup_flag = (path[0] == p) && (path[3] == o2);
                    for (size_t i = 0; i < row_num; i++) {
                        ssid_t pre_tyid = type_table.get_row_col(i, var_col);
                        double pre_count = type_table.get_row_col(i, 0);
                        int tycount = statistic->global_tyscount[pre_tyid];
                        if (dup_flag) tycount = statistic->global_tystat.get_potype_count(p, pre_tyid);
                        prune_ratio = double(statistic->global_tystat.get_potype_count(p, pre_tyid)) / tycount; // for cost model
                        double afterprune = pre_count * prune_ratio;
                        early_ret += afterprune;
                        final_ret += pre_count - afterprune;
                        vector<ty_count> tycountv = statistic->global_tystat.fine_type[make_pair(p, pre_tyid)];
                        int match_flag = 0; // for constant & var pruning
                        for (size_t k = 0; k < tycountv.size(); k++) {
                            ssid_t vtype = tycountv[k].ty;
                            double vcount = double(tycountv[k].count) / tycount * pre_count;
                            correprune_boost_results += vcount;
                            if (o1 > 0) { 
                                // for constant pruning
                                ssid_t o1type = get_type(o1);
                                if (vtype == o1type) match_flag = 1;
                            }
                            else if (var2col.find(o1) != var2col.end() && var2col[o1] > 0) { 
                                // for variable pruning
                                ssid_t pretype = type_table.get_row_col(i, var2col[o1]);
                                if (vtype == pretype) {
                                    int type_num = statistic->global_tyscount[pretype];
                                    vcount = vcount / type_num;
                                    type_table.append_newv_row_to(i, updated_result_table, vcount);
                                    condprune_results += vcount;
                                }
                            } else {
                                // normal case
                                type_table.append_newv_row_to(i, updated_result_table, vcount); 
                                updated_result_table.push_back(vtype);
                                condprune_results += vcount;
                            }
                        }
                        if (prune_flag && match_flag) {
                            type_table.append_row_to(i, updated_result_table);
                            condprune_results += pre_count;
                        }
                    }

                    if (condprune_results == 0){
                        is_empty = true;
                        return false;
                    }
                    // calculate cost
                    add_cost = AA_full*final_ret + AA_early*early_ret
                              + BB_ifor * correprune_boost_results
                              + (prune_flag ? CC_const_known : CC_unknown) * condprune_results;
                    //cout << "results: " << condprune_results << endl;
                    //cout << "add cost: " << add_cost << endl;
                    double new_cost = cost + add_cost;
                    if (new_cost > min_cost) {
                        path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                        updated_result_table.clear();
                        //cout << "this path dropped " << "-------------------------------------------------------------------" << endl;
                        continue;
                    }

                    // store to type table
                    type_table.tytable.swap(updated_result_table);
                    if (!prune_flag) {
                        var2col[o1] = type_table.get_col_num();
                        //cout << o1 << "is pushed\n";
                        //cout << "in prune flag o1: " << o1 << " var2col[o1]: " << var2col[o1] << endl;
                        type_table.set_col_num(type_table.get_col_num() + 1);
                    }

                    // next iteration
                    bool ctn = plan_enum(pt_bits | (1 << pt_pick), new_cost, condprune_results);
                    //cout << "back : " << o2 << " " << p << " " << IN << " " << o1 << "-------------------------------------" << endl;
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                    type_table.tytable.swap(updated_result_table);
                    if (!prune_flag) {
                        //cout << "out prune_flag\n";
                        var2col[o1] = -1;
                        //cout << "change var2col[o1] " << o1 << " to: " << var2col[o1] << endl;
                        type_table.set_col_num(type_table.get_col_num() - 1);
                    }
                    updated_result_table.clear();
                }
            }
        }

        return true;
    }

    // for debug single order
    bool score_order_new(unsigned int index, double cost, double pre_results) {
        if (index == _chains_size_div_4 * 4) {
            cout << "this order's cost : " << cost << endl;
            cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
            if (cost < min_cost) {
                min_cost = cost;
                min_path = path;
            }
            return true;
        }
        double add_cost = 0;
        double prune_ratio = 0;
        double correprune_boost_results = 0;
        double condprune_results = 0;
        vector<double> updated_result_table;
        ssid_t src = triples[index];
        ssid_t pred = triples[index + 1];
        ssid_t dir = triples[index + 2];
        ssid_t tgt = triples[index + 3];
        if (path.size() == 0) {
            if (pred == 0) {
                if (dir == IN) {
                    ssid_t p = src;
                    ssid_t o1 = tgt;
                    // use index vertex, find subject first
                    path.push_back(p); path.push_back(0); path.push_back(IN); path.push_back(o1);
                    cout << "pick : " << p << " " << "0" << " " << IN << " " << o1 << "-------------------------------------" << endl;

                    // initialize
                    add_cost = 0;
                    correprune_boost_results = 0;
                    condprune_results = 0;


                    // selectivity and cost estimation
                    // find all types, push into result table
                    vector<ty_count> tycountv = statistic->global_tystat.pstype[p];
                    for (size_t k = 0; k < tycountv.size(); k++) {
                        ssid_t vtype = tycountv[k].ty;
                        double vcount = double(tycountv[k].count) / global_num_servers;
                        updated_result_table.push_back(vcount);
                        updated_result_table.push_back(vtype);
                    }

                    // calculate cost
                    for (size_t j = 0; j < updated_result_table.size(); j += 2) {
                        correprune_boost_results += updated_result_table[j];
                    }
                    condprune_results = correprune_boost_results;
                    add_cost = CC_const_known*condprune_results;
                    cout << "results: " << condprune_results << endl;
                    cout << "add cost: " << add_cost << "== " << CC_const_known << " * " << condprune_results << endl;
                    double new_cost = cost + add_cost;

                    // store to type table
                    var2col[o1] = 1;
                    var2ptindex[o1] = 3;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(2);

                    // next iteration
                    bool ctn = score_order_new(index+4, new_cost, condprune_results); // next level
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                    var2col[o1] = -1;
                    var2ptindex[o1] = -1;
                    //cout << "change var2col[o1] to: " << var2col[o1] << endl;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(0);
                    updated_result_table.clear();
                } else if (dir == OUT) {
                    ssid_t p = src;
                    ssid_t o2 = tgt;
                    // different direction, find object first
                    path.push_back(p); path.push_back(0); path.push_back(OUT); path.push_back(o2);
                    cout << "pick : " << p << " " << "0" << " " << OUT << " " << o2 << "-------------------------------------" << endl;

                    // initialize
                    add_cost = 0;
                    correprune_boost_results = 0;
                    condprune_results = 0;


                    // selectivity and cost estimation
                    // find all types, push into result table
                    vector<ty_count> tycountv = statistic->global_tystat.potype[p];
                    for (size_t k = 0; k < tycountv.size(); k++) {
                        ssid_t vtype = tycountv[k].ty;
                        double vcount = double(tycountv[k].count) / global_num_servers;
                        updated_result_table.push_back(vcount);
                        updated_result_table.push_back(vtype);
                    }

                    // calculate cost
                    for (size_t j = 0; j < updated_result_table.size(); j += 2) {
                        correprune_boost_results += updated_result_table[j];
                    }
                    condprune_results = correprune_boost_results;
                    add_cost = CC_const_known*condprune_results;
                    cout << "results: " << condprune_results << endl;
                    cout << "add cost: " << add_cost << "== " << CC_const_known << " * " << condprune_results << endl;
                    double new_cost = cost + add_cost;

                    // store to type table
                    var2col[o2] = 1;
                    var2ptindex[o2] = 3;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(2);

                    // next iteration
                    bool ctn = score_order_new(index+4, new_cost, condprune_results);
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                    var2col[o2] = -1;
                    var2ptindex[o2] = -1;
                    //cout << "change var2col[o2] to: " << var2col[o2] << endl;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(0);
                    updated_result_table.clear();

                }
            }
            else if (src > 0) {
                if (dir == OUT) {
                    ssid_t o1 = src;
                    ssid_t p = pred;
                    ssid_t d = dir;
                    ssid_t o2 = tgt;

                    path.push_back(o1); path.push_back(p); path.push_back(d); path.push_back(o2);
                    cout << "pick : " << o1 << " " << p << " " << d << " " << o2 << "-------------------------------------" << endl;

                    // initialize
                    add_cost = 0;
                    correprune_boost_results = 0;
                    condprune_results = 0;


                    // selectivity and cost estimation
                    // find all types, push into result table
                    ssid_t o1type = get_type(o1);
                    int tycount = statistic->global_tystat.get_pstype_count(p, o1type);
                    vector<ty_count> tycountv = statistic->global_tystat.fine_type[make_pair(o1type, p)];
                    for (size_t k = 0; k < tycountv.size(); k++) {
                        ssid_t vtype = tycountv[k].ty;
                        double vcount = double(tycountv[k].count) / tycount;
                        updated_result_table.push_back(vcount);
                        updated_result_table.push_back(vtype);
                    }

                    // calculate cost
                    for (size_t j = 0; j < updated_result_table.size(); j += 2) {
                        correprune_boost_results += updated_result_table[j];
                    }
                    condprune_results = correprune_boost_results;
                    add_cost = CC_const_known*condprune_results;
                    cout << "results: " << condprune_results << endl;
                    cout << "add cost: " << add_cost << "== " << CC_const_known << " * " << condprune_results << endl;
                    double new_cost = cost + add_cost;

                    // store to type table
                    var2col[o2] = 1;
                    var2ptindex[o2] = 3;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(2);

                    // next iteration
                    bool ctn = score_order_new(index+4, new_cost, condprune_results);
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                    var2col[o2] = -1;
                    var2ptindex[o2] = -1;
                    //cout << "change var2col[o2] to: " << var2col[o2] << endl;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(0);
                    updated_result_table.clear();
                } else if (dir == IN) {
                    ssid_t o2 = src;
                    ssid_t p = pred;
                    ssid_t d = dir;
                    ssid_t o1 = tgt;

                    path.push_back(o2); path.push_back(p); path.push_back(IN); path.push_back(o1);
                    cout << "pick : " << o2 << " " << p << " " << IN << " " << o1 << "-------------------------------------" << endl;

                    // initialize
                    add_cost = 0;
                    correprune_boost_results = 0;
                    condprune_results = 0;


                    // selectivity and cost estimation
                    // find all types, push into result table
                    // for type triples
                    if (p == TYPE_ID) {
                        // start from ty-index vertex
                        ssid_t vtype = o2; 
                        if(statistic->global_single2complex.find(vtype) == statistic->global_single2complex.end()){
                            double vcount = double(statistic->global_tyscount[o2]) / global_num_servers;
                            updated_result_table.push_back(vcount);
                            updated_result_table.push_back(vtype);
                        }
                        else{
                            // single type o2 may not exist in muititype situation
                            if(statistic->global_tyscount.find(vtype) != statistic->global_tyscount.end()){
                                double vcount = double(statistic->global_tyscount[o2]) / global_num_servers;
                                updated_result_table.push_back(vcount);
                                updated_result_table.push_back(vtype);
                            }
                            // single type o2 may be contained in complex type
                            unordered_set<ssid_t> type_set = statistic->global_single2complex[vtype];
                            for(auto iter = type_set.cbegin(); iter != type_set.cend(); ++iter){
                                double vcount = double(statistic->global_tyscount[*iter]) / global_num_servers;
                                updated_result_table.push_back(vcount);
                                updated_result_table.push_back(*iter);
                            }
                        }

                    } else { // normal triples
                        ssid_t o2type = get_type(o2);
                        int tycount = statistic->global_tystat.get_potype_count(p, o2type);
                        vector<ty_count> tycountv = statistic->global_tystat.fine_type[make_pair(p, o2type)];
                        for (size_t k = 0; k < tycountv.size(); k++) {
                            ssid_t vtype = tycountv[k].ty;
                            double vcount = double(tycountv[k].count) / tycount;
                            updated_result_table.push_back(vcount);
                            updated_result_table.push_back(vtype);
                        }
                    }

                    // calculate cost
                    for (size_t j = 0; j < updated_result_table.size(); j += 2) {
                        correprune_boost_results += updated_result_table[j];
                    }
                    condprune_results = correprune_boost_results;
                    add_cost = CC_const_known*condprune_results;
                    cout << "results: " << condprune_results << endl;
                    cout << "add cost: " << add_cost << "== " << CC_const_known << " * " << condprune_results << endl;
                    double new_cost = cost + add_cost;

                    // store to type table
                    var2col[o1] = 1;
                    var2ptindex[o1] = 3;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(2);

                    // next iteration and backtrack
                    bool ctn = score_order_new(index+4, new_cost, condprune_results);
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                    var2col[o1] = -1;
                    var2ptindex[o1] = -1;
                    //cout << "change var2col[o1]" << o1 << " to: " << var2col[o1] << endl;
                    type_table.tytable.swap(updated_result_table);
                    type_table.set_col_num(0);
                    updated_result_table.clear();
                }
            }
        } else {
            if (dir == OUT) {
                ssid_t o1 = src;
                ssid_t p = pred;
                ssid_t d = dir;
                ssid_t o2 = tgt;

                path.push_back(o1); path.push_back(p); path.push_back(d); path.push_back(o2);
                cout << "pick : " << o1 << " " << p << " " << d << " " << o2 << "-------------------------------------" << endl;

                // initialize
                add_cost = 0;
                correprune_boost_results = 0;
                condprune_results = 0;
                double early_ret = 0;
                double final_ret = 0;


                // selectivity and cost estimation
                // find all types, push into result table
                // prune based on correlation and constant
                int row_num = type_table.get_row_num();
                int var_col = var2col[o1];
                int prune_flag = (o2 > 0) || ((var2col.find(o2) != var2col.end()) && var2col[o2] > 0);
                int dup_flag = (path[0] == p) && (path[3] == o1);
                for (size_t i = 0; i < row_num; i++) {
                    ssid_t pre_tyid = type_table.get_row_col(i, var_col);
                    double pre_count = type_table.get_row_col(i, 0);
                    // handle type predicate first
                    if (p == TYPE_ID && o2 > 0) {
                        correprune_boost_results += pre_count;
                        early_ret += pre_count;

                        // if pre_tyid do not belong to o2, prune it
                        if(pre_tyid >= 0){
                            if (pre_tyid != o2) continue;
                        }
                        else{
                            if(statistic->global_single2complex.find(o2) != statistic->global_single2complex.end()){
                                unordered_set<ssid_t> type_set = statistic->global_single2complex[o2];
                                if(type_set.count(pre_tyid) == 0) continue;
                            }
                            else continue;
                        }

                        type_table.append_row_to(i, updated_result_table);
                        condprune_results += pre_count;
                        continue;
                    }
                    int tycount = statistic->global_tyscount[pre_tyid];
                    if (dup_flag) tycount = statistic->global_tystat.get_pstype_count(p, pre_tyid);
                    prune_ratio = double(statistic->global_tystat.get_pstype_count(p, pre_tyid)) / tycount; // for cost model
                    assert(prune_ratio <= 1);
                    double afterprune = pre_count * prune_ratio;
                    early_ret += afterprune;
                    final_ret += pre_count - afterprune;
                    vector<ty_count> tycountv = statistic->global_tystat.fine_type[make_pair(pre_tyid, p)];
                    int match_flag = 0; // for constant & var pruning
                    for (size_t k = 0; k < tycountv.size(); k++) {
                        ssid_t vtype = tycountv[k].ty;
                        double vcount = double(tycountv[k].count) / tycount * pre_count;
                        correprune_boost_results += vcount;
                        if (o2 > 0) { 
                            // for constant pruning
                            ssid_t o2type = get_type(o2);
                            if (vtype == o2type) match_flag = 1;
                        }
                        else if (var2col.find(o2) != var2col.end() && var2col[o2] > 0) { 
                            // for variable pruning
                            ssid_t pretype = type_table.get_row_col(i, var2col[o2]);
                            if (vtype == pretype) {
                                int type_num = statistic->global_tyscount[pretype];
                                vcount = vcount / type_num;
                                type_table.append_newv_row_to(i, updated_result_table, vcount); 
                                condprune_results += vcount;
                            }
                        } else {
                            // normal case
                            type_table.append_newv_row_to(i, updated_result_table, vcount); 
                            updated_result_table.push_back(vtype);
                            condprune_results += vcount;
                        }
                    }
                    if (prune_flag && match_flag) {
                        type_table.append_row_to(i, updated_result_table);
                        condprune_results += pre_count;
                    }
                }

                // calculate cost
                cout << "results: " << condprune_results << endl;

                // calculate cache hit and miss rate
                if (var2ptindex.find(o1) == var2ptindex.end()) assert(false);
                // size_t fstindex = var2ptindex[o1]+1;
                // double seq_dup = 1;
                // for (; fstindex < path.size()-4; fstindex += 4) {
                //     ssid_t tmpp = path[fstindex+1];
                //     ssid_t tmpd = path[fstindex+2];
                //     double dup_ratio = 1;
                //     if (tmpd)
                //         dup_ratio = double(statistic->global_ptcount[tmpp]) / statistic->global_pscount[tmpp];
                //     else 
                //         dup_ratio = double(statistic->global_ptcount[tmpp]) / statistic->global_pocount[tmpp];
                //     seq_dup *= dup_ratio;
                // }
                // double miss_ratio = 1;
                // double hit_ratio = 0;
                // if (seq_dup > 1) {
                //     miss_ratio = 1 / seq_dup;
                //     hit_ratio = (seq_dup - 1) / seq_dup;
                // }

                // double time_get_edges = AA_full*final_ret + AA_early*early_ret;
                // double time_ifor = BB_ifor * correprune_boost_results;
                // double time_push = (prune_flag ? CC_const_known : CC_unknown) * condprune_results;
                // add_cost = time_get_edges * miss_ratio + time_get_edges * CACHE_ratio * hit_ratio
                //     + time_ifor * miss_ratio + time_ifor * CACHE_ratio * hit_ratio + time_push;

                add_cost = AA_full*final_ret + AA_early*early_ret
                 + BB_ifor * correprune_boost_results
                 + (prune_flag ? CC_const_known : CC_unknown) * condprune_results;
                cout << "add_cost: " << add_cost << " == "
                 << AA_full << "*" << final_ret << " + " << AA_early << "*" << early_ret
                 << " + " << BB_ifor << "*" << correprune_boost_results << " + "
                 << (prune_flag ? CC_const_known : CC_unknown) << "*" << condprune_results << endl;

                // cout << "three time(w/o cache): " << time_get_edges << "  " << time_ifor << "  " << time_push << endl;
                // cout << "seq_dup: " << seq_dup << endl;
                // cout << "add_cost: " << add_cost << " == " << time_get_edges * miss_ratio + time_get_edges * CACHE_ratio * hit_ratio
                //   << " + " << time_ifor * miss_ratio + time_ifor * CACHE_ratio * hit_ratio
                //   << " + " << time_push << endl;
                // cout << "add_cost(w/o cache): " << add_cost << " == "
                //   << AA_full << "*" << final_ret << " + " << AA_early << "*" << early_ret
                //   << " + " << BB_ifor << "*" << correprune_boost_results << " + "
                //   << (prune_flag ? CC_const_known : CC_unknown) << "*" << condprune_results << endl;


                double new_cost = cost + add_cost;

                // store to type table
                type_table.tytable.swap(updated_result_table);
                if (!prune_flag) {
                    var2col[o2] = type_table.get_col_num();
                    var2ptindex[o2] = path.size() - 1;
                    //cout << "in ! prune flag o2: " << o2 << " var2col[o2]: " << var2col[o2] << endl;
                    type_table.set_col_num(type_table.get_col_num() + 1);
                }

                // next iteration
                bool ctn = score_order_new(index+4, new_cost, condprune_results);
                if (!ctn) return ctn;
                path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                type_table.tytable.swap(updated_result_table);
                if (!prune_flag) {
                    var2col[o2] = -1;
                    var2ptindex[o2] = -1;
                    type_table.set_col_num(type_table.get_col_num() - 1);
                }
                updated_result_table.clear();
            } else if (dir == IN) {
                ssid_t o2 = src;
                ssid_t p = pred;
                ssid_t d = dir;
                ssid_t o1 = tgt;
                path.push_back(o2); path.push_back(p); path.push_back(IN); path.push_back(o1);
                cout << "pick : " << o2 << " " << p << " " << IN << " " << o1 << "-------------------------------------" << endl;

                // initialize
                add_cost = 0;
                correprune_boost_results = 0;
                condprune_results = 0;
                double early_ret = 0;
                double final_ret = 0;


                // selectivity and cost estimation
                // find all types, push into result table
                // prune based on correlation and constant
                int row_num = type_table.get_row_num();
                //cout << "row_num: " << row_num << endl;
                int var_col = var2col[o2];
                int prune_flag = (o1 > 0) || ((var2col.find(o1) != var2col.end()) && var2col[o1] > 0);
                int dup_flag = (path[0] == p) && (path[3] == o2);
                for (size_t i = 0; i < row_num; i++) {
                    ssid_t pre_tyid = type_table.get_row_col(i, var_col);
                    double pre_count = type_table.get_row_col(i, 0);
                    int tycount = statistic->global_tyscount[pre_tyid];
                    if (dup_flag) tycount = statistic->global_tystat.get_potype_count(p, pre_tyid);
                    prune_ratio = double(statistic->global_tystat.get_potype_count(p, pre_tyid)) / tycount; // for cost model
                    double afterprune = pre_count * prune_ratio;
                    early_ret += afterprune;
                    final_ret += pre_count - afterprune;
                    vector<ty_count> tycountv = statistic->global_tystat.fine_type[make_pair(p, pre_tyid)];
                    int match_flag = 0; // for constant & var pruning
                    for (size_t k = 0; k < tycountv.size(); k++) {
                        ssid_t vtype = tycountv[k].ty;
                        double vcount = double(tycountv[k].count) / tycount * pre_count;
                        correprune_boost_results += vcount;
                        if (o1 > 0) { 
                            // for constant pruning
                            ssid_t o1type = get_type(o1);
                            if (vtype == o1type) match_flag = 1;
                        }
                        else if (var2col.find(o1) != var2col.end() && var2col[o1] > 0) { 
                            // for variable pruning
                            ssid_t pretype = type_table.get_row_col(i, var2col[o1]);
                            if (vtype == pretype) {
                                int type_num = statistic->global_tyscount[pretype];
                                vcount = vcount / type_num;
                                type_table.append_newv_row_to(i, updated_result_table, vcount); 
                                condprune_results += vcount;
                            }
                        } else {
                            // normal case
                            type_table.append_newv_row_to(i, updated_result_table, vcount); 
                            updated_result_table.push_back(vtype);
                            condprune_results += vcount;
                        }
                    }
                    if (prune_flag && match_flag) {
                        //cout << "row " << i << " match, count: " << pre_count << endl;
                        type_table.append_row_to(i, updated_result_table);
                        condprune_results += pre_count;
                    }
                }

                // calculate cost
                cout << "results: " << condprune_results << endl;

                // calculate cache hit and miss rate
                if (var2ptindex.find(o2) == var2ptindex.end()) assert(false);
                // size_t fstindex = var2ptindex[o2]+1;
                // double seq_dup = 1;
                // for (; fstindex < path.size()-4; fstindex += 4) {
                //     ssid_t tmpp = path[fstindex+1];
                //     ssid_t tmpd = path[fstindex+2];
                //     double dup_ratio = 1;
                //     if (tmpd)
                //         dup_ratio = double(statistic->global_ptcount[tmpp]) / statistic->global_pscount[tmpp];
                //     else 
                //         dup_ratio = double(statistic->global_ptcount[tmpp]) / statistic->global_pocount[tmpp];
                //     seq_dup *= dup_ratio;
                // }
                // double miss_ratio = 1;
                // double hit_ratio = 0;
                // if (seq_dup > 1) {
                //     miss_ratio = 1 / seq_dup;
                //     hit_ratio = (seq_dup - 1) / seq_dup;
                // }

                // double time_get_edges = AA_full*final_ret + AA_early*early_ret;
                // double time_ifor = BB_ifor * correprune_boost_results;
                // double time_push = (prune_flag ? CC_const_known : CC_unknown) * condprune_results;
                // add_cost = time_get_edges * miss_ratio + time_get_edges * CACHE_ratio * hit_ratio
                //     + time_ifor * miss_ratio + time_ifor * CACHE_ratio * hit_ratio + time_push;

                add_cost = AA_full*final_ret + AA_early*early_ret
                 + BB_ifor * correprune_boost_results
                 + (prune_flag ? CC_const_known : CC_unknown) * condprune_results;
                cout << "add_cost: " << add_cost << " == " 
                 << AA_full << "*" << final_ret << " + " << AA_early << "*" << early_ret
                 << " + " << BB_ifor << "*" << correprune_boost_results << " + "
                 << (prune_flag ? CC_const_known : CC_unknown) << "*" << condprune_results << endl;

                // cout << "seq_dup: " << seq_dup << endl;
                // cout << "three time(w/o cache): " << time_get_edges << "  " << time_ifor << "  " << time_push << endl;
                // cout << "add_cost: " << add_cost << " == " << time_get_edges * miss_ratio + time_get_edges * CACHE_ratio * hit_ratio
                //   << " + " << time_ifor * miss_ratio + time_ifor * CACHE_ratio * hit_ratio
                //   << " + " << time_push << endl;
                // cout << "add_cost(w/o cache): " << add_cost << " == "
                //   << AA_full << "*" << final_ret << " + " << AA_early << "*" << early_ret
                //   << " + " << BB_ifor << "*" << correprune_boost_results << " + "
                //   << (prune_flag ? CC_const_known : CC_unknown) << "*" << condprune_results << endl;


                double new_cost = cost + add_cost;

                // store to type table
                type_table.tytable.swap(updated_result_table);
                if (!prune_flag) {
                    var2col[o1] = type_table.get_col_num();
                    var2ptindex[o1] = path.size() - 1;
                    type_table.set_col_num(type_table.get_col_num() + 1);
                }

                // next iteration
                bool ctn = score_order_new(index+4, new_cost, condprune_results);
                if (!ctn) return ctn;
                path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                type_table.tytable.swap(updated_result_table);
                if (!prune_flag) {
                    var2col[o1] = -1;
                    var2ptindex[o1] = -1;
                    type_table.set_col_num(type_table.get_col_num() - 1);
                }
                updated_result_table.clear();
            }
        }
        return true;
    }

public:
    Planner() { }
    Planner(DGraph *graph):graph(graph) { }

    bool generate_for_patterns(vector<SPARQLQuery::Pattern> &patterns){
        
        //input : patterns
        //transform to : _chains_size_div_4, triples, temp_cmd_chains
        vector<ssid_t> temp_cmd_chains;
        vector<ssid_t> attr_pattern;
        vector<int> attr_pred_chains;
        transfer_to_cmd_chains(patterns, attr_pattern, attr_pred_chains, temp_cmd_chains);

        if(temp_cmd_chains.size() == 0){
            if(attr_pattern.size() == 0) return false;
            else return true;
        }

        this->statistic = statistic;
        min_path.clear();
        path.clear();
        type_table.set_col_num(0);
        var2col.clear();
        var2ptindex.clear();
        is_empty = false;
        min_cost = std::numeric_limits<double>::max();

        // prepare for heuristic
        for (int i = 0, ilimit = temp_cmd_chains.size(); i < ilimit; i = i + 4) {
            if (temp_cmd_chains[i] >= (1 << NBITS_IDX) || temp_cmd_chains[i + 3] >= (1 << NBITS_IDX)) {
                if (i == 0) break;
                int ta, tb, tc, td;
                ta = temp_cmd_chains[i];
                tb = temp_cmd_chains[i + 1];
                tc = temp_cmd_chains[i + 2];
                td = temp_cmd_chains[i + 3];
                temp_cmd_chains[i] = temp_cmd_chains[0];
                temp_cmd_chains[i + 1] = temp_cmd_chains[1];
                temp_cmd_chains[i + 2] = temp_cmd_chains[2];
                temp_cmd_chains[i + 3] = temp_cmd_chains[3];
                temp_cmd_chains[0] = ta;
                temp_cmd_chains[1] = tb;
                temp_cmd_chains[2] = tc;
                temp_cmd_chains[3] = td;
                break;
            }
        }

        this->triples = temp_cmd_chains;
        _chains_size_div_4 = temp_cmd_chains.size() / 4 ;
        plan_enum(0, 0, 0); // the traverse function

        if (is_empty == true) {
            cout << "identified empty result query." << endl;
            cout << "query planning is finished." << endl;
            return false;
        }

        // convert
        boost::unordered_map<int, int> convert;
        for (int i = 0, ilimit = min_path.size(); i < ilimit; i++) {
            if (min_path[i] < 0 ) {
                if (convert.find(min_path[i]) == convert.end()) {
                    int value =  -1 - convert.size();
                    convert[min_path[i]] = value;
                    min_path[i] = value;
                } else {
                    min_path[i] = convert[min_path[i]];
                }
            }
        }

        // for (int i = 0, ilimit = min_path.size(); i < ilimit; i = i + 4)
        //   cout << "min_path " << " : " << min_path[i] << " "
        //     << min_path[i+1] << " "
        //     << min_path[i+2] << " "
        //     << min_path[i+3] << endl;

        // debug single order
        // triples = min_path;
        // min_path.clear();
        // path.clear();
        // type_table.set_col_num(0);
        // var2col.clear();
        // var2ptindex.clear();
        // is_empty = false;
        // min_cost = std::numeric_limits<double>::max();
        // _chains_size_div_4 = triples.size() / 4;
        // score_order_new(0,0,0);
        // min_path = triples;

        // output: min_path
        // transform min_path to patterns
        patterns.clear();
        for (int i = 0; i < min_path.size() / 4; i ++) {
            SPARQLQuery::Pattern pattern(
                min_path[4 * i],
                min_path[4 * i + 1],
                min_path[4 * i + 2],
                min_path[4 * i + 3]
            );
            pattern.pred_type = 0;
            patterns.push_back(pattern);
        }
        //add_attr_pattern to the end of patterns
        for (int i = 0 ; i < attr_pred_chains.size(); i ++) {
            SPARQLQuery::Pattern pattern(
                attr_pattern[4 * i],
                attr_pattern[4 * i + 1],
                attr_pattern[4 * i + 2],
                attr_pattern[4 * i + 3]
            );
            pattern.pred_type = attr_pred_chains[i];
            patterns.push_back(pattern);
        }

        return true;

    }

    bool generate_for_group(SPARQLQuery::PatternGroup &group) {
        bool success = true;
        if (group.patterns.size() > 0)
            success = generate_for_patterns(group.patterns);
        for (auto &g : group.unions)
            success = generate_for_group(g);
        return success;
    }

    bool generate_plan(SPARQLQuery &r, data_statistic *statistic) {
        this->statistic = statistic;
        return generate_for_group(r.pattern_group);
    }
};
