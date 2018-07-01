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

//#define COST_THRESHOLD 350

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

struct select_record {
    ssid_t p;
    ssid_t d;
    double v;
    bool operator > (const select_record &o) const {
        return v > o.v;
    }
    bool operator < (const select_record &o) const {
        return v < o.v;
    }
    bool operator == (const select_record &o) const {
        return v == o.v;
    }
};

template <class T>
class Minimum_maintenance {
private:
    T * _stack[2]; // 0 store Descending sequence, 1 store other's
    int _iter[2];    // store iter
    int * _order;    // for push and pop
    int * _fakedeepcopy ;
public:
    Minimum_maintenance() {
        logstream(LOG_ERROR) << "NOT SUPPORT" << LOG_endl;
        ASSERT(0);
    }
    Minimum_maintenance(int s) {
        if (s < 1) //protect
            s = 1;
        _iter[0] = 0;
        _iter[1] = 0;
        _stack[0] = new T[s];
        _stack[1] = new T[s];
        _order = new int[s];
        //cout<<"NEW    [] +++++++++++++++++++"<<endl;
    }
    Minimum_maintenance(const Minimum_maintenance& copy) {
        _stack[0]   = copy._stack[0];
        _stack[1]   = copy._stack[1];
        _iter [0]   = copy._iter [0];
        _iter [1]   = copy._iter [1];
        _order      = copy._order;
    }
    ~Minimum_maintenance() {
        delete [] _stack[0];
        delete [] _stack[1];
        delete [] _order;
        //cout<<"DELETE [] -------------------"<<endl;
    }
    void push(const T & ms) {
        int i;
        if (_iter[0] == 0 || _stack[0][_iter[0] - 1] > ms) {
            i = 0;
        } else {
            i = 1;
        }
        _stack[i][_iter[i]] = ms;

        _order[_iter[0] + _iter[1]] = i;
        _iter[i] ++ ;
    }
    void pop() {
        if (_iter[0] + _iter[1]) {
            _iter[_order[ _iter[0] + _iter[1] - 1 ]]--;
        }
    }
    bool top(T & ms) {
        if (!(_iter[0] + _iter[1]))
            return false;
        int o = _order[ _iter[0] + _iter[1] - 1];
        ms = _stack[0][ _iter[0] - 1];
        return true;
    }
    bool const empty() {
        return _iter[0] == 0;
    }
};

class Type_table {
    int col_num = 0;
    int row_num = 0;
public:
    vector<sid_t> tytable;
    // tytable store all the type info during planning
    // struct like
    /* |  0  |  -1  |  -2  |  -3  |
       |count|type..|type..|type..|
       |.....|......|......|......|
    */
    Type_table() { }
    int var2column(ssid_t vid) {
        //assert(vid < 0); // pattern variable
        //return (- vid);
        // false impl
        return -1;
    }
    void set_col_num(int n) { col_num = n; }
    int get_col_num() { return col_num; };
    int get_row_num() {
        if (col_num == 0) return 0;
        return tytable.size() / col_num;
    }
    sid_t get_row_col(int r, int c) {
        return tytable[col_num * r + c];
    }
    void set_row_col(int r, int c, sid_t val) {
        tytable[col_num * r + c] = val;
    }
    void append_row_to(int r, vector<sid_t> &updated_result_table) {
        for (int c = 0; c < col_num; c++)
            updated_result_table.push_back(get_row_col(r, c));
    }
    void append_newv_row_to(int r, vector<sid_t> &updated_result_table, sid_t val) {
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

    vector<int> pred_chains ; //the pred_type chains
    // for dfs
    vector<ssid_t> min_path;
    int _chains_size_div_4 ;
    int *min_select_record ;
    unordered_map<int, shared_ptr<Minimum_maintenance<select_record>>> *min_select;

    // store all orders for DP
    vector<int> subgraph[11]; // for 2^10 all orders

    // functions
    // dfs traverse , traverse all the valid orders
    bool com_traverse(unsigned int pt_bits, double cost, double pre_results) {
        if (pt_bits == ( 1 << _chains_size_div_4 ) - 1) {
            //cout << "estimated cost : " << cost << endl;
            bool ctn = true;
            if (min_cost == std::numeric_limits<double>::max()
                    && cost < COST_THRESHOLD
                    && (path[0] >= (1 << NBITS_IDX)) ) {
                ctn = false;  // small query
                logstream(LOG_DEBUG) << "small query and use heuristic." << LOG_endl;
            }
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
            ssid_t o1 = triples[i];
            ssid_t p = triples[i + 1];
            ssid_t d = triples[i + 2];
            ssid_t o2 = triples[i + 3];
            if (path.size() == 0) {
                if (o1 < 0 && o2 < 0) {
                    //continue;
                    // use index vertex
                    path.push_back(p);
                    path.push_back(0);
                    path.push_back(IN);
                    path.push_back(o1);
                    add_cost = double(statistic->global_pscount[p]) / global_num_servers; // find subjects
                    // next iteration
                    //cout << "pick : " << o1 << " " << p << " " << d << " " << o2 << endl;
                    //cout << "add_cost : " << add_cost << endl;
                    double new_cost = cost + add_cost;
                    if (new_cost > min_cost) {
                        path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                        continue;
                    }
                    double select_num_s = (double)statistic->global_pscount[p];
                    select_record srs = {p, d, select_num_s};
                    (*min_select)[o1] = std::unique_ptr<Minimum_maintenance<select_record>>
                                        (new Minimum_maintenance<select_record>(2 * _chains_size_div_4));
                    (*min_select)[o1]->push(srs);
                    bool ctn = com_traverse(pt_bits, new_cost, add_cost);
                    (*min_select)[o1]->pop();
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();

                    // different direction
                    path.push_back(p);
                    path.push_back(0);
                    path.push_back(OUT);
                    path.push_back(o2);
                    add_cost = double(statistic->global_pocount[p]) / global_num_servers; // find objects
                    // next iteration
                    //cout << "pick : " << o1 << " " << p << " " << d << " " << o2 << endl;
                    //cout << "add_cost : " << add_cost << endl;
                    new_cost = cost + add_cost;
                    if (new_cost > min_cost) {
                        path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                        continue;
                    }
                    double select_num_o = (double)statistic->global_pocount[p];
                    select_record sro = {p, IN, select_num_o};
                    (*min_select)[o2] = std::unique_ptr<Minimum_maintenance<select_record> >
                                        ( new Minimum_maintenance<select_record> (2 * _chains_size_div_4));
                    (*min_select)[o2]->push(sro);
                    ctn = com_traverse(pt_bits, new_cost, add_cost);
                    (*min_select)[o2]->pop();
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                }
                if (o1 > 0) {
                    path.push_back(o1);
                    path.push_back(p);
                    path.push_back(d);
                    path.push_back(o2);
                    add_cost = double(statistic->global_ptcount[p]) / statistic->global_pscount[p];

                    // next iteration
                    //cout << "pick : " << o1 << " " << p << " " << d << " " << o2 << endl;
                    //cout << "add_cost : " << add_cost << endl;
                    double new_cost = cost + add_cost;
                    if (new_cost > min_cost) {
                        path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                        continue;
                    }
                    new_add_selectivity(o1, p, o2);
                    bool ctn = com_traverse(pt_bits | (1 << pt_pick), new_cost, add_cost);
                    unadd_selectivity();
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();

                } else if (o2 > 0) {
                    path.push_back(o2);
                    path.push_back(p);
                    path.push_back(IN);
                    path.push_back(o1);
                    // for type triples
                    if (p == TYPE_ID) {
                        add_cost = statistic->global_pscount[o2];
                        //cout << "type " << o2 << " : " << add_cost << endl;
                    } else { // normal triples
                        add_cost = double(statistic->global_ptcount[p]) / statistic->global_pocount[p];
                    }

                    // next iteration
                    //cout << "pick : " << o1 << " " << p << " " << d << " " << o2 << endl;
                    //cout << "add_cost : " << add_cost << endl;
                    double new_cost = cost + add_cost;
                    if (new_cost > min_cost) {
                        path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                        continue;
                    }
                    new_add_selectivity(o1, p, o2);
                    bool ctn = com_traverse(pt_bits | (1 << pt_pick), new_cost, add_cost);
                    unadd_selectivity();
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();

                }
            } else {
                if (min_select->find(o1) != min_select->end() && !(*min_select)[o1]->empty()) {
                    path.push_back(o1);
                    path.push_back(p);
                    path.push_back(d);
                    path.push_back(o2);
                    double prune_result;
                    select_record sr ;
                    if (!(*min_select)[o1]->top(sr))
                        logstream(LOG_ERROR) << "o1 on top" << LOG_endl;
                    int pre_p = sr.p;
                    int pre_d = sr.d;
                    // prune based on correlation and constant
                    if (p == TYPE_ID && o2 > 0) {
                        prune_result = com_prune(pre_results, pre_p, pre_d, o2, OUT);
                        add_cost = prune_result;
                    } else if (o2 >= 0) {
                        prune_result = com_prune(pre_results, pre_p, pre_d, p, OUT);
                        prune_result = double(prune_result) / statistic->global_pocount[p];
                        add_cost = prune_result * (double(statistic->global_ptcount[p]) / statistic->global_pscount[p]);
                    } else {
                        prune_result = com_prune(pre_results, pre_p, pre_d, p, OUT);
                        add_cost = prune_result * (double(statistic->global_ptcount[p]) / statistic->global_pscount[p]);
                    }

                    // next iteration
                    //cout << "pick : " << o1 << " " << p << " " << d << " " << o2 << endl;
                    //cout << "add_cost : " << add_cost << endl;
                    double new_cost = cost + add_cost;
                    if (new_cost > min_cost) {
                        path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                        continue;
                    }
                    new_add_selectivity(o1, p, o2);
                    bool ctn = com_traverse(pt_bits | (1 << pt_pick), new_cost, add_cost);
                    unadd_selectivity();
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();

                }
                if (min_select->find(o2) != min_select->end() && !(*min_select)[o2]->empty() ) {
                    path.push_back(o2);
                    path.push_back(p);
                    path.push_back(IN);
                    path.push_back(o1);
                    double prune_result;
                    select_record sr ;
                    if (!(*min_select)[o2]->top(sr))
                        logstream(LOG_ERROR) << "o2 on top\n";
                    int pre_p = sr.p;
                    int pre_d = sr.d;
                    // prune based on correlation and constant
                    prune_result = com_prune(pre_results, pre_p, pre_d, p, IN);
                    if (o1 >= 0) {
                        prune_result = double(prune_result) / statistic->global_pscount[p];
                    }
                    add_cost = prune_result * (double(statistic->global_ptcount[p]) / statistic->global_pocount[p]);

                    // next iteration
                    //cout << "pick : " << o1 << " " << p << " " << d << " " << o2 << endl;
                    //cout << "add_cost : " << add_cost << endl;
                    double new_cost = cost + add_cost;
                    if (new_cost > min_cost) {
                        path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                        continue;
                    }
                    new_add_selectivity(o1, p, o2);
                    bool ctn = com_traverse(pt_bits | (1 << pt_pick), new_cost, add_cost);
                    unadd_selectivity();
                    if (!ctn) return ctn;
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();

                }
            }
        }

        return true;
    }

    // prune using correlation estimation
    double com_prune(double pre_results, ssid_t pre_p, ssid_t pre_d, ssid_t p, ssid_t d) {
        double result_num;
        double x, y;
        // handle corner case first
        if (pre_p == p) {
            return pre_results;
        }

        // out out correlation
        if (pre_d == OUT && d == OUT) {
            if (pre_p < p) {
                x = double(statistic->global_ppcount[make_pair(pre_p, p)].out_out);
            } else {
                x = double(statistic->global_ppcount[make_pair(p, pre_p)].out_out);
            }
            y = statistic->global_pscount[pre_p];
        }

        // in out correlation
        if (pre_d == IN && d == OUT) {
            if (pre_p < p) {
                x = double(statistic->global_ppcount[make_pair(pre_p, p)].in_out);
            } else {
                x = double(statistic->global_ppcount[make_pair(p, pre_p)].out_in);
            }
            y = statistic->global_pocount[pre_p];
        }

        // in in correlation
        if (pre_d == IN && d == IN) {
            if (pre_p < p) {
                x = double(statistic->global_ppcount[make_pair(pre_p, p)].in_in);
            } else {
                x = double(statistic->global_ppcount[make_pair(p, pre_p)].in_in);
            }
            y = statistic->global_pocount[pre_p];
        }

        // out in correlation
        if (pre_d == OUT && d == IN) {
            if (pre_p < p) {
                x = double(statistic->global_ppcount[make_pair(pre_p, p)].out_in);
            } else {
                x = double(statistic->global_ppcount[make_pair(p, pre_p)].in_out);
            }
            y = statistic->global_pscount[pre_p];
        }

        //cout << "prune ratio (x/y) : " << x << " / " << y  << endl;
        result_num = pre_results * (x / y);
        if (x == 0) is_empty = true;
        return result_num;
    }

    void new_add_selectivity(ssid_t o1, ssid_t p, ssid_t o2) {
        if (o1 < 0 && o2 > 0) {
            double select_num;
            if (p == TYPE_ID) {
                select_num = (double)statistic->global_pscount[o2];
                p = o2;
            } else {
                select_num = double(statistic->global_pscount[p]) / statistic->global_pocount[p];
            }
            select_record sr = {p, OUT, select_num};
            if (min_select->find(o1) == min_select->end())
                (*min_select)[o1] = std::unique_ptr<Minimum_maintenance<select_record>>
                                    (new Minimum_maintenance<select_record> (2 * _chains_size_div_4));
            (*min_select)[o1]->push(sr);
            min_select_record[min_select_record[0] + 1] = o1;
            min_select_record[min_select_record[0] + 2] = 1;
            min_select_record[0] += 2 ;
        } else if (o2 < 0 && o1 > 0) {
            double select_num = double(statistic->global_pocount[p]) / statistic->global_pscount[p];
            select_record sr = {p, IN, select_num};
            if (min_select->find(o2) == min_select->end())
                (*min_select)[o2] = std::unique_ptr<Minimum_maintenance<select_record>>
                                    (new Minimum_maintenance<select_record> (2 * _chains_size_div_4));
            (*min_select)[o2]->push(sr);

            min_select_record[min_select_record[0] + 1] = o2;
            min_select_record[min_select_record[0] + 2] = 1;
            min_select_record[0] += 2 ;
        } else if (o1 < 0 && o2 < 0) {
            double select_num_s = statistic->global_pscount[p];
            double select_num_o = statistic->global_pocount[p];
            select_record sro1 = {p, OUT, select_num_s};
            if (min_select->find(o1) == min_select->end())
                (*min_select)[o1] = std::unique_ptr<Minimum_maintenance<select_record>>
                                    ( new Minimum_maintenance<select_record> (2 * _chains_size_div_4));
            (*min_select)[o1]->push(sro1);

            select_record sro2 = {p, IN, select_num_o};
            if (min_select->find(o2) == min_select->end())
                (*min_select)[o2] = std::unique_ptr<Minimum_maintenance<select_record>>
                                    (new Minimum_maintenance<select_record> (2 * _chains_size_div_4));
            (*min_select)[o2]->push(sro2);

            min_select_record[min_select_record[0] + 1] = o1;
            min_select_record[min_select_record[0] + 2] = o2;
            min_select_record[min_select_record[0] + 3] = 2;
            min_select_record[0] += 3 ;
        } else {
            min_select_record[min_select_record[0] + 1] = 0;
            min_select_record[0] += 1 ;
        }
    }

    void unadd_selectivity() {
        int lastpos = min_select_record[0];
        if (lastpos == 0)
            return ;
        int lastnum = min_select_record[lastpos];
        ASSERT(lastpos > lastnum);
        int i;
        for (i = 1 ; i <= lastnum ; i++) {
            (*min_select)[min_select_record[lastpos - i]]->pop();
        }
        min_select_record[0] -= lastnum + 1;
    }

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
    
    // for test
    int count00;
    int count01;
    int count02;
    int finalpath;

    // type-centric dfs enumeration
    bool plan_enum(unsigned int pt_bits, double cost, double pre_results) {
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
            vector<sid_t> updated_result_table;
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
                    if (condprune_results == 0) condprune_results = 1;
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
                    if (condprune_results == 0) condprune_results = 1;
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

                    // TODO
                    //assert(false);
                    // selectivity and cost estimation
                    // find all types, push into result table
                    ssid_t o1type = 31; // use o1 get_global_edges
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
                    if (condprune_results == 0) condprune_results = 1;
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
                        double vcount = double(statistic->global_pscount[o2]) / global_num_servers;
                        updated_result_table.push_back(vcount);
                        updated_result_table.push_back(vtype);
                    } else { // normal triples TODO
                        //assert(false);
                        ssid_t o2type = 31; // use o2 get_global_edges
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
                    if (condprune_results == 0) condprune_results = 1;
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
                // var in the result table
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
                        sid_t pre_tyid = type_table.get_row_col(i, var_col);
                        double pre_count = type_table.get_row_col(i, 0);
                        // handle type predicate first
                        if (p == TYPE_ID && o2 > 0) {
                            correprune_boost_results += pre_count;
                            early_ret += pre_count;
                            if (pre_tyid != o2) continue;
                            type_table.append_row_to(i, updated_result_table);
                            condprune_results += pre_count;
                            continue;
                        }
                        int tycount = statistic->global_pscount[pre_tyid];
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
                                // TODO for constant pruning
                                //assert(false);
                                ssid_t o2type = 31; // TODO
                                if (vtype == o2type) match_flag = 1;
                            }
                            else if (var2col.find(o2) != var2col.end() && var2col[o2] > 0) { 
                                // for variable pruning
                                ssid_t pretype = type_table.get_row_col(i, var2col[o2]);
                                if (vtype == pretype) {
                                    int type_num = statistic->global_pscount[pretype];
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

                    if (condprune_results == 0) condprune_results = 1;
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
                // var in the result table
                if (o2 < 0 && (var2col.find(o2) != var2col.end()) && var2col[o2] > 0) {
                  //count02++;
                    if (var2col[o2] > type_table.get_col_num()) assert(false);
                    //cout << "o2: " << o2 << " var2col[o2]: " << var2col[o2] << endl;
                    path.push_back(o2); path.push_back(p); path.push_back(IN); path.push_back(o1);
                    //cout << "pick : " << o2 << " " << p << " " << IN << " " << o1 << "-------------------------------------" << endl;

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
                    int var_col = var2col[o2];
                    int prune_flag = (o1 > 0) || ((var2col.find(o1) != var2col.end()) && var2col[o1] > 0);
                    int dup_flag = (path[0] == p) && (path[3] == o2);
                    for (size_t i = 0; i < row_num; i++) {
                        sid_t pre_tyid = type_table.get_row_col(i, var_col);
                        double pre_count = type_table.get_row_col(i, 0);
                        int tycount = statistic->global_pscount[pre_tyid];
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
                                // TODO for constant pruning
                                //assert(false);
                                ssid_t o1type = 31; // TODO
                                if (vtype == o1type) match_flag = 1;
                            }
                            else if (var2col.find(o1) != var2col.end() && var2col[o1] > 0) { 
                                // for variable pruning
                                ssid_t pretype = type_table.get_row_col(i, var2col[o1]);
                                if (vtype == pretype) {
                                    int type_num = statistic->global_pscount[pretype];
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

                    if (condprune_results == 0) condprune_results = 1;
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

public:
    Planner() { }

    bool generate_for_patterns(vector<SPARQLQuery::Pattern> &patterns) {
        // transfer from patterns to temp_cmd_chains, may cause performance decrease
        vector<ssid_t> temp_cmd_chains;
        vector<ssid_t> attr_pattern;
        vector<int> attr_pred_chains;
        transfer_to_cmd_chains(patterns, attr_pattern, attr_pred_chains, temp_cmd_chains);
        min_path.clear();
        path.clear();
        is_empty = false;
        double cost = 0;
        min_cost = std::numeric_limits<double>::max();
        if (min_cost != std::numeric_limits<double>::max()) ASSERT(false);

        uint64_t t_prepare1 = timer::get_usec();
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
        uint64_t t_prepare2 = timer::get_usec();
        //cout << "prepare time : " << t_prepare2 - t_prepare1 << " us" << endl;

        uint64_t t_traverse1 = timer::get_usec();
        this->triples = temp_cmd_chains;
        this->min_select = new unordered_map<int, shared_ptr<Minimum_maintenance<select_record>>>;
        _chains_size_div_4 = temp_cmd_chains.size() / 4 ;
        min_select_record = new int[1 + 6 * _chains_size_div_4];
        min_select_record[0] = 0;
        com_traverse(0, cost, 0);
        delete [] min_select_record ;
        delete this->min_select;
        uint64_t t_traverse2 = timer::get_usec();
        //cout << "traverse time : " << t_traverse2 - t_traverse1 << " us" << endl;

        if (is_empty == true) {
            logstream(LOG_INFO) << "Identified empty result query." << LOG_endl;
            logstream(LOG_INFO) << "Query planning is finished." << LOG_endl;
            return false;
        }

        logstream(LOG_DEBUG) << "Query planning for one part is finished." << LOG_endl;
        logstream(LOG_DEBUG) << "Estimated cost: " << min_cost << LOG_endl;

        //transfer from min_path to patterns
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

    bool generate_for_patterns_new(vector<SPARQLQuery::Pattern> &patterns){
        
        cout << "generate_for_patterns_new..............." << endl;

        //input : patterns
        //transform to : _chains_size_div_4, triples, temp_cmd_chains
        vector<ssid_t> temp_cmd_chains;
        vector<ssid_t> attr_pattern;
        vector<int> attr_pred_chains;
        transfer_to_cmd_chains(patterns, attr_pattern, attr_pred_chains, temp_cmd_chains);

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

        for (int i = 0, ilimit = min_path.size(); i < ilimit; i = i + 4)
         cout << "min_path " << " : " << min_path[i] << " "
           << min_path[i+1] << " "
           << min_path[i+2] << " "
           << min_path[i+3] << endl;
        cout << "query planning is finished." << endl;
        cout << "estimated cost: " << min_cost << endl;

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
            success = generate_for_patterns_new(group.patterns);
        for (auto &g : group.unions)
            success = generate_for_group(g);
        return success;
    }

    bool generate_plan(SPARQLQuery &r, data_statistic *statistic) {
        this->statistic = statistic;
        return generate_for_group(r.pattern_group);
    }
};
