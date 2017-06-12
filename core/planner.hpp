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

#define COST_THRESHOLD 350

struct plan {
    double cost;      // now min cost
    vector<int64_t> orders; // now best orders
    double result_num;     // record intermediate results
};

struct select_record {
    int p;
    int d;
    double v;
    bool operator > (const select_record & other) const {
        return v > other.v;
    }
    bool operator < (const select_record & other) const {
        return v < other.v;
    }
    bool operator == (const select_record & other) const {
        return v == other.v;
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
        cout << "NOT SUPPORT" << endl;
        assert(0);
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

class Planner {
    // members
    data_statistic * statistic ;
    vector<int64_t> triples;
    double min_cost;
    vector<int64_t> path;
    bool is_empty;            // help identify empty queries

    // for dfs
    vector<int64_t> min_path;
    int _chains_size_div_4 ;
    int * min_select_record ;
    unordered_map<int, shared_ptr<Minimum_maintenance<select_record>> > * min_select;

    // store all orders for DP
    vector<int> subgraph[11]; // for 2^10 all orders

    // functions
    // dfs traverse , traverse all the valid orders
    bool com_traverse(unsigned int pt_bits, double cost, double pre_results) {
        if (pt_bits == ( 1 << _chains_size_div_4 ) - 1) {
            //cout << "estimated cost : " << cost << endl;
            bool ctn = true;
            if (min_cost == std::numeric_limits<double>::max() && cost < COST_THRESHOLD && (path[0] >= (1 << NBITS_IDX)) ) {
                ctn = false;  // small query
                cout << "small query and use heuristic.\n";
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
            int o1 = triples[i];
            int p = triples[i + 1];
            int d = triples[i + 2];
            int o2 = triples[i + 3];
            if (path.size() == 0) {
                if (o1 < 0 && o2 < 0) {
                    //continue;
                    // use index vertex
                    path.push_back(p);
                    path.push_back(0);
                    path.push_back(IN);
                    path.push_back(o1);
                    add_cost = double(statistic->global_pscount[p]) / statistic->world->size(); //find subjects
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
                    (*min_select)[o1] = std::unique_ptr<Minimum_maintenance<select_record> >
                                        ( new Minimum_maintenance<select_record>(2 * _chains_size_div_4));
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
                    add_cost = double(statistic->global_pocount[p]) / statistic->world->size(); //find objects
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
                        cout << "ERROR o1 on .top\n";
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
                        cout << "ERROR o2 on .top\n";
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

    // dynamic programming planner
    void dp_plan() {
        int count = triples.size() / 4; // number of triples
        int size = pow(2, count);   // size of DP state

        // initial the DP array
        plan *state = new plan[size];
        plan initial;
        initial.cost = std::numeric_limits<double>::max();
        initial.result_num = 0;
        state[0].cost = 0;
        state[0].result_num = 0;
        for (int i = 1; i < size; i++)
            state[i] = initial;

        //cout << "count:" << count << " size : " << size << endl;
        // DP
        for (int i = 1; i < count + 1; i++) {
            for (int k = 0, klimit = subgraph[i].size() ; k < klimit; k++) {
                int bit_state = subgraph[i][k];
                if (bit_state >= size) continue;
                plan& best_plan = state[bit_state];
                int best_m = -1;  // record best one
                int best_dir = OUT;
                // f[i][x] = min {f[i-1][x1] + cost, ...}
                // to compute from all the i-1 state
                for (int m = 0; m < count; m++) {
                    if ((bit_state & (1 << m)) > 0) {
                        int son_state = bit_state ^ (1 << m);
                        double pre_cost = state[son_state].cost;
                        if (pre_cost == std::numeric_limits<double>::max()) continue;  // previous order invalid
                        double pre_results = state[son_state].result_num;
                        vector<pair<double, int> > cost_dir;
                        cost_add(pre_results, son_state, m * 4, cost_dir);
                        if (cost_dir.size() == 0) continue;  // order invalid
                        //cout << "pre_results : " << pre_results << endl;
                        if (cost_dir.size() == 1) {  // one direction
                            double new_cost = cost_dir[0].first + pre_cost;
                            //cout << "new_cost : " << new_cost << endl;
                            if (best_plan.cost > new_cost) {
                                best_plan.cost = new_cost;
                                best_plan.result_num = cost_dir[0].first;
                                best_plan.orders = state[son_state].orders;
                                best_m = m;
                                best_dir = cost_dir[0].second;
                            }
                        } else if (cost_dir.size() == 2) { // both direction
                            double new_cost0 = cost_dir[0].first + pre_cost;
                            double new_cost1 = cost_dir[1].first + pre_cost;
                            //cout << "new_cost0 : " << new_cost0 << endl;
                            //cout << "new_cost1 : " << new_cost1 << endl;
                            if (new_cost0 < new_cost1) {
                                if (best_plan.cost > new_cost0) {
                                    best_plan.cost = new_cost0;
                                    best_plan.result_num = cost_dir[0].first;
                                    best_plan.orders = state[son_state].orders;
                                    best_m = m;
                                    best_dir = cost_dir[0].second;
                                }
                            } else {
                                if (best_plan.cost > new_cost1) {
                                    best_plan.cost = new_cost1;
                                    best_plan.result_num = cost_dir[1].first;
                                    best_plan.orders = state[son_state].orders;
                                    best_m = m;
                                    best_dir = cost_dir[1].second;
                                }
                            }
                            // if i == 1 then is index vertex
                            // best_m = -1
                            // push something
                            if (i == 1) {
                                best_m = -1;
                                int vertex1, vertex2;
                                if (best_dir == OUT) {
                                    vertex1 = triples[4 * m + 3];
                                    vertex2 = triples[4 * m];
                                }
                                else {
                                    vertex1 = triples[4 * m];
                                    vertex2 = triples[4 * m + 3];
                                }
                                // push index execution
                                best_plan.orders.push_back(triples[4 * m + 1]);
                                best_plan.orders.push_back(0);
                                best_plan.orders.push_back(best_dir);
                                best_plan.orders.push_back(vertex1);
                                // push the other direction
                                best_plan.orders.push_back(vertex1);
                                best_plan.orders.push_back(triples[4 * m + 1]);
                                best_plan.orders.push_back(1 - best_dir);
                                best_plan.orders.push_back(vertex2);
                                best_plan.result_num = statistic->global_ptcount[triples[4 * m + 1]]; // revise the result num
                            }

                        }
                    }
                }
                // already find the best order
                if (best_m >= 0) {
                    if (best_dir == OUT) {
                        best_plan.orders.push_back(triples[4 * best_m]);
                        best_plan.orders.push_back(triples[4 * best_m + 1]);
                        best_plan.orders.push_back(triples[4 * best_m + 2]);
                        best_plan.orders.push_back(triples[4 * best_m + 3]);
                    } else {
                        best_plan.orders.push_back(triples[4 * best_m + 3]);
                        best_plan.orders.push_back(triples[4 * best_m + 1]);
                        best_plan.orders.push_back(IN);
                        best_plan.orders.push_back(triples[4 * best_m]);
                    }
                }

            }
        }
        cout << "DP query planning is finished.\n";

        // state[size - 1] has the final best plan
        min_path = state[size - 1].orders;
        min_cost = state[size - 1].cost;

    }

    int bitcount(int i) {
        i = i - ((i >> 1) & 0x55555555);
        i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
        return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    }

    void cost_add(double pre_results, int bit_set, int next, vector<pair<double, int> >& cost_dir) {
        unordered_map<int, vector<int> > selec;
        double cost = 0;
        for (int i = 0, ilimit = triples.size(); i < ilimit; i = i + 4) {
            int tmp = i / 4;
            if ( (bit_set & (0x1 << tmp)) == false ) continue;
            add_selectivity(selec, triples[i], triples[i + 1], triples[i + 3]);
        }

        int o1 = triples[next];
        int p = triples[next + 1];
        int o2 = triples[next + 3];

        // if selec empty
        if (selec.empty()) {
            if (o1 < 0 && o2 < 0) {
                // s < 0 , o < 0, two direction
                // begin from s
                cost = double(statistic->global_pscount[p]) / statistic->world->size(); //find subjects
                cost += statistic->global_ptcount[p];
                if (cost == 0) cost = 1;
                cost_dir.push_back(make_pair(cost, IN));
                // begin from o
                cost = double(statistic->global_pocount[p]) / statistic->world->size(); //find objects
                cost += statistic->global_ptcount[p];
                if (cost == 0) cost = 1;
                cost_dir.push_back(make_pair(cost, OUT));
            }
            if (o1 > 0 && o2 < 0) {
                // s > 0, o < 0
                cost = double(statistic->global_ptcount[p]) / statistic->global_pscount[p];
                cost_dir.push_back(make_pair(cost, OUT));
            }
            if (o1 < 0 && o2 > 0) {
                // s < 0, o > 0
                if (p == TYPE_ID) {
                    cost = statistic->global_pscount[o2];
                } else { // normal triples
                    cost = double(statistic->global_ptcount[p]) / statistic->global_pocount[p];
                }
                cost_dir.push_back(make_pair(cost, IN));
            }
        }
        // not empty
        else {
            if (selec.find(o1) != selec.end()) {
                // s match previous
                double prune_result;
                int pre_p = selec[o1][0];
                int pre_d = selec[o1][1];
                if (p == TYPE_ID && o2 > 0) {
                    prune_result = com_prune(pre_results, pre_p, pre_d, o2, OUT);
                    cost = prune_result;
                } else if (o2 >= 0) {
                    prune_result = com_prune(pre_results, pre_p, pre_d, p, OUT);
                    prune_result = double(prune_result) / statistic->global_pocount[p];
                    cost = prune_result * (double(statistic->global_ptcount[p]) / statistic->global_pscount[p]);
                } else {
                    prune_result = com_prune(pre_results, pre_p, pre_d, p, OUT);
                    cost = prune_result * (double(statistic->global_ptcount[p]) / statistic->global_pscount[p]);
                }
                if (cost == 0) cost = 1;
                cost_dir.push_back(make_pair(cost, OUT));
            }

            if (selec.find(o2) != selec.end()) {
                // o match previous
                double prune_result;
                int pre_p = selec[o2][0];
                int pre_d = selec[o2][1];
                prune_result = com_prune(pre_results, pre_p, pre_d, p, IN);
                if (o1 >= 0) {
                    prune_result = double(prune_result) / statistic->global_pscount[p];
                }
                cost = prune_result * (double(statistic->global_ptcount[p]) / statistic->global_pocount[p]);
                if (cost == 0) cost = 1;
                cost_dir.push_back(make_pair(cost, IN));
            }
        }

    }

    // prune using correlation estimation
    double com_prune(double pre_results, int pre_p, int pre_d, int p, int d) {
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

    void new_add_selectivity(int o1, int p, int o2) {
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
                (*min_select)[o1] = std::unique_ptr<Minimum_maintenance<select_record> >
                                    ( new Minimum_maintenance<select_record> (2 * _chains_size_div_4));
            (*min_select)[o1]->push(sr);
            min_select_record[min_select_record[0] + 1] = o1;
            min_select_record[min_select_record[0] + 2] = 1;
            min_select_record[0] += 2 ;
        } else if (o2 < 0 && o1 > 0) {
            double select_num = double(statistic->global_pocount[p]) / statistic->global_pscount[p];
            select_record sr = {p, IN, select_num};
            if (min_select->find(o2) == min_select->end())
                (*min_select)[o2] = std::unique_ptr<Minimum_maintenance<select_record> >
                                    ( new Minimum_maintenance<select_record> (2 * _chains_size_div_4));
            (*min_select)[o2]->push(sr);

            min_select_record[min_select_record[0] + 1] = o2;
            min_select_record[min_select_record[0] + 2] = 1;
            min_select_record[0] += 2 ;
        } else if (o1 < 0 && o2 < 0) {
            double select_num_s = statistic->global_pscount[p];
            double select_num_o = statistic->global_pocount[p];
            select_record sro1 = {p, OUT, select_num_s};
            if (min_select->find(o1) == min_select->end())
                (*min_select)[o1] = std::unique_ptr<Minimum_maintenance<select_record> >
                                    ( new Minimum_maintenance<select_record> (2 * _chains_size_div_4));
            (*min_select)[o1]->push(sro1);

            select_record sro2 = {p, IN, select_num_o};
            if (min_select->find(o2) == min_select->end())
                (*min_select)[o2] = std::unique_ptr<Minimum_maintenance<select_record> >
                                    ( new Minimum_maintenance<select_record> (2 * _chains_size_div_4));
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
        assert(lastpos > lastnum);
        int i;
        for (i = 1 ; i <= lastnum ; i++) {
            (*min_select)[min_select_record[lastpos - i]]->pop();
        }
        min_select_record[0] -= lastnum + 1;
    }

    void add_selectivity(unordered_map<int, vector<int> >& tmp_selec, int o1, int p, int o2) {
        if (o1 < 0 && o2 > 0) {
            int select_num;
            if (p == TYPE_ID) {
                select_num = statistic->global_pscount[o2];
                p = o2;
            } else {
                select_num = double(statistic->global_pscount[p]) / statistic->global_pocount[p];
            }
            if (tmp_selec.find(o1) == tmp_selec.end()) {
                tmp_selec[o1].push_back(p);
                tmp_selec[o1].push_back(OUT);
                tmp_selec[o1].push_back(select_num);
            } else {
                int pre_select_num = tmp_selec[o1][2];
                if (pre_select_num > select_num) {
                    vector<int> new_tmp;
                    new_tmp.push_back(p);
                    new_tmp.push_back(OUT);
                    new_tmp.push_back(select_num);
                    tmp_selec[o1] = new_tmp;
                }
            }
        }
        if (o2 < 0 && o1 > 0) {
            int select_num = double(statistic->global_pocount[p]) / statistic->global_pscount[p];
            if (tmp_selec.find(o2) == tmp_selec.end()) {
                tmp_selec[o2].push_back(p);
                tmp_selec[o2].push_back(IN);
                tmp_selec[o2].push_back(select_num);
            } else {
                int pre_select_num = tmp_selec[o2][2];
                if (pre_select_num > select_num) {
                    vector<int> new_tmp;
                    new_tmp.push_back(p);
                    new_tmp.push_back(IN);
                    new_tmp.push_back(select_num);
                    tmp_selec[o2] = new_tmp;
                }
            }
        }
        if (o1 < 0 && o2 < 0) {
            int select_num_s = statistic->global_pscount[p];
            int select_num_o = statistic->global_pocount[p];
            if (tmp_selec.find(o1) == tmp_selec.end()) {
                tmp_selec[o1].push_back(p);
                tmp_selec[o1].push_back(OUT);
                tmp_selec[o1].push_back(select_num_s);
            } else {
                int pre_select_num = tmp_selec[o1][2];
                if (pre_select_num > select_num_s) {
                    vector<int> new_tmp;
                    new_tmp.push_back(p);
                    new_tmp.push_back(OUT);
                    new_tmp.push_back(select_num_s);
                    tmp_selec[o1] = new_tmp;
                }
            }
            if (tmp_selec.find(o2) == tmp_selec.end()) {
                tmp_selec[o2].push_back(p);
                tmp_selec[o2].push_back(IN);
                tmp_selec[o2].push_back(select_num_o);
            } else {
                int pre_select_num = tmp_selec[o2][2];
                if (pre_select_num > select_num_o) {
                    vector<int> new_tmp;
                    new_tmp.push_back(p);
                    new_tmp.push_back(IN);
                    new_tmp.push_back(select_num_o);
                    tmp_selec[o2] = new_tmp;
                }
            }
        }
    }

    // for testing purpose
    void get_all_plans(vector<int64_t> triples, unordered_map<int, vector<int> > max_selec,
                       double cost, double pre_results) {
        if (triples.empty()) {
            cout << "estimated cost : " << cost << endl;
            if (cost < min_cost) {
                min_cost = cost;
                min_path = path;
            }
            // record this new order
            plan new_order;
            new_order.cost = cost;
            new_order.orders = path;
            boost::unordered_map<int, int> convert;
            for (int i = 0, ilimit = new_order.orders.size(); i < ilimit; i++) {
                if (new_order.orders[i] < 0 ) {
                    if (convert.find(new_order.orders[i]) == convert.end()) {
                        int value =  -1 - convert.size();
                        convert[new_order.orders[i]] = value;
                        new_order.orders[i] = value;
                    } else {
                        new_order.orders[i] = convert[new_order.orders[i]];
                    }
                }
            }
            all_plans.push_back(new_order);
            return;
        }
        for (int i = 0 , ilimit = triples.size(); i < ilimit; i = i + 4) {
            double add_cost = 0;
            int o1 = triples[i];
            int p = triples[i + 1];
            int d = triples[i + 2];
            int o2 = triples[i + 3];
            if (path.size() == 0) {
                if (o1 < 0 && o2 < 0) {
                    //continue;
                    // use index vertex
                    path.push_back(p);
                    path.push_back(0);
                    path.push_back(IN);
                    path.push_back(o1);
                    add_cost = double(statistic->global_pscount[p]) / statistic->world->size(); //find subjects
                    if (add_cost == 0) add_cost = 1;
                    // next iteration
                    double new_cost = cost + add_cost;
                    unordered_map<int, vector<int> > tmp_selec = max_selec;
                    int select_num_s = statistic->global_pscount[p];
                    tmp_selec[o1].push_back(p);
                    tmp_selec[o1].push_back(d);
                    tmp_selec[o1].push_back(select_num_s);
                    get_all_plans(triples, tmp_selec, new_cost, add_cost);
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();

                    // different direction
                    path.push_back(p);
                    path.push_back(0);
                    path.push_back(OUT);
                    path.push_back(o2);
                    add_cost = double(statistic->global_pocount[p]) / statistic->world->size(); //find objects
                    if (add_cost == 0) add_cost = 1;
                    // next iteration
                    new_cost = cost + add_cost;
                    tmp_selec = max_selec;
                    int select_num_o = statistic->global_pocount[p];
                    tmp_selec[o2].push_back(p);
                    tmp_selec[o2].push_back(IN);
                    tmp_selec[o2].push_back(select_num_o);
                    get_all_plans(triples, tmp_selec, new_cost, add_cost);
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();
                }
                if (o1 > 0) {
                    path.push_back(o1);
                    path.push_back(p);
                    path.push_back(d);
                    path.push_back(o2);
                    double inter_results = 0;
                    add_cost = double(statistic->global_ptcount[p]) / statistic->global_pscount[p];
                    inter_results = double(add_cost) / statistic->world->size();

                    // next iteration
                    double new_cost = cost + add_cost;
                    vector<int64_t> tmp = triples;
                    tmp.erase(tmp.begin() + i, tmp.begin() + i + 4);
                    unordered_map<int, vector<int> > tmp_selec = max_selec;
                    add_selectivity(tmp_selec, o1, p, o2);
                    get_all_plans(tmp, tmp_selec, new_cost, inter_results);
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();

                } else if (o2 > 0) {
                    path.push_back(o2);
                    path.push_back(p);
                    path.push_back(IN);
                    path.push_back(o1);
                    double inter_results = 0;
                    // for type triples
                    if (p == TYPE_ID) {
                        add_cost = double(statistic->global_pscount[o2]) / statistic->world->size();
                        inter_results = add_cost;
                    } else { // normal triples
                        add_cost = double(statistic->global_ptcount[p]) / statistic->global_pocount[p];
                        inter_results = double(add_cost) / statistic->world->size();
                    }

                    // next iteration
                    double new_cost = cost + add_cost;
                    vector<int64_t> tmp = triples;
                    tmp.erase(tmp.begin() + i, tmp.begin() + i + 4);
                    unordered_map<int, vector<int> > tmp_selec = max_selec;
                    add_selectivity(tmp_selec, o1, p, o2);
                    get_all_plans(tmp, tmp_selec, new_cost, inter_results);
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();

                }
            } else {
                if (max_selec.find(o1) != max_selec.end()) {
                    path.push_back(o1);
                    path.push_back(p);
                    path.push_back(d);
                    path.push_back(o2);
                    double prune_result;
                    int pre_p = max_selec[o1][0];
                    int pre_d = max_selec[o1][1];
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
                    if (add_cost == 0) add_cost = 1;

                    // next iteration
                    double new_cost = cost + add_cost;
                    vector<int64_t> tmp = triples;
                    tmp.erase(tmp.begin() + i, tmp.begin() + i + 4);
                    unordered_map<int, vector<int> > tmp_selec = max_selec;
                    add_selectivity(tmp_selec, o1, p, o2);
                    get_all_plans(tmp, tmp_selec, new_cost, add_cost);
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();

                }
                if (max_selec.find(o2) != max_selec.end()) {
                    path.push_back(o2);
                    path.push_back(p);
                    path.push_back(IN);
                    path.push_back(o1);
                    double prune_result;
                    int pre_p = max_selec[o2][0];
                    int pre_d = max_selec[o2][1];
                    // prune based on correlation and constant
                    prune_result = com_prune(pre_results, pre_p, pre_d, p, IN);
                    if (o1 >= 0) {
                        prune_result = double(prune_result) / statistic->global_pscount[p];
                    }
                    add_cost = prune_result * (double(statistic->global_ptcount[p]) / statistic->global_pocount[p]);
                    if (add_cost == 0) add_cost = 1;

                    // next iteration
                    double new_cost = cost + add_cost;
                    vector<int64_t> tmp = triples;
                    tmp.erase(tmp.begin() + i, tmp.begin() + i + 4);
                    unordered_map<int, vector<int> > tmp_selec = max_selec;
                    add_selectivity(tmp_selec, o1, p, o2);
                    get_all_plans(tmp, tmp_selec, new_cost, add_cost);
                    path.pop_back(); path.pop_back(); path.pop_back(); path.pop_back();

                }
            }
        }
    }

public:
    Planner() {
        for (int i = 0; i < 1024; i++) {
            int c = bitcount(i);
            subgraph[c].push_back(i);
        }
    }

    bool generate_plan(request_or_reply& r, data_statistic* statistic) {
        this->statistic = statistic;
        min_path.clear();
        path.clear();
        is_empty = false;
        double cost = 0;
        min_cost = std::numeric_limits<double>::max();
        if (min_cost != std::numeric_limits<double>::max()) assert(false);

        uint64_t t_prepare1 = timer::get_usec();
        // prepare for heuristic
        for (int i = 0, ilimit = r.cmd_chains.size(); i < ilimit; i = i + 4) {
            if (r.cmd_chains[i] >= (1 << NBITS_IDX) || r.cmd_chains[i + 3] >= (1 << NBITS_IDX)) {
                if (i == 0) break;
                int ta, tb, tc, td;
                ta = r.cmd_chains[i];
                tb = r.cmd_chains[i + 1];
                tc = r.cmd_chains[i + 2];
                td = r.cmd_chains[i + 3];
                r.cmd_chains[i] = r.cmd_chains[0];
                r.cmd_chains[i + 1] = r.cmd_chains[1];
                r.cmd_chains[i + 2] = r.cmd_chains[2];
                r.cmd_chains[i + 3] = r.cmd_chains[3];
                r.cmd_chains[0] = ta;
                r.cmd_chains[1] = tb;
                r.cmd_chains[2] = tc;
                r.cmd_chains[3] = td;
                break;
            }
        }
        uint64_t t_prepare2 = timer::get_usec();
        //cout << "prepare time : " << t_prepare2 - t_prepare1 << " us" << endl;

        uint64_t t_traverse1 = timer::get_usec();
        this->triples = r.cmd_chains;
        this->min_select = new unordered_map<int, shared_ptr<Minimum_maintenance<select_record>>>;
        _chains_size_div_4 = r.cmd_chains.size() / 4 ;
        min_select_record = new int [ 1 + 6 * _chains_size_div_4 ];
        min_select_record[0] = 0;
        com_traverse(0, cost, 0);
        delete [] min_select_record ;
        delete this->min_select;
        uint64_t t_traverse2 = timer::get_usec();
        //cout << "traverse time : " << t_traverse2 - t_traverse1 << " us" << endl;

        if (is_empty == true) {
            cout << "identified empty result query." << endl;
            cout << "query planning is finished." << endl;
            return false;
        }

        // check for middle heuristic, swap the direction
        /*if (min_path[0] == min_path[5]) {
          int count_s = statistic->global_pscount[min_path[0]];
          int count_o = statistic->global_pocount[min_path[0]];
          if ((min_path[2] == 0 && count_s < count_o) || (min_path[2] == 1 && count_s > count_o)) {
            min_path[2] = 1 - min_path[2];
            min_path[6] = 1 - min_path[6];
            min_path[3] = min_path[7];
            min_path[7] = min_path[4];
            min_path[4] = min_path[3];
          }
        }*/

        uint64_t t_convert1 = timer::get_usec();
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
        uint64_t t_convert2 = timer::get_usec();
        //cout << "convert time : " << t_convert2 - t_convert1 << " us" << endl;

        //for (int i = 0, ilimit = r.cmd_chains.size(); i < ilimit; i = i + 4)
        //  cout << "cmd_chain " << " : " << r.cmd_chains[i] << " "
        //    << r.cmd_chains[i+1] << " "
        //    << r.cmd_chains[i+2] << " "
        //    << r.cmd_chains[i+3] << " "
        //    << "ps : " << statistic->global_pscount[r.cmd_chains[i+1]]
        //    << " pt : " << statistic->global_ptcount[r.cmd_chains[i+1]]
        //    << " po : " << statistic->global_pocount[r.cmd_chains[i+1]]
        //    << endl;

        //for (int i = 0, ilimit = min_path.size(); i < ilimit; i = i + 4)
        //  cout << "min_path " << " : " << min_path[i] << " "
        //    << min_path[i+1] << " "
        //    << min_path[i+2] << " "
        //    << min_path[i+3] << endl;
        cout << "query planning is finished." << endl;
        cout << "estimated cost: " << min_cost << endl;

        r.cmd_chains = min_path;
        return true;

    }

    bool generate_dp_plan(request_or_reply& r, data_statistic* statistic) {
        this->statistic = statistic;
        min_path.clear();
        min_cost = std::numeric_limits<double>::max();
        is_empty = false;

        cout << "using DP query planning." << endl;
        this->triples = r.cmd_chains;
        dp_plan();
        //cout << "min_path size : " << min_path.size() << endl;

        if (is_empty == true) return false;

        //cout << "before convert: " << endl;
        //for (int i = 0, ilimit = min_path.size(); i < ilimit ; i = i + 4)
        //  cout << "min_path " << " : " << min_path[i] << " "
        //    << min_path[i+1] << " "
        //    << min_path[i+2] << " "
        //    << min_path[i+3] << endl;


        boost::unordered_map<int, int> convert;
        for (int i = 0, ilimit = min_path.size(); i < ilimit ; i++) {
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
        //for (int i = 0, ilimit = r.cmd_chains.size(); i < ilimit ; i = i + 4)
        //  cout << "cmd_chain " << " : " << r.cmd_chains[i] << " "
        //    << r.cmd_chains[i+1] << " "
        //    << r.cmd_chains[i+2] << " "
        //    << r.cmd_chains[i+3] << " "
        //    << "ps : " << statistic->global_pscount[r.cmd_chains[i+1]]
        //    << " pt : " << statistic->global_ptcount[r.cmd_chains[i+1]]
        //    << " po : " << statistic->global_pocount[r.cmd_chains[i+1]]
        //    << endl;
        //for (int i = 0, ilimit = min_path.size(); i < ilimit ; i = i + 4)
        //  cout << "min_path " << " : " << min_path[i] << " "
        //    << min_path[i+1] << " "
        //    << min_path[i+2] << " "
        //    << min_path[i+3] << endl;
        cout << "estimated cost: " << min_cost << endl;

        r.cmd_chains = min_path;
        return true;
    }

    // for test
    vector<plan> all_plans;
    void test_planner(request_or_reply& r, data_statistic* statistic) {
        this->statistic = statistic;
        min_path.clear();
        path.clear();
        all_plans.clear();
        is_empty = false;
        double cost = 0;
        min_cost = std::numeric_limits<double>::max();
        unordered_map<int, vector<int> > max_selec;

        get_all_plans(r.cmd_chains, max_selec, cost, 0);
        cout << "plan space size: " << all_plans.size() << endl;

        cout << "get_all_plans is finish.\n";
    }

};

