#pragma once

#include <string>
#include <bitset>
#include <vector>
#include <iostream>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string.hpp>
#include <math.h>
#include <unistd.h>

#include "helper.hpp"
#include "timer.hpp"
#include "cost_model.hpp"

using namespace std;

vector<int> empty_ptypes_pos;

class Planner {
    // members
    int tid;
    const double BUDGET_TIME_RATIO = 0.1;
    Dgraph_helper helper;
    Stats *stats;

    vector<ssid_t> triples;
    double min_cost;
    bool no_result = false;

    bool is_empty;            // help identify empty queries
    unordered_set<ssid_t> contain_k2l;  // index which contains k2l
    //bool enable_merge;        // if non endpoint variable > 3, we enable merge

    // for type-centric method
    unordered_map<ssid_t, int> var2ptindex;  // find the first appearance for var
    unordered_map<int, int> type2sub;  // e.g. (sub -1) type (type -3)

    // for dfs
    vector<ssid_t> min_path;
    int _chains_size_div_4 ;

    vector<SPARQLQuery> all_path;

    // remove the attr pattern query before doing the planner and transfer pattern to cmd_chains
    void transfer_to_chains(vector<SPARQLQuery::Pattern> &p,
                                vector<ssid_t> &attr_pattern,
                                vector<int>& attr_pred_chains) {
        for (int i = 0; i < p.size(); i++) {
            SPARQLQuery::Pattern pattern = p[i];
            triples.push_back(pattern.subject);
            triples.push_back(pattern.predicate);
            triples.push_back((ssid_t)pattern.direction);
            triples.push_back(pattern.object);
            if (pattern.pred_type != PREDICATE_ID) {
                attr_pred_chains.push_back(pattern.pred_type);
            }
            if(pattern.subject > 0){
                contain_k2l.insert(pattern.object);
            }
            else if(pattern.object > 0 && pattern.predicate!=TYPE_ID){
                contain_k2l.insert(pattern.subject);
            }
        }

        _chains_size_div_4 = triples.size() / 4;
    }

    // test whether the variable is an end point of the graph
    inline bool is_end_point(ssid_t var, unsigned int pt_bits) {
        for (int i = 0;i < _chains_size_div_4;i++) {
            if(pt_bits & (1 << i))
                continue;
            if (triples[i*4]==var || triples[i*4+3]==var) {
                return false;
            }
        }

        return true;
    }

    CostResult cal_first_pt(ssid_t v1, ssid_t v2, ssid_t d, ssid_t v3, 
                            int mt_factor, int pick, bool from_o2=false) {
        // use index vertex, find subject first
        CostResult cost_result;
        if(v2 == PREDICATE_ID){
           cost_result = CostResult(-1);
        }else{
            cost_result = CostResult(pick);
        }
        cost_result.push_path(v1, v2, d, v3);
        vector<double> updated_result_table;
        cost_result.current_model = model_t::L2U;

        if (from_o2 && v2 == TYPE_ID){
            int tycount = Global::num_servers * mt_factor;
            cost_result.current_model = model_t::T2U;
            ssid_t vtype = v1;

            // v3 only has type v1
            if (stats->global_single2complex.find(vtype) == stats->global_single2complex.end()) {
                double vcount = double(stats->global_tyscount[v1]) / tycount;
                updated_result_table.push_back(vcount);
                updated_result_table.push_back(vtype);
                cost_result.explore_bind += vcount;

            } else {  // v3 has many types, v1 is only one of them
                if (stats->global_tyscount.find(vtype) != stats->global_tyscount.end()) {
                    double vcount = double(stats->global_tyscount[v1]) / tycount;
                    updated_result_table.push_back(vcount);
                    updated_result_table.push_back(vtype);
                    cost_result.explore_bind += vcount;
                }
                unordered_set<ssid_t> type_set = stats->global_single2complex[vtype];
                for (auto iter = type_set.cbegin(); iter != type_set.cend(); ++iter) {
                    double vcount = double(stats->global_tyscount[*iter]) / tycount;
                    updated_result_table.push_back(vcount);
                    updated_result_table.push_back(*iter);
                    cost_result.explore_bind += vcount;
                }
            }
        }else{
            double tycount = 0;
            vector<ty_count> tycountv;
            if (v2 == PREDICATE_ID && d == IN) {  // v1 is predicate
                tycount = Global::num_servers * mt_factor;
                tycountv = stats->global_tystat.pstype[v1];

            } else if (v2 == PREDICATE_ID && d == OUT){
                tycount = Global::num_servers * mt_factor;
                tycountv = stats->global_tystat.potype[v1];

            } else { // v1 > 0 || v3 > 0
                ssid_t v1type = helper.get_type(v1);
                tycountv = from_o2
                        ? stats->global_tystat.fine_type[make_pair(v2, v1type)]
                        : stats->global_tystat.fine_type[make_pair(v1type, v2)];
                for (size_t k = 0; k < tycountv.size(); k++) {
                    tycount += tycountv[k].count;
                }
                tycount = tycount / helper.get_triples_size(v1, v2, d);
            }

            // selectivity and cost estimation, find all types, push into result table
            for (size_t k = 0; k < tycountv.size(); k++) {
                ssid_t vtype = tycountv[k].ty;
                double vcount = double(tycountv[k].count) / tycount;
                updated_result_table.push_back(vcount);
                updated_result_table.push_back(vtype);
                cost_result.explore_bind += vcount;
            }
        }

        // calculate cost
        
        cost_model.calculate(cost_result);
        // store to type table
        cost_result.var2col[v3] = 1;
        cost_result.update_result(updated_result_table);
        cost_result.add_cost = cost_result.explore_bind;
        // cost_result.print();
        return cost_result;
    }

    CostResult cal_rest_pt(ssid_t o1, ssid_t p, ssid_t d, ssid_t o2, int pick,  
                            CostResult &old_result, bool from_o2=false){
        CostResult cost_result(pick);
        cost_result.init(old_result);
        cost_result.push_path(o1, p, d, o2);
        TypeTable& type_table = cost_result.typetable;
        unordered_map<ssid_t, int> &var2col = cost_result.var2col;
        vector<double> updated_result_table;

        // selectivity and cost estimation
        // find all types, push into result table
        // prune based on correlation and constant
        int row_num = type_table.get_row_num();
        int var_col = var2col[o1];
        cost_result.current_model = model_t::K2U;
        bool prune_flag = cost_result.is_known(o2) || (o2>0);
        bool o2_endpoint = (o2<0) && is_end_point(o2, cost_result.pt_bits);
        int dup_flag = cost_result.is_dup(p, o1);
        double max = 0;
        for (size_t i = 0; i < row_num; i++) {
            double pre_count = type_table.get_row_col(i, 0);
            max = (max > pre_count) ? max : pre_count;
        }

        ssid_t o2type;
        double vcount;

        if(o2>0){
            vcount = helper.get_triples_size(o2, p, 1 - d);
            o2type = helper.get_type(o2);
            // logstream(LOG_INFO) << "o2type: " << o2type << LOG_endl;
        }

        for (size_t i = 0; i < row_num; i++) {
            ssid_t pre_tyid = type_table.get_row_col(i, var_col);
            double pre_count = type_table.get_row_col(i, 0);
            cost_result.init_bind += pre_count;

            if (100 * pre_count < max || pre_count < MINIMUM_COUNT_THRESHOLD){
                cost_result.small_prune = true;
                continue;
            }

            // handle type predicate first
            if (p == TYPE_ID) {
                // o1 type const
                if (!from_o2 && o2 > 0){
                    cost_result.current_model = model_t::K2L;
                    cost_result.explore_bind += pre_count;
                    // if pre_tyid do not belong to o2, prune it
                    if (pre_tyid >= 0) {
                        if (pre_tyid != o2) continue;
                    } else {
                        if (stats->global_single2complex.find(o2) != stats->global_single2complex.end()) {
                            cost_result.small_prune = true;
                            unordered_set<ssid_t> type_set = stats->global_single2complex[o2];
                            if (type_set.count(pre_tyid) == 0) continue;
                        } else{
                            continue;
                        }
                    }
                    type_table.append_row_to(i, updated_result_table);
                    cost_result.result_bind += pre_count;
                    continue;
                }
                // o1 type o2 (o2 only show one time)
                else if (!from_o2 && o2_endpoint){
                    prune_flag = true;
                    cost_result.current_model = model_t::K2U;
                    type_table.append_row_to(i, updated_result_table);
                    cost_result.explore_bind += pre_count;
                    cost_result.result_bind += pre_count;
                    continue;
                }
                // o1 type o2 (o2 will show many times, has shown before)
                else if (!from_o2 && cost_result.is_known(o2)) {
                    int other_sub_col = type2sub[var2col[o2]];
                    ssid_t other_type = type_table.get_row_col(i, other_sub_col);
                    cost_result.explore_bind += pre_count;
                    if (helper.compare_types(pre_tyid, other_type)){
                        type_table.append_row_to(i, updated_result_table);
                        cost_result.result_bind += pre_count;
                    }
                        
                }
                // o1 type o2 (o2 will show many times, haven't shown yet)
                else if (!from_o2) {
                    type2sub[type_table.get_col_num()] = var2col[o1];
                    type_table.append_row_to(i, updated_result_table);
                    updated_result_table.push_back(0);  // placeholder
                    cost_result.explore_bind += pre_count;
                    cost_result.result_bind += pre_count;
                    continue;
                }
                //from o2, o2 type o1 (o1 has shown)
                else{
                    int other_sub_col = type2sub[var2col[o1]];
                    pre_tyid = type_table.get_row_col(i, other_sub_col);
                    int tycount = stats->global_tyscount[pre_tyid];

                    double vcount = pre_count * tycount;
                    type_table.append_newv_row_to(i, updated_result_table, vcount);
                    updated_result_table.push_back(pre_tyid);
                    cost_result.explore_bind += vcount;
                    continue;
                }
                
            }// end type situation

            double prune_ratio;
            vector<ty_count> tycountv;
            int tycount = stats->global_tyscount[pre_tyid];
            // cal prune_ratio
            if (!from_o2) {
                int pscount = stats->global_tystat.get_pstype_count(p, pre_tyid);
                if (dup_flag) tycount = pscount;
                prune_ratio = double(pscount) / tycount;  // for cost model
                tycountv = stats->global_tystat.fine_type[make_pair(pre_tyid, p)];
            } else {
                int pocount = stats->global_tystat.get_potype_count(p, pre_tyid);
                if (dup_flag) tycount = pocount;
                prune_ratio = double(pocount) / tycount;  // for cost model
                tycountv = stats->global_tystat.fine_type[make_pair(p, pre_tyid)];
            }
            cost_result.prune_bind += pre_count * (1 - prune_ratio);

            if (o2 > 0) {
                // for constant pruning
                cost_result.current_model = model_t::K2L;
                for (size_t k = 0; k < tycountv.size(); k++){
                    ssid_t vtype = tycountv[k].ty;
                    if (vtype == o2type) {
                        cost_result.result_bind += pre_count/tycount;
                        type_table.append_newv_row_to(i, updated_result_table, cost_result.result_bind);
                    }
                    cost_result.explore_bind += vcount / tycount * pre_count;
                }
            }

            for (size_t k = 0; k < tycountv.size() && o2 < 0; k++) {
                ssid_t vtype = tycountv[k].ty;
                vcount = double(tycountv[k].count) / tycount * pre_count;
                if (vcount < MINIMUM_COUNT_THRESHOLD) {
                    cost_result.small_prune = true;
                    continue;
                } 
                cost_result.explore_bind += vcount;

                if (cost_result.is_known(o2)) {
                    // for variable pruning, K2K
                    cost_result.current_model = model_t::K2K;
                    ssid_t pretype = type_table.get_row_col(i, var2col[o2]);
                    if (vtype == pretype) {
                        int type_num = stats->global_tyscount[pretype];
                        vcount = vcount / type_num;
                        type_table.append_newv_row_to(i, updated_result_table, vcount);
                        cost_result.result_bind += vcount;
                    }
                } else if (o2_endpoint) {
                    // we don't care the type of o2 if it's an end point, so add
                    // it up to one whole item to save storage and speed up
                    double vcount_sum = 0;
                    for (size_t m = 0; m < tycountv.size(); m++)
                        vcount_sum += double(tycountv[m].count) / tycount * pre_count;

                    type_table.append_newv_row_to(i, updated_result_table, vcount_sum);
                    updated_result_table.push_back(0);
                    cost_result.result_bind += vcount_sum;
                    break;
                } else {
                    // normal case unknown
                    type_table.append_newv_row_to(i, updated_result_table, vcount);
                    updated_result_table.push_back(vtype);
                    cost_result.result_bind += vcount;
                }
            }// end o2<0 situation
        }

        if(!cost_result.small_prune && cost_result.init_bind > 0 && cost_result.result_bind == 0){
            no_result = true;
        }

        // calculate cost
        if (prune_flag) {
            cost_result.match_bind = cost_result.result_bind; 
        }else{
            cost_result.var2col[o2] = type_table.get_col_num();
        }
        cost_model.calculate(cost_result);
        cost_result.update_result(updated_result_table);

        if(cost_result.path.size()!= triples.size() && is_end_point(o1, cost_result.pt_bits)){
            cost_result.typetable.merge(var2col[o1]);
        }
        // cost_result.print();
        return cost_result;
    }

    // type-centric dfs enumeration
    void plan_enum(SPARQLQuery &r) {
        uint64_t start_time = timer::get_usec();

        vector<CostResult> median_results;
        median_results.reserve(100);
        for (int pt_pick = 0; pt_pick < _chains_size_div_4; pt_pick++) {
            int i = 4 * pt_pick;
            ssid_t o1 = triples[i];
            ssid_t p = triples[i + 1];
            ssid_t d = triples[i + 2];
            ssid_t o2 = triples[i + 3];

            if (o1 < 0 && o2 < 0 && p!=TYPE_ID) {
                double result_bind = 0;
                if(!is_end_point(o1, 0)){
                    CostResult pto_o1 = cal_first_pt(p, PREDICATE_ID, IN, o1, r.mt_factor, pt_pick);
                    if (no_result) return;
                    if (contain_k2l.find(o1) == contain_k2l.end()) {
                        CostResult p_result = cal_rest_pt(o1, p, d, o2, pt_pick, pto_o1);
                        pto_o1.add_cost = pto_o1.explore_bind + p_result.result_bind;
                        result_bind = p_result.result_bind;
                    }
                    median_results.push_back(pto_o1);
                }
           
                // different direction, find object first
                if (!is_end_point(o2, 0)) {
                    CostResult pto_o2 = cal_first_pt(p, PREDICATE_ID, OUT, o2, r.mt_factor, pt_pick);
                    if (no_result) return;
                    if (contain_k2l.find(o2) == contain_k2l.end()) {
                        if(result_bind == 0){
                            CostResult p_result = cal_rest_pt(o2, p, 1-d, o1, pt_pick, pto_o2, true);
                            result_bind = p_result.result_bind;
                        }
                        pto_o2.add_cost = pto_o2.explore_bind + result_bind;
                    }
                    median_results.push_back(pto_o2);
                }
            }
            if (o1 > 0) {
                median_results.push_back(cal_first_pt(o1, p, d, o2, r.mt_factor, pt_pick));
                if (no_result) return;
            } else if (o2 > 0 && p!=TYPE_ID) {
                median_results.push_back(cal_first_pt(o2, p, IN, o1, r.mt_factor, pt_pick, true));
                if (no_result) return;
            }
        }
        sort(median_results.begin(), median_results.end());
        int path_count = 0;

        while (!median_results.empty()){
            CostResult& old_result = median_results.back();
            
            // budget: return true means continue, false means timeout
            if (old_result.pt_bits == (1 << _chains_size_div_4) - 1) {
                path_count++;
                if (old_result.all_cost < min_cost) {
                    min_cost = old_result.all_cost;
                    min_path = old_result.path;
                }
                // time
                uint64_t latency = timer::get_usec() - start_time;
                if(Global::enable_budget && latency > min_cost*BUDGET_TIME_RATIO){
                    break;
                }
                    
            }

            vector<CostResult> level_results;
            level_results.reserve(_chains_size_div_4);

            for (int pt_pick = 0; pt_pick < _chains_size_div_4; pt_pick++) {
                if (old_result.pt_bits & (1 << pt_pick)) continue;
                int i = 4 * pt_pick;
                ssid_t o1 = triples[i];
                ssid_t p = triples[i + 1];
                ssid_t d = triples[i + 2];
                ssid_t o2 = triples[i + 3];
                CostResult cost_result = CostResult(pt_pick);
                cost_result.init(old_result);

                if(old_result.is_known(o1)){
                    CostResult new_result = cal_rest_pt(o1, p, d, o2, pt_pick, old_result);
                    if (no_result) return;
                    if(min_cost > new_result.all_cost)
                        level_results.push_back(new_result);
                }
                if(old_result.is_known(o2)){
                    CostResult new_result = cal_rest_pt(o2, p, IN, o1, pt_pick, old_result, true);
                    if (no_result) return;
                    if(min_cost > new_result.all_cost)
                        level_results.push_back(new_result);
                }
            }
            median_results.pop_back();
            sort(level_results.begin(), level_results.end());
            median_results.insert(median_results.end(), level_results.begin(), level_results.end());
            level_results.clear();
        }

    }

    bool do_patterns(SPARQLQuery &r, vector<SPARQLQuery::Pattern> &patterns, bool test) {
        //input : patterns
        //transform to : _chains_size_div_4, triples
        vector<ssid_t> attr_pattern;
        vector<int> attr_pred_chains;
        contain_k2l.clear();
        triples.clear();
        transfer_to_chains(patterns, attr_pattern, attr_pred_chains);

        if (_chains_size_div_4 == 0) {
            if (attr_pattern.size() == 0) return false;
            else return true;
        }

        min_path.clear();
        var2ptindex.clear();
        type2sub.clear();
        //enable_merge = false;
        min_cost = std::numeric_limits<double>::max();   
        no_result = false;

        plan_enum(r); // greedy dps function

        if (no_result) {
            return false;
        }

        if (test) return true;

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

    // generate plan for given pattern group
    bool do_group(SPARQLQuery &r, SPARQLQuery::PatternGroup &group, bool test) {
        bool success = true;

        if (group.patterns.size() > 0)
            success = do_patterns(r, group.patterns, test);

        for (auto &g : group.unions)
            success = do_group(r, group, test);

        // FIXME: support optional, filter, etc.
        if(!test){
            logstream(LOG_DEBUG) << "estimate min cost " << min_cost << " usec." << LOG_endl;
        }
        return success;
    }

    // generate/test plan for given query
    bool do_plan(SPARQLQuery &r, bool test) {
        // FIXME: only consider pattern group now
        return do_group(r, r.pattern_group, test);
    }

    // used by set direction
    void set_ptypes_pos(vector<int> &ptypes_pos, const string &dir, int current_order, int raw_order) {
        for (int i = 0; i < ptypes_pos.size(); i ++) {
            // check if any pos need to be changed
            if (ptypes_pos[i] / 4 == raw_order) {
                if (dir == "<") {
                    switch (ptypes_pos[i] % 4) {
                    case 0:
                        ptypes_pos[i] = current_order * 4 + 3;
                        break;
                    case 1:
                        ptypes_pos[i] = current_order * 4 + 1;
                        break;
                    case 3:
                        ptypes_pos[i] = current_order * 4 + 0;
                        break;
                    default:
                        ptypes_pos[i] = current_order * 4 + ptypes_pos[i] % 4;
                    }
                } else if (dir == ">") {
                    ptypes_pos[i] = current_order * 4 + ptypes_pos[i] % 4;
                } else if (dir == "<<") {
                    switch (ptypes_pos[i] % 4) {
                    case 0:
                        ptypes_pos[i] = current_order * 4 + 3;
                        break;
                    case 1:
                        ptypes_pos[i] = current_order * 4 + 0;
                        break;
                    default:
                        ptypes_pos[i] = current_order * 4 + ptypes_pos[i] % 4;
                    }
                } else if (dir == ">>") {
                    switch (ptypes_pos[i] % 4) {
                    case 1:
                        ptypes_pos[i] = current_order * 4 + 0;
                        break;
                    default:
                        ptypes_pos[i] = current_order * 4 + ptypes_pos[i] % 4;
                    }
                }
            }
        }
    }

    // used by set plan
    void set_direction(SPARQLQuery::PatternGroup &group,
                       const vector<int> &orders,
                       const vector<string> &dirs,
                       vector<int> &ptypes_pos = empty_ptypes_pos) {
        vector<SPARQLQuery::Pattern> patterns;
        for (int i = 0; i < orders.size(); i++) {
            // number of orders starts from 1
            SPARQLQuery::Pattern pattern = group.patterns[orders[i] - 1];

            if (dirs[i] == "<") {
                pattern.direction = IN;
                ssid_t t = pattern.subject;
                pattern.subject = pattern.object;
                pattern.object = t;
            } else if (dirs[i] == ">") {
                pattern.direction = OUT;
            } else if (dirs[i] == "<<") {
                pattern.direction = IN;
                pattern.object = pattern.subject;
                pattern.subject = pattern.predicate;
                pattern.predicate = PREDICATE_ID;
            } else if (dirs[i] == ">>") {
                pattern.direction = OUT;
                pattern.subject = pattern.predicate;
                pattern.predicate = PREDICATE_ID;
            }

            if (ptypes_pos.size() != 0)
                set_ptypes_pos(ptypes_pos, dirs[i], patterns.size(), orders[i] - 1);
            patterns.push_back(pattern);
        }
        group.patterns = patterns;
    }

   public:

    Planner() { }

    Planner(int tid, DGraph *graph, Stats *stats) : stats(stats), helper(tid, graph, stats) { }

    CostModel cost_model;
    
    // generate optimal query plan by optimizer
    // @return
    bool generate_plan(SPARQLQuery &r) {
        return do_plan(r, false);
    }

    // test query optimizing (search an optimal plan)
    // @return
    bool test_plan(SPARQLQuery &r) {
        return do_plan(r, true);
    }

    // set user-defuned query plan
    // @return: false if no plan is set
    bool set_plan(SPARQLQuery::PatternGroup &group, istream &fmt_stream,
                  vector<int> &ptypes_pos = empty_ptypes_pos) {
        if (!fmt_stream.good()) {
            logstream(LOG_WARNING) << "Failed to read format file!" << LOG_endl;
            return false;
        }

        // read user-defined query plan file
        vector<int> orders;
        vector<string> dirs;

        int order;
        string dir = ">";
        int nunions = 0, noptionals = 0;
        string line;
        while (std::getline(fmt_stream, line)) {
            boost::trim(line);
            if (boost::starts_with(line, "#") || line.empty()) {
                continue; // skip comments and blank lines
            } else if (line == "{") {
                continue;
            } else if (line == "}") {
                break;
            } else if (boost::starts_with(boost::to_lower_copy(line), "union")) {
                set_plan(group.unions[nunions], fmt_stream);
                nunions ++;
                continue;
            } else if (boost::starts_with(boost::to_lower_copy(line), "optional")) {
                set_plan(group.optional[noptionals], fmt_stream);
                noptionals ++;
                continue;
            }

            istringstream iss(line);
            iss >> order >> dir;
            dirs.push_back(dir);
            orders.push_back(order);
        }

        // correctness check
        if (orders.size() < group.patterns.size()) {
            logstream(LOG_ERROR) << "wrong format file content!" << LOG_endl;
            return false;
        }

        set_direction(group, orders, dirs, ptypes_pos);
        return true;
    }
};
