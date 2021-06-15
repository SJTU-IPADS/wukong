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

#include <tbb/concurrent_queue.h>

#include <algorithm>  // sort
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include "core/common/coder.hpp"
#include "core/common/errors.hpp"
#include "core/common/global.hpp"
#include "core/common/string_server.hpp"
#include "core/common/type.hpp"
#include "core/common/bind.hpp"

#include "core/store/dgraph.hpp"

#include "core/sparql/query.hpp"

// engine
#include "core/engine/msgr.hpp"
#include "core/engine/rmap.hpp"

// utils
#include "utils/assertion.hpp"
#include "utils/math.hpp"
#include "utils/timer.hpp"

namespace wukong {

class TSPARQLEngine {
private:
    int sid;  // server id
    int tid;  // thread id

    StringServer* str_server;
    DGraph* graph;
    Coder* coder;
    Messenger* msgr;

    RMap rmap;  // a map of replies for pending (fork-join) queries
    pthread_spinlock_t rmap_lock;

    enum time_vstat {
        UNDEFINED_UNDEFINED,
        CONST_CONST,
        CONST_UNKNOWN,
        CONST_KNOWN,
        UNKNOWN_CONST,
        UNKNOWN_UNKNOWN,
        UNKNOWN_KNOWN,
        KNOWN_CONST,
        KNOWN_UNKNOWN,
        KNOWN_KNOWN,
        ERR
    };

    time_vstat get_time_vstat(SPARQLQuery& req) {
        SPARQLQuery::TimeIntervalPattern& time_pattern = req.get_pattern().time_interval;
        if(time_pattern.type == SPARQLQuery::UNDEFINED){
            return UNDEFINED_UNDEFINED;
        } else if (time_pattern.type == SPARQLQuery::VALUE_VALUE) {
            return CONST_CONST;
        } else if (time_pattern.type == SPARQLQuery::VAR_VALUE) {
            vstat stat = req.result.var_stat(time_pattern.ts_var);
            return (stat == UNKNOWN_VAR ? UNKNOWN_CONST : KNOWN_CONST);
        } else if (time_pattern.type == SPARQLQuery::VALUE_VAR) {
            vstat stat = req.result.var_stat(time_pattern.te_var);
            return (stat == UNKNOWN_VAR ? CONST_UNKNOWN : CONST_KNOWN);
        } else if (time_pattern.type == SPARQLQuery::VAR_VAR) {
            vstat stat1 = req.result.var_stat(time_pattern.ts_var);
            vstat stat2 = req.result.var_stat(time_pattern.te_var);
            if(stat1 == KNOWN_VAR && stat2 == UNKNOWN_VAR) {
                return KNOWN_UNKNOWN;
            } else if (stat1 == UNKNOWN_VAR && stat2 == UNKNOWN_VAR){
                return UNKNOWN_UNKNOWN;
            } else if (stat1 == KNOWN_VAR && stat2 == KNOWN_VAR){
                return KNOWN_KNOWN;
            } else if (stat1 == UNKNOWN_VAR && stat2 == KNOWN_VAR){
                return UNKNOWN_KNOWN;
            } else { assert(false); }
        }
    }

    /// add elements to result table
    void push_result(std::vector<sid_t>& updated_result_table, int num_id, sid_t id0 = 0, sid_t id1 = 0) {
        if (num_id >= 1) {
            updated_result_table.push_back(id0);
        }
        if (num_id >= 2) {
            updated_result_table.push_back(id1);
        }
    }

    /// set columns of result table according to vstat of ts and te
    void set_result_column(time_vstat ts_var_stat, SPARQLQuery::Result& res, 
                           SPARQLQuery::Pattern& pattern,
                           int var_num, ssid_t var0 = 0, ssid_t var1 = 0) {
        if (var_num >= 1) {
            res.add_var2col(var0, res.get_col_num());
        }
        if (var_num >= 2) {
            res.add_var2col(var1, res.get_col_num() + 1);
        }
        res.set_col_num(res.get_col_num() + var_num);

        switch (ts_var_stat) {
        case CONST_UNKNOWN:
        case KNOWN_UNKNOWN:
            res.add_var2col(pattern.time_interval.te_var, res.get_time_col_num(), TIME_t);
            res.set_time_col_num(res.get_time_col_num() + 1);
            break;
        case UNKNOWN_CONST:
        case UNKNOWN_KNOWN:
            res.add_var2col(pattern.time_interval.ts_var, res.get_time_col_num(), TIME_t);
            res.set_time_col_num(res.get_time_col_num() + 1);
            break;
        case UNKNOWN_UNKNOWN:
            res.add_var2col(pattern.time_interval.ts_var, res.get_time_col_num(), TIME_t);
            res.add_var2col(pattern.time_interval.te_var, res.get_time_col_num() + 1, TIME_t);
            res.set_time_col_num(res.get_time_col_num() + 2);
            break;
        default:
            break;
        }
        return;
    }

    /// append result table according to vstat of ts and te
    void append_result(time_vstat ts_vstat, int row_idx, SPARQLQuery::Result& res,
                       SPARQLQuery::Pattern& pattern,
                       std::vector<sid_t>& updated_result_table,
                       std::vector<int64_t>& updated_time_res_table,
                       edge_t& edge, int num_id,
                       sid_t id0 = 0, sid_t id1 = 0) {
        // edge timestamp
        int64_t idxTS, idxTE;
        if (ts_vstat != UNDEFINED_UNDEFINED) {
            idxTS = edge.ts;
            idxTE = edge.te;
        }
        if (ts_vstat == UNDEFINED_UNDEFINED) {
            res.append_row_to(row_idx, updated_result_table);
            res.append_time_row_to(row_idx, updated_time_res_table);
            push_result(updated_result_table, num_id, id0, id1);
        } else if (ts_vstat == CONST_CONST) {
            if (idxTS == pattern.time_interval.ts_value && idxTE == pattern.time_interval.te_value) {
                res.append_row_to(row_idx, updated_result_table);
                res.append_time_row_to(row_idx, updated_time_res_table);
                push_result(updated_result_table, num_id, id0, id1);
            }
        } else if (ts_vstat == CONST_UNKNOWN) {
            if (idxTS == pattern.time_interval.ts_value) {
                res.append_row_to(row_idx, updated_result_table);
                res.append_time_row_to(row_idx, updated_time_res_table);
                push_result(updated_result_table, num_id, id0, id1);
                updated_time_res_table.push_back(idxTE);
            }
        } else if (ts_vstat == CONST_KNOWN) {
            int64_t oldTE = res.get_time_row_col(row_idx, res.var2col(pattern.time_interval.te_var));
            if (idxTS == pattern.time_interval.ts_value && idxTE == oldTE) {
                res.append_row_to(row_idx, updated_result_table);
                res.append_time_row_to(row_idx, updated_time_res_table);
                push_result(updated_result_table, num_id, id0, id1);
            }
        } else if (ts_vstat == UNKNOWN_CONST) {
            if (idxTE == pattern.time_interval.te_value) {
                res.append_row_to(row_idx, updated_result_table);
                res.append_time_row_to(row_idx, updated_time_res_table);
                push_result(updated_result_table, num_id, id0, id1);
                updated_time_res_table.push_back(idxTS);
            }
        } else if (ts_vstat == UNKNOWN_UNKNOWN) {
            res.append_row_to(row_idx, updated_result_table);
            res.append_time_row_to(row_idx, updated_time_res_table);
            push_result(updated_result_table, num_id, id0, id1);
            updated_time_res_table.push_back(idxTS);
            updated_time_res_table.push_back(idxTE);
        } else if (ts_vstat == UNKNOWN_KNOWN) {
            int64_t oldTE = res.get_time_row_col(row_idx, res.var2col(pattern.time_interval.te_var));
            if (idxTE == oldTE) {
                res.append_row_to(row_idx, updated_result_table);
                res.append_time_row_to(row_idx, updated_time_res_table);
                push_result(updated_result_table, num_id, id0, id1);
                updated_time_res_table.push_back(idxTS);
            }
        } else if (ts_vstat == KNOWN_CONST) {
            int64_t oldTS = res.get_time_row_col(row_idx, res.var2col(pattern.time_interval.ts_var));
            if (idxTE == pattern.time_interval.te_value && idxTS == oldTS) {
                res.append_row_to(row_idx, updated_result_table);
                res.append_time_row_to(row_idx, updated_time_res_table);
                push_result(updated_result_table, num_id, id0, id1);
            }
        } else if (ts_vstat == KNOWN_UNKNOWN) {
            int64_t oldTS = res.get_time_row_col(row_idx, res.var2col(pattern.time_interval.ts_var));
            if (idxTS == oldTS) {
                res.append_row_to(row_idx, updated_result_table);
                res.append_time_row_to(row_idx, updated_time_res_table);
                push_result(updated_result_table, num_id, id0, id1);
                updated_time_res_table.push_back(idxTE);
            }
        } else {
            int64_t oldTS = res.get_time_row_col(row_idx, res.var2col(pattern.time_interval.ts_var));
            int64_t oldTE = res.get_time_row_col(row_idx, res.var2col(pattern.time_interval.te_var));
            if (idxTS == oldTS && idxTE == oldTE) {
                res.append_row_to(row_idx, updated_result_table);
                res.append_time_row_to(row_idx, updated_time_res_table);
                push_result(updated_result_table, num_id, id0, id1);
            }
        }
    }

    /// A query whose parent's PGType is UNION may call this pattern
    void index_to_known(SPARQLQuery& req) {
        SPARQLQuery::Pattern& pattern = req.get_pattern();
        int64_t ts = req.ts;
        int64_t te = req.te;
        time_vstat ts_vstat = get_time_vstat(req);

        ssid_t tpid = pattern.subject;
        ssid_t id01 = pattern.predicate;
        dir_t d = pattern.direction;
        ssid_t end = pattern.object;
        ASSERT_ERROR_CODE((pattern.pred_type == static_cast<char>(SID_t)), UNKNOWN_PATTERN);

        SPARQLQuery::Result& res = req.result;
        int col = res.var2col(end);

        ASSERT_ERROR_CODE(col != NO_RESULT, VERTEX_INVALID);
        ASSERT_ERROR_CODE(id01 == PREDICATE_ID || id01 == TYPE_ID, OBJ_ERROR);  // predicate or type index

        std::vector<sid_t> updated_result_table;
        std::vector<int64_t> updated_time_res_table;

        uint64_t sz = 0;
        edge_t* edges = graph->get_index(tid, tpid, d, sz);

        boost::unordered_set<sid_t> unique_set;
        // FIXME: multi-threading
        std::vector<edge_t> edge_table;
        // every thread takes a part of consecutive edges
        for (uint64_t k = 0; k < sz; k++)
            if (edges[k].valid(ts, te)) {
                unique_set.insert(edges[k].val);
                edge_table.push_back(edges[k]);
            }

        int nrows = res.get_row_num();
        for (int i = 0; i < nrows; i++) {
            sid_t cur = res.get_row_col(i, col);
            if (req.pg_type == SPARQLQuery::PGType::OPTIONAL) {
                // matched
                if (unique_set.find(cur) != unique_set.end()) {
                    res.optional_matched_rows[i] = (true && res.optional_matched_rows[i]);
                } else {
                    if (res.optional_matched_rows[i])
                        req.correct_optional_result(i);
                    res.optional_matched_rows[i] = false;
                }
            } else {
                // timestamp-related logic
                if (ts_vstat != UNDEFINED_UNDEFINED) {
                    for (int k = 0; k < edge_table.size(); k++) {
                        if (edge_table[k].val == cur) {
                            append_result(ts_vstat, i, res, pattern, 
                                          updated_result_table, updated_time_res_table, 
                                          edge_table[k], 0);
                        }
                    }
                } else if (unique_set.find(cur) != unique_set.end()) {  // matched
                    res.append_row_to(i, updated_result_table);
                    res.append_time_row_to(i, updated_time_res_table);
                }
            }
        }

        if (req.pg_type != SPARQLQuery::PGType::OPTIONAL) {
            // update result and metadata
            res.result_table.swap(updated_result_table);
            res.time_res_table.swap(updated_time_res_table);
            set_result_column(ts_vstat, res, pattern, 0);
            res.update_nrows();
        }

        req.pattern_step++;
    }

    /// A query whose parent's PGType is UNION may call this pattern
    void const_to_known(SPARQLQuery& req) {
        SPARQLQuery::Pattern& pattern = req.get_pattern();
        int64_t ts = req.ts;
        int64_t te = req.te;
        time_vstat ts_stat = get_time_vstat(req);

        ssid_t start = pattern.subject;
        ssid_t pid = pattern.predicate;
        dir_t d = pattern.direction;
        ssid_t end = pattern.object;
        ASSERT_ERROR_CODE((pattern.pred_type == static_cast<char>(SID_t)), UNKNOWN_PATTERN);

        SPARQLQuery::Result& res = req.result;
        int col = res.var2col(end);

        ASSERT_ERROR_CODE(col != NO_RESULT, VERTEX_INVALID);

        uint64_t sz = 0;
        edge_t* vids = graph->get_triples(tid, start, pid, d, sz);

        boost::unordered_set<sid_t> unique_set;

        std::vector<edge_t> edges;
        for (uint64_t k = 0; k < sz; k++) {
            if (vids[k].valid(ts, te)) {
                unique_set.insert(vids[k].val);
                edges.push_back(vids[k]);
            }
        }

        if (req.pg_type == SPARQLQuery::PGType::OPTIONAL) {
            int nrows = res.get_row_num();
            ASSERT(nrows == res.optional_matched_rows.size());
            for (uint64_t i = 0; i < nrows; i++) {
                // matched
                if (unique_set.find(res.get_row_col(i, col)) != unique_set.end()) {
                    res.optional_matched_rows[i] = (true && res.optional_matched_rows[i]);
                } else {
                    if (res.optional_matched_rows[i])
                        req.correct_optional_result(i);
                    res.optional_matched_rows[i] = false;
                }
            }
        } else {
            std::vector<sid_t> updated_result_table;
            std::vector<int64_t> updated_time_res_table;
            int nrows = res.get_row_num();
            for (uint64_t i = 0; i < nrows; i++) {
                sid_t cur = res.get_row_col(i, col);
                // timestamp-related logic
                if (ts_stat != UNDEFINED_UNDEFINED) {
                    for (int k = 0; k < edges.size(); k++) {
                        if (edges[k].val == cur) {
                            append_result(ts_stat, i, res, pattern, 
                                          updated_result_table, updated_time_res_table, 
                                          edges[k], 0);
                        }
                    }
                } else if (unique_set.find(cur) != unique_set.end()) {
                    res.append_row_to(i, updated_result_table);
                    res.append_time_row_to(i, updated_time_res_table);
                }
            }

            // update result and metadata
            res.result_table.swap(updated_result_table);
            res.time_res_table.swap(updated_time_res_table);
            set_result_column(ts_stat, res, pattern, 0);
        }

        res.update_nrows();
        req.pattern_step++;
    }

    /// IDX P ?X . (IDX and P are GIVEN, ?X is UNKNOWN)
    /// e.g., "?X __PREDICATE__ ub:subOrganizationOf" (predicate index)
    /// e.g., "?X  rdf:type  ub:GraduateStudent"      (type index)
    ///
    /// 1) Use [IDX]+[P] to retrieve all of neighbors (?X) on every node
    void index_to_unknown(SPARQLQuery& req) {
        SPARQLQuery::Pattern& pattern = req.get_pattern();
        int64_t ts = req.ts;
        int64_t te = req.te;
        time_vstat ts_vstat = get_time_vstat(req);

        ssid_t tpid = pattern.subject;
        ssid_t id01 = pattern.predicate;
        dir_t d = pattern.direction;
        ssid_t end = pattern.object;
        ASSERT_ERROR_CODE((pattern.pred_type == static_cast<char>(SID_t)), UNKNOWN_PATTERN);

        SPARQLQuery::Result& res = req.result;

        ASSERT_ERROR_CODE(id01 == PREDICATE_ID || id01 == TYPE_ID, OBJ_ERROR);  // predicate or type index
        // ASSERT_ERROR_CODE(res.get_col_num() == 0, FIRST_PATTERN_ERROR);

        std::vector<sid_t> updated_result_table;
        std::vector<int64_t> updated_time_res_table;

        uint64_t sz = 0;
        edge_t* edges = graph->get_index(tid, tpid, d, sz);
        int start = req.mt_tid % req.mt_factor;
        int length = sz / req.mt_factor;

        int nrows = res.get_row_num();
        // Fix time_stat
        if (id01 == PREDICATE_ID) ts_vstat = UNDEFINED_UNDEFINED;

        // every thread takes a part of consecutive edges
        for (uint64_t k = start * length; k < (start + 1) * length; k++) {
            if (edges[k].valid(ts, te)) {
                for (int i = 0; i < std::max(nrows, 1); i++) {
                    if (id01 == PREDICATE_ID) {
                        res.append_row_to(i, updated_result_table);
                        res.append_time_row_to(i, updated_time_res_table);
                        updated_result_table.push_back(edges[k].val);
                    } else {
                        append_result(ts_vstat, i, res, pattern, 
                                      updated_result_table, updated_time_res_table, 
                                      edges[k], 1, edges[k].val);
                    }
                }
            }
        }

        // fixup the last participant
        if (start == req.mt_factor - 1) {
            for (uint64_t k = (start + 1) * length; k < sz; k++) {
                if (edges[k].valid(ts, te)) {
                    for (int i = 0; i < std::max(nrows, 1); i++) {
                        if (id01 == PREDICATE_ID) {
                            res.append_row_to(i, updated_result_table);
                            res.append_time_row_to(i, updated_time_res_table);
                            updated_result_table.push_back(edges[k].val);
                        } else {
                            append_result(ts_vstat, i, res, pattern, 
                                          updated_result_table, updated_time_res_table, 
                                          edges[k], 1, edges[k].val);
                        }
                    }
                }
            }
        }

        // update result and metadata
        res.result_table.swap(updated_result_table);
        res.time_res_table.swap(updated_time_res_table);

        set_result_column(ts_vstat, res, pattern, 1, end);
        res.update_nrows();

        req.pattern_step++;
        req.local_var = end;
    }

    /// C P ?X . (C/P: GIVEN, ?X: UNKNOWN)
    /// e.g., "?X ub:subOrganizationOf <http://www.University0.edu>"        (Triple)
    /// e.g., "<http://www.Department7.University0.edu/Course44> ub:id ?X"  (Attribute)
    ///
    /// 1) Use [C]+[P] to retrieve all of neighbors (?X) on a certain (HASH(C)) node
    void const_to_unknown(SPARQLQuery& req) {
        SPARQLQuery::Pattern& pattern = req.get_pattern();
        int64_t ts = req.ts;
        int64_t te = req.te;
        time_vstat ts_vstat = get_time_vstat(req);

        ssid_t start = pattern.subject;
        ssid_t pid = pattern.predicate;
        dir_t d = pattern.direction;
        ssid_t end = pattern.object;

        char type = pattern.pred_type;
        ASSERT_ERROR_CODE(((Global::enable_vattr) || (type == static_cast<char>(SID_t))), ATTR_DISABLE);

        SPARQLQuery::Result& res = req.result;
        int nrows = res.get_row_num();

        if (type == SID_t) {
            // ASSERT_ERROR_CODE(res.get_col_num() == 0,
            //                   FIRST_PATTERN_ERROR);  // MUST be the first triple pattern

            uint64_t sz = 0;
            edge_t* vids = graph->get_triples(tid, start, pid, d, sz);
            std::vector<sid_t> updated_result_table;
            std::vector<int64_t> updated_time_res_table;
            for (uint64_t k = 0; k < sz; k++) {
                if (vids[k].valid(ts, te)) {
                    for (int i = 0; i < std::max(nrows, 1); i++) {
                        append_result(ts_vstat, i, res, pattern, 
                                      updated_result_table, updated_time_res_table, 
                                      vids[k], 1, vids[k].val);
                    }
                }
            }

            // update result and metadata
            res.result_table.swap(updated_result_table);
            res.time_res_table.swap(updated_time_res_table);
            set_result_column(ts_vstat, res, pattern, 1, end);
            res.update_nrows();
        } else {
            ASSERT(d == OUT);  // attribute always uses OUT
            ASSERT_ERROR_CODE(res.get_attr_col_num() == 0,
                              FIRST_PATTERN_ERROR);  // MUST be the first triple pattern

            bool has_value;
            attr_t v = graph->get_attr(tid, start, pid, d, has_value);
            ASSERT(type == boost::apply_visitor(variant_type(), v));
            std::vector<attr_t> updated_attr_table;
            if (has_value)
                updated_attr_table.push_back(v);

            // update (attribute) result and metadata
            res.attr_res_table.swap(updated_attr_table);
            res.add_var2col(end, res.get_attr_col_num(), type);
            res.set_attr_col_num(res.get_attr_col_num() + 1);
            res.update_nrows();
        }

        req.pattern_step++;
    }

    /// ?X P ?Y . (?X: KNOWN, P: GIVEN, ?X: UNKNOWN)
    /// e.g., "?X rdf:type ub:GraduateStudent"
    ///       "?X ub:memberOf ?Y"
    /// e.g., "?Z ub:undergraduateDegreeFrom ?X"
    ///       "?Z ub:memberOf ?Y"
    ///       "?Y ub:subOrganizationOf ?X"
    ///
    /// 1) Use [?X]+[P] to retrieve all of neighbors (?Y)
    void known_to_unknown(SPARQLQuery& req) {
        SPARQLQuery::Pattern& pattern = req.get_pattern();
        int64_t ts = req.ts;
        int64_t te = req.te;
        time_vstat ts_vstat = get_time_vstat(req);

        ssid_t start = pattern.subject;
        ssid_t pid = pattern.predicate;
        dir_t d = pattern.direction;
        ssid_t end = pattern.object;

        char type = pattern.pred_type;
        ASSERT_ERROR_CODE(((Global::enable_vattr) || (type == static_cast<char>(SID_t))), ATTR_DISABLE);

        SPARQLQuery::Result& res = req.result;

        std::vector<sid_t> updated_result_table;
        std::vector<int64_t> updated_time_res_table;
        std::vector<attr_t> updated_attr_table;

        // In most time, #attr_res_table table is equal to #result_table
        // reserve size of updated_result_table to #result_table
        updated_result_table.reserve(res.result_table.size());
        updated_time_res_table.reserve(res.time_res_table.size());
        if (Global::enable_vattr)
            updated_attr_table.reserve(res.result_table.size());

        if (type == SID_t) {
            std::vector<bool> updated_optional_matched_rows;
            if (req.pg_type == SPARQLQuery::PGType::OPTIONAL)
                updated_optional_matched_rows.reserve(res.optional_matched_rows.size());

            sid_t cached = BLANK_ID;  // simple dedup for consecutive same vertices
            edge_t* vids = NULL;
            uint64_t sz = 0;
            int nrows = res.get_row_num();
            for (int i = 0; i < nrows; i++) {
                sid_t cur = res.get_row_col(i, res.var2col(start));

                // optional
                if (req.pg_type == SPARQLQuery::PGType::OPTIONAL && (!res.optional_matched_rows[i] || cur == BLANK_ID)) {
                    res.append_row_to(i, updated_result_table);
                    res.append_time_row_to(i, updated_time_res_table);
                    updated_result_table.push_back(BLANK_ID);
                    updated_optional_matched_rows.push_back(res.optional_matched_rows[i]);
                    continue;
                }

                if (cur != cached) {  // new KNOWN
                    cached = cur;
                    if (pid == TYPE_ID && d == IN)
                        vids = graph->get_index(tid, cur, d, sz);
                    else
                        vids = graph->get_triples(tid, cur, pid, d, sz);
                }

                // append a new intermediate result (row)
                if (req.pg_type == SPARQLQuery::PGType::OPTIONAL) {
                    if (sz > 0) {
                        for (uint64_t k = 0; k < sz; k++) {
                            if (vids[k].valid(ts, te)) {
                                append_result(ts_vstat, i, res, pattern, 
                                              updated_result_table, updated_time_res_table, 
                                              vids[k], 1, vids[k].val);
                                updated_optional_matched_rows.push_back(true);
                            }
                        }
                    } else {
                        res.append_row_to(i, updated_result_table);
                        res.append_time_row_to(i, updated_time_res_table);
                        
                        updated_result_table.push_back(BLANK_ID);
                        updated_optional_matched_rows.push_back(true);
                    }
                } else {
                    for (uint64_t k = 0; k < sz; k++) {
                        if (vids[k].valid(ts, te)) {
                            append_result(ts_vstat, i, res, pattern, 
                                          updated_result_table, updated_time_res_table, 
                                          vids[k], 1, vids[k].val);
                            // update attribute table to map the result table
                            if (Global::enable_vattr)
                                res.append_attr_row_to(i, updated_attr_table);
                        }
                    }
                }
            }

            if (req.pg_type == SPARQLQuery::PGType::OPTIONAL)
                res.optional_matched_rows.swap(updated_optional_matched_rows);

            // update result and (attributed) result
            res.result_table.swap(updated_result_table);
            res.time_res_table.swap(updated_time_res_table);
            if (Global::enable_vattr)
                res.attr_res_table.swap(updated_attr_table);

            // update metadata
            set_result_column(ts_vstat, res, pattern, 1, end);
            res.update_nrows();
        } else {
            // TODO: TRDF dont support attr?
            ASSERT(d == OUT);  // attribute always uses OUT
            int nrows = res.get_row_num();
            for (int i = 0; i < nrows; i++) {
                sid_t prev_id = res.get_row_col(i, res.var2col(start));
                bool has_value;
                attr_t v = graph->get_attr(tid, prev_id, pid, d, has_value);
                ASSERT(type == boost::apply_visitor(variant_type(), v));
                if (has_value) {
                    res.append_row_to(i, updated_result_table);
                    res.append_time_row_to(i, updated_time_res_table);
                    res.append_attr_row_to(i, updated_attr_table);
                    updated_attr_table.push_back(v);
                }
            }

            // update result and attributed result
            res.result_table.swap(updated_result_table);
            res.time_res_table.swap(updated_time_res_table);
            res.attr_res_table.swap(updated_attr_table);

            // update metadata
            res.add_var2col(end, res.get_attr_col_num(), type);
            res.set_attr_col_num(res.get_attr_col_num() + 1);
            res.update_nrows();
        }

        req.pattern_step++;
    }

    /// ?Y P ?X . (?Y:KNOWN, ?X: KNOWN)
    /// e.g., "?Z ub:undergraduateDegreeFrom ?X"
    ///       "?Z ub:memberOf ?Y"
    ///       "?Y ub:subOrganizationOf ?X"
    ///
    /// 1) Use [?Y]+[P] to retrieve all of neighbors (X')
    /// 2) Match KNOWN X with X'
    void known_to_known(SPARQLQuery& req) {
        SPARQLQuery::Pattern& pattern = req.get_pattern();
        int64_t ts = req.ts;
        int64_t te = req.te;
        time_vstat ts_vstat = get_time_vstat(req);

        ssid_t start = pattern.subject;
        ssid_t pid = pattern.predicate;
        dir_t d = pattern.direction;
        ssid_t end = pattern.object;
        ASSERT_ERROR_CODE((pattern.pred_type == static_cast<char>(SID_t)), UNKNOWN_PATTERN);

        SPARQLQuery::Result& res = req.result;

        std::vector<sid_t> updated_result_table;
        std::vector<int64_t> updated_time_res_table;
        std::vector<attr_t> updated_attr_table;

        // simple dedup for consecutive same vertices
        sid_t cached = BLANK_ID;
        edge_t* vids = NULL;
        uint64_t sz = 0;

        int nrows = res.get_row_num();
        for (int i = 0; i < nrows; i++) {
            sid_t cur = res.get_row_col(i, res.var2col(start));
            if (cur != cached) {  // a new vertex
                cached = cur;
                vids = graph->get_triples(tid, cur, pid, d, sz);
            }

            sid_t known = res.get_row_col(i, res.var2col(end));
            if (req.pg_type == SPARQLQuery::PGType::OPTIONAL) {
                bool matched = false;
                for (uint64_t k = 0; k < sz; k++) {
                    if (vids[k].valid(ts, te) && vids[k].val == known) {
                        matched = true;
                        break;
                    }
                }
                if (res.optional_matched_rows[i] && (!matched))
                    req.correct_optional_result(i);
                res.optional_matched_rows[i] = (matched && res.optional_matched_rows[i]);
            } else {
                for (uint64_t k = 0; k < sz; k++) {
                    if (vids[k].valid(ts, te) && vids[k].val == known) {
                        // append a matched intermediate result
                        append_result(ts_vstat, i, res, pattern, 
                                      updated_result_table, updated_time_res_table, 
                                      vids[k], 0);
                        if (Global::enable_vattr)
                            res.append_attr_row_to(i, updated_attr_table);
                        break;
                    }
                }
            }
        }

        if (req.pg_type != SPARQLQuery::PGType::OPTIONAL) {
            // update result and metadata
            res.result_table.swap(updated_result_table);
            res.time_res_table.swap(updated_time_res_table);
            set_result_column(ts_vstat, res, pattern, 0);
            if (Global::enable_vattr)
                res.attr_res_table.swap(updated_attr_table);
            res.update_nrows();
        }

        req.pattern_step++;
    }

    /// ?X P C . (?X is KNOWN)
    /// e.g., "?X rdf:type ub:FullProfessor"
    ///       "?X ub:worksFor <http://www.Department0.University0.edu>"
    ///
    /// 1) Use [?X]+[P] to retrieve all of neighbors (C')
    /// 2) Match const C with C'
    void known_to_const(SPARQLQuery& req) {
        SPARQLQuery::Pattern& pattern = req.get_pattern();
        int64_t ts = req.ts;
        int64_t te = req.te;
        time_vstat ts_vstat = get_time_vstat(req);

        ssid_t start = pattern.subject;
        ssid_t pid = pattern.predicate;
        dir_t d = pattern.direction;
        ssid_t end = pattern.object;
        ASSERT_ERROR_CODE((pattern.pred_type == static_cast<char>(SID_t)), UNKNOWN_PATTERN);

        SPARQLQuery::Result& res = req.result;

        std::vector<sid_t> updated_result_table;
        std::vector<int64_t> updated_time_res_table;
        std::vector<attr_t> updated_attr_table;

        // simple dedup for consecutive same vertices
        edge_t* vids = NULL;
        uint64_t sz = 0;
        int nrows = res.get_row_num();
        for (int i = 0; i < nrows; i++) {
            sid_t cur = res.get_row_col(i, res.var2col(start));
            bool exist = false;
            vids = graph->get_triples(tid, cur, pid, d, sz);

            for (uint64_t k = 0; k < sz; k++) {
                if (vids[k].valid(ts, te) && vids[k].val == end) {
                    // append a matched intermediate result
                    exist = true;
                    if (req.pg_type != SPARQLQuery::PGType::OPTIONAL) {
                        append_result(ts_vstat, i, res, pattern, 
                                      updated_result_table, updated_time_res_table, 
                                      vids[k], 0);
                        if (Global::enable_vattr)
                            res.append_attr_row_to(i, updated_attr_table);
                    }
                    break;
                }
            }
            if (req.pg_type == SPARQLQuery::PGType::OPTIONAL) {
                if (res.optional_matched_rows[i] && (!exist)) req.correct_optional_result(i);
                res.optional_matched_rows[i] = (exist && res.optional_matched_rows[i]);
            }
        }

        if (req.pg_type != SPARQLQuery::PGType::OPTIONAL) {
            // update result and metadata
            res.result_table.swap(updated_result_table);
            res.time_res_table.swap(updated_time_res_table);
            set_result_column(ts_vstat, res, pattern, 0);
            if (Global::enable_vattr)
                res.attr_res_table.swap(updated_attr_table);
            res.update_nrows();
        }

        req.pattern_step++;
    }

    /// C ?P ?X . (?P and ?X are UNKNOWN)
    /// e.g., "?X ?P <http://www.Department0.University0.edu>"
    ///
    /// 1) Use [C]+[__PREDICATE__] to retrieve all of predicates (?P)
    /// 2) Use [C]+[?P] to retrieve all of neighbors (?X)
    void const_unknown_unknown(SPARQLQuery& req) {
        SPARQLQuery::Pattern& pattern = req.get_pattern();
        int64_t ts = req.ts;
        int64_t te = req.te;
        time_vstat ts_vstat = get_time_vstat(req);

        ssid_t start = pattern.subject;
        ssid_t pid = pattern.predicate;
        dir_t d = pattern.direction;
        ssid_t end = pattern.object;
        ASSERT_ERROR_CODE((pattern.pred_type == static_cast<char>(SID_t)), UNKNOWN_PATTERN);

        SPARQLQuery::Result& res = req.result;

        uint64_t npids = 0;
        edge_t* pids = graph->get_triples(tid, start, PREDICATE_ID, d, npids);

        // use a local buffer to store "known" predicates
        std::vector<edge_t> tpids(npids);
        std::copy(pids, pids + npids, tpids.data());

        int nrows = res.get_row_num();

        std::vector<sid_t> updated_result_table;
        std::vector<int64_t> updated_time_res_table;
        for (uint64_t p = 0; p < npids; p++) {
            if (tpids[p].valid(ts, te)) {
                uint64_t sz = 0;
                edge_t* vids = graph->get_triples(tid, start, tpids[p].val, d, sz);
                for (uint64_t k = 0; k < sz; k++) {
                    if (vids[k].valid(ts, te)) {
                        for (int i = 0; i < std::max(nrows, 1); i++) {
                            append_result(ts_vstat, i, res, pattern, 
                                          updated_result_table, updated_time_res_table, 
                                          vids[k], 2, tpids[p].val, vids[k].val);
                        }
                    }
                }
            }
        }

        // update result and metadata
        res.result_table.swap(updated_result_table);
        res.time_res_table.swap(updated_time_res_table);
        set_result_column(ts_vstat, res, pattern, 2, pid, end);
        res.update_nrows();
        req.pattern_step++;
    }

    /// ?X ?P ?Y . (?X is KNOWN; ?P and ?X are UNKNOWN)
    /// e.g., "?X ub:subOrganizationOf <http://www.University0.edu>"
    ///       "?X ?P ?Y"
    ///
    /// 1) Use [?X]+[__PREDICATE__] to retrieve all of predicates (?P)
    /// 2) Use [?X]+[?P] to retrieve all of neighbors (?Y)
    void known_unknown_unknown(SPARQLQuery& req) {
        SPARQLQuery::Pattern& pattern = req.get_pattern();
        int64_t ts = req.ts;
        int64_t te = req.te;
        time_vstat ts_vstat = get_time_vstat(req);

        ssid_t start = pattern.subject;
        ssid_t pid = pattern.predicate;
        dir_t d = pattern.direction;
        ssid_t end = pattern.object;
        ASSERT_ERROR_CODE((pattern.pred_type == static_cast<char>(SID_t)), UNKNOWN_PATTERN);

        SPARQLQuery::Result& res = req.result;

        std::vector<sid_t> updated_result_table;
        std::vector<int64_t> updated_time_res_table;
        int nrows = res.get_row_num();
        for (int i = 0; i < nrows; i++) {
            sid_t cur = res.get_row_col(i, res.var2col(start));
            uint64_t npids = 0;
            edge_t* pids = graph->get_triples(tid, cur, PREDICATE_ID, d, npids);

            // use a local buffer to store "known" predicates
            std::vector<edge_t> tpids(npids);
            std::copy(pids, pids + npids, tpids.data());

            for (uint64_t p = 0; p < npids; p++) {
                if (tpids[p].valid(ts, te)) {
                    uint64_t sz = 0;
                    edge_t* vids = graph->get_triples(tid, cur, tpids[p].val, d, sz);
                    for (uint64_t k = 0; k < sz; k++) {
                        if (vids[k].valid(ts, te)) {
                            append_result(ts_vstat, i, res, pattern, 
                                          updated_result_table, updated_time_res_table, 
                                          vids[k], 2, tpids[p].val, vids[k].val);
                        }
                    }
                }
            }
        }

        // update result and metadata
        res.result_table.swap(updated_result_table);
        res.time_res_table.swap(updated_time_res_table);
        set_result_column(ts_vstat, res, pattern, 2, pid, end);
        res.update_nrows();

        req.pattern_step++;
    }

    /// ?X ?P C . (?X is KNOWN; ?P is UNKNOWN)
    /// e.g., "<http://www.University0.edu> ub:subOrganizationOf ?X"
    ///       "?X ?P <http://www.Department4.University0.edu>"
    ///
    /// 1) Use [?X]+[__PREDICATE__] to retrieve all of predicates (?P)
    /// 2) Use [?X]+[?P] to retrieve all of neighbors (C')
    /// 3) Match const C with C'
    void known_unknown_const(SPARQLQuery& req) {
        SPARQLQuery::Pattern& pattern = req.get_pattern();
        int64_t ts = req.ts;
        int64_t te = req.te;
        time_vstat ts_vstat = get_time_vstat(req);

        ssid_t start = pattern.subject;
        ssid_t pid = pattern.predicate;
        dir_t d = pattern.direction;
        ssid_t end = pattern.object;
        ASSERT_ERROR_CODE((pattern.pred_type == static_cast<char>(SID_t)), UNKNOWN_PATTERN);

        SPARQLQuery::Result& res = req.result;

        std::vector<sid_t> updated_result_table;
        std::vector<int64_t> updated_time_res_table;
        int nrows = res.get_row_num();
        for (int i = 0; i < nrows; i++) {
            sid_t prev_id = res.get_row_col(i, res.var2col(start));
            uint64_t npids = 0;
            edge_t* pids = graph->get_triples(tid, prev_id, PREDICATE_ID, d, npids);

            // use a local buffer to store "known" predicates
            std::vector<edge_t> tpids(npids);
            std::copy(pids, pids + npids, tpids.data());

            for (uint64_t p = 0; p < npids; p++) {
                if (tpids[p].valid(ts, te)) {
                    uint64_t sz = 0;
                    edge_t* vids = graph->get_triples(tid, prev_id, tpids[p].val, d, sz);
                    for (uint64_t k = 0; k < sz; k++) {
                        if (vids[k].valid(ts, te) && vids[k].val == end) {
                            append_result(ts_vstat, i, res, pattern, 
                                          updated_result_table, updated_time_res_table, 
                                          vids[k], 1, tpids[p].val);
                            break;
                        }
                    }
                }
            }
        }

        // update result and metadata
        res.result_table.swap(updated_result_table);
        res.time_res_table.swap(updated_time_res_table);
        set_result_column(ts_vstat, res, pattern, 1, pid);
        res.update_nrows();

        req.pattern_step++;
    }

    /// C1 ?P C2 . (?P is UNKNOWN)
    /// e.g., "<http://www.University0.edu> ?P <http://www.Department4.University0.edu>"
    ///
    /// 1) Use C+[__PREDICATE__] to retrieve all of predicates (?P)
    /// 2) Use [?X]+[?P] to retrieve all of neighbors (C')
    /// 3) Match const C with C'
    void const_unknown_const(SPARQLQuery& req) {
        SPARQLQuery::Pattern& pattern = req.get_pattern();
        int64_t ts = req.ts;
        int64_t te = req.te;
        time_vstat ts_vstat = get_time_vstat(req);

        ssid_t start = pattern.subject;
        ssid_t pid = pattern.predicate;
        dir_t d = pattern.direction;
        ssid_t end = pattern.object;
        ASSERT_ERROR_CODE((pattern.pred_type == static_cast<char>(SID_t)), UNKNOWN_PATTERN);

        SPARQLQuery::Result& res = req.result;

        // the query plan is wrong
        // ASSERT_ERROR_CODE(res.get_col_num() == 0, FIRST_PATTERN_ERROR);

        uint64_t npids = 0;
        edge_t* pids = graph->get_triples(tid, start, PREDICATE_ID, d, npids);

        // use a local buffer to store "known" predicates
        std::vector<edge_t> tpids(npids);
        std::copy(pids, pids + npids, tpids.data());

        int nrows = res.get_row_num();

        std::vector<sid_t> updated_result_table;
        std::vector<int64_t> updated_time_res_table;
        for (uint64_t p = 0; p < npids; p++) {
            if (tpids[p].valid(ts, te)) {
                uint64_t sz = 0;
                edge_t* vids = graph->get_triples(tid, start, tpids[p].val, d, sz);
                for (uint64_t k = 0; k < sz; k++) {
                    if (vids[k].valid(ts, te) && vids[k].val == end) {
                        for (int i = 0; i < std::max(nrows, 1); i++) {
                            append_result(ts_vstat, i, res, pattern, 
                                          updated_result_table, updated_time_res_table, 
                                          vids[k], 1, tpids[p].val);
                        }
                    }
                }
            }
        }

        // update result and metadata
        res.result_table.swap(updated_result_table);
        res.time_res_table.swap(updated_time_res_table);
        set_result_column(ts_vstat, res, pattern, 1, pid);
        res.update_nrows();

        req.pattern_step++;
    }

    std::vector<SPARQLQuery> generate_sub_query(SPARQLQuery& req, bool need_split = true) {
        SPARQLQuery::Pattern& pattern = req.get_pattern();
        ssid_t start = pattern.subject;

        // generate sub requests for all servers
        std::vector<SPARQLQuery> sub_reqs(Global::num_servers);
        for (int i = 0; i < Global::num_servers; i++) {
            sub_reqs[i].pqid = req.qid;
            sub_reqs[i].pg_type = (req.pg_type == SPARQLQuery::PGType::UNION) ? SPARQLQuery::PGType::BASIC : req.pg_type;
            sub_reqs[i].pattern_group = req.pattern_group;
            sub_reqs[i].pattern_step = req.pattern_step;
            sub_reqs[i].corun_step = req.corun_step;
            sub_reqs[i].fetch_step = req.fetch_step;
            sub_reqs[i].local_var = start;
            sub_reqs[i].priority = req.priority + 1;

            // metadata
            sub_reqs[i].result.col_num = req.result.col_num;
            sub_reqs[i].result.time_col_num = req.result.time_col_num;
            sub_reqs[i].result.attr_col_num = req.result.attr_col_num;
            sub_reqs[i].result.blind = req.result.blind;
            sub_reqs[i].result.v2c_map = req.result.v2c_map;
            sub_reqs[i].result.nvars = req.result.nvars;
            sub_reqs[i].ts = req.ts;
            sub_reqs[i].te = req.te;
        }

        // group intermediate results to servers
        int nrows = req.result.get_row_num();
        // result table need to be duplicated to all sub-queries
        if (!need_split) {
            for (int i = 0; i < Global::num_servers; i++) {
                sub_reqs[i].result.dup_rows(req.result.result_table);
                sub_reqs[i].result.dup_time_rows(req.result.time_res_table);
                sub_reqs[i].result.dup_attr_rows(req.result.attr_res_table);
                sub_reqs[i].result.update_nrows();
            }
        } else {  // result table need to be separated to ditterent sub-queries
            for (int i = 0; i < nrows; i++) {
                int dst_sid = PARTITION(req.result.get_row_col(i, req.result.var2col(start)));
                req.result.append_row_to(i, sub_reqs[dst_sid].result.result_table);
                req.result.append_time_row_to(i, sub_reqs[dst_sid].result.time_res_table);
                req.result.append_attr_row_to(i, sub_reqs[dst_sid].result.attr_res_table);

                if (req.pg_type == SPARQLQuery::PGType::OPTIONAL)
                    sub_reqs[dst_sid].result.optional_matched_rows.push_back(req.result.optional_matched_rows[i]);
            }

            for (int i = 0; i < Global::num_servers; i++)
                sub_reqs[i].result.update_nrows();
        }

        return sub_reqs;
    }

    // fork-join or in-place execution
    bool need_fork_join(SPARQLQuery& req) {
        // always need NOT fork-join when executing on single machine
        if (Global::num_servers == 1) return false;

        // always need fork-join w/o RDMA
        if (!Global::use_rdma) return true;

        SPARQLQuery::Pattern& pattern = req.get_pattern();
        ASSERT_ERROR_CODE(req.result.var_stat(pattern.subject) == KNOWN_VAR, OBJ_ERROR);
        ssid_t start = pattern.subject;
        return ((req.local_var != start)                                   // next hop is not local
                && (req.result.get_row_num() >= Global::rdma_threshold));  // FIXME: not consider dedup
    }

    // TODO: TRDF_MODE dont support corun
    void do_corun(SPARQLQuery& req) {
        SPARQLQuery::Result& res = req.result;
        int corun_step = req.corun_step;
        int fetch_step = req.fetch_step;

        // step.1 remove dup;
        uint64_t t0 = timer::get_usec();

        boost::unordered_set<sid_t> unique_set;
        ssid_t vid = req.get_pattern(corun_step).subject;
        ASSERT_ERROR_CODE(vid < 0, VERTEX_INVALID);
        int col_idx = res.var2col(vid);
        int nrows = res.get_row_num();
        for (int i = 0; i < nrows; i++)
            unique_set.insert(res.get_row_col(i, col_idx));

        // step.2 generate cmd_chain for sub-reqs
        SPARQLQuery::PatternGroup subgroup;
        std::vector<int> pvars_map;  // from new_id to col_idx of id

        boost::unordered_map<sid_t, sid_t> sub_pvars;

        auto lambda = [&](ssid_t id) -> ssid_t {
            if (id < 0) {  // remap pattern variable
                if (sub_pvars.find(id) == sub_pvars.end()) {
                    sid_t new_id = -(sub_pvars.size() + 1);  // starts from -1
                    sub_pvars[id] = new_id;
                    pvars_map.push_back(res.var2col(id));
                }
                return sub_pvars[id];
            } else {
                return id;
            }
        };

        for (int i = corun_step; i < fetch_step; i++) {
            SPARQLQuery::Pattern& pattern = req.get_pattern(i);
            ssid_t subject = lambda(pattern.subject);
            ssid_t predicate = lambda(pattern.predicate);
            dir_t direction = pattern.direction;
            ssid_t object = lambda(pattern.object);
            SPARQLQuery::Pattern newPattern(subject, predicate, direction, object);

            newPattern.pred_type = static_cast<char>(SID_t);
            subgroup.patterns.push_back(newPattern);
        }

        // step.3 make sub-req
        SPARQLQuery sub_req;
        SPARQLQuery::Result& sub_result = sub_req.result;

        // query
        sub_req.pattern_group = subgroup;
        sub_result.nvars = pvars_map.size();

        // result
        boost::unordered_set<sid_t>::iterator iter;
        for (iter = unique_set.begin(); iter != unique_set.end(); iter++)
            sub_result.result_table.push_back(*iter);
        sub_result.col_num = 1;

        // init var_map
        sub_result.add_var2col(sub_pvars[vid], 0);

        sub_result.blind = false;         // must take back results
        uint64_t t1 = timer::get_usec();  // time to generate the sub-request

        // step.4 execute sub-req
        while (true) {
            execute_one_pattern(sub_req);
            if (sub_req.done(SPARQLQuery::SQState::SQ_PATTERN))
                break;
        }
        uint64_t t2 = timer::get_usec();  // time to run the sub-request

        uint64_t t3, t4;
        std::vector<sid_t> updated_result_table;

        if (sub_result.get_col_num() > 2) {  // qsort
            wukong::tuple::qsort_tuple(sub_result.get_col_num(), sub_result.result_table);

            t3 = timer::get_usec();
            std::vector<sid_t> tmp_vec;
            tmp_vec.resize(sub_result.get_col_num());
            int nrows = res.get_row_num();
            for (int i = 0; i < nrows; i++) {
                for (int c = 0; c < pvars_map.size(); c++)
                    tmp_vec[c] = res.get_row_col(i, pvars_map[c]);

                if (wukong::tuple::binary_search_tuple(sub_result.get_col_num(),
                                                       sub_result.result_table, tmp_vec))
                    res.append_row_to(i, updated_result_table);
            }
            t4 = timer::get_usec();
        } else {  // hash join
            boost::unordered_set<int64_pair> remote_set;
            int nrows = sub_result.get_row_num();
            for (int i = 0; i < nrows; i++)
                remote_set.insert(int64_pair(sub_result.get_row_col(i, 0),
                                             sub_result.get_row_col(i, 1)));

            t3 = timer::get_usec();
            std::vector<sid_t> tmp_vec;
            tmp_vec.resize(sub_result.get_col_num());
            nrows = res.get_row_num();
            for (int i = 0; i < nrows; i++) {
                for (int c = 0; c < pvars_map.size(); c++)
                    tmp_vec[c] = res.get_row_col(i, pvars_map[c]);

                int64_pair target = int64_pair(tmp_vec[0], tmp_vec[1]);
                if (remote_set.find(target) != remote_set.end())
                    res.append_row_to(i, updated_result_table);
            }
            t4 = timer::get_usec();
        }

        // update result and metadata
        res.result_table.swap(updated_result_table);
        res.update_nrows();

        req.pattern_step = fetch_step;
    }

    bool execute_one_pattern(SPARQLQuery& req) {
        ASSERT(!req.done(SPARQLQuery::SQState::SQ_PATTERN));

        logstream(LOG_DEBUG) << "[" << sid << "-" << tid << "]"
                             << " step=" << req.pattern_step << LOG_endl;

        SPARQLQuery::Pattern& pattern = req.get_pattern();
        ssid_t start = pattern.subject;
        ssid_t predicate = pattern.predicate;
        dir_t direction = pattern.direction;
        ssid_t end = pattern.object;

        // the first triple pattern from index
        if (req.pattern_step == 0 && req.start_from_index()) {
            if (req.result.var2col(end) != NO_RESULT)
                index_to_known(req);
            else
                index_to_unknown(req);
            return true;
        }

        // triple pattern with UNKNOWN/KNOWN predicate/attribute
        if (req.result.var_stat(predicate) != CONST_VAR) {
#ifdef VERSATILE
            /// Now unsupported UNKNOWN predicate with vertex attribute enabling.
            /// When doing the query, we judge request of vertex attribute by its predicate.
            /// Therefore we should known the predicate.
            if (Global::enable_vattr) {
                logstream(LOG_ERROR) << "Unsupported UNKNOWN predicate with vertex attribute enabling." << LOG_endl;
                logstream(LOG_ERROR) << "Please turn off the vertex attribute enabling." << LOG_endl;
                ASSERT_ERROR_CODE(false, SETTING_ERROR);
            }

            switch (const_pair(req.result.var_stat(start),
                               req.result.var_stat(end))) {
            // start from CONST
            case const_pair(CONST_VAR, UNKNOWN_VAR):
                const_unknown_unknown(req);
                break;
            case const_pair(CONST_VAR, CONST_VAR):
                const_unknown_const(req);
                break;
            case const_pair(CONST_VAR, KNOWN_VAR):
                // FIXME: possible or not?
                logstream(LOG_ERROR) << "Unsupported triple pattern [CONST|UNKNOWN|KNOWN]." << LOG_endl;
                ASSERT_ERROR_CODE(false, UNKNOWN_PATTERN);

            // start from KNOWN
            case const_pair(KNOWN_VAR, UNKNOWN_VAR):
                known_unknown_unknown(req);
                break;
            case const_pair(KNOWN_VAR, CONST_VAR):
                known_unknown_const(req);
                break;
            case const_pair(KNOWN_VAR, KNOWN_VAR):
                // FIXME: possible or not?
                logstream(LOG_ERROR) << "Unsupported triple pattern [KNOWN|UNKNOWN|KNOWN]." << LOG_endl;
                ASSERT_ERROR_CODE(false, UNKNOWN_PATTERN);

            // start from UNKNOWN (incorrect query plan)
            case const_pair(UNKNOWN_VAR, CONST_VAR):
            case const_pair(UNKNOWN_VAR, KNOWN_VAR):
            case const_pair(UNKNOWN_VAR, UNKNOWN_VAR):
                logstream(LOG_ERROR) << "Unsupported triple pattern [UNKNOWN|UNKNOWN|??]" << LOG_endl;
                ASSERT_ERROR_CODE(false, UNKNOWN_PATTERN);

            default:
                logstream(LOG_ERROR) << "Unsupported triple pattern (UNKNOWN predicate) "
                                     << "(" << req.result.var_stat(start)
                                     << "|" << req.result.var_stat(end)
                                     << ")." << LOG_endl;
                ASSERT_ERROR_CODE(false, UNKNOWN_PATTERN);
            }
#else
            logstream(LOG_ERROR) << "Unsupported variable at predicate." << LOG_endl;
            logstream(LOG_ERROR) << "Please add definition VERSATILE in CMakeLists.txt." << LOG_endl;
            ASSERT_ERROR_CODE(false, UNKNOWN_PATTERN);
#endif
        } else {
            // triple pattern with CONST predicate/attribute
            switch (const_pair(req.result.var_stat(start),
                               req.result.var_stat(end))) {
            // start from CONST
            case const_pair(CONST_VAR, CONST_VAR):
                logstream(LOG_ERROR) << "Unsupported triple pattern [CONST|KNOWN|CONST]" << LOG_endl;
                ASSERT_ERROR_CODE(false, UNKNOWN_PATTERN);
            case const_pair(CONST_VAR, KNOWN_VAR):
                const_to_known(req);
                break;
            case const_pair(CONST_VAR, UNKNOWN_VAR):
                const_to_unknown(req);
                break;

            // start from KNOWN
            case const_pair(KNOWN_VAR, CONST_VAR):
                known_to_const(req);
                break;
            case const_pair(KNOWN_VAR, KNOWN_VAR):
                known_to_known(req);
                break;
            case const_pair(KNOWN_VAR, UNKNOWN_VAR):
                known_to_unknown(req);
                break;

            // start from UNKNOWN (incorrect query plan)
            case const_pair(UNKNOWN_VAR, CONST_VAR):
            case const_pair(UNKNOWN_VAR, KNOWN_VAR):
            case const_pair(UNKNOWN_VAR, UNKNOWN_VAR):
                logstream(LOG_ERROR) << "Unsupported triple pattern [UNKNOWN|KNOWN|??]" << LOG_endl;
                ASSERT_ERROR_CODE(false, UNKNOWN_SUB);

            default:
                logstream(LOG_ERROR) << "Unsupported triple pattern with known predicate "
                                     << "(" << req.result.var_stat(start)
                                     << "|" << req.result.var_stat(end)
                                     << ")" << LOG_endl;
                ASSERT_ERROR_CODE(false, UNKNOWN_PATTERN);
            }
        }

        return true;
    }

    // deal with pattern wich start from index
    bool dispatch(SPARQLQuery& r, bool is_start = true) {
        if (Global::num_servers * r.mt_factor == 1) return false;

        // first pattern need dispatch, to all servers and multi-threads threads
        if (is_start && QUERY_FROM_PROXY(r) && r.pattern_step == 0 && r.start_from_index()) {
            logstream(LOG_DEBUG) << "[" << sid << "-" << tid << "] dispatch "
                                 << "Q(qid=" << r.qid << ", pqid=" << r.pqid
                                 << ", step=" << r.pattern_step << ")" << LOG_endl;
            rmap.put_parent_request(r, Global::num_servers * r.mt_factor);

            SPARQLQuery sub_query = r;
            for (int i = 0; i < Global::num_servers; i++) {
                for (int j = 0; j < r.mt_factor; j++) {
                    sub_query.pqid = r.qid;
                    sub_query.qid = -1;
                    sub_query.mt_tid = j;

                    int dst_tid = Global::num_proxies + (tid + j + 1 - Global::num_proxies) % Global::num_engines;

                    Bundle bundle(sub_query);
                    msgr->send_msg(bundle, i, dst_tid);
                }
            }
            return true;
        }

        // not first pattern need dispatch, to all servers but each single thread
        SPARQLQuery::Pattern& pattern = r.get_pattern();
        ssid_t start = pattern.subject;
        ssid_t p = pattern.predicate;
        dir_t d = pattern.direction;

        if (!is_start && Global::num_servers != 1 && p == TYPE_ID && d == IN) {
            std::vector<SPARQLQuery> sub_reqs = generate_sub_query(r, false);
            rmap.put_parent_request(r, sub_reqs.size());
            for (int i = 0; i < sub_reqs.size(); i++) {
                if (i != sid) {
                    Bundle bundle(sub_reqs[i]);
                    msgr->send_msg(bundle, i, tid);
                } else {
                    prior_stage.push(sub_reqs[i]);
                }
            }
            return true;
        }
        return false;
    }

    bool execute_patterns(SPARQLQuery& r) {
        uint64_t time, access;
        logstream(LOG_DEBUG) << "[" << sid << "-" << tid << "] execute patterns of "
                             << "Q(pqid=" << r.pqid << ", qid=" << r.qid
                             << ", step=" << r.pattern_step << ")"
                             << " #rows = " << r.result.get_row_num()
                             << LOG_endl;
        do {
            time = timer::get_usec();
            execute_one_pattern(r);
            logstream(LOG_DEBUG) << "[" << sid << "-" << tid << "]"
                                 << " step = " << r.pattern_step
                                 << " exec-time = " << (timer::get_usec() - time) << " usec"
                                 << " #rows = " << r.result.get_row_num()
                                 << " #cols = " << r.result.get_col_num()
                                 << LOG_endl;

            // co-run optimization
            if (r.corun_enabled && (r.pattern_step == r.corun_step))
                do_corun(r);

            if (r.done(SPARQLQuery::SQState::SQ_PATTERN))
                return true;  // done

            if (dispatch(r, false)) {
                return false;
            }

            if (need_fork_join(r)) {
                std::vector<SPARQLQuery> sub_reqs = generate_sub_query(r);
                rmap.put_parent_request(r, sub_reqs.size());
                for (int i = 0; i < sub_reqs.size(); i++) {
                    if (i != sid) {
                        Bundle bundle(sub_reqs[i]);
                        msgr->send_msg(bundle, i, tid);
                    } else {
                        prior_stage.push(sub_reqs[i]);
                    }
                }
                return false;  // outstanding
            }
        } while (true);
    }

    // relational operator: < <= > >= == !=
    void relational_filter(SPARQLQuery::Filter& filter,
                           SPARQLQuery::Result& result,
                           std::vector<bool>& is_satisfy) {
        int col1 = (filter.arg1->type == SPARQLQuery::Filter::Type::Variable)
                       ? result.var2col(filter.arg1->valueArg)
                       : -1;
        int col2 = (filter.arg2->type == SPARQLQuery::Filter::Type::Variable)
                       ? result.var2col(filter.arg2->valueArg)
                       : -1;

        auto get_str = [&](SPARQLQuery::Filter& filter, int row, int col) -> std::string {
            int id = 0;
            switch (filter.type) {
            case SPARQLQuery::Filter::Type::Variable:
                id = result.get_row_col(row, col);
                return str_server->exist(id) ? str_server->id2str(id) : "";
            case SPARQLQuery::Filter::Type::Literal:
                return "\"" + filter.value + "\"";
            default:
                logstream(LOG_ERROR) << "Unsupported FILTER type" << LOG_endl;
                ASSERT_ERROR_CODE(false, UNKNOWN_FILTER);
            }
            return "";
        };

        int nrows = result.get_row_num();
        switch (filter.type) {
        case SPARQLQuery::Filter::Type::Equal:
            for (int row = 0; row < nrows; row++)
                if (is_satisfy[row] && (get_str(*filter.arg1, row, col1) != get_str(*filter.arg2, row, col2)))
                    is_satisfy[row] = false;
            break;
        case SPARQLQuery::Filter::Type::NotEqual:
            for (int row = 0; row < nrows; row++)
                if (is_satisfy[row] && (get_str(*filter.arg1, row, col1) == get_str(*filter.arg2, row, col2)))
                    is_satisfy[row] = false;
            break;
        case SPARQLQuery::Filter::Type::Less:
            for (int row = 0; row < nrows; row++)
                if (is_satisfy[row] && (get_str(*filter.arg1, row, col1) >= get_str(*filter.arg2, row, col2)))
                    is_satisfy[row] = false;
            break;
        case SPARQLQuery::Filter::Type::LessOrEqual:
            for (int row = 0; row < nrows; row++)
                if (is_satisfy[row] && (get_str(*filter.arg1, row, col1) > get_str(*filter.arg2, row, col2)))
                    is_satisfy[row] = false;
            break;
        case SPARQLQuery::Filter::Type::Greater:
            for (int row = 0; row < nrows; row++)
                if (is_satisfy[row] && (get_str(*filter.arg1, row, col1) <= get_str(*filter.arg2, row, col2)))
                    is_satisfy[row] = false;
            break;
        case SPARQLQuery::Filter::Type::GreaterOrEqual:
            for (int row = 0; row < nrows; row++)
                if (is_satisfy[row] && get_str(*filter.arg1, row, col1) < get_str(*filter.arg2, row, col2))
                    is_satisfy[row] = false;
            break;
        }
    }

    void bound_filter(SPARQLQuery::Filter& filter,
                      SPARQLQuery::Result& result,
                      std::vector<bool>& is_satisfy) {
        int col = result.var2col(filter.arg1->valueArg);

        for (int row = 0; row < is_satisfy.size(); row++) {
            if (!is_satisfy[row])
                continue;

            if (result.get_row_col(row, col) == BLANK_ID)
                is_satisfy[row] = false;
        }
    }

    // IRI and URI are the same in SPARQL
    void isIRI_filter(SPARQLQuery::Filter& filter,
                      SPARQLQuery::Result& result,
                      std::vector<bool>& is_satisfy) {
        int col = result.var2col(filter.arg1->valueArg);

        std::string IRI_REF = R"(<([^<>\\"{}|^`\\])*>)";
        std::string prefixed_name = ".*:.*";
        std::string IRIref_str = "(" + IRI_REF + "|" + prefixed_name + ")";

        std::regex IRI_pattern(IRIref_str);
        for (int row = 0; row < is_satisfy.size(); row++) {
            if (!is_satisfy[row])
                continue;

            int id = result.get_row_col(row, col);
            std::string str = str_server->exist(id) ? str_server->id2str(id) : "";
            if (!regex_match(str, IRI_pattern))
                is_satisfy[row] = false;
        }
    }

    void isliteral_filter(SPARQLQuery::Filter& filter,
                          SPARQLQuery::Result& result,
                          std::vector<bool>& is_satisfy) {
        int col = result.var2col(filter.arg1->valueArg);

        std::string langtag_pattern_str("@[a-zA-Z]+(-[a-zA-Z0-9]+)*");

        std::string literal1_str = R"('([^\x27\x5C\x0A\x0D]|\\[tbnrf\"'])*')";
        std::string literal2_str = R"("([^\x22\x5C\x0A\x0D]|\\[tbnrf\"'])*")";
        std::string literal_long1_str = R"('''(('|'')?([^'\\]|\\[tbnrf\"']))*''')";
        std::string literal_long2_str = R"("""(("|"")?([^"\\]|\\[tbnrf\"']))*""")";
        std::string literal = "(" + literal1_str + "|" + literal2_str + "|" + literal_long1_str + "|" + literal_long2_str + ")";

        std::string IRI_REF = R"(<([^<>\\"{}|^`\\])*>)";
        std::string prefixed_name = ".*:.*";
        std::string IRIref_str = "(" + IRI_REF + "|" + prefixed_name + ")";

        std::regex RDFLiteral_pattern(literal + "(" + langtag_pattern_str + "|(\\^\\^" + IRIref_str + "))?");

        for (int row = 0; row < is_satisfy.size(); row++) {
            if (!is_satisfy[row])
                continue;

            int id = result.get_row_col(row, col);
            std::string str = str_server->exist(id) ? str_server->id2str(id) : "";
            if (!regex_match(str, RDFLiteral_pattern))
                is_satisfy[row] = false;
        }
    }

    // regex flag only support "i" option now
    void regex_filter(SPARQLQuery::Filter& filter,
                      SPARQLQuery::Result& result,
                      std::vector<bool>& is_satisfy) {
        std::regex pattern;
        if (filter.arg3 != nullptr && filter.arg3->value == "i")
            pattern = std::regex(filter.arg2->value, std::regex::icase);
        else
            pattern = std::regex(filter.arg2->value);

        int col = result.var2col(filter.arg1->valueArg);
        for (int row = 0; row < is_satisfy.size(); row++) {
            if (!is_satisfy[row])
                continue;

            int id = result.get_row_col(row, col);
            std::string str = str_server->exist(id) ? str_server->id2str(id) : "";
            if (str.front() != '\"' || str.back() != '\"')
                logstream(LOG_ERROR) << "The first parameter of function regex must be string"
                                     << LOG_endl;
            else
                str = str.substr(1, str.length() - 2);

            if (!regex_match(str, pattern))
                is_satisfy[row] = false;
        }
    }

    void general_filter(SPARQLQuery::Filter& filter,
                        SPARQLQuery::Result& result,
                        std::vector<bool>& is_satisfy) {
        if (filter.type <= SPARQLQuery::Filter::Type::And) {
            // conditional operator: Or(0), And(1)
            std::vector<bool> is_satisfy1(result.get_row_num(), true);
            std::vector<bool> is_satisfy2(result.get_row_num(), true);
            if (filter.type == SPARQLQuery::Filter::Type::And) {
                general_filter(*filter.arg1, result, is_satisfy);
                general_filter(*filter.arg2, result, is_satisfy);
            } else if (filter.type == SPARQLQuery::Filter::Type::Or) {
                general_filter(*filter.arg1, result, is_satisfy1);
                general_filter(*filter.arg2, result, is_satisfy2);
                for (int i = 0; i < is_satisfy.size(); i++)
                    is_satisfy[i] = is_satisfy[i] && (is_satisfy1[i] || is_satisfy2[i]);
            }
        } else if (filter.type <= SPARQLQuery::Filter::Type::GreaterOrEqual) {
            // relational operator: Equal(2), NotEqual(3), Less(4), LessOrEqual(5),
            //                      Greater(6), GreaterOrEqual(7)
            relational_filter(filter, result, is_satisfy);
        } else if (filter.type == SPARQLQuery::Filter::Type::Builtin_bound) {
            bound_filter(filter, result, is_satisfy);
        } else if (filter.type == SPARQLQuery::Filter::Type::Builtin_isiri) {
            isIRI_filter(filter, result, is_satisfy);
        } else if (filter.type == SPARQLQuery::Filter::Type::Builtin_isliteral) {
            isliteral_filter(filter, result, is_satisfy);
        } else if (filter.type == SPARQLQuery::Filter::Type::Builtin_regex) {
            try {
                regex_filter(filter, result, is_satisfy);
            } catch (std::regex_error err) {
                logstream(LOG_ERROR)
                    << "Something wrong with filter regex." << LOG_endl;
                throw WukongException(UNKNOWN_FILTER);
            }
        } else {
            ASSERT_ERROR_CODE(false, UNKNOWN_FILTER);  // unsupport filter type
        }
    }

    void execute_filter(SPARQLQuery& r) {
        ASSERT(r.has_filter());

        // during filtering, flag of unsatified row will be set to false one by one
        std::vector<bool> is_satisfy(r.result.get_row_num(), true);

        for (int i = 0; i < r.pattern_group.filters.size(); i++) {
            SPARQLQuery::Filter filter = r.pattern_group.filters[i];
            general_filter(filter, r.result, is_satisfy);
        }

        std::vector<sid_t> new_table;
        std::vector<int64_t> new_time_table;
        int nrows = r.result.get_row_num();
        for (int row = 0; row < nrows; row++) {
            if (is_satisfy[row]) {
                r.result.append_row_to(row, new_table);
                r.result.append_time_row_to(row, new_time_table);
            }
        }

        // uopdate result and metadata
        r.result.result_table.swap(new_table);
        r.result.time_res_table.swap(new_time_table);
        r.result.update_nrows();
    }

    class Compare {
    private:
        SPARQLQuery& query;
        StringServer* str_server;

    public:
        Compare(SPARQLQuery& query, StringServer* str_server)
            : query(query), str_server(str_server) {}

        bool operator()(const int* a, const int* b) {
            int cmp = 0;
            for (int i = 0; i < query.orders.size(); i++) {
                int col = query.result.var2col(query.orders[i].id);
                std::string str_a = str_server->exist(a[col]) ? str_server->id2str(a[col]) : "";
                std::string str_b = str_server->exist(a[col]) ? str_server->id2str(b[col]) : "";
                cmp = str_a.compare(str_b);
                if (cmp != 0) {
                    cmp = query.orders[i].descending ? -cmp : cmp;
                    break;
                }
            }
            return cmp < 0;
        }
    };

    class ReduceCmp {
    private:
        int col_num;

    public:
        explicit ReduceCmp(int col_num) : col_num(col_num) {}

        bool operator()(const int* a, const int* b) {
            for (int i = 0; i < col_num; i++) {
                if (a[i] == b[i])
                    continue;
                return a[i] < b[i];
            }
            return 0;
        }
    };

    void final_process(SPARQLQuery& r) {
        if (r.result.blind || r.result.result_table.size() == 0)
            return;

        // DISTINCT and ORDER BY
        // TODO: support them in t-sparql
        if (r.distinct || r.orders.size() > 0) {
            // initialize table
            int** table;
            int size = r.result.get_row_num();
            int new_size = size;

            table = new int*[size];
            for (int i = 0; i < size; i++)
                table[i] = new int[r.result.col_num];

            for (int i = 0; i < size; i++)
                for (int j = 0; j < r.result.col_num; j++)
                    table[i][j] = r.result.get_row_col(i, j);

            // DISTINCT
            if (r.distinct) {
                // sort and then compare
                std::sort(table, table + size, ReduceCmp(r.result.col_num));
                int p = 0, q = 1;
                auto equal = [&r](int* a, int* b) -> bool {
                    for (int i = 0; i < r.result.required_vars.size(); i++) {
                        int col = r.result.var2col(r.result.required_vars[i]);
                        if (a[col] != b[col]) return false;
                    }
                    return true;
                };

                auto swap = [](int*& a, int*& b) {
                    int* temp = a;
                    a = b;
                    b = temp;
                };

                while (q < size) {
                    while (equal(table[p], table[q])) {
                        q++;
                        if (q >= size) goto out;
                    }
                    p++;
                    swap(table[p], table[q]);
                    q++;
                }
            out:
                new_size = p + 1;
            }

            // ORDER BY
            if (r.orders.size() > 0)
                std::sort(table, table + new_size, Compare(r, str_server));

            // write back data and delete **table
            for (int i = 0; i < new_size; i++)
                for (int j = 0; j < r.result.col_num; j++)
                    r.result.result_table[r.result.col_num * i + j] = table[i][j];

            if (new_size < size)
                r.result.result_table.erase(r.result.result_table.begin() + new_size * r.result.col_num,
                                            r.result.result_table.begin() + size * r.result.col_num);

            for (int i = 0; i < size; i++)
                delete[] table[i];
            delete[] table;
        }

        // OFFSET
        if (r.offset > 0) {
            r.result.result_table.erase(r.result.result_table.begin(),
                                        std::min(r.result.result_table.begin() + r.offset * r.result.col_num,
                                                 r.result.result_table.end()));

            r.result.time_res_table.erase(r.result.time_res_table.begin(),
                                        std::min(r.result.time_res_table.begin() + r.offset * r.result.time_col_num,
                                                 r.result.time_res_table.end()));
        }

        // LIMIT
        if (r.limit >= 0) {
            r.result.result_table.erase(std::min(r.result.result_table.begin() + r.limit * r.result.col_num,
                                                 r.result.result_table.end()), r.result.result_table.end());

            r.result.time_res_table.erase(std::min(r.result.time_res_table.begin() + r.limit * r.result.time_col_num,
                                                   r.result.time_res_table.end()), r.result.time_res_table.end());
        }

        // remove unrequested variables
        // separate var to normal and attribute
        // need to think about attribute result table
        std::vector<ssid_t> normal_var;
        std::vector<ssid_t> time_var;
        std::vector<ssid_t> attr_var;
        ASSERT_ERROR_CODE(r.result.required_vars.size() != 0, NO_REQUIRED_VAR);
        for (int i = 0; i < r.result.required_vars.size(); i++) {
            ssid_t vid = r.result.required_vars[i];
            if (r.result.var_type(vid) == SID_t)
                normal_var.push_back(vid);  // entity
            else if (r.result.var_type(vid) == TIME_t)
                time_var.push_back(vid);
            else
                attr_var.push_back(vid);  // attributed
        }

        int new_row_num = r.result.get_row_num();
        int new_col_num = normal_var.size();
        int new_time_col_num = time_var.size();
        int new_attr_col_num = attr_var.size();

        // update result table
        std::vector<sid_t> new_result_table(new_row_num * new_col_num);
        for (int i = 0; i < new_row_num; i++) {
            for (int j = 0; j < new_col_num; j++) {
                int col = r.result.var2col(normal_var[j]);
                new_result_table[i * new_col_num + j] = r.result.get_row_col(i, col);
            }
        }
        r.result.result_table.swap(new_result_table);

        // update metadata
        r.result.set_col_num(new_col_num);
        r.result.update_nrows();

        // update (attribute) result
        std::vector<attr_t> new_attr_result_table(new_row_num * new_attr_col_num);
        for (int i = 0; i < new_row_num; i++) {
            for (int j = 0; j < new_attr_col_num; j++) {
                int col = r.result.var2col(attr_var[j]);
                new_attr_result_table[i * new_attr_col_num + j] = r.result.get_attr_row_col(i, col);
            }
        }
        r.result.attr_res_table.swap(new_attr_result_table);

        // update metadata
        r.result.set_attr_col_num(new_attr_col_num);

        // update (timestamp) result
        std::vector<int64_t> new_time_result_table(new_row_num * new_time_col_num);
        for (int i = 0; i < new_row_num; i++) {
            for (int j = 0; j < new_time_col_num; j++) {
                int col = r.result.var2col(time_var[j]);
                new_time_result_table[i * new_time_col_num + j] = r.result.get_time_row_col(i, col);
            }
        }
        r.result.time_res_table.swap(new_time_result_table);

        // update metadata
        r.result.set_time_col_num(new_time_col_num);
    }

public:
    tbb::concurrent_queue<SPARQLQuery> prior_stage;

    TSPARQLEngine(int sid, int tid, StringServer* str_server,
                  DGraph* graph, Coder* coder, Messenger* msgr)
        : sid(sid), tid(tid), str_server(str_server), graph(graph), coder(coder), msgr(msgr) {
        pthread_spin_init(&rmap_lock, 0);
    }

    void execute_sparql_query(SPARQLQuery& r) {
        try {
            // encode the lineage of the query (server & thread)
            if (r.qid == -1) r.qid = coder->get_and_inc_qid();

            // 0. query has done
            if (r.state == SPARQLQuery::SQState::SQ_REPLY) {
                pthread_spin_lock(&rmap_lock);
                rmap.put_reply(r);

                if (!rmap.is_ready(r.pqid)) {
                    pthread_spin_unlock(&rmap_lock);
                    return;  // not ready (waiting for the rest)
                }

                // all sub-queries have done, continue to execute
                r = rmap.get_reply(r.pqid);
                pthread_spin_unlock(&rmap_lock);
            }

            // 1. Pattern
            if (r.has_pattern() && !r.done(SPARQLQuery::SQState::SQ_PATTERN)) {
                r.state = SPARQLQuery::SQState::SQ_PATTERN;

                // exploit parallelism (multi-server and multi-threading)
                if (dispatch(r)) return;  // async waiting reply by rmap

                if (!execute_patterns(r)) return;  // outstanding
            }

            // 2. Union
            if (r.has_union() && !r.done(SPARQLQuery::SQState::SQ_UNION)) {
                r.state = SPARQLQuery::SQState::SQ_UNION;
                int size = r.pattern_group.unions.size();
                r.union_done = true;
                std::vector<SPARQLQuery> union_reqs(size);
                int sub_queries_size = 0;
                for (int i = 0; i < size; i++) {
                    union_reqs[i].inherit_union(r, i);
                    if(union_reqs[i].start_from_index()) {
                        sub_queries_size += Global::num_servers * union_reqs[i].mt_factor;
                    } else {
                        sub_queries_size++;
                    }
                }
                rmap.put_parent_request(r, sub_queries_size);

                for (int i = 0; i < size; i++) {
                    if(union_reqs[i].start_from_index()) {
                        SPARQLQuery sub_query = union_reqs[i];
                        for (int i = 0; i < Global::num_servers; i++) {
                            for (int j = 0; j < union_reqs[i].mt_factor; j++) {
                                sub_query.mt_tid = j;

                                int dst_tid = Global::num_proxies + (tid + j + 1 - Global::num_proxies) % Global::num_engines;

                                Bundle bundle(sub_query);
                                msgr->send_msg(bundle, i, dst_tid);
                            }
                        }
                    } else {
                        int dst_sid = PARTITION(union_reqs[i].pattern_group.get_start());
                        if (dst_sid != sid) {
                            Bundle bundle(union_reqs[i]);
                            msgr->send_msg(bundle, dst_sid, tid);
                        } else {
                            prior_stage.push(union_reqs[i]);
                        }
                    }
                }
                return;
            }

            // 3. Optional
            if (r.has_optional() &&
                !r.done(SPARQLQuery::SQState::SQ_OPTIONAL)) {
                r.state = SPARQLQuery::SQState::SQ_OPTIONAL;
                SPARQLQuery optional_req;
                optional_req.inherit_optional(r);
                r.optional_step++;
                if (need_fork_join(optional_req)) {
                    optional_req.qid = r.qid;
                    std::vector<SPARQLQuery> sub_reqs =
                        generate_sub_query(optional_req);
                    rmap.put_parent_request(r, sub_reqs.size());
                    for (int i = 0; i < sub_reqs.size(); i++) {
                        if (i != sid) {
                            Bundle bundle(sub_reqs[i]);
                            msgr->send_msg(bundle, i, tid);
                        } else {
                            prior_stage.push(sub_reqs[i]);
                        }
                    }
                } else {
                    rmap.put_parent_request(r, 1);
                    int dst_sid = PARTITION(optional_req.pattern_group.get_start());
                    if (dst_sid != sid) {
                        Bundle bundle(optional_req);
                        msgr->send_msg(bundle, dst_sid, tid);
                    } else {
                        prior_stage.push(optional_req);
                    }
                }
                return;
            }

            // 4. Filter
            if (r.has_filter()) {
                r.state = SPARQLQuery::SQState::SQ_FILTER;
                execute_filter(r);
            }

            // 5. Final
            if (QUERY_FROM_PROXY(r)) {
                r.state = SPARQLQuery::SQState::SQ_FINAL;
                final_process(r);
            }
        } catch (const char* msg) {
            r.result.set_status_code(UNKNOWN_ERROR);
        } catch (WukongException& ex) {
            r.result.set_status_code(ex.code());
        }
        // 6. Reply
        r.shrink();
        r.state = SPARQLQuery::SQState::SQ_REPLY;
        Bundle bundle(r);
        msgr->send_msg(bundle, coder->sid_of(r.pqid), coder->tid_of(r.pqid));
    }
};

}  // namespace wukong
