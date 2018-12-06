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
#include <boost/unordered_map.hpp>

#include "query.hpp"

using namespace std;

// The map is used to collect replies from sub_queries in fork-join execution mode
class RMap {
private:
    struct Item {
        int cnt; // #sub-queries
        SPARQLQuery parent;
        SPARQLQuery reply;
    };

    boost::unordered_map<int, Item> internal_map;

public:
    void put_parent_request(SPARQLQuery &r, int cnt) {
        logstream(LOG_DEBUG) << "add parent-qid=" << r.qid
                             << " and #sub-queries=" << cnt << LOG_endl;

        // not exist
        ASSERT(internal_map.find(r.qid) == internal_map.end());

        Item d = { .cnt = cnt, .parent = r, };
        //d.cnt = cnt;
        //d.parent = r;
        internal_map[r.qid] = d;
    }

    void put_reply(SPARQLQuery &r) {
        // exist
        ASSERT(internal_map.find(r.pqid) != internal_map.end());

        Item &d = internal_map[r.pqid];
        SPARQLQuery::Result &whole = d.reply.result;
        SPARQLQuery::Result &part = r.result;
        d.cnt--;

        // if the PatternGroup comes from a query's UNION part,
        // use merge_result to put result
        if (r.pg_type == SPARQLQuery::PGType::UNION)
            whole.merge_result(part);
        else
            whole.append_result(part);


        // NOTE: all sub-jobs have the same pattern_step, optional_step, and union_done
        // update parent's pattern step (progress)
        if (d.parent.state == SPARQLQuery::SQState::SQ_PATTERN)
            d.parent.pattern_step = r.pattern_step;

        // update parent's optional_step (avoid recursive execution)
        if (d.parent.pg_type == SPARQLQuery::PGType::OPTIONAL
                && r.done(SPARQLQuery::SQState::SQ_OPTIONAL))
            d.parent.optional_step = r.optional_step;

        // update parent's union_done (avoid recursive execution)
        if (r.done(SPARQLQuery::SQState::SQ_UNION))
            d.parent.union_done = true;
    }

    bool is_ready(int qid) {
        return internal_map[qid].cnt == 0;
    }

    SPARQLQuery get_reply(int qid) {
        SPARQLQuery r = internal_map[qid].parent;
        SPARQLQuery &reply = internal_map[qid].reply;

        // copy metadata of result
        r.result.col_num = reply.result.col_num;
        r.result.row_num = reply.result.row_num;
        r.result.attr_col_num = reply.result.attr_col_num;
        r.result.v2c_map = reply.result.v2c_map;
        // NOTE: no need to set nvars, required_vars, and blind

        // copy data of result
        r.result.result_table.swap(reply.result.result_table);
        r.result.attr_res_table.swap(reply.result.attr_res_table);

        internal_map.erase(qid);
        logstream(LOG_DEBUG) << "erase parent-qid=" << qid << LOG_endl;
        return r;
    }
};
