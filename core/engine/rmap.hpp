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

        Item d;
        d.cnt = cnt;
        d.parent = r;

        internal_map[r.qid] = d;
    }

    void put_reply(SPARQLQuery &r) {
        // exist
        ASSERT(internal_map.find(r.pqid) != internal_map.end());

        Item &d = internal_map[r.pqid];
        SPARQLQuery::Result &whole = d.reply.result;
        SPARQLQuery::Result &part = r.result;
        d.cnt--;

        // if the PatternGroup of r comes from a query's union part,
        // use merge_result to put result
        if (r.pg_type == SPARQLQuery::PGType::UNION)
            whole.merge_result(part);
        else
            whole.append_result(part);

        // keep inprogress
        if (d.parent.state == SPARQLQuery::SQState::SQ_PATTERN)
            d.reply.pattern_step = r.pattern_step;

        // if r is a forked query that has done OPTIONAL query,
        // update its parent's optional_step to avoid the parent do it again
        if (r.pg_type != SPARQLQuery::PGType::OPTIONAL
                && r.done(SPARQLQuery::SQState::SQ_OPTIONAL))
            d.parent.optional_step = r.optional_step;

        // if r is a forked query that has done UNION query,
        // update its parent's union_done to avoid the parent do it again
        if (r.union_done)
            d.parent.union_done = true;
    }

    bool is_ready(int qid) {
        return internal_map[qid].cnt == 0;
    }

    SPARQLQuery get_merged_reply(int qid) {
        SPARQLQuery r = internal_map[qid].parent;
        SPARQLQuery &reply = internal_map[qid].reply;

        // copy the result
        // FIXME: implement copy construct of SPARQLQuery::Result
        r.result.col_num = reply.result.col_num;
        r.result.blind = reply.result.blind;
        r.result.row_num = reply.result.row_num;
        r.result.attr_col_num = reply.result.attr_col_num;
        r.result.v2c_map = reply.result.v2c_map;
        r.result.result_table.swap(reply.result.result_table);
        r.result.attr_res_table.swap(reply.result.attr_res_table);

        // FIXME: need sync other fields or not
        if (r.state == SPARQLQuery::SQState::SQ_PATTERN)
            r.pattern_step = reply.pattern_step;

        internal_map.erase(qid);
        logstream(LOG_DEBUG) << "erase parent-qid=" << qid << LOG_endl;
        return r;
    }
};
