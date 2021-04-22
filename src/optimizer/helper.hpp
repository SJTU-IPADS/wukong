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

#include "core/store/dgraph.hpp"

#include "optimizer/stats_type.hpp"
#include "optimizer/stats.hpp"

namespace wukong {

class Dgraph_helper {

    int tid;
    DGraph *graph;
    Stats *stats;


   public:
    Dgraph_helper() {}
    Dgraph_helper(int tid, DGraph *graph, Stats *stats) : tid(tid), graph(graph), stats(stats) {}

    uint64_t get_triples_size(ssid_t constant, ssid_t p, ssid_t d){
        uint64_t count;
        graph->get_triples(tid, constant, p, dir_t(d), count);
        return count;
    }

    // equal is true, not equal is false
    bool compare_types(ssid_t type1, ssid_t type2){
        return type1 == type2;
    }

    // get the type of constant using get_edges
    ssid_t get_type(ssid_t constant) {
        uint64_t type_sz = 0;
        edge_t *tids = graph->get_triples(tid, constant, TYPE_ID, OUT, type_sz);
        if (type_sz == 1) {
            return tids[0].val;
        } else if (type_sz > 1) {
            std::unordered_set<int> type_composition;

            for (int i = 0; i < type_sz; i ++)
                type_composition.insert(tids[i].val);

            type_t type;
            type.set_type_composition(type_composition);
            return stats->global_type2int[type];
        } else {
            std::unordered_set<int> index_composition;

            uint64_t psize1 = 0;
            edge_t *pids1 = graph->get_triples(tid, constant, PREDICATE_ID, OUT, psize1);
            for (uint64_t k = 0; k < psize1; k++) {
                ssid_t pre = pids1[k].val;
                index_composition.insert(pre);
            }

            uint64_t psize2 = 0;
            edge_t *pids2 = graph->get_triples(tid, constant, PREDICATE_ID, IN, psize2);
            for (uint64_t k = 0; k < psize2; k++) {
                ssid_t pre = pids2[k].val;
                index_composition.insert(-pre);
            }

            type_t type;
            type.set_index_composition(index_composition);
            return stats->global_type2int[type];
        }
    }

};

} // namespace wukong