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

#include <regex>

#include "global.hpp"
#include "type.hpp"
#include "bind.hpp"
#include "coder.hpp"
#include "dgraph.hpp"
#include "query.hpp"

// engine
#include "msgr.hpp"

// utils
#include "assertion.hpp"
#include "math.hpp"
#include "timer.hpp"

using namespace std;

class RDFEngine {
private:
    int sid;    // server id
    int tid;    // thread id

    DGraph *graph;
    Coder *coder;
    Messenger *msgr;

public:

    RDFEngine(int sid, int tid, DGraph *graph, Coder *coder, Messenger *msgr)
        : sid(sid), tid(tid), graph(graph), coder(coder), msgr(msgr) { }

    void execute_gstore_check(GStoreCheck &r) {
        // unbind the core from the thread (enable OpenMPI multithreading)
        cpu_set_t mask = unbind_to_core();

        r.check_ret = graph->gstore_check(r.index_check, r.normal_check);

        // rebind the thread with the core
        bind_to_core(mask);

        Bundle bundle(r);
        msgr->send_msg(bundle, coder->sid_of(r.pqid), coder->tid_of(r.pqid));
    }

#ifdef DYNAMIC_GSTORE
    void execute_load_data(RDFLoad &r) {
        // unbind the core from the thread (enable OpenMPI multithreading)
        cpu_set_t mask = unbind_to_core();

        r.load_ret = graph->dynamic_loader->dynamic_load_data(r.load_dname, r.check_dup);

        // rebind the thread with the core
        bind_to_core(mask);

        Bundle bundle(r);
        msgr->send_msg(bundle, coder->sid_of(r.pqid), coder->tid_of(r.pqid));
    }
#endif

};
