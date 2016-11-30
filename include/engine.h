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
 *      http://ipads.se.sjtu.edu.cn/projects/wukong.html
 *
 */

#pragma once

#include "message_wrap.h"
#include "distributed_graph.h"
#include "query_basic_types.h"
#include "global_cfg.h"
#include "thread_cfg.h"
#include "wait_queue.h"

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

class Engine {
    const static uint64_t TIMEOUT_THRESHOLD = 10000; // 10 msec

    distributed_graph &g;
    thread_cfg *cfg;

    uint64_t last_time; // busy or not (work-oblige)

    pthread_spinlock_t recv_lock;
    std::vector<request_or_reply> msg_fast_path;

    pthread_spinlock_t wqueue_lock;
    wait_queue wqueue;


    // all of these means const predict
    void const_to_unknown(request_or_reply &req);
    void const_to_known(request_or_reply &req);
    void known_to_unknown(request_or_reply &req);
    void known_to_known(request_or_reply &req);
    void known_to_const(request_or_reply &req);
    void index_to_unknown(request_or_reply &req);

    // unknown_predict
    void const_unknown_unknown(request_or_reply &req);
    void known_unknown_unknown(request_or_reply &req);
    void known_unknown_const(request_or_reply &req);


    std::vector<request_or_reply> generate_sub_query(request_or_reply &r);
    std::vector<request_or_reply> generate_mt_sub_requests(request_or_reply &r);

    bool need_fork_join(request_or_reply &req);

    void do_corun(request_or_reply &req);

    bool execute_one_step(request_or_reply &req);
    void execute_request(request_or_reply &req);
    void execute(request_or_reply &r, int wid);

public:
    Engine(distributed_graph &_g, thread_cfg *_cfg);

    void run();
};

// a vector of pointers of all local engines
extern std::vector<Engine *> engines;
