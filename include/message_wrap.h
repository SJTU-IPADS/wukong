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

#include "query_basic_types.h"
#include "network_node.h"
#include "rdma_resource.h"
#include "thread_cfg.h"
#include "global_cfg.h"


void SendR(thread_cfg* cfg, int r_mid, int r_tid, request_or_reply& r);
request_or_reply RecvR(thread_cfg* cfg);
bool TryRecvR(thread_cfg* cfg, request_or_reply& r);

template<typename T>
void SendObject(thread_cfg* cfg, int r_mid, int r_tid, T& r) {
    std::stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << r;
    cfg->node->Send(r_mid, r_tid, ss.str());
}

template<typename T>
T RecvObject(thread_cfg* cfg) {
    std::string str;
    str = cfg->node->Recv();
    std::stringstream s;
    s << str;
    boost::archive::binary_iarchive ia(s);
    T r;
    ia >> r;
    return r;
}
