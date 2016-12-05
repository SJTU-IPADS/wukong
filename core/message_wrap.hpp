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

#include "config.hpp"
#include "query_basic_types.hpp"
#include "network_node.hpp"
#include "rdma_resource.hpp"

void SendR(thread_cfg *cfg, int mid, int tid, request_or_reply &r) {
    std::stringstream ss;
    boost::archive::binary_oarchive oa(ss);

    oa << r;
    if (global_use_rdma)
        cfg->rdma->rbfSend(cfg->wid, mid, tid, ss.str().c_str(), ss.str().size());
    else
        cfg->node->Send(mid, tid, ss.str());
}

request_or_reply RecvR(thread_cfg *cfg) {
    std::string str;

    if (global_use_rdma)
        str = cfg->rdma->rbfRecv(cfg->wid);
    else
        str = cfg->node->Recv();

    std::stringstream s;
    s << str;

    boost::archive::binary_iarchive ia(s);
    request_or_reply r;
    ia >> r;
    return r;
}

bool TryRecvR(thread_cfg *cfg, request_or_reply &r) {
    std::string str;
    bool ret;
    if (global_use_rdma) {
        ret = cfg->rdma->rbfTryRecv(cfg->wid, str);
        if (!ret) return false;
    } else {
        ret = cfg->node->tryRecv(str);
        if (!ret) return false;
    }

    std::stringstream s;
    s << str;

    boost::archive::binary_iarchive ia(s);
    ia >> r;
    return true;
}

template<typename T>
void SendObject(thread_cfg *cfg, int mid, int tid, T &r) {
    std::stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << r;
    cfg->node->Send(mid, tid, ss.str());
}

template<typename T>
T RecvObject(thread_cfg *cfg) {
    std::string str;
    str = cfg->node->Recv();

    std::stringstream s;
    s << str;

    boost::archive::binary_iarchive ia(s);
    T r;
    ia >> r;
    return r;
}
