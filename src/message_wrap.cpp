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

#include "message_wrap.h"

void SendR(thread_cfg* cfg, int r_mid, int r_tid, request_or_reply& r) {
    std::stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << r;
    if (global_use_rbf) {
        cfg->rdma->rbfSend(cfg->wid, r_mid, r_tid, ss.str().c_str(), ss.str().size());
    } else {
        cfg->node->Send(r_mid, r_tid, ss.str());
    }
}

request_or_reply RecvR(thread_cfg* cfg) {
    std::string str;
    if (global_use_rbf) {
        str = cfg->rdma->rbfRecv(cfg->wid);
    } else {
        str = cfg->node->Recv();
    }
    std::stringstream s;
    s << str;
    boost::archive::binary_iarchive ia(s);
    request_or_reply r;
    ia >> r;
    return r;
}

bool TryRecvR(thread_cfg* cfg, request_or_reply& r) {
    std::string str;
    if (global_use_rbf) {
        bool ret = cfg->rdma->rbfTryRecv(cfg->wid, str);
        if (!ret) {
            return false;
        }
    } else {
        str = cfg->node->tryRecv();
        if (str == "") {
            return false;
        }
    }
    std::stringstream s;
    s << str;
    boost::archive::binary_iarchive ia(s);
    ia >> r;
    return true;
};
