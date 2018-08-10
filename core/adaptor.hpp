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

#include "config.hpp"
#include "query.hpp"
#include "tcp_adaptor.hpp"
#include "rdma_adaptor.hpp"

/// TODO: define adaptor as a C++ interface and make tcp and rdma implement it
class Adaptor {
public:
    int tid; // thread id

    TCP_Adaptor *tcp;   // communicaiton by TCP/IP
    RDMA_Adaptor *rdma; // communicaiton by RDMA

    Adaptor(int tid, TCP_Adaptor *tcp, RDMA_Adaptor *rdma)
        : tid(tid), tcp(tcp), rdma(rdma) { }

    ~Adaptor() { }

    bool send(int dst_sid, int dst_tid, const string &str) {
        if (global_use_rdma && rdma->init)
            return rdma->send(tid, dst_sid, dst_tid, str);
        else
            return tcp->send(dst_sid, dst_tid, str);
    }

    bool send(int dst_sid, int dst_tid, const Bundle &b) {
        string str = b.to_str();
        return send(dst_sid, dst_tid, str);
    }

    // gpu-direct send, from gpu mem to remote ring buffer
    bool send_dev2host(int dst_sid, int dst_tid, char *data, uint64_t sz) {
#ifdef USE_GPU
        if (global_use_rdma && rdma->init)
            return rdma->send_dev2host(tid, dst_sid, dst_tid, data, sz);

        // TODO: support dev2host w/o RDMA
        logstream(LOG_ERROR) << "RDMA is required for send_dev2host." << LOG_endl;
        ASSERT (false);
#else
        logstream(LOG_ERROR) << "USE_GPU is not defined." << LOG_endl;
        ASSERT (false);
#endif
    }

    Bundle recv() {
        std::string str;
        if (global_use_rdma && rdma->init)
            str = rdma->recv(tid);
        else
            str = tcp->recv(tid);
        return Bundle(str.c_str(), str.length());
    }

    bool tryrecv(string &str) {
        if (global_use_rdma && rdma->init)
            return rdma->tryrecv(tid, str);
        else
            return tcp->tryrecv(tid, str);
    }

    bool tryrecv(Bundle &b) {
        string str;
        if (!tryrecv(str)) return false;
        b.init(str.c_str(), str.length());
        return true;
    }
};
