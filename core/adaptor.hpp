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
#include "query.hpp"
#include "tcp_adaptor.hpp"
#include "rdma_resource.hpp"

/// TODO: define adaptor as a C++ interface and make tcp and rdma implement it
class Adaptor {
public:
    int tid; // thread id

    TCP_Adaptor *tcp;  // communicaiton by TCP/IP
    RdmaResource *rdma;  // communicaiton by RDMA

    Adaptor(int tid, TCP_Adaptor *tcp, RdmaResource *rdma)
        : tid(tid), tcp(tcp), rdma(rdma) { }

    ~Adaptor() {
        delete tcp;
        // currently, 'rdma' is showed by all threads, can not delete here
    }

    void send(int dst_sid, int dst_tid, request_or_reply &r) {
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);

        oa << r;
        if (global_use_rdma)
            rdma->rbfSend(tid, dst_sid, dst_tid, ss.str().c_str(), ss.str().size());
        else
            tcp->send(dst_sid, dst_tid, ss.str());
    }

    request_or_reply recv() {
        std::string str;
        if (global_use_rdma)
            str = rdma->rbfRecv(tid);
        else
            str = tcp->recv();

        std::stringstream s;
        s << str;

        boost::archive::binary_iarchive ia(s);
        request_or_reply r;
        ia >> r;
        return r;
    }

    bool tryrecv(request_or_reply &r) {
        std::string str;
        if (global_use_rdma) {
            if (!rdma->rbfTryRecv(tid, str))
                return false;
        } else {
            if (!tcp->tryrecv(str))
                return false;
        }

        std::stringstream s;
        s << str;

        boost::archive::binary_iarchive ia(s);
        ia >> r;
        return true;
    }

    template<typename T>
    void send_object(int dst_sid, int dst_tid, T &r) {
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);
        oa << r;
        tcp->send(dst_sid, dst_tid, ss.str());
    }

    template<typename T>
    T recv_object() {
        std::string str;
        str = tcp->recv();

        std::stringstream s;
        s << str;

        boost::archive::binary_iarchive ia(s);
        T r;
        ia >> r;
        return r;
    }
};
