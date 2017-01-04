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

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "config.hpp"
#include "query.hpp"
#include "tcp_adaptor.hpp"
#include "rdma_adaptor.hpp"

/// TODO: define adaptor as a C++ interface and make tcp and rdma implement it
class Adaptor {
public:
    int tid; // thread id

    TCP_Adaptor *tcp = NULL;   // communicaiton by TCP/IP
    RDMA_Adaptor *rdma = NULL; // communicaiton by RDMA

    Adaptor(int tid, TCP_Adaptor *tcp = NULL, RDMA_Adaptor *rdma = NULL)
        : tid(tid), tcp(tcp), rdma(rdma) { }

    ~Adaptor() { }

    void send(int dst_sid, int dst_tid, request_or_reply &r) {
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);

        oa << r;
        if (global_use_rdma) {
            if (rdma) {
                rdma->send(tid, dst_sid, dst_tid, ss.str());
            } else {
                cout << "ERORR: attempting to use RDMA adaptor, "
                     << "but Wukong was built without RDMA."
                     << endl;
            }
        } else {
            tcp->send(dst_sid, dst_tid, ss.str());
        }
    }

    request_or_reply recv() {
        std::string str;
        if (global_use_rdma) {
            if (rdma) {
                str = rdma->recv(tid);
            } else {
                cout << "ERORR: attempting to use RDMA adaptor, "
                     << "but Wukong was built without RDMA."
                     << endl;
            }
        } else {
            str = tcp->recv(tid);
        }

        std::stringstream ss;
        ss << str;

        boost::archive::binary_iarchive ia(ss);
        request_or_reply r;
        ia >> r;
        return r;
    }

    bool tryrecv(request_or_reply &r) {
        std::string str;
        if (global_use_rdma) {
            if (rdma) {
                if (!rdma->tryrecv(tid, str)) return false;
            } else {
                cout << "ERORR: attempting to use RDMA adaptor, "
                     << "but Wukong was built without RDMA."
                     << endl;
            }
        } else {
            if (!tcp->tryrecv(tid, str)) return false;
        }

        std::stringstream ss;
        ss << str;

        boost::archive::binary_iarchive ia(ss);
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
        str = tcp->recv(tid);

        std::stringstream ss;
        ss << str;

        boost::archive::binary_iarchive ia(ss);
        T r;
        ia >> r;
        return r;
    }
};
