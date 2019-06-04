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

// comm
#include "tcp_broadcast.hpp"
#include "rdma_broadcast.hpp"

class Broadcast_Master {
public:
    TCP_Broadcast_Master *tcp;   // communicaiton by TCP/IP
    RDMA_Broadcast_Master *rdma; // communicaiton by RDMA

    Broadcast_Master(TCP_Broadcast_Master* tcp, RDMA_Broadcast_Master* rdma)
        : tcp(tcp), rdma(rdma) { }

    ~Broadcast_Master() { }

    bool send(int dst_sid, const string &str) {
        if (Global::use_rdma && rdma->init)
            return rdma->send(dst_sid, str);
        else
            return tcp->send(dst_sid, str);
    }

    bool send(int dst_sid, const Bundle &b) {
        string str = b.to_str();
        return send(dst_sid, str);
    }

    Bundle recv() {
        std::string str;
        if (Global::use_rdma && rdma->init)
            str = rdma->recv();
        else
            str = tcp->recv();
        return Bundle(str);
    }

    bool tryrecv(string &str) {
        if (Global::use_rdma && rdma->init)
            return rdma->tryrecv(str);
        else
            return tcp->tryrecv(str);
    }

    bool tryrecv(Bundle &b) {
        string str;
        if (!tryrecv(str)) return false;
        b.init(str);
        return true;
    }
};

class Broadcast_Slave {
public:
    TCP_Broadcast_Slave *tcp;    // communicaiton by TCP/IP
    RDMA_Broadcast_Slave *rdma;  // communicaiton by RDMA

    Broadcast_Slave(TCP_Broadcast_Slave* tcp, RDMA_Broadcast_Slave* rdma)
        : tcp(tcp), rdma(rdma) { }

    ~Broadcast_Slave() { }

    bool send(const string &str) {
        if (Global::use_rdma && rdma->init)
            return rdma->send(str);
        else
            return tcp->send(str);
    }

    bool send(const Bundle &b) {
        string str = b.to_str();
        return send(str);
    }

    Bundle recv() {
        std::string str;
        if (Global::use_rdma && rdma->init)
            str = rdma->recv();
        else
            str = tcp->recv();
        return Bundle(str);
    }

    bool tryrecv(string &str) {
        if (Global::use_rdma && rdma->init)
            return rdma->tryrecv(str);
        else
            return tcp->tryrecv(str);
    }

    bool tryrecv(Bundle &b) {
        string str;
        if (!tryrecv(str)) return false;
        b.init(str);
        return true;
    }
};
