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

#include <zmq.hpp>
#include <string>
#include <iostream>
#include <unistd.h>
#include <unordered_map>
#include <fstream>
#include <errno.h>
#include <sstream>

#include "config.hpp"

using namespace std;

/**
 * The communication over ZeroMQ, a distributed messaging lib
 */
class TCP_Adaptor {
private:
    zmq::context_t context;

    vector<zmq::socket_t *> receivers;
    unordered_map<int, zmq::socket_t *> senders;

    vector<string> ipset;

    inline int port_code(int sid, int tid) { return sid * 200 + tid; }

public:

    TCP_Adaptor(int sid, string fname): context(1) {
        ifstream hostfile(fname);
        string ip;
        while (hostfile >> ip)
            ipset.push_back(ip);

        receivers.resize(global_num_threads);
        for (int tid = 0; tid < global_num_threads; tid++) {
            receivers[tid] = new zmq::socket_t(context, ZMQ_PULL);
            char address[32] = "";
            snprintf(address, 32, "tcp://*:%d", global_eth_port_base + port_code(sid, tid));
            receivers[tid]->bind(address);
        }
    }

    ~TCP_Adaptor() {
        for (auto r : receivers)
            if (r != NULL) delete r;

        for (auto s : senders) {
            if (s.second != NULL) {
                delete s.second;
                s.second = NULL;
            }
        }
    }

    string ip_of(int sid) { return ipset[sid]; }

    void send(int sid, int tid, string str) {
        int pid = port_code(sid, tid);

        // new socket if needed
        if (senders.find(pid) == senders.end()) {
            senders[pid] = new zmq::socket_t(context, ZMQ_PUSH);
            char address[32] = "";
            snprintf(address, 32, "tcp://%s:%d", ipset[sid].c_str(), global_eth_port_base + pid);
            senders[pid]->connect(address);
        }

        zmq::message_t msg(str.length());
        memcpy((void *)msg.data(), str.c_str(), str.length());
        senders[pid]->send(msg);
    }

    string recv(int tid) {
        zmq::message_t msg;
        if (receivers[tid]->recv(&msg) < 0) {
            cout << "recv with error " << strerror(errno) << endl;
            exit(-1);
        }
        return string((char *)msg.data(), msg.size());
    }

    bool tryrecv(int tid, string &str) {
        zmq::message_t msg;
        bool success = false;
        if (success = receivers[tid]->recv(&msg, ZMQ_NOBLOCK))
            str = string((char *)msg.data(), msg.size());
        return success;
    }
};
