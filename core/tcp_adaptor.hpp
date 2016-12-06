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
 * The connection over ZeroMQ, a distributed messaging lib
 */
class TCP_Adaptor {
private:
    zmq::context_t context;

    zmq::socket_t *receiver;
    unordered_map<int, zmq::socket_t *> senders;

    vector<string> ipset;

    inline int code(int sid, int wid) { return sid * 200 + wid; }

public:

    TCP_Adaptor(int sid, int wid, string fname): context(1) {
        ifstream hostfile(fname);
        string ip;
        while (hostfile >> ip)
            ipset.push_back(ip);

        receiver = new zmq::socket_t(context, ZMQ_PULL);
        char address[32] = "";
        snprintf(address, 32, "tcp://*:%d", global_eth_port_base + code(sid, wid));
        receiver->bind(address);
    }

    ~TCP_Adaptor() {
        delete receiver;
        for (auto s : senders) {
            if (s.second != NULL) {
                delete s.second;
                s.second = NULL;
            }
        }
    }

    string ip_of(int sid) { return ipset[sid]; }

    void send(int sid, int wid, string str) {
        int id = code(sid, wid);

        // new socket if needed
        if (senders.find(id) == senders.end()) {
            senders[id] = new zmq::socket_t(context, ZMQ_PUSH);
            char address[32] = "";
            snprintf(address, 32, "tcp://%s:%d", ipset[sid].c_str(), global_eth_port_base + id);
            senders[id]->connect(address);
        }

        zmq::message_t msg(str.length());
        memcpy((void *)msg.data(), str.c_str(), str.length());
        senders[id]->send(msg);
    }

    string recv() {
        zmq::message_t msg;
        if (receiver->recv(&msg) < 0) {
            cout << "recv with error " << strerror(errno) << endl;
            exit(-1);
        }
        return string((char *)msg.data(), msg.size());
    }

    bool tryrecv(string &str) {
        zmq::message_t msg;
        bool success = false;
        if (success = receiver->recv(&msg, ZMQ_NOBLOCK))
            str = string((char *)msg.data(), msg.size());
        return success;
    }
};
