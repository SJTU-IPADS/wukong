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

#include <zmq.hpp>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <iostream>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <tbb/concurrent_unordered_map.h>

#include "config.hpp"

using namespace std;

/**
 * The communication over ZeroMQ, a distributed messaging lib
 */
class TCP_Adaptor {
private:
    typedef tbb::concurrent_unordered_map<int, zmq::socket_t *> tbb_unordered_map;

    int port_base;
    zmq::context_t context;

    vector<zmq::socket_t *> receivers;  // exclusive
    tbb_unordered_map senders;          // shared

    pthread_spinlock_t *locks;

    vector<string> ipset;

    inline int port_code(int sid, int tid) { return sid * 200 + tid; }

public:

    TCP_Adaptor(int sid, string fname, int num_threads, int port_base)
        : port_base(port_base), context(1) {

        ifstream hostfile(fname);
        string ip;
        while (hostfile >> ip)
            ipset.push_back(ip);

        receivers.resize(global_num_threads);
        for (int tid = 0; tid < global_num_threads; tid++) {
            receivers[tid] = new zmq::socket_t(context, ZMQ_PULL);
            char address[32] = "";
            snprintf(address, 32, "tcp://*:%d", port_base + port_code(sid, tid));
            receivers[tid]->bind(address);
        }

        locks = (pthread_spinlock_t *)malloc(sizeof(pthread_spinlock_t) * num_threads);
        for (int i = 0; i < num_threads; i++)
            pthread_spin_init(&locks[i], 0);
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

    bool send(int sid, int tid, string str) {
        int pid = port_code(sid, tid);

        // new socket if needed
        pthread_spin_lock(&locks[tid]);
        if (senders.find(pid) == senders.end()) {
            senders[pid] = new zmq::socket_t(context, ZMQ_PUSH);
            char address[32] = "";
            snprintf(address, 32, "tcp://%s:%d", ipset[sid].c_str(), port_base + pid);
            senders[pid]->connect(address);
        }

        zmq::message_t msg(str.length());
        memcpy((void *)msg.data(), str.c_str(), str.length());
        bool result = senders[pid]->send(msg, ZMQ_DONTWAIT);
        pthread_spin_unlock(&locks[tid]);
        return result;
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
