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

#include "global.hpp"

using namespace std;

class TCP_Adaptor {
private:
    typedef tbb::concurrent_unordered_map<int, zmq::socket_t *> socket_map;
    typedef vector<zmq::socket_t *> socket_vector;

    int port_base;

    // The communication over zeromq, a socket library.
    zmq::context_t context;
    socket_vector receivers;     // static allocation
    socket_map senders;          // dynamic allocation

    pthread_spinlock_t *send_locks;
    pthread_spinlock_t *receive_locks;

    vector<string> ipset;

    inline int port_code(int sid, int tid) { return sid * 200 + tid; }

public:
    TCP_Adaptor(int sid, string fname, int num_threads, int port_base)
        : port_base(port_base), context(1) {

        ifstream hostfile(fname);
        string ip;
        while (hostfile >> ip)
            ipset.push_back(ip);

        receivers.resize(num_threads);
        for (int tid = 0; tid < num_threads; tid++) {
            receivers[tid] = new zmq::socket_t(context, ZMQ_PULL);
            char address[32] = "";
            snprintf(address, 32, "tcp://*:%d", port_base + port_code(sid, tid));
            receivers[tid]->bind(address);
        }

        send_locks = (pthread_spinlock_t *)malloc(sizeof(pthread_spinlock_t) * num_threads);
        for (int i = 0; i < num_threads; i++)
            pthread_spin_init(&send_locks[i], 0);

        receive_locks = (pthread_spinlock_t *)malloc(sizeof(pthread_spinlock_t) * num_threads);
        for (int i = 0; i < num_threads; i++)
            pthread_spin_init(&receive_locks[i], 0);
    }

    ~TCP_Adaptor() {
        for (auto &r : receivers)
            if (r != NULL) delete r;

        for (auto &s : senders) {
            if (s.second != NULL) {
                delete s.second;
                s.second = NULL;
            }
        }
    }

    string ip_of(int sid) { return ipset[sid]; }

    bool send(int sid, int tid, const string &str) {
        int pid = port_code(sid, tid);

        zmq::message_t msg(str.length());
        memcpy((void *)msg.data(), str.c_str(), str.length());

        // avoid two contentions
        // 1) add the 'equal' sockets to the set (overwrite)
        // 2) use the same socket by multiple proxy threads simultaneously.
        pthread_spin_lock(&send_locks[tid]);
        if (senders.find(pid) == senders.end()) {
            // new socket on-demand
            char address[32] = "";
            snprintf(address, 32, "tcp://%s:%d", ipset[sid].c_str(), port_base + pid);
            senders[pid] = new zmq::socket_t(context, ZMQ_PUSH);
            /// FIXME: check return value
            senders[pid]->connect(address);
        }

        bool result = senders[pid]->send(msg, ZMQ_DONTWAIT);
        pthread_spin_unlock(&send_locks[tid]);

        return result;
    }

    string recv(int tid) {
        zmq::message_t msg;

        // multiple engine threads may recv the same msg simultaneously (no case)
        pthread_spin_lock(&receive_locks[tid]);
        if (receivers[tid]->recv(&msg) < 0) {
            logstream(LOG_ERROR) << "Failed to recv msg ("
                                 << strerror(errno) << ")" << LOG_endl;
            assert(false);
        }
        pthread_spin_unlock(&receive_locks[tid]);

        return string((char *)msg.data(), msg.size());
    }

    bool tryrecv(int tid, string &str) {
        zmq::message_t msg;
        bool success = false;

        // multiple engine threads may recv the same msg simultaneously
        // (work-stealing is the only case now)
        pthread_spin_lock(&receive_locks[tid]);
        if (success = receivers[tid]->recv(&msg, ZMQ_NOBLOCK))
            str = string((char *)msg.data(), msg.size());
        pthread_spin_unlock(&receive_locks[tid]);

        return success;
    }
};
