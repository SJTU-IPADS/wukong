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

// utils
#include "logger2.hpp"
#include "assertion.hpp"

using namespace std;

class TCP_Adaptor {
private:

    int sid;            // unused in TCP communication
    int num_servers;    // used to check parameter
    int num_threads;

    int port_base;

    typedef tbb::concurrent_unordered_map<int, zmq::socket_t *> socket_map;
    typedef vector<zmq::socket_t *> socket_vector;

    // The communication over zeromq, a socket library.
    socket_vector receivers;     // static allocation
    socket_map senders;          // dynamic allocation

    zmq::context_t context;

    pthread_spinlock_t *send_locks;
    pthread_spinlock_t *receive_locks;

    vector<string> ipset;

    inline int port_code(int dst_tid) { return port_base + dst_tid; }
    inline int socket_code(int dst_sid, int dst_tid) { return dst_sid * num_threads + dst_tid; }

public:
    TCP_Adaptor(int sid, string fname, int port_base, int nsrvs, int nthds)
        : sid(sid), port_base(port_base), context(1),
          num_servers(nsrvs), num_threads(nthds) {

        ifstream hostfile(fname);
        string ip;
        while (hostfile >> ip)
            ipset.push_back(ip);

        receivers.resize(num_threads);
        for (int tid = 0; tid < num_threads; tid++) {
            receivers[tid] = new zmq::socket_t(context, ZMQ_PULL);
            char address[32] = "";
            snprintf(address, 32, "tcp://*:%d", port_code(tid));
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

    string ip_of(int dst_sid) { return ipset[dst_sid]; }

    bool send(int dst_sid, int dst_tid, const string &str) {
        // check parameters
        ASSERT_MSG((dst_sid >= 0 && dst_sid < num_servers),
                   "server ID: %d (#servers: %d)\n", dst_sid, num_servers);
        ASSERT_MSG((dst_tid >= 0 && dst_tid < num_threads),
                   "thread ID: %d (#threads: %d)\n", dst_tid, num_threads);

        int pid = port_code(dst_tid);
        int id = socket_code(dst_sid, dst_tid); // socket id

        zmq::message_t msg(str.length());
        memcpy((void *)msg.data(), str.c_str(), str.length());

        // avoid two contentions
        // 1) add the 'equal' sockets to the set (overwrite)
        // 2) use the same socket by multiple proxy threads simultaneously.
        pthread_spin_lock(&send_locks[dst_tid]);
        if (senders.find(id) == senders.end()) {
            // new socket on-demand
            char address[32] = "";
            snprintf(address, 32, "tcp://%s:%d", ipset[dst_sid].c_str(), pid);
            senders[id] = new zmq::socket_t(context, ZMQ_PUSH);
            senders[id]->connect(address);
        }

        int result = senders[id]->send(msg, ZMQ_DONTWAIT);
        if (result < 0) {
            logstream(LOG_ERROR) << "failed to send msg to ["
                                 << dst_sid << ", " << dst_tid << "] "
                                 << strerror(errno) << LOG_endl;
        }
        pthread_spin_unlock(&send_locks[dst_tid]);

        return result;
    }

    string recv(int tid) {
        ASSERT_MSG((tid >= 0 && tid < num_threads),
                   "thread ID: %d (#threads: %d)\n", tid, num_threads);

        zmq::message_t msg;

        // multiple engine threads may recv the same msg simultaneously (no case)
        pthread_spin_lock(&receive_locks[tid]);
        if (receivers[tid]->recv(&msg) < 0) {
            logstream(LOG_ERROR) << "failed to recv msg ("
                                 << strerror(errno) << ")" << LOG_endl;
            assert(false);
        }
        pthread_spin_unlock(&receive_locks[tid]);

        return string((char *)msg.data(), msg.size());
    }


    string recv(int tid, int src_sid) {
        logstream(LOG_WARNING) << "recv() from a specified server (sid=" << src_sid
                               << ") is unsupported by TCP adaptor now!"
                               << LOG_endl;
        return recv(tid);
    }

    bool tryrecv(int tid, string &str) {
        ASSERT_MSG((tid >= 0 && tid < num_threads),
                   "thread ID: %d (#threads: %d)\n", tid, num_threads);

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

    bool tryrecv(int tid, string &str, int &src_sid) {
        logstream(LOG_WARNING) << "tryrecv() and retrieve the server ID "
                               << "is unsuppored by TCP adaptor now!"
                               << LOG_endl;
        src_sid = -1;
        return tryrecv(tid, str);
    }

};
