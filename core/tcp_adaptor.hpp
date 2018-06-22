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

//#include <zmq.hpp>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <iostream>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>

//#include <nanomsg/nn.h>
#include <nanomsg/nn.hpp>
#include <nanomsg/pipeline.h>
#include <nanomsg/tcp.h>

#include <tbb/concurrent_unordered_map.h>

using namespace std;


class TCP_Adaptor {
private:
    typedef tbb::concurrent_unordered_map<int, nn::socket *> socket_map;
    typedef vector<nn::socket *> socket_vector;

    int port_base;

    // The communication over nanomsg (https://nanomsg.org/), a socket library.
    socket_vector receivers;  // static allocation
    socket_map senders;       // dynamic allocation

    pthread_spinlock_t *locks;

    vector<string> ipset;

    inline int port_code(int sid, int tid) { return sid * 200 + tid; }

public:

    TCP_Adaptor(int sid, string fname, int num_threads, int port_base)
        : port_base(port_base) {

        ifstream hostfile(fname);
        string ip;
        while (hostfile >> ip)
            ipset.push_back(ip);

        receivers.resize(num_threads);
        for (int tid = 0; tid < num_threads; tid++) {
            char address[128] = "";
            snprintf(address, 128, "tcp://*:%d", port_base + port_code(sid, tid));

            try {
                receivers[tid] = new nn::socket(AF_SP, NN_PULL);

                /// NN_RCVMAXSIZE: Maximum message size that can be received, in bytes.
                /// Negative value means that the received size is limited only by available
                /// addressable memory. The default type is 1024KB (too small for WUKONG)
                int limit = -1;
                receivers[tid]->setsockopt(NN_SOL_SOCKET, NN_RCVMAXSIZE, &limit, sizeof(limit));

                receivers[tid]->bind(address);
            } catch (nn::exception e) {
                logstream(LOG_FATAL) << "Failed to init a recv-side socket on " << address
                                     << " (" << e.what() << ")" << LOG_endl;
                assert(false);
            }
        }

        locks = (pthread_spinlock_t *)malloc(sizeof(pthread_spinlock_t) * num_threads);
        for (int i = 0; i < num_threads; i++)
            pthread_spin_init(&locks[i], 0);
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

    bool send(int sid, int tid, string str) {
        int pid = port_code(sid, tid);

        void *msg = nn::allocmsg(str.length(), 0);
        memcpy(msg, str.c_str(), str.length());

        // FIXME: need lock or not? what to protect?
        pthread_spin_lock(&locks[tid]);
        if (senders.find(pid) == senders.end()) {
            // new socket on-demand
            char address[128] = "";
            snprintf(address, 128, "tcp://%s:%d", ipset[sid].c_str(), port_base + pid);

            try {
                senders[pid] = new nn::socket(AF_SP, NN_PUSH);
                senders[pid]->connect(address);
            } catch (nn::exception e) {
                logstream(LOG_FATAL) << "Failed to new a send-side socket on " << address
                                     << " (" << e.what() << ")" << LOG_endl;
                assert(false);
            }
        }

        int n = 0;
        try {
            n = senders[pid]->send(&msg, NN_MSG, 0);
        } catch (nn::exception e) {
            logstream(LOG_FATAL) << "Failed to send msg (" << e.what() << ")" << LOG_endl;
            assert(false);
        }
        pthread_spin_unlock(&locks[tid]);

        return (n == str.length());
    }

    string recv(int tid) {
        try {
            void *msg = NULL;
            int n = receivers[tid]->recv(&msg, NN_MSG, 0);

            string s((char *)msg, n);
            nn::freemsg(msg);
            return s;
        } catch (nn::exception e) {
            logstream(LOG_FATAL) << "Failed to recv msg (" << e.what() << ")" << LOG_endl;
            assert(false);
        }
    }

    bool tryrecv(int tid, string &s) {
        try {
            void *msg = NULL;
            int n = receivers[tid]->recv(&msg, NN_MSG, NN_DONTWAIT);
            if (n < 0)
                return false;
            s = string((char *)msg, n);
            nn::freemsg(msg);
            return true;
        } catch (nn::exception e) {
            logstream(LOG_FATAL) << "Failed to tryrecv msg (" << e.what() << ")" << LOG_endl;
            assert(false);
        }
    }
};
