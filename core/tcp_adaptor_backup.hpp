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

#include <nng/nng.h>
#include <nng/protocol/pipeline0/push.h>
#include <nng/protocol/pipeline0/pull.h>

#include <tbb/concurrent_unordered_map.h>

using namespace std;

#define SUCC (0)
//number of message received
#define RECVBUF_NUM (10)

class TCP_Adaptor {
private:
    typedef tbb::concurrent_unordered_map<int, nng_socket *> socket_map;
    typedef vector<nng_socket *> socket_vector;

    int port_base;

    // The communication over nng (https://nanomsg.github.io/nng), a socket library.
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
            receivers[tid] = new nng_socket(); 
            int rv = 0;
            if ((rv = nng_pull0_open(receivers[tid])) != SUCC) {
                logstream(LOG_FATAL) << "Failed to init a recv-side socket on " << address << LOG_endl;
                assert(false);
            }
            // set recv size to unlimit
            nng_setopt_size(*(receivers[tid]),NNG_OPT_RECVMAXSZ,0);
            // default value is 1, it may hang with default value if dataset is large
            // so set it to RECVBUF_NUM
            nng_setopt_int(*(receivers[tid]),NNG_OPT_RECVBUF,RECVBUF_NUM);

            if ((rv = nng_listen(*(receivers[tid]), address, NULL, 0)) != SUCC) {
                logstream(LOG_FATAL) << "Failed to bind  a recv-side socket on " << address << LOG_endl;
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

        // alloc msg, nng_send is responsible to free it 
        nng_msg* msg = NULL ;
        nng_msg_alloc(&msg, str.length());
        memcpy((nng_msg_body(msg)), str.c_str(), str.length());

        int rv = 0;
        // FIXME: need lock or not? what to protect?
        pthread_spin_lock(&locks[tid]);
        if (senders.find(pid) == senders.end()) {
            // new socket on-demand
            char address[128] = "";
            snprintf(address, 128, "tcp://%s:%d", ipset[sid].c_str(), port_base + pid);

            senders[pid] = new nng_socket();
            if ((rv = nng_push0_open(senders[pid])) != SUCC) {
                logstream(LOG_FATAL) << "Failed to new a send-side socket on " << address << LOG_endl;
                assert(false);
            }
            if ((rv = nng_dial(*(senders[pid]), address,  NULL, 0)) != SUCC) {
                logstream(LOG_FATAL) << "Failed to dial at send-side socket on " << address << LOG_endl;
                assert(false);
            }
        }
       int n = 0;
       pthread_spin_unlock(&locks[tid]);

       // if succ, msg will be free by nng_sendmsg
       n = nng_sendmsg(*(senders[pid]), msg, 0);

        if(n != SUCC) {
            nng_msg_free(msg);
            logstream(LOG_FATAL) << "Failed to send the msg at send-side "<< LOG_endl;
            return false;
        }

        return true ;
    }

    string recv(int tid) {
        nng_msg* msg = NULL;
        int n = nng_recvmsg(*(receivers[tid]), &msg,0);

        if(n != SUCC ) {
            logstream(LOG_FATAL) << "Failed to recv msg " << LOG_endl;
            assert(false);
        }
        string s((char *)(nng_msg_body(msg)), nng_msg_len(msg));
        nng_msg_free(msg);
        return s;
    }

    bool tryrecv(int tid, string &s) {
        nng_msg* msg = NULL;

        int n = nng_recvmsg(*(receivers[tid]), &msg, NNG_FLAG_NONBLOCK);

        if(n != SUCC ) {
            return false;
        }
        s= string((char *)(nng_msg_body(msg)), nng_msg_len(msg));
        nng_msg_free(msg);
        return true;
    }
};
