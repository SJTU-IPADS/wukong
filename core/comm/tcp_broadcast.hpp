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

#include "query.hpp"

using namespace std;

// zeromq version one-to-many communication
class TCP_Broadcast_Master {
private:
    int sid;
    int tid;
    int slave_tid;
    int port_base;

    typedef tbb::concurrent_unordered_map<int, zmq::socket_t *> socket_map;

    socket_map senders;          // dynamic allocation
    zmq::socket_t * receiver;    // static allocation, only one receiver

    zmq::context_t context;

    vector<string> ipset;

    inline int port_code(int dst_sid, int dst_tid) { return dst_sid * 200 + dst_tid; }

public:
    TCP_Broadcast_Master(int sid, int tid, int port_base, string fname, int s_tid)
        :sid(sid), tid(tid), port_base(port_base), slave_tid(s_tid), context(1) {

        ifstream hostfile(fname);
        string ip;
        while (hostfile >> ip)
            ipset.push_back(ip);

        receiver = new zmq::socket_t(context, ZMQ_PULL);
        char address[32] = "";
        snprintf(address, 32, "tcp://*:%d", port_base + port_code(sid, tid));
        receiver->bind(address);
    }

    bool send(int dst_sid, const string &str) {
        int pid = port_code(dst_sid, slave_tid);

        zmq::message_t msg(str.length());
        memcpy((void *)msg.data(), str.c_str(), str.length());

        if (senders.find(pid) == senders.end()) {
            // new socket on-demand
            char address[32] = "";
            snprintf(address, 32, "tcp://%s:%d", ipset[dst_sid].c_str(), port_base + pid);
            senders[pid] = new zmq::socket_t(context, ZMQ_PUSH);
            /// FIXME: check return value
            senders[pid]->connect(address);
        }
        bool result = senders[pid]->send(msg, ZMQ_DONTWAIT);
        return result;
    }

    string recv() {
        zmq::message_t msg;
        if (receiver->recv(&msg) < 0) {
            logstream(LOG_ERROR) << "Failed to recv msg ("
                                 << strerror(errno) << ")" << LOG_endl;
            assert(false);
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

class TCP_Broadcast_Slave {
private:
    int sid;
    int tid;
    int port_base;

    zmq::socket_t * receiver;    // static allocation, only one receiver
    zmq::socket_t * sender;      // static allocation, only one sender

    zmq::context_t context;

    vector<string> ipset;

    inline int port_code(int dst_sid, int dst_tid) { return dst_sid * 200 + dst_tid; }
public:
    TCP_Broadcast_Slave(int sid, int tid, string fname, int port_base, int m_sid, int m_tid)
        :sid(sid), tid(tid), port_base(port_base), context(1) {

        ifstream hostfile(fname);
        string ip;
        while (hostfile >> ip)
            ipset.push_back(ip);

        //sender should be connected to string server
        sender = new zmq::socket_t(context, ZMQ_PUSH);
        int pid = port_code(m_sid, m_tid);
        char sender_address[32] = "";
        snprintf(sender_address, 32, "tcp://%s:%d", ipset[m_sid].c_str(), port_base + pid);
        /// FIXME: check return value
        sender->connect(sender_address);

        receiver = new zmq::socket_t(context, ZMQ_PULL);
        char receiver_address[32] = "";
        snprintf(receiver_address, 32, "tcp://*:%d", port_base + port_code(sid, tid));
        receiver->bind(receiver_address);
    }

    bool send(const string &str) {
        zmq::message_t msg(str.length());
        memcpy((void *)msg.data(), str.c_str(), str.length());

        bool result = sender->send(msg, ZMQ_DONTWAIT);
        return result;
    }

    string recv() {
        zmq::message_t msg;
        if (receiver->recv(&msg) < 0) {
            logstream(LOG_ERROR) << "Failed to recv msg ("
                                 << strerror(errno) << ")" << LOG_endl;
            assert(false);
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
