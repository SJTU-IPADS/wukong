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
#include <zhelpers.hpp>
#include <string>
#include <sstream>

#include "cs_basic_type.hpp"

class Server_Socket {
public:
  zmq::context_t context;
  zmq::socket_t replyer;
  std::string identity;
  Server_Socket(std::string _broker_name, int _broker_port):
    context(1), replyer(zmq::socket_t(context, ZMQ_REP)) {
    s_set_id(replyer);
    char address[30] = "";
    sprintf(address, "tcp://%s:%d", _broker_name.c_str(), _broker_port);
    //fprintf(stdout,"tcp binding address %s\n",address);
    replyer.connect(address);
  }

  void send(string msg) {
    assert(identity.size() != 0);
    s_sendmore(replyer, identity);
    identity = "";
    s_send(replyer, msg);
  }

  string recv() {
    assert(identity.size() == 0);
    identity = s_recv(replyer);
    string empty = s_recv(replyer);
    assert(empty.size() == 0);
    return s_recv(replyer);;
  }

  void send_reply(CS_Reply &reply) {
    stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << reply;
    assert(identity.size() != 0);
    s_sendmore(replyer, identity);
    identity = "";
    s_send(reply, ss.str());
  }

  CS_Request recv_request() {
    string result = s_recv(replyer);
    sstringstream s;
    s << result;
    boost::archive::binary_iarchive ia(s);
    CS_Request request;
    ia >> request;
    return request;
  }
};