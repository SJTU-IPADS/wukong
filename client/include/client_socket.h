#pragma once

#include <zhelpers.hpp>
#include <zmq.hpp>
#include <sstream>
#include "cs_basic_type.h"

class Client_Socket {
public:
  zmq::context_t context;
  zmq::socket_t requester;

  Client_Socket(std::string _broker_name, int _broker_port): context(1),
    requester(zmq::socket_t(context, ZMQ_REQ)) {
    s_set_id(requester);

    char address[30] = "";
    sprintf(address, "tcp://%s:%d", _broker_name.c_str(), _broker_port);
    //fprintf(stdout,"tcp binding address %s\n",address);
    requester.connect(address);
  }

  inline void send(string msg) {
    s_send(requester, msg);
  }

  inline string recv() {
    return s_recv(requester);
  }

  void send_request(CS_Request &request) {
    std::stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << request;
    s_send(requester, ss.str());
  }

  CS_Reply recv_reply() {
    string result = s_recv(requester);
    std::stringstream s;
    s << result;
    boost::archive::binary_iarchive ia(s);
    CS_Reply reply;
    ia >> reply;
    return reply;
  }
};