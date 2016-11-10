#pragma once
#include <zmq.hpp>
#include <zhelpers.hpp>
#include <string>

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

  void send(std::string msg) {
    assert(identity.size() != 0);
    s_sendmore(replyer, identity);
    identity = "";
    s_send(replyer, msg);
  }

  std::string recv() {
    assert(identity.size() == 0);
    identity = s_recv(replyer);
    std::string empty = s_recv(replyer);
    assert(empty.size() == 0);
    return s_recv(replyer);;
  }

};