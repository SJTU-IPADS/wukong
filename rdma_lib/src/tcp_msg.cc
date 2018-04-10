#include "tcp_msg.h"


namespace rdmaio {

  static zmq::context_t context(12); // each RDMA lib instance uses one zmq context

  namespace tcpmsg {
#define CALCULATE_PORT(nid,tid) ((nid) * 200 + (tid))

    __thread unordered_map<int,zmq::socket_t *> *senders_; // sender's sockets
    __thread int thread_id;
    __thread zmq::socket_t *recv_socket;

    TCPMessage::TCPMessage(const vector<string> &network,int nid,int num_threads,int base,msg_func_t callback) :
      node_id_(nid),
      num_threads_(num_threads),
      port_base_(base),
      network_(network.begin(),network.end()),
      callback_(callback)
    {
    }

    TCPMessage::~TCPMessage() {
    }

    void TCPMessage::thread_local_init(int tid) {
#if 1
      recv_socket = new zmq::socket_t(context, ZMQ_PULL);
      assert(recv_socket != NULL);
      char address[32] = "";
      snprintf(address, 32, "tcp://*:%d", port_base_ + tid);
      try {
        recv_socket->bind(address);
      } catch (...) {
        assert(false);
      }

      senders_ = new unordered_map<int,zmq::socket_t *>();
      assert(senders_ != NULL);
      for(uint i = 0;i < network_.size();++i) {
        auto s = new zmq::socket_t(context, ZMQ_PUSH);
        assert(s != NULL);
        char address[32] = "";
        snprintf(address, 32, "tcp://%s:%d", network_[i].c_str(), port_base_ + tid);
        s->connect(address);
        senders_->insert(make_pair(i,s));
      }

      thread_id = tid; // init thread local tid
#endif
    }

    Qp::IOStatus TCPMessage::send_to(int node,int tid,char *msg,int len) {
      return send_to(node,msg,len);
    }

    Qp::IOStatus TCPMessage::send_to(int node_id,char *msg,int len) {

      auto s = (*senders_)[node_id];

      zmq::message_t m(len + sizeof(uint8_t));
      *((uint8_t *)(m.data())) = node_id_;
      memcpy((char *)(m.data()) + sizeof(uint8_t),msg,len);
      try {
        s->send(m);
      } catch (...) {
        assert(false);
      }
      return Qp::IO_SUCC;
    }

    Qp::IOStatus TCPMessage::broadcast_to(int *node_ids, int num_of_node, char *msg,int len) {
#if 0
      zmq::message_t m(len + sizeof(uint8_t));
      memcpy((char *)(m.data()) + sizeof(uint8_t),msg,len);
      *((uint8_t *)(m.data())) = node_id_;
#endif
      for(uint i = 0;i < num_of_node;++i) {
        //auto s = (*senders_)[node_ids[i]];
        //s->send(m);
        send_to(node_ids[i],msg,len);
      }
      return Qp::IO_SUCC;
    }

    bool  TCPMessage::try_recv_from(int from_mac,char *buffer) {
      assert(false); // not supported
      return false;
    }

    char *TCPMessage::try_recv_from(int from_mac) {
      assert(false); // not supported
      return NULL;
    }

    void TCPMessage::poll_comps() {
#if 1
      // actually i donot know how to configure the retry number
      for(uint i = 0; i < 4;++i) {
        zmq::message_t msg;
        if(recv_socket->recv(&msg,ZMQ_NOBLOCK)) {
          int nid = *((uint8_t *)(msg.data()));
          callback_((char *)(msg.data()) + sizeof(uint8_t), nid,thread_id);
        }
      }
#endif // end receive logic
    }

  };
};
