#ifndef RDMA_TCP_MSG_
#define RDMA_TCP_MSG_

// implement the msg using TCP

#include <string>
#include <vector>
#include <unordered_map>

#include <zmq.hpp>    // a wrapper over zeromq
#include "rdma_msg.h" // abstract interfaces
#include "rdmaio.h"   // for some constants

using namespace std;

namespace rdmaio {

  namespace tcpmsg {

    class TCPMessage : public RDMA_msg {

    public:
      TCPMessage(const vector<string> &network,int nid,int num_threads,int port,msg_func_t callback);
      ~TCPMessage();

      void thread_local_init(int tid);

      Qp::IOStatus send_to(int node_id,char *msg,int len);
      Qp::IOStatus send_to(int node_id,int tid,char *msg,int len);
      Qp::IOStatus broadcast_to(int *node_ids, int num_of_node, char *msg,int len);
      void poll_comps();

      bool  try_recv_from(int from_mac,char *buffer);
      char *try_recv_from(int from_mac); // return: NULL no msg found, otherwise a po

      void check() {}
      int  get_num_nodes() { return network_.size();}
      int  get_thread_id() { assert(false); return 0; }

    private:
      int port_base_;
      zmq::context_t context_;
      vector<string> network_;
      int num_threads_;
      int node_id_;             // current machine id
      msg_func_t callback_;     // msg callback
    }; // end class
  };
};

#endif
