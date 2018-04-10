#ifndef RDMA_MSG_
#define RDMA_MSG_

// an abstraction of RDMA message passing interface

#include "rdmaio.h"

namespace rdmaio {
    typedef std::function<void(char *,int,int)> msg_func_t;
    class RDMA_msg {
    public:
        virtual Qp::IOStatus send_to(int node_id,char *msg,int len) = 0;
        virtual Qp::IOStatus send_to(int node_id,int tid,char *msg,int len) { assert(false);}
        virtual Qp::IOStatus broadcast_to(int *node_ids, int num_of_node, char *msg,int len) = 0;

        // delayed methods
        virtual Qp::IOStatus prepare_pending() { assert(false); }
        virtual Qp::IOStatus post_pending(int node_id,char *msg,int len) { assert(false); }
        virtual Qp::IOStatus post_pending(int node_id,int tid,char *msg,int len) { assert(false); }
        virtual Qp::IOStatus flush_pending() { assert(false); }

        virtual void force_sync(int *node_id,int num_of_node) { assert(false);}
        virtual bool  try_recv_from(int from_mac,char *buffer) { assert(false);}
        virtual char* try_recv_from(int from_mac) {assert(false); }
        virtual void  ack_msg(int) { assert(false);}
        virtual void  poll_comps() { assert(false);}

        virtual int get_num_nodes() = 0;
        virtual int get_thread_id() = 0;

        // print debug msg
        virtual void check() = 0;
        virtual void report() { } // report running statistics
    };

};

#endif
