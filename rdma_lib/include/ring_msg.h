#ifndef RDMA_RING_MSG_
#define RDMA_RING_MSG_

#include "rdmaio.h"
#include "rdma_msg.h"

/*
 * Layout of message buffer in RDMA registered region
 * | meta data | ring buffer | overflow padding |
 */


namespace rdmaio {

  namespace ringmsg {

    // constants
    const uint8_t MAX_BROADCAST_SERVERS = 32;
    const uint8_t MSG_META_SZ = sizeof(uint64_t);
    const uint8_t  MSG_MAX_MAC_SUPPORTED = 64;

    const uint32_t MSG_DEFAULT_PADDING = (16 * 1024);
    const uint32_t MSG_DEFAULT_SZ = (4 * 1024 * 1024 - MSG_META_SZ - MSG_DEFAULT_PADDING);

    class RingMessage : public RDMA_msg {

    public:
      /*
       * ringSz:      The buffer for receiving messages.
       * ringPadding: The overflow buffer for one message. msgsize must <= ringPadding
       * basePtr:     The start pointer of the total message buffer used at one server
       */
      RingMessage(uint64_t ring_size,uint64_t ring_padding,int thread_id,RdmaCtrl *cm,char *base_ptr,msg_func_t callback);
      RingMessage(int thread_id,RdmaCtrl *cm,char *base_ptr,msg_func_t callback);

      Qp::IOStatus send_to(int node_id,char *msg,int len);
      Qp::IOStatus send_to(int node,int tid,char *msg,int len);
      Qp::IOStatus broadcast_to(int *node_ids, int num_of_node, char *msg,int len);

      virtual void poll_comps();

      // force a sync among all current in-flight messages, return when all these msgs are ready
      void force_sync(int *node_id,int num_of_node);

      // Return true if one message is received
      bool  try_recv_from(int from_mac,char *buffer);
      char *try_recv_from(int from_mac); // return: NULL no msg found, otherwise a pointer to the msg

      // if we receive one
      void inline __attribute__((always_inline))
        ack_msg(int from_mac) {
        //*((uint64_t *)poll_ptr_) = 0;
        //*((uint64_t *)((char *)poll_ptr_ + sizeof(uint64_t) + msg_size_)) = 0;
        memset((char *)poll_ptr_,0,sizeof(uint64_t) + msg_size_ + sizeof(uint64_t));
        headers_[from_mac] += (msg_size_ + sizeof(uint64_t) + sizeof(uint64_t));
        // printf("next poll_offset: %lu\n", from_mac * (total_buf_size_ + MSG_META_SZ) + headers_[from_mac] % ring_size_);
      }

      int   get_num_nodes() { return num_nodes_; }
      int   get_thread_id() { return thread_id_; }

      virtual void check();

    private:
      std::vector<Qp *> qp_vec_;
      // The ring buffer size
      const uint64_t ring_size_;
      const uint64_t ring_padding_;
      const uint64_t total_buf_size_;

      // The base offset used to send message
      uint64_t base_offset_;

      // Receive side buffering
      uint64_t *poll_ptr_;
      uint64_t  msg_size_;
      //      int from_mac_;

      // num nodes in total
      int num_nodes_;

      msg_func_t callback_;
    public:
      // my node id
      int node_id_;

      /* Local base offset */
      char *base_ptr_;
      RdmaCtrl *cm_;

      /* Local offsets used for polling */
      uint64_t offsets_[MSG_MAX_MAC_SUPPORTED];

      /* Remote offsets used for sending messages */
      uint64_t headers_[MSG_MAX_MAC_SUPPORTED];

      /* The thread id */
      int thread_id_;
    };

  }
};

#endif
