#include "ring_msg.h"

#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>

#define FORCE 0 //Whether to force poll completion
#define FORCE_REPLY 0

namespace rdmaio {

  namespace ringmsg {

    RingMessage::RingMessage(int thread_id,RdmaCtrl *cm,char *base_ptr,msg_func_t callback)
      :RingMessage(MSG_DEFAULT_SZ,MSG_DEFAULT_PADDING,thread_id,cm,base_ptr,callback) {
    }


    RingMessage::RingMessage(uint64_t ring_size,uint64_t ring_padding,
                             int thread_id,RdmaCtrl *cm,char *base_ptr,msg_func_t callback)
      :ring_size_(ring_size),
       ring_padding_(ring_padding),
       thread_id_(thread_id),
       num_nodes_(cm->get_num_nodes()),
       total_buf_size_(ring_size + ring_padding),
       cm_(cm),
       node_id_(cm->get_nodeid()),
       callback_(callback)
    {
#if FORCE == 1
      // warning!
#endif
      //fprintf(stdout,"ring message @ %d init\n",thread_id);
      base_ptr_ = (base_ptr + (total_buf_size_ + MSG_META_SZ) * num_nodes_ * thread_id);

      for(uint i = 0;i < num_nodes_;++i) {
        offsets_[i] = 0;
        headers_[i] = 0;
      }

      /* do we really need to clean the buffer here ? */
      //  memset(base_ptr_,0, (total_buf_size_ + MSG_META_SZ) * num_nodes_ );

      char *start_ptr = (char *)(cm_->conn_buf_);

      /* The baseoffsets must be synchronized at all machine */
      base_offset_ = base_ptr_ - start_ptr;

      // init qp vector
      for(uint i = 0;i < num_nodes_;++i) {
        Qp *qp = cm_->get_rc_qp(thread_id,i,1);
        assert(qp != NULL);
        qp_vec_.push_back(qp);
      }
    }

    Qp::IOStatus RingMessage::send_to(int node,int tid,char *msg,int len) { return send_to(node,msg,len);}

    Qp::IOStatus
    RingMessage::send_to(int node,char *msgp,int len) {

      int ret = (int) Qp::IO_SUCC;

      // calculate offset
      uint64_t offset = base_offset_ + node_id_ * (total_buf_size_ + MSG_META_SZ) +
        (offsets_[node] % ring_size_) + MSG_META_SZ;
      // printf("send_offset: %lu to: %d, r_off:%lu\n", node_id_ * (total_buf_size_ + MSG_META_SZ) +
      //   (offsets_[node] % ring_size_), node, offset);
      offsets_[node] += (len + sizeof(uint64_t) + sizeof(uint64_t));

      assert( (len + sizeof(uint64_t ) + sizeof(uint64_t)) <= ring_padding_);

      // get qp
      Qp *qp = qp_vec_[node];

      // calculate send flag
      int send_flag = (len < 64) ? (IBV_SEND_INLINE) : 0;

#if FORCE_REPLY == 1
      send_flag |= IBV_SEND_SIGNALED;
#else
      if(qp->first_send()) {
        send_flag |= IBV_SEND_SIGNALED;
      }

      if(qp->need_poll())
        ret |= qp->poll_completion();
#endif

      // post the request
      ret |= qp->rc_post_send(IBV_WR_RDMA_WRITE,msgp,len + sizeof(uint64_t) + sizeof(uint64_t),
                              offset,send_flag);
#if FORCE_REPLY == 1
      ret |= qp->poll_completion();
#endif
      assert(ret == Qp::IO_SUCC);
      return (Qp::IOStatus)ret;
    }

    Qp::IOStatus
    RingMessage::broadcast_to(int *nodeids, int num, char *msg,int len) {

      int ret = (int)(Qp::IO_SUCC);

      /* maybe we shall avoid this?*/
      uint64_t remote_offsets[MAX_BROADCAST_SERVERS];
      //fprintf(stdout,"prepare offs @%d\n",thread_id_);
      for(uint i = 0;i < num;++i) {
        // calculate the offset
        //fprintf(stdout,"start send to  @%d\n",thread_id_);
        uint64_t off = base_offset_ +  node_id_ * (total_buf_size_ + MSG_META_SZ) +
          offsets_[nodeids[i]] % ring_size_ + MSG_META_SZ;
        // printf("send_offset_broad : %lu to: %d, r_off:%lu\n", node_id_ * (total_buf_size_ + MSG_META_SZ) +
        //   (offsets_[nodeids[i]] % ring_size_), nodeids[i], off);
        offsets_[nodeids[i]] += (len + sizeof(uint64_t) + sizeof(uint64_t));

        //fprintf(stdout,"start send to %d, off %lu @%d\n",nodeids[i],off,thread_id_);

        remote_offsets[i] = off;
      }


      for(uint i = 0;i < num;++i) {

        Qp *qp = qp_vec_[nodeids[i]];

        int send_flag = 0;
#if FORCE == 1
        send_flag = IBV_SEND_SIGNALED;
#else
        if(qp->first_send()) {
          send_flag = IBV_SEND_SIGNALED;
        }
        if(qp->need_poll()) {
          Qp::IOStatus s = qp->poll_completion();
          ret |= (int)s;
        }
#endif

        ret |= qp->rc_post_send(IBV_WR_RDMA_WRITE,msg,len + sizeof(uint64_t) + sizeof(uint64_t),
                                remote_offsets[i],send_flag);
#if FORCE == 1
        Qp::IOStatus s = qp->poll_completion();
#endif
        ///ret |= (int)s;
        assert(ret == Qp::IO_SUCC);
      }
      return (Qp::IOStatus)ret;
    }

    void RingMessage::poll_comps() {

      uint64_t polled = 0;

      int total_servers = get_num_nodes();

      for(uint i = 0;i < total_servers;++i) {

        char *msg;
      RETRY:
        if ( (msg = try_recv_from(i)) != NULL) {
          callback_(msg,i,thread_id_);
          ack_msg(i);
          goto RETRY;
        }
        // failed, go to next
      RECV_END:
        ; /*pass */
      }
    }

    void RingMessage::force_sync(int *node_ids, int num_of_node) {
      for(uint i = 0;i < num_of_node;++i) {
        qp_vec_[node_ids[i]]->force_poll();
      }
    }

    bool
    RingMessage::try_recv_from(int from_mac, char *buffer) {

      uint64_t poll_offset = from_mac * (total_buf_size_ + MSG_META_SZ) + headers_[from_mac] % ring_size_;
      poll_ptr_ = (uint64_t *)(base_ptr_ + poll_offset + MSG_META_SZ);

      if(*poll_ptr_ != 0) {
        msg_size_ = *poll_ptr_;
        //assert(msg_size_ < ring_padding_);
        if(msg_size_ >= ring_padding_) {
          fprintf(stdout,"recv msg size %lu @%d, header %lu, total_buf_size %lu, from %d\n",msg_size_,thread_id_,
                  headers_[from_mac], total_buf_size_,from_mac);
          this->check();
          assert(false);
        }
        uint64_t *end_ptr = (uint64_t *)((char *)poll_ptr_ + msg_size_ + sizeof(uint64_t));

        if(*end_ptr != msg_size_){
          return false;
        }

        memcpy(buffer,(char *)poll_ptr_ + sizeof(uint64_t),msg_size_);
        //    memset((char *)poll_ptr,0,msg_size + sizeof(uint64_t) + sizeof(uint64_t));
        //    headers_[from_mac] += (msg_size + sizeof(uint64_t) + sizeof(uint64_t));
        //    assert((poll_offset + MSG_META_SZ ) % 8 == 0);
        //    if((poll_offset + MSG_META_SZ) % 8 != 0) {
        //      fprintf(stdout,"poll_offset %lu, msg size %lu\n",poll_offset,msg_size_);
        //      assert(false);
        //    };
        memset((char *)poll_ptr_,0,msg_size_ + sizeof(uint64_t) + sizeof(uint64_t));
        headers_[from_mac] += (msg_size_ + sizeof(uint64_t) + sizeof(uint64_t));
        return true;
      } else
        return false;
    }


    char *
    RingMessage::try_recv_from(int from_mac) {

      // static int counts[2] = {0,0};
      // counts[from_mac]++;
      uint64_t poll_offset = from_mac * (total_buf_size_ + MSG_META_SZ) + headers_[from_mac] % ring_size_;
      poll_ptr_ = (uint64_t *)(base_ptr_ + poll_offset + MSG_META_SZ);
      uint64_t *end_ptr = NULL;

      if(*poll_ptr_ != 0) {
        msg_size_ = *poll_ptr_;
        //assert(msg_size_ < ring_padding_);
        if(msg_size_ >= ring_padding_) {
          fprintf(stdout,"recv msg size %lu @%d, header %lu, total_buf_size %lu, from %d\n",msg_size_,thread_id_,
                  headers_[from_mac], total_buf_size_,from_mac);
          this->check();
          assert(false);
        }
        end_ptr = (uint64_t *)((char *)poll_ptr_ + msg_size_ + sizeof(uint64_t));

        if(*end_ptr == msg_size_){
          // counts[from_mac] = 0;
          return (char *)poll_ptr_ + sizeof(uint64_t);
        }
      }

      // if(counts[from_mac] >= 500000){
      //   printf("----no message from %u, the header is %lu, poll_ptr_ is %lu, off:%lu\n",
      //     from_mac,from_mac * (total_buf_size_ + MSG_META_SZ) + headers_[from_mac] % ring_size_, *poll_ptr_, poll_offset+ MSG_META_SZ+ base_offset_);
      //   if(end_ptr != NULL)
      //     printf("------, end_ptr is %lu\n",  *end_ptr);
      //   counts[from_mac] = 0;
      // }
      return NULL;
    }


    void RingMessage::check() {

    }
    // end namespace msg
  }
};
