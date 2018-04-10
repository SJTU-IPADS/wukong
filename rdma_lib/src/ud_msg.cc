#include "ud_msg.h"
#include "ralloc.h"
#include "utils/utils.h"

#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>

#define STATICS 1

namespace rdmaio {

  namespace udmsg {

    __thread std::vector<Qp *> *send_qps;

    bool bootstrap_ud_qps(RdmaCtrl *cm,int tid,int total,int dev_id,int port_idx,int send_qp_num) {

      assert(send_qp_num + 1 <= cm->num_ud_qps_); // +1 is the qp used to recv
      assert(total >= 1);

      for(uint j = 0;j < send_qps->size();++j) {

        Qp *send_qp = (*send_qps)[j];
        while(1) {
          int connected = 0;
          for(uint i = 0;i < cm->get_num_nodes();++i) {
            if(total == 1) {
              if(send_qp->get_ud_connect_info_specific(i,tid,RECV_QP_IDX))
                connected += 1;
              else {
                usleep(200000);
              }
            } else {

              for(uint k = 0;k < total;++k) {
                if(send_qp->get_ud_connect_info_specific(i,k,RECV_QP_IDX)) {
                  connected += 1;
                }
                else
                  usleep(200000);
              }
            } // end multi connection case
          }
          if(connected == cm->get_num_nodes() * total)
            goto UD_QP_CREATE_END;
        }
      UD_QP_CREATE_END:
        j += 0; // dummy place holder
      }

      return true; //FIXME!! no error detection
    }

    UDMsg::UDMsg(RdmaCtrl *cm,int thread_id,int total_threads,
                 int max_recv_num,msg_func_t fun,
                 int dev_id,int port_idx,int send_qp_num)
      : cm_(cm),
        recv_qp_(cm->create_ud_qp(thread_id,dev_id,port_idx,RECV_QP_IDX)),
        thread_id_(thread_id),
        num_nodes_(cm->get_num_nodes()),
        my_node_id_(cm->get_nodeid()),
        max_recv_num_(max_recv_num),
        callback_(fun),
        sender_batch_size_(0),
        send_qp_idx_(0),
        total_send_qps_(send_qp_num),
        // statics init
        total_costs_(0),pre_total_costs_(0),
        counts_(0),pre_counts_(0)
    {
      assert(recv_qp_ != NULL);
      send_qps = new std::vector<Qp *>();

      for(uint i = 0;i < send_qp_num;++i) {
        send_qps->push_back(cm->create_ud_qp(thread_id,dev_id,port_idx,SEND_QP_IDX + i));
      }

      assert(max_recv_num_ <= MAX_RECV_SIZE);
      init();
      bootstrap_ud_qps(cm,thread_id_,total_threads,dev_id,port_idx,send_qp_num); // make connections
    }

    void UDMsg::init() {

      RThreadLocalInit();
      idle_recv_num_ = 0;
      // calculate the recv_buf_size
      recv_buf_size_ = 0;
      while(recv_buf_size_ < MAX_PACKET_SIZE + GRH_SIZE){
        recv_buf_size_ += MIN_STEP_SIZE;
      }

      // init recv relate data structures
      for(int i = 0; i < max_recv_num_; i++) {
        sge_[i].length = recv_buf_size_;
        sge_[i].lkey   = recv_qp_->dev_->conn_buf_mr->lkey;
        sge_[i].addr   = (uintptr_t)(Rmalloc(recv_buf_size_));

        assert(sge_[i].addr != 0);

        rr_[i].wr_id   = sge_[i].addr;
        rr_[i].sg_list = &sge_[i];
        rr_[i].num_sge = 1;

        rr_[i].next    = (i < max_recv_num_ - 1) ?
          &rr_[i + 1] : &rr_[0];
      }

      // init sender relate data structures
      for(uint64_t i = 0;i < MAX_DOORBELL_SIZE;i++) {
        //sr_[i].opcode = IBV_WR_SEND;
        sr_[i].opcode = IBV_WR_SEND_WITH_IMM;
        sr_[i].num_sge = 1;
        sr_[i].imm_data = my_node_id_;
        sr_[i].next = &sr_[i+1];

        ssge_[i].lkey = (*send_qps)[send_qp_idx_]->dev_->conn_buf_mr->lkey;
      }

      recv_head_ = 0;

      // post these recvs
      post_recvs(max_recv_num_);
      recv_qp_->inited_ = true;
    }

    inline void UDMsg::post_recvs(int recv_num) {

      int tail   = recv_head_ + recv_num - 1;
      if(tail >= max_recv_num_) {
        tail -= max_recv_num_;
      }
      ibv_recv_wr  *head_rr = rr_ + recv_head_;
      ibv_recv_wr  *tail_rr = rr_ + tail;

      ibv_recv_wr  *temp = tail_rr->next;
      tail_rr->next = NULL;

      int rc = ibv_post_recv(recv_qp_->qp,head_rr,&bad_rr_);
      CE_1(rc, "[UDMSG] qp: Failed to post_recvs, %s\n", strerror(errno));

      recv_head_ = tail;
      tail_rr->next = temp; // restore the recv chain
      recv_head_ = (recv_head_ + 1) % max_recv_num_;

    }

    Qp::IOStatus UDMsg::send_to(int nid,char *msg,int len) {
      return send_to(nid,thread_id_,msg,len);
    }

    Qp::IOStatus UDMsg::send_to(int node_id,int tid,char *msg,int len) {
      //Qp::IOStatus UDMsg::send_to(int node_id,char *msg,int len) {

      Qp *send_qp = (*send_qps)[send_qp_idx_];
      int ret = (int) Qp::IO_SUCC;
      auto key = _QP_ENCODE_ID(node_id,tid);
      //assert(send_qp->ahs_.find(key) != send_qp->ahs_.end());

      sr_[0].wr.ud.ah = send_qp->ahs_[key];
      sr_[0].wr.ud.remote_qpn  = send_qp->ud_attrs_[key].qpn;
      sr_[0].wr.ud.remote_qkey = DEFAULT_QKEY;
      //
      sr_[0].next = NULL;

      sr_[0].sg_list = &ssge_[0];

      sr_[0].send_flags = ((send_qp->first_send()) ? IBV_SEND_SIGNALED : 0)
        | ((len < 64) ? IBV_SEND_INLINE : 0);

      ssge_[0].addr = (uint64_t)msg;
      ssge_[0].length = len;
      sr_[0].imm_data = _QP_ENCODE_ID(my_node_id_,tid);

      if(send_qp->need_poll())
        ret |= send_qp->poll_completion();

      ret |= ibv_post_send(send_qp->qp, &sr_[0], &bad_sr_);
      assert(ret == 0);
      //reset next ptr
      sr_[0].next = &sr_[1];
      send_qp_idx_ = (send_qp_idx_ + 1) % total_send_qps_;

      return (Qp::IOStatus)ret;
    }

    Qp::IOStatus UDMsg::broadcast_to(int *node_ids, int num_of_node, char *msg,int len) {
#if 0
      Qp *send_qp = (*send_qps)[send_qp_idx_];
      int ret = (int) Qp::IO_SUCC;

      assert(num_of_node < MAX_DOORBELL_SIZE);

      for(uint i = 0;i < num_of_node;++i) {

        sr_[i].wr.ud.ah = send_qp->ahs_[node_ids[i]];
        sr_[i].wr.ud.remote_qpn  = send_qp->ud_attrs_[node_ids[i]].qpn;
        sr_[i].wr.ud.remote_qkey = DEFAULT_QKEY;

        sr_[i].sg_list = &ssge_[i];

        sr_[i].send_flags = ((send_qp->first_send()) ? IBV_SEND_SIGNALED : 0)
            | ((len < 64) ? IBV_SEND_INLINE : 0);

        ssge_[i].addr = (uint64_t)msg;
        ssge_[i].length = len;

        if(send_qp->need_poll())
          ret |= send_qp->poll_completion();
      }

      sr_[num_of_node - 1].next = NULL;
      ret |= ibv_post_send(send_qp->qp, &sr_[0], &bad_sr_);
      sr_[num_of_node - 1].next = &sr_[num_of_node];

      send_qp_idx_ = (send_qp_idx_ + 1) % total_send_qps_;
#endif
      prepare_pending();
      for(uint i = 0;i < num_of_node;++i) {
        post_pending(node_ids[i],msg,len);
      }
      flush_pending();
      return Qp::IO_SUCC;
    }

    Qp::IOStatus UDMsg::prepare_pending() {
      send_qp_ = (*send_qps)[send_qp_idx_];
      assert(current_idx_ == 0);
    }

    Qp::IOStatus UDMsg::post_pending(int node_id,int tid,char *msg,int len) {
      return post_pending_helper(_QP_ENCODE_ID(node_id,tid),msg,len);
    }

    Qp::IOStatus UDMsg::post_pending(int node_id,char *msg,int len) {
      return post_pending_helper(_QP_ENCODE_ID(node_id,thread_id_),msg,len);
    }

    Qp::IOStatus UDMsg::post_pending_helper(int key,char *msg,int len) {

      auto ret = Qp::IO_SUCC;
      auto i = current_idx_++;

      sr_[i].wr.ud.ah = send_qp_->ahs_[key];
      sr_[i].wr.ud.remote_qpn  = send_qp_->ud_attrs_[key].qpn;
      sr_[i].wr.ud.remote_qkey = DEFAULT_QKEY;
      sr_[i].sg_list = &ssge_[i];
      sr_[i].imm_data = _QP_ENCODE_ID(my_node_id_,thread_id_);

      sr_[i].send_flags = ((send_qp_->first_send()) ? IBV_SEND_SIGNALED : 0)
        | ((len < 64) ? IBV_SEND_INLINE : 0);

      if(send_qp_->need_poll())
        send_qp_->poll_completion();

      ssge_[i].addr = (uintptr_t)msg;
      ssge_[i].length = len;

      if(current_idx_ >= UD_MAX_DOORBELL_SIZE)
        flush_pending();

      return ret;
    }

    Qp::IOStatus UDMsg::flush_pending() {
      if(current_idx_ > 0) {
        sr_[current_idx_ - 1].next = NULL;
        auto ret = ibv_post_send(send_qp_->qp, &sr_[0], &bad_sr_);
        sr_[current_idx_ - 1].next = &sr_[current_idx_];
        current_idx_ = 0;
        return (Qp::IOStatus)ret;
      }
      return Qp::IO_SUCC;
    }

    void UDMsg::force_sync(int *node_id,int num_of_node) {
      return;
    }


    void UDMsg::check() {
      // TODO
    }

    void UDMsg::report() {
      auto counts = counts_ - pre_counts_;
      pre_counts_ = counts_;

      counts = counts == 0?1:counts;

      fprintf(stdout,"RPC poll costs %f\n",(total_costs_ - pre_total_costs_) / (double)counts);
      pre_total_costs_ = total_costs_;
    }

    void UDMsg::poll_comps() {

#if STATICS == 1
      auto start = rdtsc();
#endif
      prepare_pending();
      int poll_result = ibv_poll_cq(recv_qp_->recv_cq, MAX_RECV_SIZE,wc_);

      // prepare for replies
      assert(poll_result >= 0); // FIXME: ignore error
      for(uint i = 0;i < poll_result;++i) {
        // msg_num: poll_result
        if(wc_[i].status != IBV_WC_SUCCESS) assert(false); // FIXME!
        callback_((char *)(wc_[i].wr_id + GRH_SIZE),_QP_DECODE_MAC(wc_[i].imm_data),_QP_DECODE_INDEX(wc_[i].imm_data));
      }
      flush_pending(); // send replies

      idle_recv_num_ += poll_result;
      if(idle_recv_num_ > max_idle_recv_num_) {
        // re-post
        post_recvs(idle_recv_num_);
        idle_recv_num_ = 0;
      }
#if STATICS == 1
      auto total = rdtsc() - start;
      if(poll_result > 0) {
        counts_      += 1;
        total_costs_ += total;
      }
#endif
      // end UD polls comps
    }

    // dummy methods for backward compatibility
    bool  UDMsg::try_recv_from(int from_mac,char *buffer) {
      assert(false); // cannot be implemented in UD msg
      return false;
    }

    void  ack_msg() {

    }
  }; // end namespace ud msg

}; // end namespace rdmaio
