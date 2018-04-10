// TODO: + namespace, naming

#ifndef RDMA_IO
#define RDMA_IO

#include <unistd.h>
#include <infiniband/verbs.h>
#include <zmq.hpp>
#include <vector>
#include <string>
#include <mutex>
#include <map>

#include "rdma_header.h"
#include "simple_map.h"

// #define PER_QP_PD

namespace rdmaio {

  extern int num_rc_qps;
  extern int num_uc_qps;
  extern int num_ud_qps;

  // rdma device info
  struct RdmaDevice {
    int dev_id;
    struct ibv_context *ctx;

    struct ibv_pd *pd;
    struct ibv_mr *conn_buf_mr;
    struct ibv_mr *dgram_buf_mr;

    struct ibv_port_attr *port_attrs;
    //used for ud QPs
    //key: _QP_ENCODE_ID(dlid, dev_id)
    SimpleMap<struct ibv_ah*> ahs;

    RdmaDevice():ahs(NULL){}
  };

  struct RdmaQpAttr {
      uint64_t checksum;
      uintptr_t buf;
      uint32_t buf_size;
      uint32_t rkey;
      uint16_t lid;
      uint64_t qpn;
      RdmaQpAttr() { }
  } __attribute__ ((aligned (CACHE_LINE_SZ)));

  struct RdmaReq {
    enum ibv_wr_opcode opcode;
    int length;
    int flags;
    int rid;
    uint64_t buf;
    union {
      struct {
        uint64_t remote_offset;
      } rdma;
      struct {
        int nid;
        int remote_qid;
      } ud;
      struct{
        uint64_t remote_offset;
        uint64_t compare_add;
        uint64_t swap;
      } atomic;
    }wr;
  };

  struct RdmaRecvHelper{
    int recv_head = 0, recv_step = 0, idle_recv_num = 0;
    int max_idle_recv_num = 1, max_recv_num;
    struct ibv_recv_wr rr[UD_MAX_RECV_SIZE];
    struct ibv_sge sge[UD_MAX_RECV_SIZE];
    struct ibv_wc wc[UD_MAX_RECV_SIZE];
  };

  extern __thread RdmaDevice **rdma_devices_;
  // A wrapper over the ibv_qp, which makes it easy to do rdma read,write and check completions
  class Qp {

  public:
    enum IOStatus {
      IO_SUCC = 0,
      IO_TIMEOUT,
      IO_ERR,
      IO_NULL // used to indicate a null IO req
    };

#ifdef PER_QP_PD
    struct ibv_pd *pd;
    struct ibv_mr *mr;
#endif

    // members
    struct ibv_qp *qp;
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    RdmaDevice *dev_; // device which it belongs to
    int port_id_;    //  port id of the qp

    int tid = 0;
    int nid = 0;
    int idx_ = 0;
    int port_idx;
    int pendings = 0;

    int current_idx; // pending req idx

    struct ibv_send_wr sr[MAX_DOORBELL_SIZE], *bad_sr;
    struct ibv_sge sge[MAX_DOORBELL_SIZE];

    bool inited_ = false;
    RdmaQpAttr remote_attr_;

    // XD: do we need to record the QP states? e.g, whether is RC,UC,UD
    // DZY : no, ibv_qp has specific state!
    Qp();
    ~Qp() { //TODO!!
    }

    // initilization method
    void init_rc(RdmaDevice *rdma_device, int port_id);
    void init_uc(RdmaDevice *rdma_device, int port_id);
    void init_ud(RdmaDevice *rdma_device, int port_id);

    //return true if the connection is succesfull
    bool connect_rc();
    bool connect_uc();
    // return true if the connection is succesfull,
    // unlike rc, a ud qp can be used to connnect many destinations, so this method can be called many times
    bool get_ud_connect_info(int remote_id,int idx);
    bool get_ud_connect_info_specific(int remote_id,int thread_id,int idx);

    // change rc,uc QP's states to ready
    void change_qp_states(RdmaQpAttr *remote_qp_attr, int dev_port_id);

    // post and poll wrapper
    IOStatus rc_post_send(ibv_wr_opcode op,char *local_buf,int len,uint64_t off,int flags,int wr_id = 0);
    IOStatus rc_post_doorbell(RdmaReq *reqs, int batch_size);
    IOStatus rc_post_compare_and_swap(char *local_buf,uint64_t off,
                    uint64_t compare_value, uint64_t swap_value, int flags,int wr_id = 0);
    IOStatus rc_post_fetch_and_add(char *local_buf,uint64_t off,
                    uint64_t add_value, int flags,int wr_id= 0);

    IOStatus rc_post_pending(ibv_wr_opcode op,char *local_buf,int len,uint64_t off,int flags,int wr_id = 0);
    bool     rc_flush_pending();

    IOStatus uc_post_send(ibv_wr_opcode op,char *local_buf,int len,uint64_t off,int flags);
    IOStatus uc_post_doorbell(RdmaReq *reqs, int batch_size);


    IOStatus poll_completion(uint64_t *rid = NULL);
    IOStatus poll_completions(int cq_num, uint64_t *rid = NULL);
    int      try_poll(); // return: -1 on NULL, otherwise req wr_id

    inline bool first_send(){
      return pendings == 0;
    }

    inline bool need_poll(){
        // whether the post operation need poll completions
        if(pendings >= POLL_THRSHOLD) {
            pendings += 1;
            return true;
        } else {
            pendings += 1;
            return false;
        }
    }

    inline bool force_poll() { pendings = POLL_THRSHOLD; }

  public:
    // ud routing info
    //struct ibv_ah *ahs_[16]; //FIXME!, currently we only have 16 servers ..
    //RdmaQpAttr     ud_attrs_[16];
    std::map<uint64_t,struct ibv_ah *> ahs_;
    std::map<uint64_t,RdmaQpAttr>      ud_attrs_;
  };


  // A simple rdma connection manager
  class RdmaCtrl {
  public:
    RdmaCtrl(int node_id, const std::vector<std::string> network,
             int tcp_base_port, bool enable_single_thread_mr = false);
    ~RdmaCtrl();

    // XD: why volatile?
    void set_dgram_mr(volatile void *dgram_buf, int dgram_buf_size);
    void set_connect_mr(volatile void *conn_buf, uint64_t conn_buf_size);//huge page?

    // query methods
    void query_devinfo();
    int get_active_dev(int port_index);
    int get_active_port(int port_index);

    // simple wrapper over ibv_query_device
    int query_specific_dev(int dev_id,struct ibv_device_attr *device_attr) {
        auto dev = rdma_devices_[dev_id]; // FIXME: no checks here
        return ibv_query_device(dev->ctx,device_attr);
    }

    //-----------------------------------------------
    // thread local methods, which means the behavior will change depends on the execution threads
    // thread specific initilization
    void thread_local_init();

    // open devices for process
    void open_device(int dev_id = 0);

    // register memory buffer to a device, shall be called after the set_connect_mr and open_device
    void register_connect_mr(int dev_id = 0);
    void register_dgram_mr(int dev_id = 0);

    //-----------------------------------------------

    // background threads to handle QP exchange information
    static void* recv_thread(void *arg);
    // start the background listening thread
    void start_server();

    //-----------------------------------------------

    // qp creation
    // creates a connected QP, this method will block if necessary.
    // Input:  remote server id defined in the network, local thread id, the port which QP is created on.
    // Output: a connected ready to use QP.
    // DZY:assume that one thread only has one corresponding QP
    Qp  *create_rc_qp(int tid, int remote_id,int dev_id,int port_idx, int idx = 0);
    Qp  *create_uc_qp(int tid, int remote_id,int dev_id,int port_idx, int idx = 0);

    //  unlike rc qp, a thread may create multiple ud qp, so an idx will identify which ud qp to use
    //Qp  *create_ud_qp(int tid, int remote_id,int dev_id,int port_idx,int idx);
    Qp  *create_ud_qp(int tid,int dev_id,int port_idx,int idx);

    void link_connect_qps(int tid, int dev_id, int port_idx,int idx, ibv_qp_type qp_type);

    //rdma device query
    inline RdmaDevice* get_rdma_device(int dev_id = 0){
      return enable_single_thread_mr_ ? rdma_single_device_ : rdma_devices_[dev_id];
    }


    // qp query
    inline Qp *get_rc_qp(int tid,int remote_id, int idx = 0) {
        mtx_->lock();
        uint64_t qid = (uint64_t)(_QP_ENCODE_ID(remote_id,RC_ID_BASE + tid * num_rc_qps_+ idx));
        // fprintf(stdout,"find qp %d %d %d, qid %lu\n",tid,remote_id,idx,qid);
        assert(qps_.find(qid) != qps_.end());
        if(qps_.find(qid) == qps_.end()) { mtx_->unlock(); return NULL;}
        Qp *res = qps_[qid];
        mtx_->unlock();
        return res;
    }

    inline Qp *get_ud_qp(int tid,int idx = 0) {
        mtx_->lock();
        uint64_t qid = (uint64_t)(_QP_ENCODE_ID(tid + UD_ID_BASE,  UD_ID_BASE + idx));
        assert(qps_.find(qid) != qps_.end());
        Qp *res = qps_[qid];
        mtx_->unlock();
        return res;
    }

    inline Qp *get_uc_qp(int tid,int remote_id, int idx = 0) {
        mtx_->lock();
        uint64_t qid = (uint64_t)(_QP_ENCODE_ID(remote_id,UC_ID_BASE + tid * num_uc_qps_+ idx));
        // fprintf(stdout,"find qp %d %d %d, qid %lu\n",tid,remote_id,idx,qid);
        assert(qps_.find(qid) != qps_.end());
        Qp *res = qps_[qid];
        mtx_->unlock();
        return res;
    }

    inline Qp *get_local_ud_qp(int tid) {
      return qps_[_QP_ENCODE_ID(node_id_,tid + UD_ID_BASE)];
    }
    //-----------------------------------------------

    // minor functions
    // number of nodes in the cluster
    inline int get_num_nodes() { return network_.size(); }
    inline int get_nodeid() { return node_id_; };

    //-----------------------------------------------

    static ibv_ah* create_ah(int dlid,int port_index, RdmaDevice* rdma_device);
    void init_conn_recv_qp(int qid);
    void init_dgram_recv_qp(int qid);

    RdmaQpAttr get_local_qp_attr(int qid);
    RdmaQpAttr get_remote_qp_attr(int nid, uint64_t qid);

    int post_ud(int qid, RdmaReq* req);
    int post_ud_doorbell(int qid, int batch_size, RdmaReq* reqs);

    int post_conn_recvs(int qid, int recv_num);
    int post_ud_recv(struct ibv_qp *qp, void *buf_addr, int len, int lkey);
    int post_ud_recvs(int qid, int recv_num);

    int poll_recv_cq(int qid);
    int poll_recv_cq(Qp* qp);
    int poll_cqs(int qid, int cq_num);
    int poll_conn_recv_cqs(int qid);
    int poll_dgram_recv_cqs(int qid);


  private:
    // global mtx to protect qp_vector
    std::mutex *mtx_;
    // global mtx to protect ud_attr (routing info)
    std::mutex *ud_mtx_;

    // current node id
    int node_id_;


    // TCP listening port
    int tcp_base_port_;

    const bool enable_single_thread_mr_;
  public:

    // global network topology
    const std::vector<std::string> network_;

    // which device and port to use
    int dev_id_;

    RdmaDevice *rdma_single_device_;
    int num_devices_, num_ports_;
    struct ibv_device **dev_list_;
    int* active_ports_;

    std::map<uint64_t,Qp*> qps_;
    //SimpleMap<Qp *>qps_;
    int num_rc_qps_;
    int num_uc_qps_;
    int num_ud_qps_;

    volatile uint8_t *conn_buf_;
    uint64_t conn_buf_size_;
    volatile uint8_t *dgram_buf_;
    uint64_t dgram_buf_size_;

    SimpleMap<RdmaQpAttr*> remote_ud_qp_attrs_;
    SimpleMap<RdmaRecvHelper*> recv_helpers_;
  };

}

#endif
