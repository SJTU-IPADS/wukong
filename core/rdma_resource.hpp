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

#include "tcp_adaptor.hpp"

#pragma GCC diagnostic warning "-fpermissive"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>
#include <endian.h>
#include <byteswap.h>
#include <getopt.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <vector>
#include <pthread.h>

#include "timer.hpp"

#include "rdmaio.h"
using namespace rdmaio;

#ifdef HAS_RDMA

struct config_t {
    const char *dev_name;         /* IB device name */
    char *server_name;            /* server host name */
    u_int32_t tcp_port;           /* server TCP port */
    int ib_port;                  /* local IB port to work with */
    int gid_idx;                  /* gid index to use */
};

/* structure to exchange data which is needed to connect the QPs */
struct cm_con_data_t {
    uint64_t addr;                /* Buffer address */
    uint32_t rkey;                /* Remote key */
    uint32_t qp_num;              /* QP number */
    uint16_t lid;                 /* LID of the IB port */
    uint8_t gid[16];              /* gid */
} __attribute__ ((packed));

/* structure of system resources */
struct dev_resource {
    struct ibv_device_attr device_attr; /* Device attributes */
    struct ibv_port_attr port_attr;     /* IB port attributes */
    struct ibv_context *ib_ctx;         /* device handle */

    struct ibv_pd *pd;                  /* PD handle */
    struct ibv_mr *mr;                  /* MR handle for buf */
    char *buf;                          /* memory buffer pointer, used for RDMA and send*/
};

struct QP {
    struct cm_con_data_t remote_props;  /* values to connect to remote side */
    struct ibv_pd *pd;            /* PD handle */
    struct ibv_cq *cq;            /* CQ handle */
    struct ibv_qp *qp;            /* QP handle */
    struct ibv_mr *mr;            /* MR handle for buf */

    struct dev_resource *dev;
};

struct normal_op_req {
    ibv_wr_opcode opcode;
    char *local_buf;
    int size; // default set to sizeof(uint64_t)
    int remote_offset;

    //for atomicity operations
    uint64_t compare_and_add;
    uint64_t swap;

    //for internal usage!!
    struct ibv_send_wr sr;
    struct ibv_sge sge;
};

struct config_t rdma_config = {
    NULL,                         /* dev_name */
    NULL,                         /* server_name */
    19875,                        /* tcp_port */
    1,                            /* ib_port */
    -1                            /* gid_idx */
};

#if __BYTE_ORDER == __LITTLE_ENDIAN

static inline uint64_t htonll(uint64_t x) { return bswap_64(x); }
static inline uint64_t ntohll(uint64_t x) { return bswap_64(x); }

#elif __BYTE_ORDER == __BIG_ENDIAN

static inline uint64_t htonll(uint64_t x) { return x; }
static inline uint64_t ntohll(uint64_t x) { return x; }

#else

#error __BYTE_ORDER is neither __LITTLE_ENDIAN nor __BIG_ENDIAN

#endif


static void dev_resources_init(struct dev_resource *res) {
    memset(res, 0, sizeof * res);
}


static void QP_init(struct QP *res) {
    memset(res, 0, sizeof * res);
}


static int modify_qp_to_rtr(struct ibv_qp *qp, uint32_t remote_qpn, uint16_t dlid, uint8_t * dgid) {
    struct ibv_qp_attr attr;
    int flags;
    int rc;

    memset(&attr, 0, sizeof (attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_256;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 16;
    attr.min_rnr_timer = 0x12;
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = dlid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = rdma_config.ib_port;

    if (rdma_config.gid_idx >= 0) {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.port_num = 1;
        memcpy (&attr.ah_attr.grh.dgid, dgid, 16);
        attr.ah_attr.grh.flow_label = 0;
        attr.ah_attr.grh.hop_limit = 1;
        attr.ah_attr.grh.sgid_index = rdma_config.gid_idx;
        attr.ah_attr.grh.traffic_class = 0;
    }

    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    rc = ibv_modify_qp(qp, &attr, flags);
    if (rc)
        fprintf(stderr, "failed to modify QP state to RTR\n");

    return rc;
}

static int modify_qp_to_rts(struct ibv_qp *qp) {
    struct ibv_qp_attr attr;
    int flags;
    int rc;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 0x12;
    attr.retry_cnt = 6;
    attr.rnr_retry = 0;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 16;
    attr.max_dest_rd_atomic = 16;

    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    rc = ibv_modify_qp (qp, &attr, flags);
    if (rc)
        fprintf (stderr, "failed to modify QP state to RTS\n");

    return rc;
}

int post_send(struct QP *res, ibv_wr_opcode opcode, char *local_buf, size_t size, size_t remote_offset, Qp*qp) {
    // struct ibv_send_wr sr;
    // struct ibv_sge sge;
    // struct ibv_send_wr *bad_wr = NULL;
    // int rc;
    // /* prepare the scatter/gather entry */
    // sge.addr = (uintptr_t) local_buf;
    // sge.length = size;
    // // sge.lkey = res->mr->lkey;
    // sge.lkey = qp->dev_->conn_buf_mr->lkey;
    // /* prepare the send work request */
    // sr.next = NULL;
    // sr.wr_id = 0;
    // sr.sg_list = &sge;
    // sr.num_sge = 1;
    // sr.opcode = opcode;
    // sr.send_flags = IBV_SEND_SIGNALED ;

    // // sr.wr.rdma.remote_addr = res->remote_props.addr + remote_offset;
    // // sr.wr.rdma.rkey = res->remote_props.rkey;
    // sr.wr.rdma.remote_addr = qp->remote_attr_.buf + remote_offset;
    // sr.wr.rdma.rkey = qp->remote_attr_.rkey;

    // /* there is a Receive Request in the responder side,
    // so we won't get any into RNR flow */
    // // rc = ibv_post_send(res->qp, &sr, &bad_wr);

    // rc = ibv_post_send(qp->qp, &sr, &bad_wr);
    // if (rc)
    //     fprintf (stderr, "failed to post SR\n");
    struct ibv_send_wr sr;
    struct ibv_sge sge;
    struct ibv_send_wr *bad_wr = NULL;
    int rc;
        assert(qp->qp->qp_type == IBV_QPT_RC);
        sr.opcode = opcode;
        sr.num_sge = 1;
        sr.next = NULL;
        sr.sg_list = &sge;

        sr.send_flags = IBV_SEND_SIGNALED;

        sge.addr = (uint64_t)local_buf;
        sge.length = size;
        sge.lkey = qp->dev_->conn_buf_mr->lkey;

        sr.wr.rdma.remote_addr =
            qp->remote_attr_.buf + remote_offset;
        sr.wr.rdma.rkey = qp->remote_attr_.rkey;

        rc = ibv_post_send(qp->qp, &sr, &bad_wr);
    if (rc)
        fprintf (stderr, "failed to post SR\n");
    return rc;
}

static int poll_completion(struct QP *res) {
    struct ibv_wc wc;
    unsigned long start_time_msec;
    unsigned long cur_time_msec;
    struct timeval cur_time;
    int poll_result;
    int rc = 0;

    /* poll the completion for a while before giving up of doing it .. */
    do {
        poll_result = ibv_poll_cq (res->cq, 1, &wc);
    } while ((poll_result == 0));

    if (poll_result < 0) {
        /* poll CQ failed */
        fprintf (stderr, "poll CQ failed\n");
        rc = 1;
    } else if (poll_result == 0) {
        /* the CQ is empty */
        fprintf (stderr, "completion wasn't found in the CQ after timeout\n");
        rc = 1;
    } else {
        /* CQE found */
        //      fprintf (stdout, "completion was found in CQ with status 0x%x\n",
        //               wc.status);
        /* check the completion status (here we don't care about the completion opcode) */
        if (wc.status != IBV_WC_SUCCESS) {
            //fprintf (stderr,
             //        "got bad completion with status: 0x%x, vendor syndrome: 0x%x\n",
             //        wc.status, wc.vendor_err);
		fprintf (stderr,
						 "got bad completion with status: 0x%x, vendor syndrome: 0x%x, with error %s\n",
						 wc.status, wc.vendor_err,ibv_wc_status_str(wc.status));
            rc = 1;
        }
    }

    return rc;
}

class RDMA {
    class RDMA_Device {
        const static uint64_t BATCH_FACTOR = 32;
        const static int SERVICE_PORT_BASE = 19975;

        int num_nodes;
        int num_threads;
        int node_id;
        char *mem;
        uint64_t mem_sz;

        vector<string> ipset;

        struct dev_resource *dev0; //for remote usage
        struct dev_resource *dev1; //for local usage

        struct QP **res;
        int rdmaOp(int dst_tid, int dst_nid, char *buf, uint64_t size,
                   uint64_t off, ibv_wr_opcode op) {
            if(off > mem_sz)printf("off : %ld, mem_sz: %ld\n", off,mem_sz);
            assert(off < mem_sz);
            assert(dst_nid < num_nodes);
            assert(dst_tid < num_threads);

            Qp* qp = ctrl->get_rc_qp(dst_tid, dst_nid);

            // struct QP* qpres = res[dst_tid] + dst_nid;
            // assert(qpres->mr->lkey == qp->dev_->conn_buf_mr->lkey);
            // assert(qpres->remote_props.addr == qp->remote_attr_.buf);
            // assert(qpres->remote_props.rkey == qp->remote_attr_.rkey); 
            // assert(qpres->qp == qp->qp); 

    //  struct ibv_send_wr sr;
    // struct ibv_sge sge;
    // struct ibv_send_wr *bad_wr = NULL;
    // int rc;
    //     assert(qp->qp->qp_type == IBV_QPT_RC);
    //     sr.opcode = op;
    //     sr.num_sge = 1;
    //     sr.next = NULL;
    //     sr.sg_list = &sge;

    //     sr.send_flags = IBV_SEND_SIGNALED;

    //     sge.addr = (uint64_t)buf;
    //     sge.length = size;
    //     sge.lkey = qp->dev_->conn_buf_mr->lkey;

    //     sr.wr.rdma.remote_addr =
    //         qp->remote_attr_.buf + off;
    //     sr.wr.rdma.rkey = qp->remote_attr_.rkey;

    //     rc = ibv_post_send(qp->qp, &sr, &bad_wr);
    // if (rc)
    //     fprintf (stderr, "failed to post SR\n");
            // if (post_send(res[dst_tid] + dst_nid, op, buf, size, off, qp)) {
            //     cout << "ERROR: failed to post request!" << endl;
            //     assert(false);
            // }
            qp->rc_post_send(op,buf,size,off,IBV_SEND_SIGNALED);
            // if (poll_completion(res[dst_tid] + dst_nid)) {
            //     cout << "poll completion failed!" << endl;
            //     assert(false);
            // }
            qp->poll_completion();
            return 0;
        }

        void init() {
            assert(num_nodes > 0 && num_threads > 0 && node_id >= 0);
            cout << "init RDMA devs" << endl;

            dev0 = new dev_resource;

            dev_resources_init(dev0);
            dev0->ib_ctx = ctrl->rdma_single_device_->ctx;
            dev0->port_attr = ctrl->rdma_single_device_->port_attrs[1];
            dev0->pd = ctrl->rdma_single_device_->pd;
            dev0->buf = ctrl->rdma_single_device_->conn_buf_mr->addr;
            dev0->mr = ctrl->rdma_single_device_->conn_buf_mr;

            ctrl->start_server();
            for(uint j = 0; j < num_threads; ++j){
                for(uint i = 0;i < num_nodes;++i) {
                    ctrl->create_rc_qp(j,i,0,1);
                }
            }

            //cout << "creating remote QPs" << endl;
            res = new struct QP *[num_threads];
            for (int i = 0; i < num_threads ; i++) {
                res[i] = new struct QP[num_nodes];
                for (int j = 0; j < num_nodes; j++) {
                    QP_init(res[i] + j);
                    struct QP* qp_ptr = res[i] + j;
                    qp_ptr->dev = dev0;
                    qp_ptr->pd = dev0->pd;
                    qp_ptr->mr = dev0->mr;
                    qp_ptr->cq = ctrl->get_rc_qp(i,j)->send_cq;
                    qp_ptr->qp = ctrl->get_rc_qp(i,j)->qp;
                }
            }
        }

    public:
        RdmaCtrl* ctrl = NULL;
        RDMA_Device(int num_nodes, int num_threads, int node_id,
                    char *mem, uint64_t mem_sz, string ipfn)
            : num_nodes(num_nodes), num_threads(num_threads), node_id(node_id),
              mem(mem), mem_sz(mem_sz) {

            // record IPs of ndoes
            ifstream ipfile(ipfn);
            string ip;
            while (ipfile >> ip)
                ipset.push_back(ip);

    // //node_id, ipset, port, thread_id-no use, enable single memory region 
            ctrl = new RdmaCtrl(node_id, ipset, 19344,0, true);
            ctrl->open_device();
            ctrl->set_connect_mr(mem, mem_sz);
            ctrl->register_connect_mr();//single
            ctrl->start_server();
            for(uint j = 0; j < num_threads; ++j){
                for(uint i = 0;i < num_nodes;++i) {
                    ctrl->create_rc_qp(j,i,0,1);
                }
            }
    // RdmaCtrl* ctrl = rdma.dev->ctrl;
    
            // init();//DZY modify
            // connect();
        }
        void connect() {
            // rolling start from next node, i.e., (node_id + j) % num_nodes
            for (int j = 1; j < num_nodes; j++) {
                int id = (node_id + j) % num_nodes;

                // request QP info from all threads of other nodes and build an one-to-one connect
                for (int tid = 0; tid < num_threads; tid++) {

                    struct cm_con_data_t remote_con_data;
                    // memcpy(&remote_con_data, (char *)reply.data(), sizeof(remote_con_data));
                    Qp* qp = ctrl->get_rc_qp(tid, id);
                    remote_con_data.addr = qp->remote_attr_.buf;
                    remote_con_data.rkey = qp->remote_attr_.rkey;
                    remote_con_data.qp_num = qp->remote_attr_.qpn;
                    remote_con_data.lid = qp->remote_attr_.lid;
                    (res[tid] + id)->remote_props = remote_con_data;
                    // a one-to-one mapping between the same thread on each node
                    // if (connect_qp(res[tid] + id, remote_con_data) ) {
                    //     fprintf (stderr, "failed to connect QPs\n");
                    //     assert(false);
                    //     exit(-1);
                    // }
                }
            }

            cout << "RDMA connect QP done." << endl;
        }

        string ip_of(int sid) { return ipset[sid]; }

        // 0 on success, -1 otherwise
        int RdmaRead(int dst_tid, int dst_nid, char *local, uint64_t size, uint64_t off) {
            Qp* qp = ctrl->get_rc_qp(dst_tid,dst_nid);
            qp->rc_post_send(IBV_WR_RDMA_READ,local,size,off,IBV_SEND_SIGNALED);
            qp->poll_completion();
            return 0;
            return rdmaOp(dst_tid, dst_nid, local, size, off, IBV_WR_RDMA_READ);
        }

        int RdmaWrite(int dst_tid, int dst_nid, char *local, uint64_t size, uint64_t off) {
            Qp* qp = ctrl->get_rc_qp(dst_tid,dst_nid);
            qp->rc_post_send(IBV_WR_RDMA_WRITE,local,size,off,IBV_SEND_SIGNALED);
            qp->poll_completion();
            return 0;
        }

    }; // end of class RdmaResource

public:
    RDMA_Device *dev = NULL;

    RDMA() { }

    ~RDMA() { if (dev != NULL) delete dev; }

    void init_dev(int num_nodes, int num_threads, int node_id,
                  char *mem, uint64_t mem_sz, string ipfn) {
        dev = new RDMA_Device(num_nodes, num_threads, node_id, mem, mem_sz, ipfn);
    }

    inline static bool has_rdma() { return true; }

    static RDMA &get_rdma() {
        static RDMA rdma;
        return rdma;
    }
}; // end of clase RDMA

void RDMA_init(int num_nodes, int num_threads, int node_id, char *mem, uint64_t mem_sz, string ipfn) {
    uint64_t t = timer::get_usec();

    RDMA &rdma = RDMA::get_rdma();

    // init RDMA device
    rdma.init_dev(num_nodes, num_threads, node_id, mem, mem_sz, ipfn);

    t = timer::get_usec() - t;
    cout << "INFO: initializing RMDA done (" << t / 1000  << " ms)" << endl;

    
}

#else

class RDMA {
    class RDMA_Device {
    public:
        RDMA_Device(int num_nodes, int num_threads, int node_id,
                    string fname, char *mem, uint64_t mem_sz) {
            cout << "This system is compiled without RDMA support." << endl;
            assert(false);
        }

        void servicing() {
            cout << "This system is compiled without RDMA support." << endl;
            assert(false);
        }

        void connect() {
            cout << "This system is compiled without RDMA support." << endl;
            assert(false);
        }

        string ip_of(int sid) {
            cout << "This system is compiled without RDMA support." << endl;
            assert(false);
            return string();
        }

        int RdmaRead(int dst_tid, int dst_nid, char *local,
                     uint64_t size, uint64_t remote_offset) {
            cout << "This system is compiled without RDMA support." << endl;
            assert(false);
            return 0;
        }

        int RdmaWrite(int dst_tid, int dst_nid, char *local,
                      uint64_t size, uint64_t remote_offset) {
            cout << "This system is compiled without RDMA support." << endl;
            assert(false);
            return 0;
        }

        int RdmaCmpSwap(int dst_tid, int dst_nid, char *local,
                        uint64_t compare, uint64_t swap,
                        uint64_t size, uint64_t off) {
            cout << "This system is compiled without RDMA support." << endl;
            assert(false);
            return 0;
        }
    }; // end of class RdmaResource

public:
    RDMA_Device *dev = NULL;

    RDMA() {
        std::cout << "This system is compiled without RDMA support."
                  << std::endl;
    }

    ~RDMA() { }

    void init_dev(int num_nodes, int num_threads, int node_id,
                  char *mem, uint64_t mem_sz, string ipfn) {
        std::cout << "This system is compiled without RDMA support."
                  << std::endl;
    }

    inline static bool has_rdma() { return false; }

    static RDMA &get_rdma() {
        static RDMA rdma;
        return rdma;
    }

};

void RDMA_init(int num_nodes, int num_threads, int node_id,
               char *mem, uint64_t mem_sz, string ipfn) {
    std::cout << "This system is compiled without RDMA support."
              << std::endl;
}

#endif
