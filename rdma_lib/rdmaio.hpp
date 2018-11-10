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
 *      http://ipads.se.sjtu.edu.cn/projects/wukong
 *
 */

#pragma once

#ifndef RDMA_IO
#define RDMA_IO

#include <unistd.h>
#include <infiniband/verbs.h>
#include <vector>
#include <string>
#include <mutex>
#include <map>
#include <sys/time.h>  // gettimeofday
#include <malloc.h>
#include <arpa/inet.h> //used for checksum

#include "utils.hpp"
#include "pre_connector.hpp"
#include "rdma_header.hpp"
#include "simple_map.hpp"

#ifdef USE_GPU
// utils
#include "gpu.hpp"
#endif

// #define PER_QP_PD

#define MAX_POLL_CQ_TIMEOUT 4000 // Time-out of the poll complection

static volatile bool running;

// helper functions to change the state of qps  ///////////////////////////////
static void rc_ready2init(ibv_qp * qp, int port_id);
static void rc_init2rtr(ibv_qp * qp, int port_id, int qpn, int dlid);
static void rc_rtr2rts(ibv_qp * qp);

static void uc_ready2init(ibv_qp * qp, int port_id);
static void uc_init2rtr(ibv_qp * qp, int port_id, int qpn, int dlid);
static void uc_rtr2rts(ibv_qp * qp);

static void ud_ready2init(ibv_qp * qp, int port_id);
static void ud_init2rtr(ibv_qp * qp);
static void ud_rtr2rts(ibv_qp * qp);


namespace rdmaio {

// extern int num_rc_qps;
// extern int num_uc_qps;
// extern int num_ud_qps;

// rdma device info
struct RdmaDevice {
    int dev_id;
    struct ibv_context *ctx;

    struct ibv_pd *pd;
    struct ibv_mr *conn_buf_mr;
    struct ibv_mr *dgram_buf_mr;
#ifdef USE_GPU
    struct ibv_mr *conn_buf_mr_gpu;
#endif

    struct ibv_port_attr *port_attrs;
    //used for ud QPs
    //key: _QP_ENCODE_ID(dlid, dev_id)
    SimpleMap<struct ibv_ah*> ahs;

    RdmaDevice(): ctx(NULL), pd(NULL), conn_buf_mr(NULL),
        dgram_buf_mr(NULL), port_attrs(NULL), ahs(NULL) { }
};

struct RdmaQpAttr {
    uint64_t checksum;
    uintptr_t buf;
    uint32_t buf_size;
    uint32_t rkey;

#ifdef USE_GPU
    // buffer on GPU
    uintptr_t gpu_buf;
    uint32_t gpu_buf_size;
    uint32_t gpu_rkey;
#endif

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
        struct {
            uint64_t remote_offset;
            uint64_t compare_add;
            uint64_t swap;
        } atomic;
    } wr;
};

struct RdmaRecvHelper {
    int recv_head = 0, recv_step = 0, idle_recv_num = 0;
    int max_idle_recv_num = 1, max_recv_num;
    struct ibv_recv_wr rr[UD_MAX_RECV_SIZE];
    struct ibv_sge sge[UD_MAX_RECV_SIZE];
    struct ibv_wc wc[UD_MAX_RECV_SIZE];
};

// extern __thread RdmaDevice **rdma_devices_;
int tcp_base_port;
int num_rc_qps;
int num_uc_qps;
int num_ud_qps;
int node_id;

std::vector<std::string> network;

// per-thread allocator
__thread RdmaDevice **rdma_devices_;


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
    Qp() {
        // zeroing ud connection parameter
        for (uint i = 0; i < 16; ++i)
            ahs_[i] = NULL;
    }

    ~Qp() { //TODO!!
    }

    // initilization method
    void init_rc(RdmaDevice *rdma_device, int port_id) {

        assert(rdma_device != NULL && rdma_device->ctx != NULL);
        dev_ = rdma_device;
        port_id_ = port_id;

#ifdef PER_QP_PD
        pd = ibv_alloc_pd(rdma_device->ctx);
        mr = ibv_reg_mr(pd, (char *)rdma_device->conn_buf_mr->addr, rdma_device->conn_buf_mr->length,
                        DEFAULT_PROTECTION_FLAG);
#endif

        recv_cq = send_cq = ibv_create_cq(rdma_device->ctx, RC_MAX_RECV_SIZE, NULL, NULL, 0);
        if (send_cq == NULL) {
            fprintf(stderr, "[librdma] qp: Failed to create cq, %s\n", strerror(errno));
        }
        assert(send_cq != NULL);

        struct ibv_qp_init_attr qp_init_attr;
        memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
        qp_init_attr.send_cq = send_cq;
        qp_init_attr.recv_cq = recv_cq;
        qp_init_attr.qp_type = IBV_QPT_RC;

        qp_init_attr.cap.max_send_wr = RC_MAX_SEND_SIZE;
        qp_init_attr.cap.max_recv_wr = 1;   /* Can be set to 1, if RC Two-sided is not required */
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        qp_init_attr.cap.max_inline_data = MAX_INLINE_SIZE;
        assert(rdma_device->pd != NULL);
#ifdef PER_QP_PD
        qp = ibv_create_qp(pd, &qp_init_attr);
#else
        qp = ibv_create_qp(rdma_device->pd, &qp_init_attr);
#endif
        CE(!qp, "qp failure!!!");

        rc_ready2init(qp, port_id);
    }

    void init_uc(RdmaDevice *rdma_device, int port_id) {

        dev_ = rdma_device;
        port_id_ = port_id;

        recv_cq = send_cq = ibv_create_cq(rdma_device->ctx, UC_MAX_SEND_SIZE, NULL, NULL, 0);
        assert(send_cq != NULL);

        struct ibv_qp_init_attr qp_init_attr;
        memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
        qp_init_attr.send_cq = send_cq;
        qp_init_attr.recv_cq = recv_cq;
        qp_init_attr.qp_type = IBV_QPT_UC;

        qp_init_attr.cap.max_send_wr = UC_MAX_SEND_SIZE;
        qp_init_attr.cap.max_recv_wr = UC_MAX_RECV_SIZE;    /* We don't do RECVs on conn QPs */
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        qp_init_attr.cap.max_inline_data = MAX_INLINE_SIZE;

        qp = ibv_create_qp(rdma_device->pd, &qp_init_attr);
        assert(qp != NULL);

        uc_ready2init(qp, port_id);
    }

    void init_ud(RdmaDevice *rdma_device, int port_id) {

        dev_ = rdma_device;
        port_id_ = port_id;

        send_cq = ibv_create_cq(rdma_device->ctx, UD_MAX_SEND_SIZE, NULL, NULL, 0);
        assert(send_cq != NULL);

        recv_cq = ibv_create_cq(rdma_device->ctx, UD_MAX_RECV_SIZE, NULL, NULL, 0);
        assert(recv_cq != NULL);

        /* Initialize creation attributes */
        struct ibv_qp_init_attr qp_init_attr;
        memset((void *) &qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
        qp_init_attr.send_cq = send_cq;
        qp_init_attr.recv_cq = recv_cq;
        qp_init_attr.qp_type = IBV_QPT_UD;

        qp_init_attr.cap.max_send_wr = UD_MAX_SEND_SIZE;
        qp_init_attr.cap.max_recv_wr = UD_MAX_RECV_SIZE;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        qp_init_attr.cap.max_inline_data = MAX_INLINE_SIZE;

        qp = ibv_create_qp(rdma_device->pd, &qp_init_attr);
        assert(qp != NULL);

        ud_ready2init(qp, port_id);
        ud_init2rtr(qp);
        ud_rtr2rts(qp);

        ahs_.clear();
        ud_attrs_.clear();
    }

    //return true if the connection is succesfull
    bool connect_rc() {
        if (inited_) {
            return true;
        } else {
            //          fprintf(stdout,"qp %d %d not connected\n",tid,nid);
        }

        int remote_qid = _QP_ENCODE_ID(node_id, RC_ID_BASE + tid * num_rc_qps + idx_);

        char address[30];

        QPConnArg arg; memset((char *)(&arg), 0, sizeof(QPConnArg));
        arg.qid = remote_qid;
        arg.sign = MAGIC_NUM;
        arg.tid  = tid;
        arg.nid  = nid;
        arg.calculate_checksum();

        // prepare socket to remote
        auto socket = PreConnector::get_send_socket(network[nid], tcp_base_port);
        if (socket < 0) {
            // cannot establish the connection, shall retry
            return false;
        }
        auto n = send(socket, (char *)(&arg), sizeof(QPConnArg), 0);
        if (n != sizeof(QPConnArg)) {
            close(socket);
            return false;
        }

        // receive reply
        if (!PreConnector::wait_recv(socket)) {
            close(socket);
            return false;
        }

        int buf_size = sizeof(QPReplyHeader) + sizeof(RdmaQpAttr); // format: header | QP attr
        char *reply_buf = new char[buf_size];

        n = recv(socket, reply_buf, buf_size, MSG_WAITALL);

        if (n != sizeof(RdmaQpAttr) + sizeof(QPReplyHeader)) {
            close(socket);
            delete reply_buf;
            usleep(1000);
            return false;
        }

        // close connection
        close(socket);

        QPReplyHeader *hdr = (QPReplyHeader *)(reply_buf);

        if (hdr->status == TCPSUCC) {

        } else if (hdr->status == TCPFAIL) {
            delete reply_buf;
            return false;
        } else {
            fprintf(stdout, "QP connect fail!, val %d\n", ((char *)reply_buf)[0]);
            assert(false);
        }

        RdmaQpAttr qp_attr;
        memcpy(&qp_attr, (char *)reply_buf + sizeof(QPReplyHeader), sizeof(RdmaQpAttr));

        // verify the checksum
        uint64_t checksum = ip_checksum((void *)(&(qp_attr.buf)), sizeof(RdmaQpAttr) - sizeof(uint64_t));
        assert(checksum == qp_attr.checksum);

        change_qp_states(&qp_attr, port_idx);
        inited_ = true;

        delete reply_buf;

        return true;
    }

    bool connect_uc() {
        if (inited_) {
            return true;
        } else {
            //          fprintf(stdout,"qp %d %d not connected\n",tid,nid);
        }

        int remote_qid = _QP_ENCODE_ID(node_id, UC_ID_BASE + tid * num_uc_qps + idx_);

        char address[30];
        int address_len = snprintf(address, 30, "tcp://%s:%d", network[nid].c_str(), tcp_base_port);
        assert(address_len < 30);

        QPConnArg arg;
        arg.qid = remote_qid;
        arg.sign = MAGIC_NUM;
        arg.calculate_checksum();

        auto socket = PreConnector::get_send_socket(network[nid], tcp_base_port);
        if (socket < 0) {
            // cannot establish the connection, shall retry
            return false;
        }

        auto n = PreConnector::send_to(socket, (char *)(&arg), sizeof(QPConnArg));
        if (n != sizeof(QPConnArg)) {
            close(socket);
            return false;
        }

        if (!PreConnector::wait_recv(socket)) {
            close(socket);
            return false;
        }

        int buf_size = sizeof(QPReplyHeader) + sizeof(RdmaQpAttr); // format: header | QP attr
        char *reply_buf = new char[buf_size];

        n = recv(socket, reply_buf, buf_size, MSG_WAITALL);
        if (n != sizeof(RdmaQpAttr) + sizeof(QPReplyHeader)) {
            close(socket);
            delete reply_buf;
            usleep(1000);
            return false;
        }

        // close connection
        close(socket);

        QPReplyHeader *hdr = (QPReplyHeader *)(reply_buf);

        if (hdr->status == TCPSUCC) {

        } else if (hdr->status == TCPFAIL) {
            delete reply_buf;
            return false;
        } else {
            assert(false);
        }

        RdmaQpAttr qp_attr;
        memcpy(&qp_attr, (char *)reply_buf + sizeof(QPReplyHeader), sizeof(RdmaQpAttr));

        // verify thise checksum
        uint64_t checksum = ip_checksum((void *)(&(qp_attr.buf)), sizeof(RdmaQpAttr) - sizeof(uint64_t));
        assert(checksum == qp_attr.checksum);

        change_qp_states(&qp_attr, port_idx);

        inited_ = true;

        delete reply_buf;
        return true;
    }

    // return true if the connection is succesfull,
    // unlike rc, a ud qp can be used to connnect many destinations, so this method can be called many times,
    // for a specific QP
    bool get_ud_connect_info_specific(int remote_id, int thread_id, int idx);

    // change rc,uc QP's states to ready
    void change_qp_states(RdmaQpAttr *remote_qp_attr, int dev_port_id) {

        assert(remote_qp_attr != NULL);
        assert(dev_port_id >= 1);

        if (qp->qp_type == IBV_QPT_RC) {
            rc_init2rtr(qp, dev_port_id, remote_qp_attr->qpn, remote_qp_attr->lid);
            rc_rtr2rts(qp);
        } else if (qp->qp_type == IBV_QPT_UC) {
            uc_init2rtr(qp, dev_port_id, remote_qp_attr->qpn, remote_qp_attr->lid);
            uc_rtr2rts(qp);
        } else {
            assert(false);
        }
        remote_attr_ = *remote_qp_attr;

    }

    // post and poll wrapper
    IOStatus rc_post_send(ibv_wr_opcode op, char *local_buf, int len, uint64_t off, int flags, int wr_id = 0) {

        IOStatus rc = IO_SUCC;
        struct ibv_send_wr sr, *bad_sr;
        struct ibv_sge sge;

        assert(this->qp->qp_type == IBV_QPT_RC);

        sge.addr = (uint64_t)local_buf;
        sge.length = len;

#ifdef PER_QP_PD
        sge.lkey = mr->lkey;
#else
        sge.lkey = dev_->conn_buf_mr->lkey;
#endif
        sr.wr_id = wr_id;

        sr.opcode = op;
        sr.num_sge = 1;
        sr.next = NULL;
        sr.sg_list = &sge;

        sr.send_flags = flags;

        sr.wr.rdma.remote_addr =
            remote_attr_.buf + off;
        sr.wr.rdma.rkey = remote_attr_.rkey;
        // printf("rkey:%lu\n", remote_attr_.rkey);

        rc = (IOStatus)ibv_post_send(qp, &sr, &bad_sr);
        //  CE(rc, "ibv_post_send error\n");
        //this->pendings += 1;
        return rc;
    }

    IOStatus rc_post_doorbell(RdmaReq *reqs, int batch_size) {

        IOStatus rc = IO_SUCC;
        assert(batch_size <= MAX_DOORBELL_SIZE);
        assert(this->qp->qp_type == IBV_QPT_RC);

        bool poll = false;
        for (uint i = 0; i < batch_size; i++) {
            // fill in the requests
            sr[i].opcode = reqs[i].opcode;
            sr[i].num_sge = 1;
            sr[i].next = (i == batch_size - 1) ? NULL : &sr[i + 1];
            sr[i].sg_list = &sge[i];
            sr[i].send_flags = reqs[i].flags;

            if (first_send()) {
                sr[i].send_flags |= IBV_SEND_SIGNALED;
            }
            if (need_poll()) {
                poll_completion();
            }

            sge[i].addr = reqs[i].buf;
            sge[i].length = reqs[i].length;
#ifdef PER_QP_PD
            sge[i].lkey = mr->lkey;
#else
            sge[i].lkey = dev_->conn_buf_mr->lkey;
#endif

            sr[i].wr.rdma.remote_addr =
                remote_attr_.buf + reqs[i].wr.rdma.remote_offset;
            sr[i].wr.rdma.rkey = remote_attr_.rkey;
        }
        rc = (IOStatus)ibv_post_send(qp, &sr[0], &bad_sr);
        CE(rc, "ibv_post_send doorbell error");
        return rc;
    }

    IOStatus rc_post_compare_and_swap(char *local_buf, uint64_t off,
                                      uint64_t compare_value, uint64_t swap_value, int flags, int wr_id = 0) {

        IOStatus rc = IO_SUCC;
        struct ibv_send_wr sr, *bad_sr;
        struct ibv_sge sge;

        assert(this->qp->qp_type == IBV_QPT_RC);
        sr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
        sr.num_sge = 1;
        sr.next = NULL;
        sr.sg_list = &sge;
        sr.send_flags = flags;
        sr.wr_id = wr_id;

        sge.addr = (uint64_t)local_buf;
        sge.length = sizeof(uint64_t);
#ifdef PER_QP_PD
        sge.lkey = mr->lkey;
#else
        sge.lkey = dev_->conn_buf_mr->lkey;
#endif

        sr.wr.atomic.remote_addr = remote_attr_.buf + off;
        sr.wr.atomic.rkey = remote_attr_.rkey;
        sr.wr.atomic.compare_add = compare_value;
        sr.wr.atomic.swap = swap_value;
        rc = (IOStatus)ibv_post_send(this->qp, &sr, &bad_sr);
        CE(rc, "ibv_post_send error");
        return rc;
    }

    IOStatus rc_post_fetch_and_add(char *local_buf, uint64_t off,
                                   uint64_t add_value, int flags, int wr_id = 0) {

        IOStatus rc = IO_SUCC;
        struct ibv_send_wr sr, *bad_sr;
        struct ibv_sge sge;

        assert(this->qp->qp_type == IBV_QPT_RC);
        sr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
        sr.num_sge = 1;
        sr.next = NULL;
        sr.sg_list = &sge;
        sr.send_flags = flags;
        sr.wr_id = wr_id;

        sge.addr = (uint64_t)local_buf;
        sge.length = sizeof(uint64_t);
#ifdef PER_QP_PD
        sge.lkey = mr->lkey;
#else
        sge.lkey = dev_->conn_buf_mr->lkey;
#endif

        sr.wr.atomic.remote_addr = remote_attr_.buf + off;
        sr.wr.atomic.rkey = remote_attr_.rkey;
        sr.wr.atomic.compare_add = add_value;
        rc = (IOStatus)ibv_post_send(this->qp, &sr, &bad_sr);
        CE(rc, "ibv_post_send error");
        return rc;
    }

    IOStatus rc_post_pending(ibv_wr_opcode op, char *local_buf, int len, uint64_t off, int flags, int wr_id = 0) {
        int i = current_idx++;
        sr[i].opcode  = op;
        sr[i].num_sge = 1;
        sr[i].next    = &sr[i + 1];
        sr[i].sg_list = &sge[i];
        sr[i].wr_id   = wr_id;
        sr[i].send_flags = flags;

        sge[i].addr   = (uintptr_t)local_buf;
        sge[i].length = len;
#ifdef PER_QP_PD
        sge[i].lkey = mr->lkey;
#else
        sge[i].lkey = dev_->conn_buf_mr->lkey;
#endif

        //if(need_poll()) poll_completion();

        sr[i].wr.rdma.remote_addr =
            remote_attr_.buf + off;
        sr[i].wr.rdma.rkey = remote_attr_.rkey;
        return IO_SUCC;
    }

    bool rc_flush_pending() {

        if (current_idx > 0) {
            sr[current_idx - 1].next    = NULL;
            sr[current_idx - 1].send_flags |= IBV_SEND_SIGNALED;
            ibv_post_send(qp, &sr[0], &bad_sr);
            current_idx = 0;
            return true;
        }
        return false;
    }

    IOStatus uc_post_send(ibv_wr_opcode op, char *local_buf, int len, uint64_t off, int flags) {

        IOStatus rc = IO_SUCC;
        struct ibv_send_wr sr, *bad_sr;
        struct ibv_sge sge;

        assert(this->qp->qp_type == IBV_QPT_UC);
        sr.opcode = op;
        sr.num_sge = 1;
        sr.next = NULL;
        sr.sg_list = &sge;

        sr.send_flags = flags;

        sge.addr = (uint64_t)local_buf;
        sge.length = len;
        sge.lkey = dev_->conn_buf_mr->lkey;

        sr.wr.rdma.remote_addr = remote_attr_.buf + off;
        sr.wr.rdma.rkey = remote_attr_.rkey;

        rc = (IOStatus)ibv_post_send(qp, &sr, &bad_sr);
        CE(rc, "ibv_post_send error\n");
        return rc;
    }

    IOStatus uc_post_doorbell(RdmaReq *reqs, int batch_size) {

        IOStatus rc = IO_SUCC;
        assert(batch_size <= MAX_DOORBELL_SIZE);

        struct ibv_send_wr sr[MAX_DOORBELL_SIZE], *bad_sr;
        struct ibv_sge sge[MAX_DOORBELL_SIZE];

        assert(this->qp->qp_type == IBV_QPT_UC);
        bool poll = false;
        for (uint i = 0; i < batch_size; i++) {
            // fill in the requests
            sr[i].opcode = reqs[i].opcode;
            sr[i].num_sge = 1;
            sr[i].next = (i == batch_size - 1) ? NULL : &sr[i + 1];
            sr[i].sg_list = &sge[i];
            sr[i].send_flags = reqs[i].flags;

            if (first_send()) {
                sr[i].send_flags |= IBV_SEND_SIGNALED;
            }
            if (need_poll()) {
                poll = true;
            }

            sge[i].addr = reqs[i].buf;
            sge[i].length = reqs[i].length;
            sge[i].lkey = dev_->conn_buf_mr->lkey;

            sr[i].wr.rdma.remote_addr =
                remote_attr_.buf + reqs[i].wr.rdma.remote_offset;
            sr[i].wr.rdma.rkey = remote_attr_.rkey;
        }
        if (poll) rc = poll_completion();
        rc = (IOStatus)ibv_post_send(qp, &sr[0], &bad_sr);
        CE(rc, "ibv_post_send error");
        return rc;
    }

    // poll complection of a cq
    IOStatus poll_completion(uint64_t *rid = NULL) {

        struct ibv_wc wc;
        unsigned long start_time_msec;
        unsigned long cur_time_msec;
        struct timeval cur_time;

        int poll_result;
        /* poll the completion for a while before giving up of doing it .. */
        gettimeofday (&cur_time, NULL);
        start_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);

        this->pendings = 0;

        do {
            poll_result = ibv_poll_cq (this->send_cq, 1, &wc);

            gettimeofday (&cur_time, NULL);
            cur_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
        } while (poll_result == 0);
        // && ((cur_time_msec - start_time_msec) < MAX_POLL_CQ_TIMEOUT));

        if (unlikely(rid != NULL))
            *rid = wc.wr_id;

        // check the result
        if (poll_result < 0) {
            assert(false);
            /* poll CQ failed */
            return IO_ERR;
        } else if (poll_result == 0) {
            /* the CQ is empty */
            fprintf (stderr, "completion wasn't found in the CQ after timeout\n");
            return IO_TIMEOUT;
        } else {
            /* CQE found */
            // fprintf (stdout, "completion was found in CQ with status 0x%x\n",
            //          wc.status);
            /* check the completion status (here we don't care about the completion opcode */
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf (stderr,
                         "got bad completion with status: 0x%x, vendor syndrome: 0x%x, with error %s, qp n:%d t:%d\n",
                         wc.status, wc.vendor_err, ibv_wc_status_str(wc.status), nid, tid);
                assert(false);
                return IO_ERR;
            } else {
                // success, just pass
            }
        }

        return IO_SUCC;
    }

    // poll complections of a cq
    IOStatus poll_completions(int cq_num, uint64_t *rid = NULL) {
        struct ibv_wc wc[RC_MAX_SEND_SIZE];
        unsigned long start_time_msec;
        unsigned long cur_time_msec;
        struct timeval cur_time;

        int poll_result = 0;
        IOStatus rc = IO_SUCC;
        /* poll the completion for a while before giving up of doing it .. */
        gettimeofday (&cur_time, NULL);
        start_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
        this->pendings = 0;

        do {
            int poll_once = ibv_poll_cq(this->send_cq, cq_num - poll_result, &wc[poll_result]);
            if (poll_once < 0) {
                assert(false);
            }
            poll_result += poll_once;

            gettimeofday (&cur_time, NULL);
            cur_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
        }
        while ((poll_result < cq_num));
        // && ((cur_time_msec - start_time_msec) < MAX_POLL_CQ_TIMEOUT));

        if (poll_result != cq_num) {
            return IO_TIMEOUT;
        } else {
            for (int cq_id = 0 ; cq_id < cq_num; cq_id ++) {
                if (wc[cq_id].status != IBV_WC_SUCCESS) {
                    fprintf (stderr,
                             "got bad completion with status: 0x%x, vendor syndrome: 0x%x, with error %s\n",
                             wc[cq_id].status, wc[cq_id].vendor_err, ibv_wc_status_str(wc[poll_result].status));
                    return IO_ERR;
                    // exit(-1);
                }
            }
        }

        return rc;
    }

    int try_poll() {  // return: -1 on NULL, otherwise req wr_id
        struct ibv_wc wc;
        auto poll_result = ibv_poll_cq(this->send_cq, 1, &wc);
        if (poll_result > 0) {
            assert(wc.status == IBV_WC_SUCCESS);
            return wc.wr_id;
        } else if (poll_result < 0) {
            // FIXME: not handled yet
            assert(false);
        }

        return -1;
    }

    inline bool first_send() { return pendings == 0; }

    inline bool need_poll() {
        // whether the post operation need poll completions
        if (pendings >= POLL_THRSHOLD) {
            pendings += 1;
            return true;
        } else {
            pendings += 1;
            return false;
        }
    }

    inline bool force_poll() { pendings = POLL_THRSHOLD; }

#ifdef USE_GPU
    // data is from gpu mem.
    // if to_gpu is true, send to remote gpu mem
    // else send to cpu mem
    IOStatus rc_post_send_gpu(ibv_wr_opcode op, char *local_gpu_buf,
                              int len, uint64_t off, int flags, bool to_gpu, int wr_id = 0) {
        IOStatus rc = IO_SUCC;
        struct ibv_send_wr sr, *bad_sr;
        struct ibv_sge sge;
        assert(this->qp->qp_type == IBV_QPT_RC);

        sge.addr = (uint64_t)local_gpu_buf;
        sge.length = len;

        if (op == IBV_WR_RDMA_WRITE) {
            sge.lkey = dev_->conn_buf_mr_gpu->lkey;
        } else if (op == IBV_WR_RDMA_READ) {
            fprintf(stdout, "rc_post_send_gpu: not support opcode! op: %d\n", op);
            assert(false);
        } else {
            fprintf(stdout, "rc_post_send_gpu: not support opcode! op: %d\n", op);
            assert(false);
        }
        sr.wr_id = wr_id;
        sr.opcode = op;
        sr.num_sge = 1;
        sr.next = NULL;
        sr.sg_list = &sge;
        sr.send_flags = flags;

        if (to_gpu) {
            sr.wr.rdma.remote_addr = remote_attr_.gpu_buf + off;
            sr.wr.rdma.rkey = remote_attr_.gpu_rkey;
        } else {
            sr.wr.rdma.remote_addr = remote_attr_.buf + off;
            sr.wr.rdma.rkey = remote_attr_.rkey;
        }

        rc = (IOStatus)ibv_post_send(qp, &sr, &bad_sr);
        return rc;
    }

#endif

public:
    // ud routing info
    //struct ibv_ah *ahs_[16]; //FIXME!, currently we only have 16 servers ..
    //RdmaQpAttr     ud_attrs_[16];
    std::map<uint64_t, struct ibv_ah *> ahs_;
    std::map<uint64_t, RdmaQpAttr>      ud_attrs_;
};

// A simple rdma connection manager
class RdmaCtrl {
public:
    RdmaCtrl(int id, const std::vector<std::string> net,
             int port, bool enable_single_thread_mr = false):
        node_id_(id), network_(net.begin(), net.end()), tcp_base_port_(port),
        recv_helpers_(NULL), remote_ud_qp_attrs_(NULL), //qps_(NULL),
        rdma_single_device_(NULL),
        num_rc_qps_(100), num_uc_qps_(1), num_ud_qps_(4),
        enable_single_thread_mr_(enable_single_thread_mr) {

        assert(node_id >= 0);

        // init global locks
        mtx_ = new std::mutex();
        ud_mtx_ = new std::mutex();

        qps_.clear();

        // record
        tcp_base_port = tcp_base_port_;
        node_id = node_id_;
        num_rc_qps = num_rc_qps_;
        num_uc_qps = num_uc_qps_;
        num_ud_qps = num_ud_qps_;
        network = std::vector<std::string>(net.begin(), net.end());

        query_devinfo();

        running = true;
    }

    ~RdmaCtrl() {
        // free some resources, may be we does need to do this,
        // since when RDMA device is closed, the app shall close
        delete mtx_; delete ud_mtx_; qps_.clear();

        // TODO!! free RDMA related devices

        running = false; // close listening threads
    }

    // XD: why volatile?
    void set_connect_mr(volatile void *conn_buf, uint64_t conn_buf_size) {
        if (conn_buf == NULL)
            conn_buf = (volatile uint8_t *)memalign(4096, conn_buf_size);

        assert(conn_buf != NULL);
        memset((char *) conn_buf, 0, conn_buf_size);

        conn_buf_ = (volatile uint8_t *)conn_buf;
        conn_buf_size_ = conn_buf_size;
    }

    // register memory buffer to a device, shall be called after the set_connect_mr and open_device
    void register_connect_mr(int dev_id = 0) {
        RdmaDevice *rdma_device = get_rdma_device(dev_id);
        assert(rdma_device->pd != NULL);
        if (enable_single_thread_mr_ && (rdma_device->conn_buf_mr != NULL)) {
            assert(false);
            return;
        }
        rdma_device->conn_buf_mr = ibv_reg_mr(rdma_device->pd, (char *)conn_buf_, conn_buf_size_,
                                              DEFAULT_PROTECTION_FLAG);
        CE_2(!rdma_device->conn_buf_mr,
             "[librdma]: Connect Memory Region failed at dev %d, err %s\n", dev_id, strerror(errno));
    }

#ifdef USE_GPU
    void set_connect_mr_gpu(volatile void *gpu_buf, uint64_t gpu_buf_size) {
        assert(gpu_buf != NULL);
        CUDA_ASSERT(cudaMemset((char *) gpu_buf, 0, gpu_buf_size));
        conn_buf_gpu_ = (volatile uint8_t *)gpu_buf;
        conn_buf_size_gpu_ = gpu_buf_size;
    }

    void register_connect_mr_gpu(int dev_id = 0) {
        RdmaDevice *rdma_device = get_rdma_device(dev_id);
        assert(rdma_device->pd != NULL);

        rdma_device->conn_buf_mr_gpu = ibv_reg_mr(
                                           rdma_device->pd,
                                           (char *)conn_buf_gpu_,
                                           conn_buf_size_gpu_,
                                           DEFAULT_PROTECTION_FLAG);
        CE_2(!rdma_device->conn_buf_mr_gpu,
             "[librdma]: Connect Device Memory Region failed at dev %d, err %s\n", dev_id,
             strerror(errno));
    }
#endif

    void set_dgram_mr(volatile void *dgram_buf, int dgram_buf_size) {
        if (dgram_buf == NULL) {
            dgram_buf = (volatile uint8_t *) memalign(4096, dgram_buf_size);
        }
        assert(dgram_buf != NULL);
        memset((char *) dgram_buf, 0, dgram_buf_size);

        dgram_buf_ = (volatile uint8_t *)dgram_buf;
        dgram_buf_size_ = dgram_buf_size;
    }

    void register_dgram_mr(int dev_id = 0) {
        RdmaDevice *rdma_device = get_rdma_device(dev_id);
        assert(rdma_device->pd != NULL);
        rdma_device->dgram_buf_mr = ibv_reg_mr(rdma_device->pd, (char *)dgram_buf_, dgram_buf_size_,
                                               DEFAULT_PROTECTION_FLAG);
        CE_2(!rdma_device->dgram_buf_mr
             , "[librdma]: Datagram Memory Region failed at dev %d, err %s\n", dev_id, strerror(errno));
    }

    // query methods
    void query_devinfo() {

        int rc;

        dev_list_ = ibv_get_device_list (&num_devices_);
        CE(!num_devices_, "[librdma] : failed to get IB devices list\n");
        // printf("[librdma] : Total %d devices!\n", num_devices_);

        active_ports_ = new int[num_devices_];
        num_ports_ = 0;
        for (int device_id = 0; device_id < num_devices_; device_id++) {

            // printf("[librdma] get device name %s, idx %d\n",dev_list_[device_id]->name,device_id);
            struct ibv_context *ib_ctx = ibv_open_device(dev_list_[device_id]);
            CE_1(!ib_ctx, "[librdma] : Failed to open device %d\n", device_id);

            struct ibv_device_attr device_attr;
            memset(&device_attr, 0, sizeof(device_attr));

            rc = ibv_query_device(ib_ctx, &device_attr);
            CE_1(rc, "[librdma] : Failed to query device %d\n", device_id);

            int port_num = 0, port_count = device_attr.phys_port_cnt;
            for (int port_id = 1; port_id <= port_count; port_id++) {
                struct ibv_port_attr port_attr;
                rc = ibv_query_port(ib_ctx, port_id, &port_attr);
                CE_2(rc, "[librdma] : Failed to query port %d on device %d\n ", port_id, device_id);

                if (port_attr.phys_state != IBV_PORT_ACTIVE &&
                        port_attr.phys_state != IBV_PORT_ACTIVE_DEFER) {
                    // printf("\n[librdma] Ignoring port %d on device %d. State is %s\n",
                    //   port_id, device_id, ibv_port_state_str((ibv_port_state) port_attr.phys_state));
                    continue;
                }
                port_num++;
            }
            // printf("[librdma] : Device %d has %d ports\n", device_id, port_num);
            active_ports_[device_id] = port_num;
            num_ports_ += port_num;

            rc = ibv_close_device(ib_ctx);
            CE_1(rc, "[librdma] : Failed to close device %d", device_id);
        }
        // printf("[librdma] : Total %d Ports!\n", num_ports_);
    }

    int get_active_dev(int port_index) {
        assert(port_index >= 0 && port_index < num_ports_);
        for (int device_id = 0; device_id < num_devices_; device_id++) {
            int port_num = active_ports_[device_id];
            for (int port_id = 1; port_id <= port_num; port_id++) {
                if (port_index == 0)return device_id;
                port_index--;
            }
        }
        return -1;
    }

    int get_active_port(int port_index) {

        assert(port_index >= 0 && port_index < num_ports_);
        for (int device_id = 0; device_id < num_devices_; device_id++) {
            int port_num = active_ports_[device_id];
            for (int port_id = 1; port_id <= port_num; port_id++) {
                if (port_index == 0)return port_id;
                port_index--;
            }
        }
        return -1;
    }

    // simple wrapper over ibv_query_device
    int query_specific_dev(int dev_id, struct ibv_device_attr *device_attr) {
        auto dev = rdma_devices_[dev_id]; // FIXME: no checks here
        return ibv_query_device(dev->ctx, device_attr);
    }

    //-----------------------------------------------
    // thread local methods, which means the behavior will change depends on the execution threads
    // thread specific initilization
    void thread_local_init() {
        //single memory region
        if (enable_single_thread_mr_) return;
        // the device related object shall be created locally
        rdma_devices_ = new RdmaDevice*[num_devices_];
        for (uint i = 0; i < num_devices_; ++i)
            rdma_devices_[i] = NULL;
    }

    // open devices for process
    void open_device(int dev_id = 0) {

        int rc;

        struct ibv_device *device = dev_list_[dev_id];

        RdmaDevice *rdma_device;
        if (enable_single_thread_mr_) {
            if (rdma_single_device_ == NULL) {
                rdma_single_device_ = new RdmaDevice();
                rdma_device = rdma_single_device_;
            } else {
                return;
            }
        } else {
            if (rdma_devices_[dev_id] == NULL) {
                rdma_device = rdma_devices_[dev_id] = new RdmaDevice();
            } else {
                return;
            }
        }

        rdma_device->dev_id = dev_id;
        rdma_device->ctx = ibv_open_device(device);
        assert(rdma_device->ctx);

        struct ibv_device_attr device_attr;
        rc = ibv_query_device(rdma_device->ctx, &device_attr);

        int port_count = device_attr.phys_port_cnt;
        rdma_device->port_attrs = (struct ibv_port_attr*)
                                  malloc(sizeof(struct ibv_port_attr) * (port_count + 1));
        for (int port_id = 1; port_id <= port_count; port_id++) {
            rc = ibv_query_port (rdma_device->ctx, port_id, rdma_device->port_attrs + port_id);
        }

        rdma_device->pd = ibv_alloc_pd(rdma_device->ctx);
        assert(rdma_device->pd != 0);
    }


    //-----------------------------------------------

    // background threads to handle QP exchange information
    static void* recv_thread(void *arg) {

        pthread_detach(pthread_self());
        struct RdmaCtrl *rdma = (struct RdmaCtrl*) arg;

        int port = rdma->tcp_base_port_;

        auto listenfd = PreConnector::get_listen_socket(rdma->network_[rdma->node_id_], port);

        int opt = 1;
        CE(setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(int)) != 0,
           "[RDMA pre connector] set reused socket error!");

        CE(listen(listenfd, rdma->network_.size() * 24) < 0, "[RDMA pre connector] bind TCP port error.");

        int num = 0;

        try {
            while (running) {
                // accept a request
                struct sockaddr_in cli_addr;
                socklen_t clilen;
                auto csfd = accept(listenfd, (struct sockaddr *) &cli_addr, &clilen);
                QPConnArg arg;

                if (!PreConnector::wait_recv(csfd)) { // timeout
                    close(csfd);
                    continue;
                }
                auto n = recv(csfd, (char *)(&arg), sizeof(QPConnArg), MSG_WAITALL);

                if (n != sizeof(QPConnArg)) { // an invalid message
                    close(csfd);
                    continue;
                }

                // check that the arg is correct
                assert(arg.sign = MAGIC_NUM);
                assert(arg.get_checksum() == arg.checksum);

                uint64_t qid = arg.qid;
                uint64_t nid = _QP_DECODE_MAC(qid);
                uint64_t idx = _QP_DECODE_INDEX(qid);

                char *reply_buf = new char[sizeof(QPReplyHeader) + sizeof(RdmaQpAttr)];
                memset(reply_buf, 0, sizeof(QPReplyHeader) + sizeof(RdmaQpAttr));

                rdma->mtx_->lock();
                if (rdma->qps_.find(qid) == rdma->qps_.end()) {
                    (*(QPReplyHeader *)(reply_buf)).status = TCPFAIL;
                } else {
                    if (IS_UD(qid)) {
                        (*(QPReplyHeader *)(reply_buf)).qid = qid;
                        // further check whether QPs are initilized or not
                        Qp *ud_qp = rdma->qps_[qid];
                        if (ud_qp->inited_ == false) {
                            (*(QPReplyHeader *)(reply_buf)).status = TCPFAIL;
                        } else {
                            (*(QPReplyHeader *)(reply_buf)).status = TCPSUCC;
                            num++;
                            RdmaQpAttr qp_attr = rdma->get_local_qp_attr(qid);
                            memcpy((char *)(reply_buf) + sizeof(QPReplyHeader),
                                   (char *)(&qp_attr), sizeof(RdmaQpAttr));
                        }
                    } else {
                        RdmaQpAttr qp_attr = rdma->get_local_qp_attr(qid);
                        (*(QPReplyHeader *)(reply_buf)).status = TCPSUCC;
                        memcpy((char *)(reply_buf) + sizeof(QPReplyHeader), (char *)(&qp_attr), sizeof(RdmaQpAttr));
                    }
                }

                rdma->mtx_->unlock();

                // reply with the QP attribute
                PreConnector::send_to(csfd, reply_buf, sizeof(RdmaQpAttr) + sizeof(QPReplyHeader));
                PreConnector::wait_close(csfd); // wait for the client to close the connection
                delete reply_buf;
            }   // while receiving reqests
            close(listenfd);
        } catch (...) {
            // pass
        }

        printf("[librdma] : recv thread exit!\n");
    }

    // start the background listening thread
    void start_server() {

        pthread_t tid;
        pthread_attr_t attr;

        int rc = pthread_attr_init(&attr);
        assert(rc == 0);
        //rc = pthread_attr_setschedpolicy(&attr,1); // min priority
        //assert(rc == 0);

        pthread_create(&tid, &attr, recv_thread, (void *)this);
    }

    //-----------------------------------------------

    // qp creation
    // creates a connected QP, this method will block if necessary.
    // Input:  remote server id defined in the network, local thread id, the port which QP is created on.
    // Output: a connected ready to use QP.
    // DZY:assume that one thread only has one corresponding QP
    Qp *create_rc_qp(int tid, int remote_id, int dev_id, int port_idx, int idx = 0) {

        // TODO: check device
        // compute local qp id
        assert(num_rc_qps_ != 0);
        assert(idx >= 0 && idx < num_rc_qps_);
        uint64_t qid = _QP_ENCODE_ID(remote_id, RC_ID_BASE + tid * num_rc_qps_ + idx);
        Qp *res = NULL;

        mtx_->lock();
        //fprintf(stdout,"create qp %d %d using dev_id %d\n",tid,remote_id,dev_id);
        if (qps_.find(qid) != qps_.end() && qps_[qid] != nullptr) {
            res = qps_[qid];
            mtx_->unlock();
            return res;
        }
        res = new Qp();
        // set ids
        res->tid  = tid;
        res->idx_ = idx;
        res->nid = remote_id;
        res->port_idx = enable_single_thread_mr_ ? 1 : port_idx;

        res->init_rc(get_rdma_device(dev_id), port_idx);
        qps_.insert(std::make_pair(qid, res));
        //fprintf(stdout,"create qp %d %d done %p\n",tid,remote_id,res);
        mtx_->unlock();

        // done
        return res;
    }

    Qp *create_uc_qp(int tid, int remote_id, int dev_id, int port_idx, int idx = 0) {
        // TODO: check device

        // compute local qp id
        assert(num_uc_qps_ != 0);
        assert(idx >= 0 && idx < num_uc_qps_);
        int32_t qid = _QP_ENCODE_ID(remote_id, UC_ID_BASE + tid * num_uc_qps_ + idx);
        Qp *res = NULL;

        mtx_->lock();
        if (qps_.find(qid) != qps_.end() && qps_[qid] != nullptr) {
            res = qps_[qid];
            mtx_->unlock();
            return res;
        }
        res = new Qp();
        // set ids
        res->tid = tid;
        res->idx_ = idx;
        res->nid = remote_id;
        res->port_idx = port_idx;

        res->init_uc(get_rdma_device(dev_id), port_idx);
        qps_.insert(std::make_pair(qid, res));
        //fprintf(stdout,"create qp %d %d done %p\n",tid,remote_id,res);
        mtx_->unlock();
        // done
        return res;
    }

    //  unlike rc qp, a thread may create multiple ud qp, so an idx will identify which ud qp to use
    //Qp  *create_ud_qp(int tid, int remote_id,int dev_id,int port_idx,int idx);

    Qp *create_ud_qp(int tid, int dev_id, int port_idx, int idx) {

        RdmaDevice *rdma_device = get_rdma_device(dev_id);

        // the unique id which identify this QP
        assert(num_ud_qps_ != 0);
        assert(idx >= 0 && idx < num_ud_qps_);
        uint64_t qid = _QP_ENCODE_ID(UD_ID_BASE + tid , UD_ID_BASE + idx);

        Qp *res = NULL;

        mtx_->lock();
        if (qps_.find(qid) != qps_.end()) {
            res = qps_[qid];
            mtx_->unlock();
            assert(false);
            return res;
        }

        res = new Qp();
        res->init_ud(get_rdma_device(dev_id), port_idx);
        res->tid = tid;
        res->port_idx = port_idx;
        res->dev_ = rdma_device;

        //qps_.insert(qid,res);
        qps_.insert(std::make_pair(qid, res));
        mtx_->unlock();
        return res;
    }

    void link_connect_qps(int tid, int dev_id, int port_idx, int idx, ibv_qp_type qp_type) {

        Qp* (RdmaCtrl::* create_qp_func)(int, int, int, int, int);
        bool (Qp::* connect_qp_func)();
        int num_qps;

        switch (qp_type) {
        case IBV_QPT_RC:
            create_qp_func = &RdmaCtrl::create_rc_qp;
            connect_qp_func = &Qp::connect_rc;
            num_qps = num_rc_qps_;
            break;
        case IBV_QPT_UC:
            create_qp_func = &RdmaCtrl::create_uc_qp;
            connect_qp_func = &Qp::connect_uc;
            num_qps = num_uc_qps_;
            break;
        default:
            CE(true, "link_connect_qp: error qp type");
        }
        for (uint i = 0; i < get_num_nodes(); ++i) {
            Qp *qp = (this->*create_qp_func)(tid, i, dev_id, port_idx, idx);
            assert(qp != NULL);
        }
        // {
        //     Qp *qp = (this->*create_qp_func)(tid,1,dev_id,port_idx,idx);
        //     assert(qp != NULL);
        //     sleep(2);
        // }
        // {
        //     Qp *qp = (this->*create_qp_func)(tid,0,dev_id,port_idx,idx);
        //     assert(qp != NULL);
        // }

        while (1) {
            int connected = 0;
            for (uint i = 0; i < get_num_nodes(); ++i) {
                Qp *qp = (this->*create_qp_func)(tid, i, dev_id, port_idx, idx);
                if (qp->inited_)
                    connected += 1;
                else if ((qp->*connect_qp_func)())
                    connected += 1;
            }

            if (connected == get_num_nodes())
                break;
            else
                usleep(10000);
        }
    }

    //rdma device query
    inline RdmaDevice* get_rdma_device(int dev_id = 0) {
        return enable_single_thread_mr_ ? rdma_single_device_ : rdma_devices_[dev_id];
    }

    // qp query
    inline Qp *get_rc_qp(int tid, int remote_id, int idx = 0) {
        mtx_->lock();
        uint64_t qid = (uint64_t)(_QP_ENCODE_ID(remote_id, RC_ID_BASE + tid * num_rc_qps_ + idx));
        // fprintf(stdout,"find qp %d %d %d, qid %lu\n",tid,remote_id,idx,qid);
        assert(qps_.find(qid) != qps_.end());
        if (qps_.find(qid) == qps_.end()) { mtx_->unlock(); return NULL;}
        Qp *res = qps_[qid];
        mtx_->unlock();
        return res;
    }

    inline Qp *get_ud_qp(int tid, int idx = 0) {
        mtx_->lock();
        uint64_t qid = (uint64_t)(_QP_ENCODE_ID(tid + UD_ID_BASE,  UD_ID_BASE + idx));
        assert(qps_.find(qid) != qps_.end());
        Qp *res = qps_[qid];
        mtx_->unlock();
        return res;
    }

    inline Qp *get_uc_qp(int tid, int remote_id, int idx = 0) {
        mtx_->lock();
        uint64_t qid = (uint64_t)(_QP_ENCODE_ID(remote_id, UC_ID_BASE + tid * num_uc_qps_ + idx));
        // fprintf(stdout,"find qp %d %d %d, qid %lu\n",tid,remote_id,idx,qid);
        assert(qps_.find(qid) != qps_.end());
        Qp *res = qps_[qid];
        mtx_->unlock();
        return res;
    }

    inline Qp *get_local_ud_qp(int tid) {
        return qps_[_QP_ENCODE_ID(node_id_, tid + UD_ID_BASE)];
    }
    //-----------------------------------------------

    // minor functions
    // number of nodes in the cluster
    inline int get_num_nodes() { return network_.size(); }
    inline int get_nodeid() { return node_id_; };

    //-----------------------------------------------

    static ibv_ah* create_ah(int dlid, int port_index, RdmaDevice* rdma_device) {
        struct ibv_ah_attr ah_attr;
        ah_attr.is_global = 0;
        ah_attr.dlid = dlid;
        ah_attr.sl = 0;
        ah_attr.src_path_bits = 0;
        ah_attr.port_num = port_index;

        struct ibv_ah *ah;
        ah = ibv_create_ah(rdma_device->pd, &ah_attr);
        assert(ah != NULL);
        return ah;
    }

    void init_conn_recv_qp(int qid) {
        RdmaRecvHelper *recv_helper = new RdmaRecvHelper;
        RdmaDevice* rdma_device = qps_[qid]->dev_;
        int recv_step = 0;
        int max_recv_num = RC_MAX_RECV_SIZE;
        while (recv_step < MAX_PACKET_SIZE) {
            recv_step += MIN_STEP_SIZE;
        }
        assert(recv_step > 0 && recv_step % MIN_STEP_SIZE == 0);

        printf("recv_step: %d\n", recv_step);
        for (int i = 0; i < max_recv_num; i++) {
            int offset = i * recv_step;

            recv_helper->sge[i].length = recv_step;
            recv_helper->sge[i].lkey = rdma_device->conn_buf_mr->lkey;
            recv_helper->sge[i].addr = (uintptr_t) &conn_buf_[offset];

            recv_helper->rr[i].wr_id = recv_helper->sge[i].addr;/* Debug */
            recv_helper->rr[i].sg_list = &recv_helper->sge[i];
            recv_helper->rr[i].num_sge = 1;

            recv_helper->rr[i].next = (i < max_recv_num - 1) ?
                                      &recv_helper->rr[i + 1] : &recv_helper->rr[0];
        }
        recv_helper->recv_step = recv_step;
        recv_helper->max_recv_num = max_recv_num;
        recv_helpers_.insert(qid, recv_helper);
        post_conn_recvs(qid, max_recv_num);
    }

    void init_dgram_recv_qp(int qid) {

        RdmaRecvHelper *recv_helper = new RdmaRecvHelper;
        RdmaDevice* rdma_device = qps_[qid]->dev_;
        int recv_step = 0;
        int max_recv_num = UD_MAX_RECV_SIZE;
        while (recv_step < MAX_PACKET_SIZE + GRH_SIZE) {
            recv_step += MIN_STEP_SIZE;
        }
        assert(recv_step > 0 && recv_step % MIN_STEP_SIZE == 0);

        printf("recv_step: %d\n", recv_step);
        for (int i = 0; i < max_recv_num; i++) {
            int offset = MIN_STEP_SIZE - GRH_SIZE + i * recv_step;

            recv_helper->sge[i].length = recv_step;
            recv_helper->sge[i].lkey = rdma_device->dgram_buf_mr->lkey;
            recv_helper->sge[i].addr = (uintptr_t) &dgram_buf_[offset];

            recv_helper->rr[i].wr_id = recv_helper->sge[i].addr;/* Debug */
            recv_helper->rr[i].sg_list = &recv_helper->sge[i];
            recv_helper->rr[i].num_sge = 1;

            recv_helper->rr[i].next = (i < max_recv_num - 1) ?
                                      &recv_helper->rr[i + 1] : &recv_helper->rr[0];
        }
        recv_helper->recv_step = recv_step;
        recv_helper->max_recv_num = max_recv_num;
        recv_helpers_.insert(qid, recv_helper);
        post_ud_recvs(qid, max_recv_num);
    }

    RdmaQpAttr get_local_qp_attr(int qid) {

        RdmaQpAttr qp_attr;
        Qp *local_qp = qps_[qid];
        assert(local_qp != NULL);
        //uint64_t begin = rdtsc();
        if (IS_CONN(qid)) {

            qp_attr.buf = (uint64_t) (uintptr_t) conn_buf_;
            qp_attr.buf_size = conn_buf_size_;

#ifdef PER_QP_PD
            qp_attr.rkey = local_qp->mr->rkey;
#else
            assert(local_qp->dev_ != NULL);
            assert(local_qp->dev_->conn_buf_mr != NULL);
            qp_attr.rkey = local_qp->dev_->conn_buf_mr->rkey;
#endif

#ifdef USE_GPU
            assert(local_qp->dev_ != NULL);
            assert(local_qp->dev_->conn_buf_mr_gpu != NULL);
            qp_attr.gpu_buf = (uint64_t) (uintptr_t) conn_buf_gpu_;
            qp_attr.gpu_buf_size = conn_buf_size_gpu_;
            qp_attr.gpu_rkey = local_qp->dev_->conn_buf_mr_gpu->rkey;
#endif
            //qp_attr.rkey = rdma_device_->conn_buf_mr->rkey;
            //qp_attr.rkey = qps_[qid]->reg_mr->rkey;
        }
        //qp_attr.lid = qps_[qid]->dev_->port_attrs[dev_port_id_].lid;
        qp_attr.lid = local_qp->dev_->port_attrs[local_qp->port_id_].lid;
        qp_attr.qpn = local_qp->qp->qp_num;
        //fprintf(stdout,"get local qp costs %lu\n",rdtsc() - begin);

        // calculate the checksum
        uint64_t checksum = ip_checksum((void *)(&(qp_attr.buf)), sizeof(RdmaQpAttr) - sizeof(uint64_t));
        qp_attr.checksum = checksum;
        return qp_attr;
    }


    int post_ud(int qid, RdmaReq* reqs) {
        int rc = 0;
        struct ibv_send_wr sr, *bad_sr;
        struct ibv_sge sge;
        Qp *qp = qps_[qid];
        assert(qp->qp->qp_type == IBV_QPT_UD);
        RdmaQpAttr* qp_attr = remote_ud_qp_attrs_[reqs->wr.ud.remote_qid];
        sr.wr.ud.ah = qp->dev_->ahs[_QP_ENCODE_ID(qp_attr->lid, qp->port_id_)];
        sr.wr.ud.remote_qpn = qp_attr->qpn;
        sr.wr.ud.remote_qkey = DEFAULT_QKEY;

        sr.opcode = IBV_WR_SEND;
        sr.num_sge = 1;
        sr.next = NULL;
        sr.sg_list = &sge;
        sr.send_flags = reqs->flags;
        // sr[i].send_flags |= IBV_SEND_INLINE;

        sge.addr = reqs->buf;
        sge.length = reqs->length;
        sge.lkey = qp->dev_->dgram_buf_mr->lkey;

        rc = ibv_post_send(qp->qp, &sr, &bad_sr);
        CE(rc, "ibv_post_send error");
        return rc;
    }

    int post_ud_doorbell(int qid, int batch_size, RdmaReq* reqs) {

        int rc = 0;
        struct ibv_send_wr sr[MAX_DOORBELL_SIZE], *bad_sr;
        struct ibv_sge sge[MAX_DOORBELL_SIZE];
        Qp *qp = qps_[qid];
        assert(qp->qp->qp_type == IBV_QPT_UD);
        bool needpoll = false;

        for (int i = 0; i < batch_size; i++) {
            RdmaQpAttr* qp_attr = remote_ud_qp_attrs_[reqs[i].wr.ud.remote_qid];
            if (qp_attr == NULL) {
                fprintf(stdout, "qid %u\n", reqs[i].wr.ud.remote_qid);
                assert(false);
            }
            sr[i].wr.ud.ah = qp->dev_->ahs[_QP_ENCODE_ID(qp_attr->lid, qp->port_id_)];
            sr[i].wr.ud.remote_qpn = qp_attr->qpn;
            sr[i].wr.ud.remote_qkey = DEFAULT_QKEY;

            sr[i].opcode = IBV_WR_SEND;
            sr[i].num_sge = 1;
            sr[i].next = (i == batch_size - 1) ? NULL : &sr[i + 1];
            sr[i].sg_list = &sge[i];

            sr[i].send_flags = reqs[i].flags;
            if (qp->first_send()) {
                sr[i].send_flags |= IBV_SEND_SIGNALED;
            }
            if (qp->need_poll()) {
                needpoll = true;
            }
            // sr[i].send_flags |= IBV_SEND_INLINE;

            sge[i].addr = reqs[i].buf;
            sge[i].length = reqs[i].length;
            sge[i].lkey = qp->dev_->dgram_buf_mr->lkey;
        }
        if (needpoll)qp->poll_completion();
        rc = ibv_post_send(qp->qp, &sr[0], &bad_sr);
        CE(rc, "ibv_post_send error");
        return rc;
    }

    int post_conn_recvs(int qid, int recv_num) {
        struct ibv_recv_wr *head_rr, *tail_rr, *temp_rr, *bad_rr = NULL;
        RdmaRecvHelper *recv_helper = recv_helpers_[qid];

        int rc = 0;
        int head = recv_helper->recv_head;
        int tail = head + recv_num - 1;
        if (tail >= recv_helper->max_recv_num) {
            tail -= recv_helper->max_recv_num;
        }

        head_rr = recv_helper->rr + head;//&recvWRs[head];
        tail_rr = recv_helper->rr + tail;//&recvWRs[tail];
        temp_rr = tail_rr->next;
        tail_rr->next = NULL;

        rc = ibv_post_recv(qps_[qid]->qp, head_rr, &bad_rr);
        CE(rc, "ibv_post_recvs error");
        tail_rr->next = temp_rr;

        /* Update recv head: go to the last wr posted and take 1 more step */
        recv_helper->recv_head = tail;
        MOD_ADD(recv_helper->recv_head, recv_helper->max_recv_num); /* 1 step */
        return rc;
    }

    int post_ud_recv(struct ibv_qp *qp, void *buf_addr, int len, int lkey) {
        int rc = 0;
        struct ibv_recv_wr *bad_wr;
        struct ibv_sge sge;
        memset(&sge, 0, sizeof(struct ibv_sge));
        struct ibv_recv_wr rr;
        memset(&rr, 0, sizeof(struct ibv_recv_wr));

        sge.addr = (uintptr_t) buf_addr;
        sge.length = len;
        sge.lkey = lkey;

        rr.wr_id = (uint64_t) buf_addr;
        rr.sg_list = &sge;
        rr.num_sge = 1;

        rc = ibv_post_recv(qp, &rr, &bad_wr);
        CE(rc, "Failed to  posting datagram recv.\n");

        return rc;
    }

    int post_ud_recvs(int qid, int recv_num) {
        struct ibv_recv_wr *head_rr, *tail_rr, *temp_rr, *bad_rr;
        RdmaRecvHelper *recv_helper = recv_helpers_[qid];
        // recv_num > 0 && recv_num <= MAX_RECV_SIZE;
        // fprintf(stdout, "Node %d: Posting %d RECVs \n",nodeId, recv_num);

        int rc = 0;
        int head = recv_helper->recv_head;
        int tail = head + recv_num - 1;
        if (tail >= recv_helper->max_recv_num) {
            tail -= recv_helper->max_recv_num;
        }

        head_rr = recv_helper->rr + head;//&recvWRs[head];
        tail_rr = recv_helper->rr + tail;//&recvWRs[tail];
        temp_rr = tail_rr->next;
        tail_rr->next = NULL;

        rc = ibv_post_recv(qps_[qid]->qp, head_rr, &bad_rr);
        CE(rc, "ibv_post_recvs error");
        tail_rr->next = temp_rr;

        /* Update recv head: go to the last wr posted and take 1 more step */
        recv_helper->recv_head = tail;
        MOD_ADD(recv_helper->recv_head, recv_helper->max_recv_num); /* 1 step */
        return rc;
    }

    int poll_recv_cq(int qid) {
        Qp *qp = qps_[qid];
        struct ibv_wc wc;
        int rc = 0;
        int poll_result;

        do {
            poll_result = ibv_poll_cq (qp->recv_cq, 1, &wc);
        } while (poll_result == 0);
        assert(poll_result == 1);

        if (wc.status != IBV_WC_SUCCESS) {
            fprintf (stderr,
                     "got bad completion with status: 0x%x, vendor syndrome: 0x%x, with error %s\n",
                     wc.status, wc.vendor_err, ibv_wc_status_str(wc.status));
        }
        // fprintf(stdout,"poll Recv imm %d, buffer data: %d\n",wc.imm_data,
        //       (*(uint32_t*)(wc.wr_id+GRH_SIZE)));
        return rc;
    }

    int poll_recv_cq(Qp* qp) {
        struct ibv_wc wc;
        int rc = 0;
        int poll_result;

        do {
            poll_result = ibv_poll_cq (qp->recv_cq, 1, &wc);
        } while (poll_result == 0);
        assert(poll_result == 1);

        if (wc.status != IBV_WC_SUCCESS) {
            fprintf (stderr,
                     "got bad completion with status: 0x%x, vendor syndrome: 0x%x, with error %s\n",
                     wc.status, wc.vendor_err, ibv_wc_status_str(wc.status));
        }
        // fprintf(stdout,"poll Recv imm %d, buffer data: %d\n",wc.imm_data,
        //       (*(uint32_t*)(wc.wr_id+GRH_SIZE)));
        return rc;
    }

    int poll_cqs(int qid, int cq_num) {
        struct ibv_wc wc[RC_MAX_SEND_SIZE];
        int rc = 0;
        int poll_result = 0;
        Qp *qp = qps_[qid];
        while (poll_result < cq_num) {
            int poll_once = ibv_poll_cq(qp->send_cq, cq_num - poll_result, &wc[poll_result]);
            if (poll_once != 0) {
                if (wc[poll_result].status != IBV_WC_SUCCESS) {
                    fprintf (stderr,
                             "got bad completion with status: 0x%x, vendor syndrome: 0x%x, with error %s\n",
                             wc[poll_result].status, wc[poll_result].vendor_err, ibv_wc_status_str(wc[poll_result].status));
                    // exit(-1);
                }
            }
            poll_result += poll_once;
        }
        qp->pendings = 0;
        return rc;
    }

    int poll_conn_recv_cqs(int qid) {
        Qp *qp = qps_[qid];
        RdmaRecvHelper *recv_helper = recv_helpers_[qid];
        int poll_result, rc;
        struct ibv_wc* wc = recv_helper->wc;
        poll_result = ibv_poll_cq (qp->recv_cq, recv_helper->max_recv_num, wc);
        rc = poll_result;
        CE(poll_result < 0, "poll CQ failed\n");
        for (int i = 0; i < poll_result; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                fprintf (stderr,
                         "got bad completion with status: 0x%x, vendor syndrome: 0x%x, with error %s\n",
                         wc[i].status, wc[i].vendor_err, ibv_wc_status_str(wc[i].status));
                rc = -1;
            }
            // fprintf(stdout,"poll Recv imm %d, buffer data: %d\n",ntohl(qp->recvWCs[i].imm_data),
            //   (*(uint32_t*)(qp->recvWCs[i].wr_id)));
        }
        recv_helper->idle_recv_num += poll_result;
        if (recv_helper->idle_recv_num > recv_helper->max_idle_recv_num) {
            post_conn_recvs(qid, recv_helper->idle_recv_num);
            recv_helper->idle_recv_num = 0;
        }
        return rc;
    }

    int poll_dgram_recv_cqs(int qid) {
        Qp *qp = qps_[qid];
        RdmaRecvHelper *recv_helper = recv_helpers_[qid];
        int poll_result, rc;
        struct ibv_wc* wc = recv_helper->wc;
        poll_result = ibv_poll_cq (qp->recv_cq, recv_helper->max_recv_num, wc);
        rc = poll_result;
        CE(poll_result < 0, "poll CQ failed\n");
        for (int i = 0; i < poll_result; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                fprintf (stderr,
                         "got bad completion with status: 0x%x, vendor syndrome: 0x%x, with error %s\n",
                         wc[i].status, wc[i].vendor_err, ibv_wc_status_str(wc[i].status));
                rc = -1;
            }
            // fprintf(stdout,"poll Recv imm %d, buffer data: %d\n",ntohl(qp->recvWCs[i].imm_data),
            //   (*(uint32_t*)(qp->recvWCs[i].wr_id+GRH_SIZE)));
        }
        recv_helper->idle_recv_num += poll_result;
        if (recv_helper->idle_recv_num > recv_helper->max_idle_recv_num) {
            post_ud_recvs(qid, recv_helper->idle_recv_num);
            recv_helper->idle_recv_num = 0;
        }
        return rc;
    }

private:
    // global mtx to protect qp_vector
    std::mutex *mtx_ = NULL;
    // global mtx to protect ud_attr (routing info)
    std::mutex *ud_mtx_ = NULL;

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

    RdmaDevice *rdma_single_device_ = NULL;
    int num_devices_, num_ports_;
    struct ibv_device **dev_list_ = NULL;
    int* active_ports_;

    std::map<uint64_t, Qp*> qps_;
    int num_rc_qps_;
    int num_uc_qps_;
    int num_ud_qps_;

    volatile uint8_t *conn_buf_ = NULL;
    uint64_t conn_buf_size_ = 0;
#ifdef USE_GPU
    volatile uint8_t *conn_buf_gpu_ = NULL;
    uint64_t conn_buf_size_gpu_ = 0;
#endif
    volatile uint8_t *dgram_buf_ = NULL;
    uint64_t dgram_buf_size_ = 0;

    SimpleMap<RdmaQpAttr*> remote_ud_qp_attrs_;
    SimpleMap<RdmaRecvHelper*> recv_helpers_;

};

bool Qp::get_ud_connect_info_specific(int remote_id, int thread_id, int idx) {

    auto key = _UD_ENCODE_ID(remote_id, thread_id);
    if (ahs_[key] != NULL) return true;

    uint64_t qid = _QP_ENCODE_ID(thread_id + UD_ID_BASE, UD_ID_BASE + idx);

    QPConnArg arg; memset((char *)(&arg), 0, sizeof(QPConnArg));
    arg.qid = qid;
    arg.sign = MAGIC_NUM;
    arg.tid  = thread_id;
    arg.nid  = node_id;
    arg.calculate_checksum();

    auto socket = PreConnector::get_send_socket(network[remote_id], tcp_base_port);

    if (socket < 0) {
        // cannot establish the connection, shall retry
        return false;
    }

    auto n = PreConnector::send_to(socket, (char *)(&arg), sizeof(QPConnArg));
    if (n != sizeof(QPConnArg)) {
        close(socket);
        return false;
    }

    if (!PreConnector::wait_recv(socket)) {
        close(socket);
        return false;
    }

    int buf_size = sizeof(QPReplyHeader) + sizeof(RdmaQpAttr);
    char *reply_buf = new char[buf_size];

    n = recv(socket, reply_buf, buf_size, MSG_WAITALL);
    if (n != sizeof(RdmaQpAttr) + sizeof(QPReplyHeader)) {
        fprintf(stdout, "Receive ud connection content %d by %s\n", n, network[remote_id].c_str());
        close(socket);
        delete reply_buf;
        usleep(1000);
        return false;
    }

    // close connection
    close(socket);

    QPReplyHeader *hdr = (QPReplyHeader *)(reply_buf);

    // the first byte of the message is used to identify the status of the request
    if (hdr->status == TCPSUCC) {

        if (hdr->qid != qid) { // sanity checks
            assert(false);
        }

    } else if (hdr->status == TCPFAIL) {
        delete reply_buf;
        return false;
    } else {
        fprintf(stdout, "QP connect fail!, val %d\n", reply_buf[0]);
        assert(false);
    }
    RdmaQpAttr qp_attr;
    memcpy(&qp_attr, reply_buf + sizeof(QPReplyHeader), sizeof(RdmaQpAttr));

    // verify the checksum
    uint64_t checksum = ip_checksum((void *)(&(qp_attr.buf)), sizeof(RdmaQpAttr) - sizeof(uint64_t));
    assert(checksum == qp_attr.checksum);
    int dlid = qp_attr.lid;

    ahs_[key] = RdmaCtrl::create_ah(dlid, port_idx, dev_);
    memcpy(&(ud_attrs_[key]), reply_buf + sizeof(QPReplyHeader), sizeof(RdmaQpAttr));

    delete reply_buf;
    return true;
}

} // end namespace rdmaio

// private helper functions ////////////////////////////////////////////////////////////////////

static void rc_ready2init(ibv_qp * qp, int port_id) {
    int rc, flags;
    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = port_id;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_REMOTE_ATOMIC;

    flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    rc = ibv_modify_qp(qp, &qp_attr, flags);
    CE_1(rc, "[librdma] qp: Failed to modify RC to INIT state, %s\n", strerror(errno));
}

static void rc_init2rtr(ibv_qp * qp, int port_id, int qpn, int dlid) {
    int rc, flags;
    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;
    qp_attr.dest_qp_num = qpn;
    qp_attr.rq_psn = DEFAULT_PSN;
    qp_attr.max_dest_rd_atomic = 16;
    qp_attr.min_rnr_timer = 12;

    qp_attr.ah_attr.is_global = 0;
    qp_attr.ah_attr.dlid = dlid;
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = port_id; /* Local port! */

    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN
            | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    rc = ibv_modify_qp(qp, &qp_attr, flags);
    CE_1(rc, "[librdma] qp: Failed to modify RC to RTR state, %s\n", strerror(errno));
}

static void rc_rtr2rts(ibv_qp * qp) {
    int rc, flags;
    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = DEFAULT_PSN;
    qp_attr.timeout = 15;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.max_rd_atomic = 16;
    qp_attr.max_dest_rd_atomic = 16;

    flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_MAX_QP_RD_ATOMIC;
    rc = ibv_modify_qp(qp, &qp_attr, flags);
    CE_1(rc, "[librdma] qp: Failed to modify RC to RTS state, %s\n", strerror(errno));
}

static void uc_ready2init(ibv_qp * qp, int port_id) {
    int rc, flags;
    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = port_id;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;

    flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    rc = ibv_modify_qp(qp, &qp_attr, flags);
    CE_1(rc, "[librdma] qp: Failed to modify UC to INIT state, %s\n", strerror(errno));
}

static void uc_init2rtr(ibv_qp * qp, int port_id, int qpn, int dlid) {
    int rc, flags;
    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;
    qp_attr.dest_qp_num = qpn;
    qp_attr.rq_psn = DEFAULT_PSN;

    qp_attr.ah_attr.is_global = 0;
    qp_attr.ah_attr.dlid = dlid;
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = port_id; /* Local port! */

    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
    rc = ibv_modify_qp(qp, &qp_attr, flags);

    CE_1(rc, "[librdma] qp: Failed to modify UC to RTR state, %s\n", strerror(errno));
}

static void uc_rtr2rts(ibv_qp * qp) {
    int rc, flags;
    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = DEFAULT_PSN;

    flags = IBV_QP_STATE | IBV_QP_SQ_PSN;
    rc = ibv_modify_qp(qp, &qp_attr, flags);
    CE_1(rc, "[librdma] qp: Failed to modify RC to RTS state, %s\n", strerror(errno));
}

static void ud_ready2init(ibv_qp * qp, int port_id) {
    int rc, flags;
    struct ibv_qp_attr qp_attr;
    memset((void *) &qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = port_id;
    qp_attr.qkey = DEFAULT_QKEY;

    flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY;
    rc = ibv_modify_qp(qp, &qp_attr, flags);
    CE_1(rc, "[librdma] qp: Failed to modify UD to INIT state, %s\n", strerror(errno));
}

static void ud_init2rtr(ibv_qp * qp) {
    int rc, flags;
    struct ibv_qp_attr qp_attr;
    memset((void *) &qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;

    flags = IBV_QP_STATE;
    rc = ibv_modify_qp(qp, &qp_attr, flags);
    CE_1(rc, "[librdma] qp: Failed to modify UD to RTR state, %s\n", strerror(errno));
}

static void ud_rtr2rts(ibv_qp * qp) {
    int rc, flags;
    struct ibv_qp_attr qp_attr;
    memset((void *) &qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = DEFAULT_PSN;

    flags = IBV_QP_STATE | IBV_QP_SQ_PSN;
    rc = ibv_modify_qp(qp, &qp_attr, flags);
    CE_1(rc, "[librdma] qp: Failed to modify UD to RTS state, %s\n", strerror(errno));
}

#endif
