// implementation of the RdmaCtrl class

#include <malloc.h>

#include <arpa/inet.h> //used for checksum

#include "rdmaio.h"
#include "../utils/utils.h"
#include "helper_func.hpp"

static volatile bool running;

namespace rdmaio {

    int tcp_base_port;
    int num_rc_qps;
    int num_uc_qps;
    int num_ud_qps;
    int node_id;

    std::vector<std::string> network;

    // seems that zeromq requires to use one context per process
    zmq::context_t context(12);


    // per-thread allocator
    __thread RdmaDevice **rdma_devices_;

    RdmaCtrl::RdmaCtrl(int id, const std::vector<std::string> net,
                       int port, bool enable_single_thread_mr):
        node_id_(id),network_(net.begin(),net.end()),tcp_base_port_(port),
        recv_helpers_(NULL),remote_ud_qp_attrs_(NULL),//qps_(NULL),
        rdma_single_device_(NULL),
        num_rc_qps_(100),num_uc_qps_(1),num_ud_qps_(4),
        enable_single_thread_mr_(enable_single_thread_mr)
    {

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
        network = std::vector<std::string>(net.begin(),net.end());

        query_devinfo();

        running = true;
    }

    RdmaCtrl::~RdmaCtrl() {
        // free some resources, may be we does need to do this,
        // since when RDMA device is closed, the app shall close
        delete mtx_; delete ud_mtx_; qps_.clear();

        // TODO!! free RDMA related devices

        running = false; // close listening threads
    }

    void RdmaCtrl::thread_local_init() {
        //single memory region
        if(enable_single_thread_mr_) return;
        // the device related object shall be created locally
        rdma_devices_ = new RdmaDevice*[num_devices_];
        for(uint i = 0;i < num_devices_;++i)
            rdma_devices_[i] = NULL;
    }

    void RdmaCtrl::query_devinfo() {

        int rc;

        dev_list_ = ibv_get_device_list (&num_devices_);
        CE(!num_devices_,"[librdma] : failed to get IB devices list\n");
        // printf("[librdma] : Total %d devices!\n", num_devices_);

        active_ports_ = new int[num_devices_];
        num_ports_ = 0;
        for(int device_id = 0; device_id < num_devices_; device_id++){

            // printf("[librdma] get device name %s, idx %d\n",dev_list_[device_id]->name,device_id);
            struct ibv_context *ib_ctx = ibv_open_device(dev_list_[device_id]);
            CE_1(!ib_ctx, "[librdma] : Failed to open device %d\n", device_id);

            struct ibv_device_attr device_attr;
            memset(&device_attr, 0, sizeof(device_attr));

            rc = ibv_query_device(ib_ctx, &device_attr);
            CE_1(rc, "[librdma] : Failed to query device %d\n", device_id);

            int port_num = 0, port_count = device_attr.phys_port_cnt;
            for(int port_id = 1; port_id <= port_count; port_id++){
                struct ibv_port_attr port_attr;
                rc = ibv_query_port(ib_ctx, port_id, &port_attr);
                CE_2(rc, "[librdma] : Failed to query port %d on device %d\n ", port_id, device_id);

                if(port_attr.phys_state != IBV_PORT_ACTIVE &&
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

    int RdmaCtrl::get_active_dev(int port_index){
        assert(port_index >= 0 && port_index < num_ports_);
        for(int device_id = 0; device_id < num_devices_; device_id++){
            int port_num = active_ports_[device_id];
            for(int port_id = 1; port_id <= port_num; port_id++){
                if(port_index == 0)return device_id;
                port_index--;
            }
        }
        return -1;
    }

    int RdmaCtrl::get_active_port(int port_index){

        assert(port_index >= 0 && port_index < num_ports_);
        for(int device_id = 0; device_id < num_devices_; device_id++){
            int port_num = active_ports_[device_id];
            for(int port_id = 1; port_id <= port_num; port_id++){
                if(port_index == 0)return port_id;
                port_index--;
            }
        }
        return -1;
    }

    void RdmaCtrl::open_device(int dev_id) {

        int rc;

        struct ibv_device *device = dev_list_[dev_id];
        //CE_2(!device,"[librdma]: IB device %d wasn't found\n",dev_id);
        RdmaDevice *rdma_device;
        if(enable_single_thread_mr_){
            if(rdma_single_device_ == NULL) {
                rdma_single_device_ = new RdmaDevice();
                rdma_device = rdma_single_device_;
            } else {
                return;
            }
        } else {
            if(rdma_devices_[dev_id] == NULL) {
                rdma_device = rdma_devices_[dev_id] = new RdmaDevice();
            } else {
                return;
            }
        }

        rdma_device->dev_id = dev_id;
        rdma_device->ctx = ibv_open_device(device);
        //CE_2(!rdma_device->ctx,"[librdma] : failed to open device %d\n",dev_id);

        struct ibv_device_attr device_attr;
        rc = ibv_query_device(rdma_device->ctx,&device_attr);
        //CE_2(rc,"[librdma]: failed to query device %d\n",dev_id);

        int port_count = device_attr.phys_port_cnt;
        rdma_device->port_attrs =(struct ibv_port_attr*)
            malloc(sizeof(struct ibv_port_attr) * (port_count + 1));
        for(int port_id = 1; port_id <= port_count; port_id++){
            rc = ibv_query_port (rdma_device->ctx, port_id, rdma_device->port_attrs + port_id);
            //      CE_2(rc,"[librdma]: ibv_query_port on port %u failed\n",port_id);
        }

        rdma_device->pd = ibv_alloc_pd(rdma_device->ctx);
        assert(rdma_device->pd != 0);
        // CE_1(!rdma_device_->pd, "[librdma]: ibv_alloc prodection doman failed at dev %d\n",dev_id);
    }

    void RdmaCtrl::set_connect_mr(volatile void *conn_buf, uint64_t conn_buf_size){
        if(conn_buf == NULL) {
            conn_buf = (volatile uint8_t *) memalign(4096, conn_buf_size);
        }
        assert(conn_buf != NULL);
        memset((char *) conn_buf, 0, conn_buf_size);

        conn_buf_ = (volatile uint8_t *)conn_buf;
        conn_buf_size_ = conn_buf_size;
    }

    void RdmaCtrl::set_dgram_mr(volatile void *dgram_buf, int dgram_buf_size){
        if(dgram_buf == NULL) {
            dgram_buf = (volatile uint8_t *) memalign(4096, dgram_buf_size);
        }
        assert(dgram_buf != NULL);
        memset((char *) dgram_buf, 0, dgram_buf_size);

        dgram_buf_ = (volatile uint8_t *)dgram_buf;
        dgram_buf_size_ = dgram_buf_size;
    }


    void RdmaCtrl::register_connect_mr(int dev_id) {
        RdmaDevice *rdma_device = get_rdma_device(dev_id);
        assert(rdma_device->pd != NULL);
        if(enable_single_thread_mr_ && rdma_device->conn_buf_mr != NULL) {
            return;
        }
        rdma_device->conn_buf_mr = ibv_reg_mr(rdma_device->pd,(char *)conn_buf_, conn_buf_size_,
                                              DEFAULT_PROTECTION_FLAG);
        CE_2(!rdma_device->conn_buf_mr,
             "[librdma]: Connect Memory Region failed at dev %d, err %s\n",dev_id,strerror(errno));
    }

    void RdmaCtrl::register_dgram_mr(int dev_id) {
        RdmaDevice *rdma_device = get_rdma_device(dev_id);
        assert(rdma_device->pd != NULL);
        rdma_device->dgram_buf_mr = ibv_reg_mr(rdma_device->pd,(char *)dgram_buf_, dgram_buf_size_,
                                               DEFAULT_PROTECTION_FLAG);
        CE_2(!rdma_device->dgram_buf_mr
             ,"[librdma]: Datagram Memory Region failed at dev %d, err %s\n",dev_id,strerror(errno));
    }

    Qp *RdmaCtrl::create_rc_qp(int tid, int remote_id,int dev_id,int port_idx, int idx) {

        // TODO: check device
        // compute local qp id
        assert(num_rc_qps_ != 0);
        assert(idx >= 0 && idx < num_rc_qps_);
        uint64_t qid = _QP_ENCODE_ID(remote_id, RC_ID_BASE + tid * num_rc_qps_ + idx);
        Qp *res = NULL;

        mtx_->lock();
        // fprintf(stdout,"create qp %d %d %d, qid %lu\n",tid,remote_id,idx,qid);
        if(qps_.find(qid) != qps_.end() && qps_[qid] != nullptr) {
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

        res->init_rc(get_rdma_device(dev_id),port_idx);
        qps_.insert(std::make_pair(qid,res));
        //fprintf(stdout,"create qp %d %d done %p\n",tid,remote_id,res);
        mtx_->unlock();

        // done
        return res;
    }

    Qp *RdmaCtrl::create_uc_qp(int tid, int remote_id,int dev_id,int port_idx, int idx) {
        // TODO: check device

        // compute local qp id
        assert(num_uc_qps_ != 0);
        assert(idx >= 0 && idx < num_uc_qps_);
        int32_t qid = _QP_ENCODE_ID(remote_id, UC_ID_BASE + tid * num_uc_qps_ + idx);
        Qp *res = NULL;

        mtx_->lock();
        if(qps_.find(qid) != qps_.end() && qps_[qid] != nullptr) {
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

        res->init_uc(get_rdma_device(dev_id),port_idx);
        qps_.insert(std::make_pair(qid,res));
        //fprintf(stdout,"create qp %d %d done %p\n",tid,remote_id,res);
        mtx_->unlock();
        // done
        return res;
    }

    Qp *RdmaCtrl::create_ud_qp(int tid,int dev_id,int port_idx,int idx) {

        RdmaDevice *rdma_device = get_rdma_device(dev_id);

        // the unique id which identify this QP
        assert(num_ud_qps_ != 0);
        assert(idx >= 0 && idx < num_ud_qps_);
        uint64_t qid = _QP_ENCODE_ID(UD_ID_BASE + tid ,UD_ID_BASE + idx);

        Qp *res = NULL;

        mtx_->lock();
        if(qps_.find(qid) != qps_.end()) {
            res = qps_[qid];
            mtx_->unlock();
            assert(false);
            return res;
        }

        res = new Qp();
        res->init_ud(get_rdma_device(dev_id),port_idx);
        res->tid = tid;
        res->port_idx = port_idx;
        res->dev_ = rdma_device;

        //qps_.insert(qid,res);
        qps_.insert(std::make_pair(qid,res));
        mtx_->unlock();
        return res;
    }

    void RdmaCtrl::link_connect_qps(int tid, int dev_id, int port_idx, int idx, ibv_qp_type qp_type){

        Qp* (RdmaCtrl::* create_qp_func)(int,int,int,int,int);
        bool (Qp::* connect_qp_func)();
        int num_qps;

        switch(qp_type){
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
            CE(true,"link_connect_qp: error qp type");
        }
        for(uint i = 0; i < get_num_nodes();++i) {
            Qp *qp = (this->*create_qp_func)(tid,i,dev_id,port_idx,idx);
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

        while(1) {
            int connected = 0;
            for(uint i = 0;i < get_num_nodes();++i) {
                Qp *qp = (this->*create_qp_func)(tid,i,dev_id,port_idx,idx);
                if(qp->inited_)
                    connected += 1;
                else if((qp->*connect_qp_func)())
                    connected += 1;
            }

            if(connected == get_num_nodes())
                break;
            else
                usleep(200000);
        }
    }

    RdmaQpAttr RdmaCtrl::get_remote_qp_attr(int nid, uint64_t qid) {
        assert(false);
        int retry_count = 0;
    retry:
        char address[30];
        snprintf(address,30,"tcp://%s:%d",network_[nid].c_str(),tcp_base_port_);
        zmq::context_t context(1);
        zmq::socket_t socket(context,ZMQ_REQ);
        socket.connect(address);

        zmq::message_t request(sizeof(QPConnArg));
        fprintf(stdout,"conn to %s\n",address);
        //    fprintf(stdout,"encode id %d %d\n",this->nodeId,_QP_DECODE_INDEX(qid));
        //*((int *)request.data()) = qid;//////wa! o !
        QPConnArg *argp = (QPConnArg *)(request.data());
        argp->qid = qid;
        argp->sign = MAGIC_NUM;
        argp->calculate_checksum();

        socket.send(request);

        zmq::message_t reply;
        socket.recv(&reply);

        if(((char *)reply.data())[0] == TCPSUCC) {

        } else if(((char *)reply.data())[0] == TCPFAIL) {

            if(retry_count > 10) {
                fprintf(stdout,"response %d, try connect to %d\n",((char *)reply.data())[0],nid);
                assert(false);
            }
            sleep(1);
            retry_count += 1;
            goto retry;

        } else {
            fprintf(stdout,"QP connect fail!, val %d\n",((char *)reply.data())[0]);
            assert(false);
        }

        RdmaQpAttr qp_attr;
        memcpy(&qp_attr,(char *)reply.data() + 1,sizeof(RdmaQpAttr));

        // verify the checksum
        uint64_t checksum = ip_checksum((void *)(&(qp_attr.buf)),sizeof(RdmaQpAttr) - sizeof(uint64_t));
        assert(checksum == qp_attr.checksum);
        return qp_attr;
    }


    RdmaQpAttr RdmaCtrl::get_local_qp_attr(int qid){

        RdmaQpAttr qp_attr;
        Qp *local_qp = qps_[qid];
        assert(local_qp != NULL);
        //uint64_t begin = rdtsc();
        if(IS_CONN(qid)){

            qp_attr.buf = (uint64_t) (uintptr_t) conn_buf_;
            qp_attr.buf_size = conn_buf_size_;

#ifdef PER_QP_PD
            qp_attr.rkey = local_qp->mr->rkey;
#else
            assert(local_qp->dev_ != NULL);
            assert(local_qp->dev_->conn_buf_mr != NULL);
            qp_attr.rkey = local_qp->dev_->conn_buf_mr->rkey;
#endif

            //qp_attr.rkey = rdma_device_->conn_buf_mr->rkey;
            //qp_attr.rkey = qps_[qid]->reg_mr->rkey;
        }
        //qp_attr.lid = qps_[qid]->dev_->port_attrs[dev_port_id_].lid;
        qp_attr.lid = local_qp->dev_->port_attrs[local_qp->port_id_].lid;
        qp_attr.qpn = local_qp->qp->qp_num;
        //fprintf(stdout,"get local qp costs %lu\n",rdtsc() - begin);

        // calculate the checksum
        uint64_t checksum = ip_checksum((void *)(&(qp_attr.buf)),sizeof(RdmaQpAttr) - sizeof(uint64_t));
        qp_attr.checksum = checksum;
        return qp_attr;
    }

    void RdmaCtrl::start_server() {

        pthread_t tid;
        pthread_attr_t attr;

        int rc = pthread_attr_init(&attr);
        assert(rc == 0);
        //rc = pthread_attr_setschedpolicy(&attr,1); // min priority
        //assert(rc == 0);

        pthread_create(&tid, &attr, recv_thread, (void *)this);
    }

    void* RdmaCtrl::recv_thread(void *arg){

        pthread_detach(pthread_self());
        struct RdmaCtrl *rdma = (struct RdmaCtrl*) arg;

        zmq::socket_t socket(context,ZMQ_REP);

        char address[30] = "";
        int port = rdma->tcp_base_port_;
        //        char address[30]="";
        //int port = rdma->`tcp_base_port_ + rdma->node_id_;
        sprintf(address,"tcp://*:%d",port);
        //	DEBUG(rdma->thread_id_,
		printf("[librdma] : listener binding: %s\n", address);
        socket.bind(address);

        try {
            while(running) {
                zmq::message_t request;
                socket.recv(&request);

                //int qid =  *((int *)(request.data()));
                QPConnArg *arg = (QPConnArg *)(request.data());
                // check that the arg is correct
                assert(arg->sign = MAGIC_NUM);
                assert(arg->get_checksum() == arg->checksum);

                uint64_t qid = arg->qid;
                uint64_t nid = _QP_DECODE_MAC(qid);
                uint64_t idx = _QP_DECODE_INDEX(qid);

                zmq::message_t reply(sizeof(RdmaQpAttr) + 1);

                rdma->mtx_->lock();
                if(rdma->qps_.find(qid) == rdma->qps_.end()) {
                    *(char *)(reply.data()) = TCPFAIL;
                } else {
                    if(IS_UD(qid)) {
                        // further check whether receives are posted
                        Qp *ud_qp = rdma->qps_[qid];
                        if(ud_qp->inited_ == false) {
                            *(char *)(reply.data()) = TCPFAIL;
                        } else {
                            RdmaQpAttr qp_attr = rdma->get_local_qp_attr(qid);
                            *(char *)(reply.data()) = TCPSUCC;
                            memcpy((char *)(reply.data()) + 1,(char *)(&qp_attr),sizeof(RdmaQpAttr));
                        }
                    } else {
                        RdmaQpAttr qp_attr = rdma->get_local_qp_attr(qid);
                        *(char *)(reply.data()) = TCPSUCC;
                        memcpy((char *)(reply.data()) + 1,(char *)(&qp_attr),sizeof(RdmaQpAttr));
                    }
                }
                rdma->mtx_->unlock();
                // reply with the QP attribute
                socket.send(reply);
                //if(rdma->response_times_ < 0) continue;
                //if(--rdma->response_times_ == 0)break;
            }   // while receiving reqests
            context.close();
        } catch (...) {
            // pass
        }
        printf("[librdma] : recv thread exit!\n");
    }

    ibv_ah* RdmaCtrl::create_ah(int dlid, int port_index, RdmaDevice* rdma_device){
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

    void RdmaCtrl::init_conn_recv_qp(int qid){
        RdmaRecvHelper *recv_helper = new RdmaRecvHelper;
        RdmaDevice* rdma_device = qps_[qid]->dev_;
        int recv_step = 0;
        int max_recv_num = RC_MAX_RECV_SIZE;
        while(recv_step < MAX_PACKET_SIZE){
            recv_step += MIN_STEP_SIZE;
        }
        assert(recv_step > 0 && recv_step % MIN_STEP_SIZE == 0);

        printf("recv_step: %d\n", recv_step);
        for(int i = 0; i < max_recv_num; i++) {
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

    void RdmaCtrl::init_dgram_recv_qp(int qid){

        RdmaRecvHelper *recv_helper = new RdmaRecvHelper;
        RdmaDevice* rdma_device = qps_[qid]->dev_;
        int recv_step = 0;
        int max_recv_num = UD_MAX_RECV_SIZE;
        while(recv_step < MAX_PACKET_SIZE + GRH_SIZE){
            recv_step += MIN_STEP_SIZE;
        }
        assert(recv_step > 0 && recv_step % MIN_STEP_SIZE == 0);

        printf("recv_step: %d\n", recv_step);
        for(int i = 0; i < max_recv_num; i++) {
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

    int RdmaCtrl::poll_recv_cq(int qid){
        Qp *qp = qps_[qid];
        struct ibv_wc wc;
        int rc = 0;
        int poll_result;

        do {
            poll_result = ibv_poll_cq (qp->recv_cq, 1, &wc);
        } while(poll_result == 0);
        assert(poll_result == 1);

        if (wc.status != IBV_WC_SUCCESS) {
            fprintf (stderr,
                     "got bad completion with status: 0x%x, vendor syndrome: 0x%x, with error %s\n",
                     wc.status, wc.vendor_err,ibv_wc_status_str(wc.status));
        }
        // fprintf(stdout,"poll Recv imm %d, buffer data: %d\n",wc.imm_data,
        //       (*(uint32_t*)(wc.wr_id+GRH_SIZE)));
        return rc;
    }


    int RdmaCtrl::poll_recv_cq(Qp* qp){
        struct ibv_wc wc;
        int rc = 0;
        int poll_result;

        do {
            poll_result = ibv_poll_cq (qp->recv_cq, 1, &wc);
        } while(poll_result == 0);
        assert(poll_result == 1);

        if (wc.status != IBV_WC_SUCCESS) {
            fprintf (stderr,
                     "got bad completion with status: 0x%x, vendor syndrome: 0x%x, with error %s\n",
                     wc.status, wc.vendor_err,ibv_wc_status_str(wc.status));
        }
        // fprintf(stdout,"poll Recv imm %d, buffer data: %d\n",wc.imm_data,
        //       (*(uint32_t*)(wc.wr_id+GRH_SIZE)));
        return rc;
    }

    int RdmaCtrl::poll_cqs(int qid, int cq_num){
        struct ibv_wc wc[RC_MAX_SEND_SIZE];
        int rc = 0;
        int poll_result = 0;
        Qp *qp = qps_[qid];
        while(poll_result < cq_num) {
            int poll_once = ibv_poll_cq(qp->send_cq, cq_num - poll_result, &wc[poll_result]);
            if(poll_once != 0) {
                if(wc[poll_result].status != IBV_WC_SUCCESS) {
                    fprintf (stderr,
                             "got bad completion with status: 0x%x, vendor syndrome: 0x%x, with error %s\n",
                             wc[poll_result].status, wc[poll_result].vendor_err,ibv_wc_status_str(wc[poll_result].status));
                    // exit(-1);
                }
            }
            poll_result += poll_once;
        }
        qp->pendings = 0;
        return rc;
    }

}
