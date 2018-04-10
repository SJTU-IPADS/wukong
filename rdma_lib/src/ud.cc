#include <arpa/inet.h> //used for checksum

#include "rdmaio.h"
#include "../utils/utils.h"

#include "helper_func.hpp"

namespace rdmaio {

	extern int tcp_base_port; // tcp listening port
	extern int node_id; // this instance's node id
	extern std::vector<std::string> network; // topology

	extern zmq::context_t context;

	bool Qp::get_ud_connect_info_specific(int remote_id,int thread_id,int idx) {

		auto key = _QP_ENCODE_ID(remote_id,thread_id);
		if(ahs_.find(key) != ahs_.end()) {
			return true;
		}

		uint64_t qid = _QP_ENCODE_ID(thread_id + UD_ID_BASE,UD_ID_BASE + idx);

        char address[30];
        snprintf(address,30,"tcp://%s:%d",network[remote_id].c_str(),tcp_base_port);

		// prepare tcp connection
        zmq::socket_t socket(context,ZMQ_REQ);
        socket.connect(address);

        zmq::message_t request(sizeof(QPConnArg));

        QPConnArg *argp = (QPConnArg *)(request.data());
        argp->qid = qid;
        argp->sign = MAGIC_NUM;
        argp->calculate_checksum();

        socket.send(request);

        zmq::message_t reply;
        socket.recv(&reply);

		// the first byte of the message is used to identify the status of the request
        if(((char *)reply.data())[0] == TCPSUCC) {

        } else if(((char *)reply.data())[0] == TCPFAIL) {
			return false;

        } else {
            fprintf(stdout,"QP connect fail!, val %d\n",((char *)reply.data())[0]);
            assert(false);
        }
		zmq_close(&socket);
        RdmaQpAttr qp_attr;
        memcpy(&qp_attr,(char *)reply.data() + 1,sizeof(RdmaQpAttr));

        // verify the checksum
        uint64_t checksum = ip_checksum((void *)(&(qp_attr.buf)),sizeof(RdmaQpAttr) - sizeof(uint64_t));
        assert(checksum == qp_attr.checksum);

        int dlid = qp_attr.lid;

		ahs_.insert(std::make_pair(key,RdmaCtrl::create_ah(dlid,port_idx,dev_)));
		ud_attrs_.insert(std::make_pair(key,RdmaQpAttr()));
		memcpy(&(ud_attrs_[key]),(char *)reply.data() + 1,sizeof(RdmaQpAttr));

		return true;
	}

	bool Qp::get_ud_connect_info(int remote_id,int idx) {

		// already gotten
		if(ahs_[remote_id] != NULL) {
			return true;
		}

		//int qid = _QP_ENCODE_ID(0,UD_ID_BASE + tid * num_ud_qps + idx);
		uint64_t qid = _QP_ENCODE_ID(tid + UD_ID_BASE,UD_ID_BASE + idx);

        char address[30];
        snprintf(address,30,"tcp://%s:%d",network[remote_id].c_str(),tcp_base_port);

		// prepare tcp connection
        zmq::socket_t socket(context,ZMQ_REQ);
        socket.connect(address);

        zmq::message_t request(sizeof(QPConnArg));

        QPConnArg *argp = (QPConnArg *)(request.data());
        argp->qid = qid;
        argp->sign = MAGIC_NUM;
        argp->calculate_checksum();

        socket.send(request);

        zmq::message_t reply;
        socket.recv(&reply);

		// the first byte of the message is used to identify the status of the request
        if(((char *)reply.data())[0] == TCPSUCC) {

        } else if(((char *)reply.data())[0] == TCPFAIL) {
			return false;

        } else {
            fprintf(stdout,"QP connect fail!, val %d\n",((char *)reply.data())[0]);
            assert(false);
        }
		zmq_close(&socket);
        RdmaQpAttr qp_attr;
        memcpy(&qp_attr,(char *)reply.data() + 1,sizeof(RdmaQpAttr));

        // verify the checksum
        uint64_t checksum = ip_checksum((void *)(&(qp_attr.buf)),sizeof(RdmaQpAttr) - sizeof(uint64_t));
        assert(checksum == qp_attr.checksum);

        int dlid = qp_attr.lid;

		ahs_[remote_id] = RdmaCtrl::create_ah(dlid,port_idx,dev_);
		memcpy(&(ud_attrs_[remote_id]),(char *)reply.data() + 1,sizeof(RdmaQpAttr));

		return true;
	}


	int RdmaCtrl::post_ud(int qid, RdmaReq* reqs){
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

	int RdmaCtrl::post_ud_doorbell(int qid, int batch_size, RdmaReq* reqs){

		int rc = 0;
		struct ibv_send_wr sr[MAX_DOORBELL_SIZE], *bad_sr;
		struct ibv_sge sge[MAX_DOORBELL_SIZE];
		Qp *qp = qps_[qid];
		assert(qp->qp->qp_type == IBV_QPT_UD);
		bool needpoll = false;

		for(int i = 0; i < batch_size; i++) {
			RdmaQpAttr* qp_attr = remote_ud_qp_attrs_[reqs[i].wr.ud.remote_qid];
			if(qp_attr == NULL) {
				fprintf(stdout,"qid %lu\n",reqs[i].wr.ud.remote_qid);
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
			if(qp->first_send()){
				sr[i].send_flags |= IBV_SEND_SIGNALED;
			}
			if(qp->need_poll()){
				needpoll = true;
			}
			// sr[i].send_flags |= IBV_SEND_INLINE;

			sge[i].addr = reqs[i].buf;
			sge[i].length = reqs[i].length;
			sge[i].lkey = qp->dev_->dgram_buf_mr->lkey;
		}
		if(needpoll)qp->poll_completion();
		rc = ibv_post_send(qp->qp, &sr[0], &bad_sr);
		CE(rc, "ibv_post_send error");
		return rc;
	}

	int RdmaCtrl::post_ud_recv(struct ibv_qp *qp, void *buf_addr, int len, int lkey) {
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

	int RdmaCtrl::post_ud_recvs(int qid, int recv_num) {
		struct ibv_recv_wr *head_rr, *tail_rr, *temp_rr, *bad_rr;
		RdmaRecvHelper *recv_helper = recv_helpers_[qid];
		// recv_num > 0 && recv_num <= MAX_RECV_SIZE;
		// fprintf(stdout, "Node %d: Posting %d RECVs \n",nodeId, recv_num);

		int rc = 0;
		int head = recv_helper->recv_head;
		int tail = head + recv_num - 1;
		if(tail >= recv_helper->max_recv_num) {
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

	int RdmaCtrl::poll_dgram_recv_cqs(int qid){
		Qp *qp = qps_[qid];
		RdmaRecvHelper *recv_helper = recv_helpers_[qid];
		int poll_result, rc;
		struct ibv_wc* wc = recv_helper->wc;
		poll_result = ibv_poll_cq (qp->recv_cq, recv_helper->max_recv_num, wc);
		rc = poll_result;
		CE(poll_result < 0,"poll CQ failed\n");
		for(int i = 0; i < poll_result; i++){
			if (wc[i].status != IBV_WC_SUCCESS) {
				fprintf (stderr,
						 "got bad completion with status: 0x%x, vendor syndrome: 0x%x, with error %s\n",
						 wc[i].status, wc[i].vendor_err,ibv_wc_status_str(wc[i].status));
				rc = -1;
			}
			// fprintf(stdout,"poll Recv imm %d, buffer data: %d\n",ntohl(qp->recvWCs[i].imm_data),
			//   (*(uint32_t*)(qp->recvWCs[i].wr_id+GRH_SIZE)));
		}
		recv_helper->idle_recv_num += poll_result;
		if(recv_helper->idle_recv_num > recv_helper->max_idle_recv_num){
			post_ud_recvs(qid, recv_helper->idle_recv_num);
			recv_helper->idle_recv_num = 0;
		}
		return rc;
	}

}
