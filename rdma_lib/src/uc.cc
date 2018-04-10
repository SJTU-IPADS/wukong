#include <arpa/inet.h> //used for checksum

#include "rdmaio.h"
#include "../utils/utils.h"
#include "helper_func.hpp"

namespace rdmaio {

	extern int tcp_base_port; // tcp listening port
	extern int node_id; // this instance's node id
	extern std::vector<std::string> network; // topology
	extern int num_uc_qps;

	extern zmq::context_t context;
	
	bool Qp::connect_uc() {

		if(inited_) {
			return true;
		} else {
			//			fprintf(stdout,"qp %d %d not connected\n",tid,nid);
		}

		int remote_qid = _QP_ENCODE_ID(node_id,UC_ID_BASE + tid * num_uc_qps + idx_);

        char address[30];
        int address_len = snprintf(address,30,"tcp://%s:%d",network[nid].c_str(),tcp_base_port);
		assert(address_len < 30);

		// prepare tcp connection
        zmq::socket_t socket(context,ZMQ_REQ);
        socket.connect(address);

        zmq::message_t request(sizeof(QPConnArg));

        QPConnArg *argp = (QPConnArg *)(request.data());
        argp->qid = remote_qid;
        argp->sign = MAGIC_NUM;
        argp->calculate_checksum();
        socket.send(request);

        zmq::message_t reply;
        socket.recv(&reply);
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

		change_qp_states(&qp_attr,port_idx);

		inited_ = true;
		return true;
	}

	Qp::IOStatus Qp::uc_post_send(ibv_wr_opcode op,char *local_buf,int len,uint64_t off,int flags) {

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

	Qp::IOStatus Qp::uc_post_doorbell(RdmaReq *reqs, int batch_size) {

		IOStatus rc = IO_SUCC;
		assert(batch_size <= MAX_DOORBELL_SIZE);

		struct ibv_send_wr sr[MAX_DOORBELL_SIZE], *bad_sr;
		struct ibv_sge sge[MAX_DOORBELL_SIZE];

		assert(this->qp->qp_type == IBV_QPT_UC);
		bool poll = false;
		for(uint i = 0; i < batch_size; i++) {
			// fill in the requests
			sr[i].opcode = reqs[i].opcode;
			sr[i].num_sge = 1;
			sr[i].next = (i == batch_size - 1) ? NULL : &sr[i + 1];
			sr[i].sg_list = &sge[i];
			sr[i].send_flags = reqs[i].flags;

			if(first_send()){
				sr[i].send_flags |= IBV_SEND_SIGNALED;
			}
			if(need_poll()){
				poll = true;
			}

			sge[i].addr = reqs[i].buf;
			sge[i].length = reqs[i].length;
			sge[i].lkey = dev_->conn_buf_mr->lkey;

			sr[i].wr.rdma.remote_addr =
				remote_attr_.buf + reqs[i].wr.rdma.remote_offset;
			sr[i].wr.rdma.rkey = remote_attr_.rkey;
		}
		if(poll) rc = poll_completion();
		rc = (IOStatus)ibv_post_send(qp, &sr[0], &bad_sr);
		CE(rc, "ibv_post_send error");
		return rc;
	}


}
