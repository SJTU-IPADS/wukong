#include "rdmaio.h"
#include "../utils/utils.h"

#include <sys/time.h>  // gettimeofday

#define MAX_POLL_CQ_TIMEOUT 4000 // Time-out of the poll complection

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

// comment ////////////////////////////////////////////////////////////////////

namespace rdmaio {

	Qp::Qp(){
		// zeroing ud connection parameter
		for(uint i = 0;i < 16;++i)
			ahs_[i] = NULL;
	}

	//todo:use send?
	void Qp::init_rc(RdmaDevice *rdma_device, int port_id){

		assert(rdma_device != NULL && rdma_device->ctx != NULL);
		dev_ = rdma_device;
		port_id_ = port_id;

#ifdef PER_QP_PD
        pd = ibv_alloc_pd(rdma_device->ctx);
		mr = ibv_reg_mr(pd,(char *)rdma_device->conn_buf_mr->addr, rdma_device->conn_buf_mr->length,
                                              DEFAULT_PROTECTION_FLAG);
#endif


		recv_cq = send_cq = ibv_create_cq(rdma_device->ctx, RC_MAX_SEND_SIZE, NULL, NULL, 0);
		if(send_cq == NULL) {
			fprintf(stderr,"[librdma] qp: Failed to create cq, %s\n", strerror(errno));
		}
		assert(send_cq != NULL);

		struct ibv_qp_init_attr qp_init_attr;
		memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
		qp_init_attr.send_cq = send_cq;
		qp_init_attr.recv_cq = recv_cq;
		qp_init_attr.qp_type = IBV_QPT_RC;

		qp_init_attr.cap.max_send_wr = RC_MAX_SEND_SIZE;
		qp_init_attr.cap.max_recv_wr = 1;	/* Can be set to 1, if RC Two-sided is not required */
		qp_init_attr.cap.max_send_sge = 1;
		qp_init_attr.cap.max_recv_sge = 1;
		qp_init_attr.cap.max_inline_data = MAX_INLINE_SIZE;
		assert(rdma_device->pd != NULL);
#ifdef PER_QP_PD
		qp = ibv_create_qp(pd, &qp_init_attr);
#else
		qp = ibv_create_qp(rdma_device->pd, &qp_init_attr);
#endif
		CE(!qp,"qp failure!!!");

		rc_ready2init(qp, port_id);
	}

	void Qp::init_uc(RdmaDevice *rdma_device, int port_id){

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
		qp_init_attr.cap.max_recv_wr = UC_MAX_RECV_SIZE;	/* We don't do RECVs on conn QPs */
		qp_init_attr.cap.max_send_sge = 1;
		qp_init_attr.cap.max_recv_sge = 1;
		qp_init_attr.cap.max_inline_data = MAX_INLINE_SIZE;

		qp = ibv_create_qp(rdma_device->pd, &qp_init_attr);
		assert(qp != NULL);

		uc_ready2init(qp, port_id);
	}

	void Qp::init_ud(RdmaDevice *rdma_device, int port_id){

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

	void Qp::change_qp_states(RdmaQpAttr *remote_qp_attr, int dev_port_id) {

		assert(remote_qp_attr != NULL);
		assert(dev_port_id >= 1);

		if(qp->qp_type == IBV_QPT_RC){
			rc_init2rtr(qp, dev_port_id, remote_qp_attr->qpn, remote_qp_attr->lid);
			rc_rtr2rts(qp);
		} else if(qp->qp_type == IBV_QPT_UC){
			uc_init2rtr(qp, dev_port_id, remote_qp_attr->qpn, remote_qp_attr->lid);
			uc_rtr2rts(qp);
		} else {
			assert(false);
		}
        remote_attr_ = *remote_qp_attr;

	}

	int Qp::try_poll() {
		struct ibv_wc wc;
		auto poll_result = ibv_poll_cq(this->send_cq, 1, &wc);
		if(poll_result > 0) {
			assert(wc.status == IBV_WC_SUCCESS);
			return wc.wr_id;
		} else if(poll_result < 0) {
			// FIXME: not handled yet
			assert(false);
		}
		return -1;
	}

	// poll complection of a cq
	Qp::IOStatus Qp::poll_completion(uint64_t *rid) {

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
		}
		while ((poll_result == 0));
			   // && ((cur_time_msec - start_time_msec) < MAX_POLL_CQ_TIMEOUT));

		if(unlikely(rid != NULL))
			*rid = wc.wr_id;

		// check the result
		if(poll_result < 0) {
			assert(false);
			/* poll CQ failed */
			return IO_ERR;
		} else if (poll_result == 0){
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
						 wc.status, wc.vendor_err,ibv_wc_status_str(wc.status),nid,tid);
				assert(false);
				return IO_ERR;
			} else {
				// success, just pass
			}
		}

		return IO_SUCC;
	}

	// poll complections of a cq
	Qp::IOStatus Qp::poll_completions(int cq_num, uint64_t *rid) {
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
			if(poll_once < 0) {
				assert(false);
			}
            poll_result += poll_once;

			gettimeofday (&cur_time, NULL);
			cur_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
		}
		while ((poll_result < cq_num));
			   // && ((cur_time_msec - start_time_msec) < MAX_POLL_CQ_TIMEOUT));

		if (poll_result != cq_num){
			return IO_TIMEOUT;
		} else {
			for (int cq_id = 0 ; cq_id < cq_num; cq_id ++){
			    if(wc[cq_id].status != IBV_WC_SUCCESS) {
	                fprintf (stderr,
	                         "got bad completion with status: 0x%x, vendor syndrome: 0x%x, with error %s\n",
	                         wc[cq_id].status, wc[cq_id].vendor_err,ibv_wc_status_str(wc[poll_result].status));
	                return IO_ERR;
	                // exit(-1);
	            }
			}
        }

		return rc;
	}
	// end namespace
};
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
	rc = ibv_modify_qp(qp, &qp_attr,flags);
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
	rc = ibv_modify_qp(qp, &qp_attr,flags);
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
	rc = ibv_modify_qp(qp, &qp_attr,flags);
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
	rc = ibv_modify_qp(qp, &qp_attr,flags);
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
	rc = ibv_modify_qp(qp, &qp_attr,flags);

	CE_1(rc, "[librdma] qp: Failed to modify UC to RTR state, %s\n", strerror(errno));
}

static void uc_rtr2rts(ibv_qp * qp) {
	int rc, flags;
	struct ibv_qp_attr qp_attr;
	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
	qp_attr.qp_state = IBV_QPS_RTS;
	qp_attr.sq_psn = DEFAULT_PSN;

	flags = IBV_QP_STATE | IBV_QP_SQ_PSN;
	rc = ibv_modify_qp(qp, &qp_attr,flags);
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
