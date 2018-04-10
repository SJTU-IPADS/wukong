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

#define DUAL_PORT 1

#define _RDMA_INDEX_MASK (0xffff)
#define _RDMA_MAC_MASK (_RDMA_INDEX_MASK << 16)
#define _QP_ENCODE_ID(mac,index) ((mac) << 16 | (index))
#define _QP_DECODE_MAC(qid) (((qid) & _RDMA_MAC_MASK) >> 16 )
#define _QP_DECODE_INDEX(qid) ((qid) & _RDMA_INDEX_MASK)

#define IS_RC(qid) (_QP_DECODE_INDEX(qid)>=RC_ID_BASE && _QP_DECODE_INDEX(qid) < UC_ID_BASE)
#define IS_UC(qid) (_QP_DECODE_INDEX(qid)>=UC_ID_BASE && _QP_DECODE_INDEX(qid) < UD_ID_BASE)
#define IS_CONN(qid) (_QP_DECODE_INDEX(qid)>=RC_ID_BASE && _QP_DECODE_INDEX(qid) < UD_ID_BASE)
#define IS_UD(qid) (_QP_DECODE_INDEX(qid)>=UD_ID_BASE)
#define RC_ID_BASE 0
#define UC_ID_BASE 10000
#define UD_ID_BASE 20000

#define DEFAULT_PROTECTION_FLAG ( IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | \
                                  IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC)

// XD: can we clean these flags?
#define GRH_SIZE 40
#define MIN_STEP_SIZE 64
#define MAX_PACKET_SIZE 4032 //4096 - 64
// #define MAX_PACKET_SIZE 64

#define MAX_QP_NUM 128 //todo
#define MAX_DOORBELL_SIZE 16 //todo
#define MAXTHREAD 32
#define MAXPORTS 2


#define M_2 2097152
#define M_2_ 2097151

#define M_8 8388608
#define M_8_ 8388607

#define M_1024 1073741824
#define M_1024_ 1073741823

#define RC_MAX_SEND_SIZE 256
#define RC_MAX_RECV_SIZE 128
#define UC_MAX_SEND_SIZE 128
#define UC_MAX_RECV_SIZE 128
#define UD_MAX_SEND_SIZE 128
#define UD_MAX_RECV_SIZE 2048

#define POLL_THRSHOLD 64

// XD: is it configurable?
#define DEFAULT_PSN 3185  /* PSN for all queues */
#define DEFAULT_QKEY 0x11111111

#define MAX_INLINE_SIZE 64


#define CACHE_LINE_SZ 64
