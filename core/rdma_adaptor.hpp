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

#include <string>
#include <iostream>
#include <unistd.h>
#include <unordered_map>
#include <fstream>
#include <errno.h>
#include <sstream>

#include "config.hpp"
#include "rdma.hpp"
#include "mem.hpp"
#include "query.hpp"

#ifdef USE_GPU
#include "gpu_utils.hpp"
#include "gpu_mem.hpp"
#include "gpu.hpp"
#endif

using namespace std;

#define WK_CLINE 64

enum MemTypes { GPU_DRAM = 1, CPU_DRAM };

// The communication over RDMA-based ring buffer
class RDMA_Adaptor {
private:
    int sid;
    int num_servers;
    int num_threads;
    Mem *mem;
    #ifdef USE_GPU
    GPUMem *gmem;
    #endif

    /// The ring-buffer space contains #threads logical-queues.
    /// Each logical-queue contains #servers physical queues (ring-buffer).
    /// The X physical-queue (ring-buffer) of thread(tid) is written by the responding threads
    /// (proxies and engine with the same "tid") on the X server.
    /// Access mode of physical queue is N writers (from the same server) and 1 reader.

    // track tail of ring buffer for writer
    struct rbf_rmeta_t {
        uint64_t tail;
        pthread_spinlock_t lock;
    } __attribute__ ((aligned (WK_CLINE)));

    // track head of ring buffer for reader
    struct rbf_lmeta_t {
        uint64_t head;
        pthread_spinlock_t lock;
    } __attribute__ ((aligned (WK_CLINE)));

    rbf_rmeta_t *rmetas = NULL;
    rbf_lmeta_t *lmetas = NULL;

    // each thread uses a round-robin strategy to check its physical-queues
    struct scheduler_t {
        uint64_t rr_cnt; // round-robin
    } __attribute__ ((aligned (WK_CLINE)));

    scheduler_t *schedulers;

    // Align given value down to given alignment
    uint64_t inline floor(uint64_t val, uint64_t alignment) {
        ASSERT(alignment != 0);
        return val - val % alignment;
    }

    // Align given value up to given alignment
    uint64_t inline ceil(uint64_t val, uint64_t alignment) {
        ASSERT(alignment != 0);
        if (val % alignment == 0)
            return val;
        return val - val % alignment + alignment;
    }

    // Check if there is data from threads in dst_sid to tid
    bool check(int tid, int dst_sid) {
        rbf_lmeta_t *lmeta = &lmetas[tid * num_servers + dst_sid];
        char *rbf = mem->ring(tid, dst_sid); // ring buffer for tid to recv data from threads in dst_sid
        uint64_t rbf_sz = mem->ring_size();
        volatile uint64_t data_sz = *(volatile uint64_t *)(rbf + lmeta->head % rbf_sz);  // header

        return (data_sz != 0);
    }

    /* Fetch data from threads in dst_sid to tid
     * for GPU, the data will be copied to GPU mem
     */
    bool fetch(int tid, int dst_sid, string &result, MemTypes memtype) {
        // step 1: get lmeta of dst rbf
        rbf_lmeta_t *lmeta = &lmetas[tid * num_servers + dst_sid];
        // step 2: calculate mem location and data size
        char * rbf = mem->ring(tid, dst_sid);
        uint64_t rbf_sz = mem->ring_size();
        // struct of data: [size | data | size]
        volatile uint64_t data_sz = *(volatile uint64_t *)(rbf + lmeta->head % rbf_sz);  // header
        *(uint64_t *)(rbf + lmeta->head % rbf_sz) = 0;  // clean header

        uint64_t to_footer = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t));
        volatile uint64_t * footer = (volatile uint64_t *)(rbf + (lmeta->head + to_footer) % rbf_sz); // footer
        // step 3: spin-wait rdma-write done
        while (*footer != data_sz) {
            _mm_pause();
            /* If RDMA-WRITE is done, footer == header == size
             * Otherwise, footer == 0
             */
            ASSERT(*footer == 0 || *footer == data_sz);
        }
        *footer = 0;  // clean footer

        // step 4: actually read data
        uint64_t start = (lmeta->head + sizeof(uint64_t)) % rbf_sz; // start offset of data
        uint64_t end = (lmeta->head + sizeof(uint64_t) + data_sz) % rbf_sz;  // end offset of data

        result.reserve(data_sz);
        if (memtype == GPU_DRAM) {
            #ifdef USE_GPU
            GPU &gpu = GPU::instance();
            char *history_buf = gpu.history_inbuf();
            if (start < end) {
                CUDA_ASSERT( cudaMemcpy(history_buf, rbf + start, data_sz, cudaMemcpyHostToDevice) );
                memset(rbf + start, 0, ceil(data_sz, sizeof(uint64_t)));  // clean data
            } else {
                CUDA_ASSERT( cudaMemcpy(history_buf, rbf + start, data_sz - end, cudaMemcpyHostToDevice) );
                CUDA_ASSERT( cudaMemcpy(history_buf + (data_sz - end), rbf, end, cudaMemcpyHostToDevice) );
                memset(rbf + start, 0, data_sz - end);                    // clean data
                memset(rbf, 0, ceil(end, sizeof(uint64_t)));              // clean data
            }
            gpu.set_history_size(data_sz / sizeof(sid_t));
            // caution: after copying data to gpu mem, the caller need to set r.result.gpu_history_ptr, r.result.gpu_history_size
            # else
            logstream(LOG_ERROR) << "USE_GPU is undefined. Memtype should not be GPU_DRAM." << LOG_endl;
            ASSERT(false);
            #endif
        } else {
            if (start < end) {
                result.append(rbf + start, data_sz);
                memset(rbf + start, 0, ceil(data_sz, sizeof(uint64_t)));  // clean data
            } else { // overwrite from the start
                result.append(rbf + start, data_sz - end);
                result.append(rbf, end);
                memset(rbf + start, 0, data_sz - end);                    // clean data
                memset(rbf, 0, ceil(end, sizeof(uint64_t)));              // clean data
            }
        }
        // step 5: move forward rbf head
        lmeta->head += 2 * sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t));

        // step 6: update heads of ring buffer to writer to help it detect overflow
        char *head = mem->local_ring_head(tid, dst_sid);
        const uint64_t threshold = rbf_sz / 8;
        if (lmeta->head - * (uint64_t *)head > threshold) {
            *(uint64_t *)head = lmeta->head;
            // update to remote server
            if (sid != dst_sid) {
                RDMA &rdma = RDMA::get_rdma();
                uint64_t remote_head = mem->remote_ring_head_offset(tid, sid);
                rdma.dev->RdmaWrite(tid, dst_sid, head, mem->remote_ring_head_size(), remote_head);
            } else {
                *(uint64_t *)mem->remote_ring_head(tid, sid) = lmeta->head;
            }
        }
        return true;
    } // end of fetch

    /* Check whether overflow occurs if given msg is sent
     * @tid tid of writer
     * @dst_sid, @dst_tid sid and tid of reader
     * @msg_sz size of msg to send
    */
    inline bool rbf_full(int tid, int dst_sid, int dst_tid, uint64_t msg_sz) {
        uint64_t rbf_sz = mem->ring_size();
        uint64_t tail = rmetas[dst_sid * num_threads + dst_tid].tail;
        uint64_t head = *(uint64_t *)mem->remote_ring_head(dst_tid, dst_sid);

        return (rbf_sz < (tail - head + msg_sz));
    }

public:
    bool init = false;

    #ifdef USE_GPU
    RDMA_Adaptor(int sid, Mem *mem, GPUMem *gmem, int num_servers, int num_threads)
        : sid(sid), mem(mem), gmem(gmem), num_servers(num_servers), num_threads(num_threads) {
    #else
    RDMA_Adaptor(int sid, Mem *mem, int num_servers, int num_threads)
        : sid(sid), mem(mem), num_servers(num_servers), num_threads(num_threads) {
    #endif
        // no RDMA device
        if (!RDMA::get_rdma().has_rdma()) return;

        // init the metadata of remote and local ring-buffers
        int nrbfs = num_servers * num_threads;

        rmetas = (rbf_rmeta_t *)malloc(sizeof(rbf_rmeta_t) * nrbfs);
        memset(rmetas, 0, sizeof(rbf_rmeta_t) * nrbfs);
        for (int i = 0; i < nrbfs; i++) {
            rmetas[i].tail = 0;
            pthread_spin_init(&rmetas[i].lock, 0);
        }

        lmetas = (rbf_lmeta_t *)malloc(sizeof(rbf_lmeta_t) * nrbfs);
        memset(lmetas, 0, sizeof(rbf_lmeta_t) * nrbfs);
        for (int i = 0; i < nrbfs; i++) {
            lmetas[i].head = 0;
            pthread_spin_init(&lmetas[i].lock, 0);
        }

        schedulers = (scheduler_t *)malloc(sizeof(scheduler_t) * num_threads);
        memset(schedulers, 0, sizeof(scheduler_t) * num_threads);

        init = true;
    }

    ~RDMA_Adaptor() { }  //TODO

    // Send given string to (dst_sid, dst_tid) by thread(tid)
    // Return false if failed . Otherwise, return true.
    bool send(int tid, int dst_sid, int dst_tid, const string &str) {
        ASSERT(init);

        rbf_rmeta_t *rmeta = &rmetas[dst_sid * num_threads + dst_tid];
        uint64_t rbf_sz = mem->ring_size();

        // struct of data: [size | data | size] (use size of bundle as header and footer)
        uint64_t msg_sz = sizeof(uint64_t) + ceil(str.length(), sizeof(uint64_t)) + sizeof(uint64_t);

        ASSERT(msg_sz < rbf_sz);

        pthread_spin_lock(&rmeta->lock);
        if (rbf_full(tid, dst_sid, dst_tid, msg_sz)) { // detect overflow
            pthread_spin_unlock(&rmeta->lock);
            return false;
        }

        if (sid == dst_sid) { // local physical-queue
            uint64_t off = rmeta->tail;
            rmeta->tail += msg_sz;
            pthread_spin_unlock(&rmeta->lock);

            // write msg to the local physical-queue
            char *ptr = mem->ring(dst_tid, sid);

            *((uint64_t *)(ptr + off % rbf_sz)) = str.length();       // header
            off += sizeof(uint64_t);

            if (off / rbf_sz == (off + str.length() - 1) / rbf_sz ) {    // data
                memcpy(ptr + (off % rbf_sz), str.c_str(), str.length());
            } else {
                uint64_t _sz = rbf_sz - (off % rbf_sz);
                memcpy(ptr + (off % rbf_sz), str.c_str(), _sz);
                memcpy(ptr, str.c_str() + _sz, str.length() - _sz);
            }
            off += ceil(str.length(), sizeof(uint64_t));

            *((uint64_t *)(ptr + off % rbf_sz)) = str.length();       // footer
        } else { // remote physical-queue
            uint64_t off = rmeta->tail;
            rmeta->tail += msg_sz;
            pthread_spin_unlock(&rmeta->lock);

            // prepare RDMA buffer for RDMA-WRITE
            char *rdma_buf = mem->buffer(tid);
            uint64_t buf_sz = mem->buffer_size();
            ASSERT(msg_sz < buf_sz); // enough space to buffer the msg

            *((uint64_t *)rdma_buf) = str.length();  // header
            rdma_buf += sizeof(uint64_t);

            memcpy(rdma_buf, str.c_str(), str.length());    // data
            rdma_buf += ceil(str.length(), sizeof(uint64_t));
            *((uint64_t*)rdma_buf) = str.length();   // footer

            // write msg to the remote physical-queue
            RDMA &rdma = RDMA::get_rdma();
            uint64_t rdma_off = mem->ring_offset(dst_tid, sid);
            if (off / rbf_sz == (off + msg_sz - 1) / rbf_sz ) {
                rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid), msg_sz, rdma_off + (off % rbf_sz));
            } else {
                uint64_t _sz = rbf_sz - (off % rbf_sz);
                rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid), _sz, rdma_off + (off % rbf_sz));
                rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid) + _sz, msg_sz - _sz, rdma_off);
            }
        }

        return true;
    } // end of send

    string recv(int tid) {
        ASSERT(init);

        while (true) {
            // each thread has a logical-queue (#servers physical-queues)
            int dst_sid = (schedulers[tid].rr_cnt++) % num_servers; // round-robin
            if (check(tid, dst_sid)) {
                string str;
                bool ret = fetch(tid, dst_sid, str, CPU_DRAM);
                assert(ret == true);
                return str;
            }
        }
    }

    // Try to recv data of given thread
    bool tryrecv(int tid, int &dst_sid_out, string &str) {
        ASSERT(init);

        // check all physical-queues of tid once
        for (int dst_sid = 0; dst_sid < num_servers; dst_sid++) {
            if (check(tid, dst_sid)) {
                dst_sid_out = dst_sid;
                return fetch(tid, dst_sid, str, CPU_DRAM);
            }
        }
        return false;
    }

    #ifdef USE_GPU
    // recv the data from specified rbf and copy to gpu mem, return the history_size
    int recv_by_gpu(int tid, int dst_sid, string &str) {
        while (true) {
            if (check(tid, dst_sid)) {
                bool ret;
                ret = fetch(tid, dst_sid, str, GPU_DRAM);
                assert(ret == true);
                return GPU::instance().history_size();
            }
        }
        return -1;
    }

    // for adapter sending split query
    bool send_split(int tid, int dst_sid, int dst_tid, const string &ctrl, const char *data, uint64_t data_sz) {
        // step1: get rmeta of dst rbf
        rbf_rmeta_t *rmeta = &rmetas[dst_sid * num_threads + dst_tid];
        pthread_spin_lock(&rmeta->lock);
        // step2: calculate msg size [data_sz | (type_sz) | data | data_sz]
        uint64_t ctrl_sz = ctrl.length();
        uint64_t ctrl_msg_sz = sizeof(uint64_t) + ceil(ctrl_sz, sizeof(uint64_t)) + sizeof(uint64_t);
        uint64_t data_msg_sz = sizeof(uint64_t) + sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t)) + sizeof(uint64_t);

        // return false if rbf is full
        if (rbf_full(tid, dst_sid, dst_tid, ctrl_msg_sz + data_msg_sz)) {
            pthread_spin_unlock(&rmeta->lock);
            return false;
        }

        // step3: send ctrl object
        uint64_t rbf_sz = mem->ring_size();
        uint64_t off = rmeta->tail;

        if (sid == dst_sid) { // local physical-queue
            // write msg to the local physical-queue
            char *ptr = mem->ring(dst_tid, sid);
            *((uint64_t *)(ptr + off % rbf_sz)) = ctrl_sz;       // header
            off += sizeof(uint64_t);

            if (off / rbf_sz == (off + ctrl_sz - 1) / rbf_sz ) { // data
                memcpy(ptr + (off % rbf_sz), ctrl.c_str(), ctrl_sz);
            } else {
                uint64_t _sz = rbf_sz - (off % rbf_sz);
                memcpy(ptr + (off % rbf_sz), ctrl.c_str(), _sz);
                memcpy(ptr, ctrl.c_str() + _sz, ctrl_sz - _sz);
            }
            off += ceil(ctrl_sz, sizeof(uint64_t));

            *((uint64_t *)(ptr + off % rbf_sz)) = ctrl_sz;       // footer
        } else { // remote physical-queue
            // prepare RDMA buffer for RDMA-WRITE
            char *rdma_buf = mem->buffer(tid);
            *((uint64_t *)rdma_buf) = ctrl_sz;  // header
            rdma_buf += sizeof(uint64_t);

            memcpy(rdma_buf, ctrl.c_str(), ctrl_sz);    // data
            rdma_buf += ceil(ctrl_sz, sizeof(uint64_t));

            *((uint64_t*)rdma_buf) = ctrl_sz;   // footer

            // write msg to the remote physical-queue
            RDMA &rdma = RDMA::get_rdma();
            uint64_t rdma_off = mem->ring_offset(dst_tid, sid);
            if (off / rbf_sz == (off + ctrl_msg_sz - 1) / rbf_sz ) {
                rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid), ctrl_msg_sz, rdma_off + (off % rbf_sz));
            } else {
                uint64_t _sz = rbf_sz - (off % rbf_sz);
                rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid), _sz, rdma_off + (off % rbf_sz));
                rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid) + _sz, ctrl_msg_sz - _sz, rdma_off);
            }
        }
        rmeta->tail += ctrl_msg_sz;
        // step4: send history
        gdr_send(tid, dst_sid, dst_tid, data, data_sz, rmeta->tail);
        rmeta->tail += data_msg_sz;

        pthread_spin_unlock(&rmeta->lock);
        return true;
    }

    // GPUDirect send, from local GPU mem to remote CPU mem (remote rbf)
    void gdr_send(int tid, int dst_sid, int dst_tid, const char *data, uint64_t data_sz, uint64_t offset) {
        uint64_t rbf_sz = mem->ring_size();
        uint64_t off = offset;
        // msg: header + data + footer (use bundle_sz as header and footer)
        uint64_t bundle_sz = sizeof(uint64_t) + data_sz;
        uint64_t msg_sz = sizeof(uint64_t) + ceil(bundle_sz, sizeof(uint64_t)) + sizeof(uint64_t);
        ASSERT(msg_sz < rbf_sz);
        // must send to remote host
        ASSERT(sid != dst_sid);
        // prepare RDMA buffer for RDMA-WRITE
        char *rdma_buf = gmem->buffer(tid);
        // copy header(bundle_sz) to rdma_buf(on local GPU mem)
        CUDA_ASSERT( cudaMemcpy(rdma_buf, &bundle_sz, sizeof(uint64_t), cudaMemcpyHostToDevice) );
        rdma_buf += sizeof(uint64_t);

        // copy type
        uint64_t msg_type = SPARQL_HISTORY;
        CUDA_ASSERT( cudaMemcpy(rdma_buf, &msg_type, sizeof(uint64_t), cudaMemcpyHostToDevice) );
        rdma_buf += sizeof(uint64_t);

        // copy data(on local GPU mem) to rdma_buf(on local GPU mem)
        CUDA_ASSERT( cudaMemcpy(rdma_buf, data, data_sz, cudaMemcpyDeviceToDevice) );    // data
        rdma_buf += ceil(data_sz, sizeof(uint64_t));

        // copy footer(bundle_sz) to rdma_buf(on local GPU mem)
        CUDA_ASSERT( cudaMemcpy(rdma_buf, &bundle_sz, sizeof(uint64_t), cudaMemcpyHostToDevice) );  // footer

        // write msg to the remote physical-queue
        RDMA &rdma = RDMA::get_rdma();
        uint64_t rdma_off = mem->ring_offset(dst_tid, sid);

        if (off / rbf_sz == (off + msg_sz - 1) / rbf_sz ) {
            rdma.dev->GPURdmaWrite(tid, dst_sid, gmem->buffer(tid), msg_sz, rdma_off + (off % rbf_sz));
        } else {
            uint64_t _sz = rbf_sz - (off % rbf_sz);
            rdma.dev->GPURdmaWrite(tid, dst_sid, gmem->buffer(tid), _sz, rdma_off + (off % rbf_sz));
            rdma.dev->GPURdmaWrite(tid, dst_sid, gmem->buffer(tid) + _sz, msg_sz - _sz, rdma_off);
        }
    }

    // send message from gpu to remote ring buffer
    bool send_device2host(int tid, int dst_sid, int dst_tid, const char *data, uint64_t data_sz) {
        // step1: get rmeta of dst rbf
        rbf_rmeta_t *rmeta = &rmetas[dst_sid * num_threads + dst_tid];
        pthread_spin_lock(&rmeta->lock);
        // step2: calculate msg size [data_sz | data | data_sz]
        uint64_t data_msg_sz = sizeof(uint64_t) + sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t)) + sizeof(uint64_t);

        // return false if rbf is full
        if (rbf_full(tid, dst_sid, dst_tid, data_msg_sz)) {
            pthread_spin_unlock(&rmeta->lock);
            return false;
        }
        // step3: send data
        gdr_send(tid, dst_sid, dst_tid, data, data_sz, rmeta->tail);
        rmeta->tail += data_msg_sz;

        pthread_spin_unlock(&rmeta->lock);
        return true;
    }
    #endif
};
