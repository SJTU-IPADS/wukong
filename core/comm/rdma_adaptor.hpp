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
#include <assert.h>

#include "global.hpp"
#include "rdma.hpp"
#include "mem.hpp"

#ifdef USE_GPU
#include "gpu/gpu_mem.hpp"
#include "gpu.hpp"
#endif

// utils
#include "assertion.hpp"
#include "atomic.hpp"

using namespace std;

#define WK_CLINE 64

// The communication over RDMA-based ring buffer
class RDMA_Adaptor {
private:

    int sid;
    int num_servers;
    int num_threads;

    Mem *mem = nullptr;   // (Host) CPU memory
#ifdef USE_GPU
    GPUMem *gmem = nullptr; // (Device) GPU memory
#endif

    /// The ring-buffer space contains #threads logical-queues.
    /// Each logical-queue contains #servers physical queues (ring-buffer).
    /// The X physical-queue (ring-buffer) of thread(tid) is written by the responding threads
    /// (proxies and engine with the same "tid") on the X server.
    /// Access mode of physical queue is N writers (from the same server) and 1 reader.

    // track tail of ring buffer for remote writer
    struct rbf_rmeta_t {
        uint64_t tail;
        pthread_spinlock_t lock;
    } __attribute__ ((aligned (WK_CLINE)));

    // track head of ring buffer for local reader
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
    uint64_t check(int tid, int dst_sid) {
        rbf_lmeta_t *lmeta = &lmetas[tid * num_servers + dst_sid];
        char *rbf = mem->ring(tid, dst_sid); // ring buffer for tid to recv data from threads in dst_sid
        uint64_t rbf_sz = mem->ring_size();
        return *(volatile uint64_t *)(rbf + lmeta->head % rbf_sz);  // header (data size)
    }

    // Fetch data from threads in dst_sid to tid
    bool fetch(int tid, int dst_sid, std::string &data, uint64_t data_sz) {
        // 1. validate and acquire the message
        char * rbf = mem->ring(tid, dst_sid);
        uint64_t rbf_sz = mem->ring_size();

        // layout of msg: [size | data | size]
        rbf_lmeta_t *lmeta = &lmetas[tid * num_servers + dst_sid];
        uint64_t *head_ptr = (uint64_t *)(rbf + lmeta->head % rbf_sz);

        // validate: data_sz is not changed; acquire: zeroing the size in header
        // (NOTE: data_sz is read in check())
        if (wukong::atomic::compare_and_swap(head_ptr, data_sz, 0) != data_sz)
            return false; // msg has been acquired by another concurrent engine

        // 2. wait the entire msg has been written
        uint64_t to_footer = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t));
        volatile uint64_t * footer = (volatile uint64_t *)(rbf + (lmeta->head + to_footer) % rbf_sz); // footer

        // spin-wait RDMA WRITE done
        while (*footer != data_sz) {
            _mm_pause();
            // If RDMA-WRITE is done, then footer == header == size. Otherwise, footer == 0
            ASSERT(*footer == 0 || *footer == data_sz);
        }
        *footer = 0;  // clean footer

        // 3. actually fetch data
        uint64_t start = (lmeta->head + sizeof(uint64_t)) % rbf_sz; // start of data
        uint64_t end = (lmeta->head + sizeof(uint64_t) + data_sz) % rbf_sz;  // end of data
        if (start < end) {
            data.append(rbf + start, data_sz);
            memset(rbf + start, 0, ceil(data_sz, sizeof(uint64_t)));  // clean data
        } else { // overwrite from the start
            data.append(rbf + start, data_sz - end);
            data.append(rbf, end);
            memset(rbf + start, 0, data_sz - end);                    // clean data
            memset(rbf, 0, ceil(end, sizeof(uint64_t)));              // clean data
        }

        // 4. notify sender the header of ring buffer (detect overflow)
        uint64_t real_head = lmeta->head + 2 * sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t));
        char *head = mem->local_ring_head(tid, dst_sid);
        const uint64_t threshold = rbf_sz / 8; // a threshold for lazy update

        if (real_head - * (uint64_t *)head > threshold) {
            // remote.is_queue_full may be false positive since remote use old value of head
            *(uint64_t *)head = real_head; // update local ring head
            if (sid != dst_sid) {  // update remote ring head via RDMA
                RDMA &rdma = RDMA::get_rdma();
                uint64_t remote_head = mem->remote_ring_head_offset(tid, sid);
                rdma.dev->RdmaWrite(tid, dst_sid, head, mem->remote_ring_head_size(), remote_head);
            } else { // direct update remote ring head
                *(uint64_t *)mem->remote_ring_head(tid, sid) = real_head;
            }
        }

        // 5. update the metadata of ring buffer (done)
        lmeta->head += 2 * sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t));

        return true;
    } // end of fetch

    /* Check whether overflow occurs if given msg is sent
     * @tid tid of writer
     * @dst_sid, @dst_tid sid and tid of reader
     * @msg_sz size of msg to send
    */
    inline bool rbf_full(int tid, int dst_sid, int dst_tid, uint64_t msg_sz) {
        uint64_t rbf_sz = mem->ring_size();
        // tail of remote queue can access via rmeta
        uint64_t tail = rmetas[dst_sid * num_threads + dst_tid].tail;
        // head of remote queue must read from remote ring head
        // since lmeta is used to check head of local queue
        uint64_t head = *(uint64_t *)mem->remote_ring_head(dst_tid, dst_sid);

        return (rbf_sz < (tail - head + msg_sz));
    }

    void native_send(int tid, const char *data, uint64_t data_sz,
                     int dst_sid, int dst_tid, uint64_t off, uint64_t sz) {
        if (sid == dst_sid) {                                    // send to local server
            // write msg to local ring buffer
            char *ptr = mem->ring(dst_tid, sid);
            uint64_t rbf_sz = mem->ring_size();
            ASSERT(sz < rbf_sz); // enough space (remote ring buffer)

            *((uint64_t *)(ptr + off % rbf_sz)) = data_sz;       // header

            off += sizeof(uint64_t);
            if (off / rbf_sz == (off + data_sz - 1) / rbf_sz ) { // data
                memcpy(ptr + (off % rbf_sz), data, data_sz);
            } else {
                uint64_t _sz = rbf_sz - (off % rbf_sz);
                memcpy(ptr + (off % rbf_sz), data, _sz);
                memcpy(ptr, data + _sz, data_sz - _sz);
            }

            off += ceil(data_sz, sizeof(uint64_t));
            *((uint64_t *)(ptr + off % rbf_sz)) = data_sz;       // footer
        } else {                                                 // send to remote server
            // copy msg to local RDMA buffer
            uint64_t buf_sz = mem->buffer_size();
            ASSERT(sz < buf_sz); // enough space (local RDMA buffer)

            char *rdma_buf = mem->buffer(tid);
            *((uint64_t *)rdma_buf) = data_sz;                   // header

            rdma_buf += sizeof(uint64_t);
            memcpy(rdma_buf, data, data_sz);                     // data

            rdma_buf += ceil(data_sz, sizeof(uint64_t));
            *((uint64_t*)rdma_buf) = data_sz;                    // footer


            // write msg to remote ring buffer
            uint64_t rbf_sz = mem->ring_size();
            ASSERT(sz < rbf_sz); // enough space (remote ring buffer)

            RDMA &rdma = RDMA::get_rdma();
            uint64_t rdma_off = mem->ring_offset(dst_tid, sid);
            if (off / rbf_sz == (off + sz - 1) / rbf_sz) {
                rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid), sz, rdma_off + (off % rbf_sz));
            } else {
                uint64_t _sz = rbf_sz - (off % rbf_sz);
                rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid), _sz, rdma_off + (off % rbf_sz));
                rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid) + _sz, sz - _sz, rdma_off);
            }
        }
    }

#ifdef USE_GPU
    // GPUDirect send, from local GPU mem to remote CPU mem (remote rbf)
    void gdr_send(int tid, const char *data, uint64_t data_sz,
                  int dst_sid, int dst_tid, uint64_t off) {
        // TODO: only support send local data (GPU) to ring buffer on remote host (CPU)
        ASSERT(sid != dst_sid);

        // msg: header + data + footer (use data_sz as header and footer)
        uint64_t sz = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t)) + sizeof(uint64_t);

        // prepare RDMA buffer for RDMA-WRITE
        // copy header(data_sz) to rdma_buf(on local GPU mem)
        CUDA_ASSERT( cudaMemcpy(gmem->rdma_buf_hdr(tid), &data_sz, sizeof(uint64_t), cudaMemcpyHostToDevice) );

        char *rdma_buf = gmem->rdma_buf_body(tid);
        // copy data(on local GPU mem) to rdma_buf_body(on local GPU mem)
        CUDA_ASSERT( cudaMemcpy(rdma_buf, data, data_sz, cudaMemcpyDeviceToDevice) );    // data
        rdma_buf += ceil(data_sz, sizeof(uint64_t));

        // copy footer(data_sz) to rdma_buf(on local GPU mem)
        CUDA_ASSERT( cudaMemcpy(rdma_buf, &data_sz, sizeof(uint64_t), cudaMemcpyHostToDevice) );  // footer

        // for safety
        CUDA_DEVICE_SYNC;

        // write msg to remote ring buffer (CPU)
        uint64_t rbf_sz = mem->ring_size();
        ASSERT(sz < rbf_sz); // enough space (remote ring buffer)

        RDMA &rdma = RDMA::get_rdma();
        uint64_t rdma_off = mem->ring_offset(dst_tid, sid);

        if (off / rbf_sz == (off + sz - 1) / rbf_sz ) {
            rdma.dev->GPURdmaWrite(tid, dst_sid, gmem->rdma_buf_hdr(tid), sz, rdma_off + (off % rbf_sz));
        } else {
            uint64_t _sz = rbf_sz - (off % rbf_sz);
            rdma.dev->GPURdmaWrite(tid, dst_sid, gmem->rdma_buf_hdr(tid), _sz, rdma_off + (off % rbf_sz));
            rdma.dev->GPURdmaWrite(tid, dst_sid, gmem->rdma_buf_hdr(tid) + _sz, sz - _sz, rdma_off);
        }
    }
#endif // end of USE_GPU

public:
    bool init = false;

    RDMA_Adaptor(int sid, vector<RDMA::MemoryRegion> &mrs, int nsrvs, int nthds)
        : sid(sid), num_servers(nsrvs), num_threads(nthds) {
        // no RDMA device
        if (!RDMA::get_rdma().has_rdma()) return;

        // init memory regions
        assert(mrs.size() <= 2); // only support at most one CPU memory region and one GPU memory region
        for (auto mr : mrs) {
            switch (mr.type) {
            case RDMA::MemType::CPU:
                mem = (Mem *)mr.mem;
                break;
            case RDMA::MemType::GPU:
#ifdef USE_GPU
                gmem = (GPUMem *)mr.mem;
                break;
#else
                logstream(LOG_ERROR) << "Build wukong w/o GPU support." << LOG_endl;
                ASSERT(false);
#endif
            }
        }

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

#ifdef USE_GPU
    bool send_dev2host(int tid, int dst_sid, int dst_tid, char *data, uint64_t data_sz) {
        ASSERT(init);

        // 1. calculate msg size
        // struct of msg: [data_sz | data | data_sz] (use size of data as header and footer)
        uint64_t msg_sz = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t)) + sizeof(uint64_t);

        // 2. reserve space in ring-buffer
        rbf_rmeta_t *rmeta = &rmetas[dst_sid * num_threads + dst_tid];

        pthread_spin_lock(&rmeta->lock);
        if (rbf_full(tid, dst_sid, dst_tid, msg_sz)) { // detect overflow
            pthread_spin_unlock(&rmeta->lock);
            return false; // return false if rbf is full
        }
        uint64_t off = rmeta->tail;
        rmeta->tail += msg_sz;
        pthread_spin_unlock(&rmeta->lock);


        // 3. (real) send data
        // local data:  <tid, data, data_sz>; remote buffer: <dst_sid, dst_tid, off, msg_sz>
        gdr_send(tid, data, data_sz, dst_sid, dst_tid, off);

        return true;
    }
#endif

    // Send given string to (dst_sid, dst_tid) by thread(tid)
    // Return false if failed . Otherwise, return true.
    bool send(int tid, int dst_sid, int dst_tid, const string &str) {
        ASSERT(init);

        const char *data = str.c_str();
        uint64_t data_sz = str.length();

        // 1. calculate msg size
        // struct of msg: [size | data | size] (use size of data as header and footer)
        uint64_t msg_sz = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t)) + sizeof(uint64_t);


        // 2. reserve space in ring-buffer
        rbf_rmeta_t *rmeta = &rmetas[dst_sid * num_threads + dst_tid];

        pthread_spin_lock(&rmeta->lock);
        if (rbf_full(tid, dst_sid, dst_tid, msg_sz)) { // detect overflow
            pthread_spin_unlock(&rmeta->lock);

            //avoid not enough space (remote ring buffer)
            //without this, it will always fail, and retry,
            //which will cause dead loop
            uint64_t rbf_sz = mem->ring_size();
            if ( msg_sz > rbf_sz ) {
                logstream(LOG_ERROR) << "Size of ring buffer is smaller than message, please check your configure" << LOG_endl;
                ASSERT(false);
            }

            // enough space (remote ring buffer)
            return false;
        }
        uint64_t off = rmeta->tail;
        rmeta->tail += msg_sz;
        pthread_spin_unlock(&rmeta->lock);


        // 3. (real) send data
        // local data:  <tid, data, data_sz>; remote buffer: <dst_sid, dst_tid, off, msg_sz>
        native_send(tid, data, data_sz, dst_sid, dst_tid, off, msg_sz);

        return true;
    }

    std::string recv(int tid) {
        ASSERT(init);

        while (true) {
            // each thread has a logical-queue (#servers physical-queues)
            int src_sid = (schedulers[tid].rr_cnt++) % num_servers; // round-robin
            uint64_t data_sz = check(tid, src_sid);
            if (data_sz != 0) {
                std::string data;
                if (fetch(tid, src_sid, data, data_sz))
                    return data;
            }
        }
    }

    std::string recv(int tid, int src_sid) {
        ASSERT(init);
        ASSERT(src_sid >= 0);

        while (true) {
            // each thread has a logical-queue (#servers physical-queues)
            uint64_t data_sz = check(tid, src_sid);
            if (data_sz != 0) {
                std::string data;
                if (fetch(tid, src_sid, data, data_sz))
                    return data;
            }
        }
    }

    // try to recv data of given thread
    bool tryrecv(int tid, std::string &data) {
        ASSERT(init);

        // check all physical-queues of tid once
        for (int sid = 0; sid < num_servers; sid++) {
            uint64_t data_sz = check(tid, sid);
            if (data_sz != 0)
                if (fetch(tid, sid, data, data_sz))
                    return true;
        }

        return false;
    }

    // try to recv data of given thread and retrieve the server ID
    bool tryrecv(int tid, std::string &data, int &src_sid) {
        ASSERT(init);

        // check all physical-queues of tid once
        for (int sid = 0; sid < num_servers; sid++) {
            uint64_t data_sz = check(tid, sid);
            if (data_sz != 0) {
                src_sid = sid;
                if (fetch(tid, sid, data, data_sz))
                    return true;
            }
        }

        return false;
    }
};
