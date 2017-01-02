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

#include <string>
#include <iostream>
#include <unistd.h>
#include <unordered_map>
#include <fstream>
#include <errno.h>
#include <sstream>

#include "config.hpp"
#include "rdma_resource.hpp"

using namespace std;

#define WK_CLINE 64

/**
 * The communication over RDMA-based logical queue
 */
class RDMA_Adaptor {
private:
    Mem *mem;

    int sid;
    int num_nodes;
    int num_threads;

    char *rdma_mem;
    uint64_t rbf_sz;  // split a logical-queue into num_servers physical queues

    struct scheduler_t {
        uint64_t rr_cnt; // round-robin
    } __attribute__ ((aligned (WK_CLINE)));

    scheduler_t *schedulers;

    struct rbf_rmeta_t {
        uint64_t tail; // write from here
        pthread_spinlock_t lock;
    } __attribute__ ((aligned (WK_CLINE)));

    struct rbf_lmeta_t {
        uint64_t head; // read from here
        pthread_spinlock_t lock;
    } __attribute__ ((aligned (WK_CLINE)));

    rbf_rmeta_t *rmetas;
    rbf_lmeta_t *lmetas;


    uint64_t inline floor(uint64_t original, uint64_t n) {
        assert(n != 0);
        return original - original % n;
    }

    uint64_t inline ceil(uint64_t original, uint64_t n) {
        assert(n != 0);
        if (original % n == 0)
            return original;
        return original - original % n + n;
    }

public:
    RDMA_Adaptor(int sid, Mem *mem, int num_nodes, int num_threads)
        : sid(sid), mem(mem), num_nodes(num_nodes), num_threads(num_threads) {

        rdma_mem = mem->kvstore();
        rbf_sz = mem->ring_size();

        schedulers = (scheduler_t *)malloc(sizeof(scheduler_t) * num_threads);
        memset(schedulers, 0, sizeof(scheduler_t) * num_threads);

        // init the metadata of remote and local ring-buffers
        int nrbfs = num_nodes * num_threads;

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
    }

    ~RDMA_Adaptor() { }

    void send(int local_tid, int remote_mid, int remote_tid, const char *start, uint64_t size) {
        // msg: header + string + footer (use size as header and footer)
        rbf_rmeta_t *rmeta = &rmetas[remote_mid * num_threads + remote_tid];

        pthread_spin_lock(&rmeta->lock);
        uint64_t remote_off = mem->ring_offset(remote_tid, sid);
        if (sid == remote_mid) {  // MT
            char *ptr = rdma_mem + remote_off;
            uint64_t tail = rmeta->tail;
            rmeta->tail += sizeof(uint64_t) * 2 + ceil(size, sizeof(uint64_t));
            pthread_spin_unlock(&rmeta->lock);

            // write msg to physical queue
            *((uint64_t*)(ptr + (tail) % rbf_sz)) = size;
            tail += sizeof(uint64_t);
            for (uint64_t i = 0; i < size; i++)
                *(ptr + (tail + i) % rbf_sz) = start[i];
            tail += ceil(size, sizeof(uint64_t));
            *((uint64_t*)(ptr + (tail) % rbf_sz)) = size;
        } else {
            uint64_t total_write_size = sizeof(uint64_t) * 2 + ceil(size, sizeof(uint64_t));

            char* local_buffer = mem->buffer(local_tid);
            *((uint64_t*)local_buffer) = size;
            local_buffer += sizeof(uint64_t);
            memcpy(local_buffer, start, size);
            local_buffer += ceil(size, sizeof(uint64_t));
            *((uint64_t*)local_buffer) = size;

            uint64_t tail = rmeta->tail;
            rmeta->tail += total_write_size;
            pthread_spin_unlock(&rmeta->lock);

            /// TODO: check the overflow of physical queue
            assert(total_write_size < rbf_sz);
            RDMA &rdma = RDMA::get_rdma();
            if (tail / rbf_sz == (tail + total_write_size - 1) / rbf_sz ) {
                uint64_t remote_msg_offset = remote_off + (tail % rbf_sz);
                rdma.dev->RdmaWrite(local_tid, remote_mid, mem->buffer(local_tid), total_write_size, remote_msg_offset);
            } else {
                uint64_t first = rbf_sz - (tail % rbf_sz);
                uint64_t second = total_write_size - first;
                uint64_t first_off = remote_off + (tail % rbf_sz);
                uint64_t second_off = remote_off;
                rdma.dev->RdmaWrite(local_tid, remote_mid, mem->buffer(local_tid), first, first_off);
                rdma.dev->RdmaWrite(local_tid, remote_mid, mem->buffer(local_tid) + first, second, second_off);
            }
        }
    }

    bool check_rbf_msg(int local_tid, int mid) {
        rbf_lmeta_t *lmeta = &lmetas[local_tid * num_nodes + mid];
        char *rbf_ptr = rdma_mem + mem->ring_offset(local_tid, mid);

        volatile uint64_t msg_size = *(volatile uint64_t *)(rbf_ptr + lmeta->head % rbf_sz);
        //uint64_t skip_size = sizeof(uint64_t) + ceil(msg_size, sizeof(uint64_t));
        //volatile uint64_t * msg_end_ptr=(uint64_t*)(rbf_ptr+ (lmeta->head+skip_size)%rbf_sz);
        //   wait for longer time
        //   if(msg_size==0 || *msg_end_ptr !=msg_size){
        //       return false;
        //   }

        return (msg_size != 0);
    }

    std::string fetch_rbf_msg(int local_tid, int mid) {
        rbf_lmeta_t *lmeta = &lmetas[local_tid * num_nodes + mid];
        char * rbf_ptr = rdma_mem + mem->ring_offset(local_tid, mid);

        volatile uint64_t msg_size = *(volatile uint64_t *)(rbf_ptr + lmeta->head % rbf_sz);
        uint64_t t1 = timer::get_usec();
        //clear head
        *(uint64_t *)(rbf_ptr + lmeta->head % rbf_sz) = 0;

        uint64_t skip_size = sizeof(uint64_t) + ceil(msg_size, sizeof(uint64_t));
        volatile uint64_t * msg_end_ptr = (volatile uint64_t *)(rbf_ptr + (lmeta->head + skip_size) % rbf_sz);
        while (*msg_end_ptr != msg_size) {
            //timer::cpu_relax(10);
            uint64_t tmp = *msg_end_ptr;
            if (tmp != 0 && tmp != msg_size) {
                printf("waiting for %ld, but actually %ld\n", msg_size, tmp);
                exit(0);
            }
        }
        //clear tail
        *msg_end_ptr = 0;
        uint64_t t2 = timer::get_usec();
        //copy from (lmeta->head+sizeof(uint64_t) , lmeta->head+sizeof(uint64_t)+ msg_size )
        //      or
        std::string result;
        result.reserve(msg_size);
        {
            size_t msg_head = (lmeta->head + sizeof(uint64_t)) % rbf_sz;
            size_t msg_tail = (lmeta->head + sizeof(uint64_t) + msg_size) % rbf_sz;
            if (msg_head < msg_tail) {
                result.append(rbf_ptr + msg_head, msg_size);
                memset(rbf_ptr + msg_head, 0, ceil(msg_size, sizeof(uint64_t)));
            } else {
                result.append(rbf_ptr + msg_head, msg_size - msg_tail);
                result.append(rbf_ptr, msg_tail);
                memset(rbf_ptr + msg_head, 0, msg_size - msg_tail);
                memset(rbf_ptr, 0, ceil(msg_tail, sizeof(uint64_t)));
            }
        }
        //   for(uint64_t i=0;i<ceil(msg_size,sizeof(uint64_t));i++){
        //     char * tmp=rbf_ptr+(lmeta->head+sizeof(uint64_t)+i)%rbf_sz;
        //     if(i<msg_size)
        //       result.push_back(*tmp);
        //     //clear data
        //     *tmp=0;
        //   }

        lmeta->head += 2 * sizeof(uint64_t) + ceil(msg_size, sizeof(uint64_t));
        uint64_t t3 = timer::get_usec();
        return result;
    }

    std::string recv(int tid) {
        while (true) {
            // NOTE: a logical queue = N * physical queue
            // check the queues in round robin
            int nid = (schedulers[tid].rr_cnt++) % num_nodes;
            if (check_rbf_msg(tid, nid))
                return fetch_rbf_msg(tid, nid);
        }
    }

    bool tryrecv(int tid, std::string &msg) {
        for (int nid = 0; nid < num_nodes; nid++) {
            if (check_rbf_msg(tid, nid)) {
                msg = fetch_rbf_msg(tid, nid);
                return true;
            }
        }

        return false;
    }
};
