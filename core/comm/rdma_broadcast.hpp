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

using namespace std;

// rdma version one-to-many communication
class RDMA_Broadcast {
public:

    Broadcast_Mem *mem;
    int tid;
    int sid;
    int num_servers;
    // track tail of ring buffer for writer
    uint64_t *tail = NULL;
    // track head of ring buffer for reader
    uint64_t *head = NULL;

    bool init = false;

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

    // Check if there is new data
    inline bool check(char *rbf, uint64_t head) {
        uint64_t rbf_sz = mem->ring_size();
        volatile uint64_t data_sz = *(volatile uint64_t *)(rbf + head % rbf_sz);  // header
        return (data_sz != 0);
    }

    // Check whether overwrite occurs if msg is sent
    inline bool rbf_full(uint64_t remote_head, uint64_t tail, uint64_t msg_sz) {
        uint64_t rbf_sz = mem->ring_size();
        return (rbf_sz < (tail - remote_head + msg_sz));
    }

    // Return the pointer of data fetched from ring buffer
    string fetch(char *rbf, uint64_t *head) {
        uint64_t rbf_sz = mem->ring_size();
        // struct of data: [size | data | size]
        volatile uint64_t data_sz = *(volatile uint64_t *)(rbf + *head % rbf_sz);  // header
        *(uint64_t *)(rbf + *head % rbf_sz) = 0;  // clean header

        uint64_t to_footer = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t));
        volatile uint64_t * footer = (volatile uint64_t *)(rbf + (*head + to_footer) % rbf_sz); // footer
        while (*footer != data_sz) { // spin-wait RDMA-WRITE done
            _mm_pause();
            /* If RDMA-WRITE is done, footer == header == size
             * Otherwise, footer == 0
             */
            ASSERT(*footer == 0 || *footer == data_sz);
        }
        *footer = 0;  // clean footer

        // actually read data
        string result;
        result.reserve(data_sz);
        uint64_t start = (*head + sizeof(uint64_t)) % rbf_sz; // start of data
        uint64_t end = (*head + sizeof(uint64_t) + data_sz) % rbf_sz;  // end of data
        if (start < end) {
            result.append(rbf + start, data_sz);
            memset(rbf + start, 0, ceil(data_sz, sizeof(uint64_t)));  // clean data
        } else { // overwrite from the start
            result.append(rbf + start, data_sz - end);
            result.append(rbf, end);
            memset(rbf + start, 0, data_sz - end);                    // clean data
            memset(rbf, 0, ceil(end, sizeof(uint64_t)));              // clean data
        }
        *head += 2 * sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t));  //update head
        return result;
    } // end of fetch

    // Update heads of ring buffer to writer to help it detect overflow
    void update_remote_head(int dst_sid, uint64_t head,
            char *lrhd, uint64_t rrhd_off, char *rrhd) {

        uint64_t rbf_sz = mem->ring_size();
        const uint64_t threshold = rbf_sz / 8;
        if (head - *(uint64_t *)lrhd > threshold) {
            *(uint64_t *)lrhd = head;
            if (sid != dst_sid) {  // update to remote server
                RDMA &rdma = RDMA::get_rdma();
                rdma.dev->RdmaWrite(tid, dst_sid, lrhd, mem->remote_ring_head_size(), rrhd_off);
            } else {
                *(uint64_t *)rrhd = *(uint64_t *)lrhd;
            }
        }
    }

    /* Send msg to local ring buffer
     * @dst_ptr: destination pointer
     * @off: tail of ring buffer
     * @data: data to send
     * @data_sz: size of data
     */
    void send_local(char *dst_ptr, uint64_t off, const char *data, int data_sz) {
        uint64_t rbf_sz = mem->ring_size();
        *((uint64_t *)(dst_ptr + off % rbf_sz)) = data_sz;       // header
        off += sizeof(uint64_t);

        if (off / rbf_sz == (off + data_sz - 1) / rbf_sz ) { // data
            memcpy(dst_ptr + (off % rbf_sz), data, data_sz);
        } else {
            uint64_t _sz = rbf_sz - (off % rbf_sz);
            memcpy(dst_ptr + (off % rbf_sz), data, _sz);
            memcpy(dst_ptr, data + _sz, data_sz - _sz);
        }
        off += ceil(data_sz, sizeof(uint64_t));

        *((uint64_t *)(dst_ptr + off % rbf_sz)) = data_sz;       // footer
    }

    /* Send msg to remote ring buffer
     * @dst_sid: destination server
     * @rdma_off: offset of destination rdma buffer
     * @off: tail of ring buffer
     * @data: data to send
     * @data_sz: size of data
     */
    void send_remote(int dst_sid, uint64_t rdma_off, uint64_t off, const char *data, int data_sz) {
        uint64_t rbf_sz = mem->ring_size();
        uint64_t msg_sz = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t)) + sizeof(uint64_t);
        // prepare RDMA buffer for RDMA-WRITE
        char *rdma_buf = mem->buffer(tid);
        uint64_t buf_sz = mem->buffer_size();
        ASSERT(msg_sz < buf_sz);

        *((uint64_t *)rdma_buf) = data_sz;  // header
        rdma_buf += sizeof(uint64_t);
        memcpy(rdma_buf, data, data_sz);    // data
        rdma_buf += ceil(data_sz, sizeof(uint64_t));
        *((uint64_t*)rdma_buf) = data_sz;   // footer

        // write msg to the remote physical-queue
        RDMA &rdma = RDMA::get_rdma();
        if (off / rbf_sz == (off + msg_sz - 1) / rbf_sz ) {
            rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid), msg_sz, rdma_off + (off % rbf_sz));
        } else {
            uint64_t _sz = rbf_sz - (off % rbf_sz);
            rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid), _sz, rdma_off + (off % rbf_sz));
            rdma.dev->RdmaWrite(tid, dst_sid, mem->buffer(tid) + _sz, msg_sz - _sz, rdma_off);
        }
    }

    RDMA_Broadcast(int sid, int tid, Broadcast_Mem *mem, int num_servers)
        : sid(sid), tid(tid), mem(mem), num_servers(num_servers) {

        if (RDMA::get_rdma().has_rdma())
            init = true;
    }

    ~RDMA_Broadcast() { }  //TODO
};

// communication of master side
class RDMA_Broadcast_Master : public RDMA_Broadcast {
private:
    int slave_tid;

public:
    RDMA_Broadcast_Master(int sid, int tid, Broadcast_Mem *mem, int num_servers, int slave_tid)
        : RDMA_Broadcast(sid, tid, mem, num_servers), slave_tid(slave_tid) {

        tail = new uint64_t[num_servers];
        head = new uint64_t[num_servers];
        for (int i = 0; i < num_servers; i++) {
            tail[i] = head[i] = 0;
        }
    }

    // Send given @str to slave on @dst_sid server
    // Return false if failed. Otherwise, return true.
    bool send(int dst_sid, string str) {
        ASSERT(init);
        const char *data = str.c_str();
        uint64_t data_sz = str.length();
        uint64_t rbf_sz = mem->ring_size();
        // struct of data: [size | data | size] (size of data serves as header and footer)
        uint64_t msg_sz = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t)) + sizeof(uint64_t);
        ASSERT(msg_sz < rbf_sz);

        uint64_t remote_head = *(uint64_t *)mem->remote_slave_ring_head(dst_sid);
        if (rbf_full(remote_head, tail[dst_sid], msg_sz))
            return false;

        uint64_t off = tail[dst_sid];
        tail[dst_sid] += msg_sz;

        if (dst_sid == sid) {  // local
            char *ptr = mem->slave_ring();
            send_local(ptr, off, data, data_sz);
        } else { // remote
            uint64_t rdma_off = mem->slave_ring_offset();
            send_remote(dst_sid, rdma_off, off, data, data_sz);
        }
        return true;
    }

    // Try to recv data from slaves
    bool tryrecv(string &str) {
        ASSERT(init);
        // check all physical-queues of tid once
        for (int src_sid = 0; src_sid < num_servers; src_sid++) {
            char *rbf = mem->master_ring(src_sid); // ring buffer to recv data from slave in src_sid
            if (check(rbf, head[src_sid])) {
                str = fetch(rbf, &head[src_sid]);

                char *lrhd = mem->local_master_ring_head(src_sid);
                char *rrhd = mem->remote_master_ring_head();
                uint64_t rrhd_off = mem->remote_master_ring_head_offset();
                update_remote_head(src_sid, head[src_sid], lrhd, rrhd_off, rrhd);

                return true;
            }
        }
        return false;
    }

    // Simple recv which calls tryrecv
    string recv() {
        string data;
        while (!tryrecv(data));
        return data;
    }
};

// communication of slave side
class RDMA_Broadcast_Slave : public RDMA_Broadcast {
private:
    int master_sid;
    int master_tid;

public:
    RDMA_Broadcast_Slave(int sid, int tid, Broadcast_Mem *mem, int num_servers, int m_sid, int m_tid)
        : RDMA_Broadcast(sid, tid, mem, num_servers), master_sid(m_sid), master_tid(m_tid) {

        tail = new uint64_t;
        head = new uint64_t;
        *tail = *head = 0;
    }

    // Send given @str to master
    // Return false if failed. Otherwise, return true.
    bool send(string str) {
        ASSERT(init);
        const char *data = str.c_str();
        uint64_t data_sz = str.length();
        uint64_t rbf_sz = mem->ring_size();
        // struct of data: [size | data | size] (use size of data as header and footer)
        uint64_t msg_sz = sizeof(uint64_t) + ceil(data_sz, sizeof(uint64_t)) + sizeof(uint64_t);
        ASSERT(msg_sz < rbf_sz);

        uint64_t remote_head = *(uint64_t *)mem->remote_master_ring_head();
        if (rbf_full(remote_head, *tail, msg_sz)) // detect overflow
            return false;

        uint64_t off = *tail;
        *tail += msg_sz;

        if (master_sid == sid) {  // local
            char *ptr = mem->master_ring(sid);
            send_local(ptr, off, data, data_sz);
        } else { // remote
            uint64_t rdma_off = mem->master_ring_offset(sid);
            send_remote(master_sid, rdma_off, off, data, data_sz);
        }
        return true;
    }

    // Try to recv data from slaves
    bool tryrecv(string &str) {
        ASSERT(init);

        char *rbf = mem->slave_ring(); // ring buffer to recv data from master
        if (check(rbf, *head)) {
            str = fetch(rbf, head);

            char *lrhd = mem->local_slave_ring_head();
            char *rrhd = mem->remote_slave_ring_head(sid);
            uint64_t rrhd_off = mem->remote_slave_ring_head_offset(sid);
            update_remote_head(master_sid, *head, lrhd, rrhd_off, rrhd);

            return true;
        }
        return false;
    }

    // Simple recv which calls tryrecv
    string recv() {
        string data;
        while (!tryrecv(data));
        return data;
    }
};
