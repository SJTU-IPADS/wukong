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

#include "global.hpp"
#include "rdma.hpp"

// utils
#include "unit.hpp"

using namespace std;

#define ADDR_PER_SRV(_addr, _sz, _tid) ((_addr) + ((_sz) * (_tid)));
#define OFFSET_PER_SRV(_off, _sz, _tid) ((_off) + ((_sz) * (_tid)));

#define ADDR_PER_TH(_addr, _sz, _tid, _sid) ((_addr) + (((_sz) * num_servers) * (_tid)) + ((_sz) * (_sid)));
#define OFFSET_PER_TH(_off, _sz, _tid, _sid) ((_off) + (((_sz) * num_servers) * (_tid)) + ((_sz) * (_sid)));

// memory layer of broadcast
class Broadcast_Mem {
private:
    int num_servers;
    int num_threads;
    int start_tid; // start id of threads sharing this mem

    uint64_t mem_sz;

    char *buf; // #threads
    uint64_t buf_sz;
    uint64_t buf_off;

    uint64_t rbf_sz;
    char *m_rbf; // master's ring buffer, #servers
    uint64_t m_rbf_off;
    char *s_rbf; // slave's ring buffer
    uint64_t s_rbf_off;

    // NOTE: To maintain the head of remote ring buffer, the reciever
    // actively pushes the head of (local) ring buffer to the (remote)
    // sender by RDMA WRITE (broadcast_adpator.lmeta.head > lrbf_hds > rrbf_hds).

    // the head of local RDMA ring buffer
    uint64_t lrbf_hd_sz;
    char *m_lrbf_hd; // #servers, written and read by reciever (local master)
    uint64_t m_lrbf_hd_off;
    char *s_lrbf_hd; // 1, written and read by reciever (local slave)
    uint64_t s_lrbf_hd_off;

    // the head of remote RDMA ring buffer
    uint64_t rrbf_hd_sz;
    char *m_rrbf_hd; // 1, written by reciever (remote master) and read by sender (local slave)
    uint64_t m_rrbf_hd_off;
    char *s_rrbf_hd; // #servers, written by reciever (remote slave) and read by sender (local master)
    uint64_t s_rrbf_hd_off;

public:
    Broadcast_Mem (int num_servers, int start_tid)
        : num_servers(num_servers), start_tid(start_tid) {
        if (RDMA::get_rdma().has_rdma())
            // only used by RDMA device
            buf_sz = rbf_sz = MiB2B(Global::rdma_rbf_size_mb);
        else
            buf_sz = rbf_sz = 0;

        lrbf_hd_sz = rrbf_hd_sz = sizeof(uint64_t);

        num_threads = 2;
        mem_sz = buf_sz * num_threads
                 + rbf_sz * (num_servers + 1)
                 + lrbf_hd_sz * (num_servers + 1)
                 + rrbf_hd_sz * (num_servers + 1);
    }

    // init poiters and offset
    void init(char *mem, uint64_t mem_off) {
        uint64_t off = 0;
        buf_off = mem_off + off;
        buf = mem + off;

        off += buf_sz * num_threads;
        m_rbf_off = mem_off + off;
        m_rbf = mem + off;

        off += rbf_sz * num_servers;
        s_rbf_off = mem_off + off;
        s_rbf = mem + off;

        off += rbf_sz;
        m_lrbf_hd_off = mem_off + off;
        m_lrbf_hd = mem + off;

        off += lrbf_hd_sz * num_servers;
        s_lrbf_hd_off = mem_off + off;
        s_lrbf_hd = mem + off;

        off += lrbf_hd_sz;
        m_rrbf_hd_off = mem_off + off;
        m_rrbf_hd = mem + off;

        off += rrbf_hd_sz;
        s_rrbf_hd_off = mem_off + off;
        s_rrbf_hd = mem + off;
    }

    uint64_t mem_size() { return mem_sz; }

    // buffer
    inline char *buffer(int tid) { return buf + buf_sz * (tid - start_tid); }
    inline uint64_t buffer_size() { return buf_sz; }
    inline uint64_t buffer_offset(int tid) { return buf_off + buf_sz * (tid - start_tid); }

    // ring-buffer
    inline uint64_t ring_size() { return rbf_sz; }
    inline char *master_ring(int sid) { return m_rbf + rbf_sz * sid; }
    inline char *slave_ring() { return s_rbf; }
    inline uint64_t master_ring_offset(int sid) { return m_rbf_off + rbf_sz * sid; }
    inline uint64_t slave_ring_offset() { return s_rbf_off; }

    // head of local ring-buffer
    inline uint64_t local_ring_head_size() { return lrbf_hd_sz; }
    inline char *local_master_ring_head(int sid) { return m_lrbf_hd + lrbf_hd_sz * sid; }
    inline char *local_slave_ring_head() { return s_lrbf_hd; }
    inline uint64_t local_master_ring_head_offset(int sid) { return m_lrbf_hd_off + lrbf_hd_sz * sid; }
    inline uint64_t local_slave_ring_head_offset() { return s_lrbf_hd_off; }

    // head of remote ring-buffer
    inline uint64_t remote_ring_head_size() { return rrbf_hd_sz; }
    inline char *remote_master_ring_head() { return m_rrbf_hd; }
    inline char *remote_slave_ring_head(int sid) { return s_rrbf_hd + rrbf_hd_sz * sid; }
    inline uint64_t remote_master_ring_head_offset() { return m_rrbf_hd_off; }
    inline uint64_t remote_slave_ring_head_offset(int sid) { return s_rrbf_hd_off + rrbf_hd_sz * sid; }
};

class Mem {
private:
    int num_servers;
    int num_threads;

    // The Wukong's (host) CPU memory layout: kvstore | rdma-buffer | ring-buffer
    // The rdma-buffer and ring-buffer are only used when HAS_RDMA
    char *mem;
    uint64_t mem_sz;


    // Key-value (graph) store
    char *kvs;
    uint64_t kvs_off;
    uint64_t kvs_sz;

    // RDMA buffer (#threads)
    char *buf;
    uint64_t buf_off;
    uint64_t buf_sz; // per thread

    // Ring buffer (#thread x #servers)
    char *rbf;
    uint64_t rbf_off;
    uint64_t rbf_sz; // per thread x server

    // To maintain the head of remote ring buffer, the reciever actively pushes
    // the head of (local) ring buffer to the (remote) sender by RDMA WRITE
    // NOTE: rdma_adpator.lmeta.head > lrbf_hds > rrbf_hds).

    // (local) recieve-side head of ring buffer (#thread x #servers)
    char *lrbf_hd; // written and read by reciever (local)
    uint64_t lrbf_hd_off;
    uint64_t lrbf_hd_sz;

    // (remote) send-side head of ring buffer (#thread x #servers)
    char *rrbf_hd; // written by reciever (remote) and read by sender (local)
    uint64_t rrbf_hd_off;
    uint64_t rrbf_hd_sz;

    vector<Broadcast_Mem *> bc_mems;
public:
    Mem(int num_servers, int num_threads, vector<Broadcast_Mem *> bc_ms = vector<Broadcast_Mem *>())
        : num_servers(num_servers), num_threads(num_threads), bc_mems(bc_ms) {

        // calculate memory usage
        kvs_sz = GiB2B(Global::memstore_size_gb);

        // only used by RDMA device (NOTE: global variable should be set to 0 if no RDMA)
        if (RDMA::get_rdma().has_rdma()) {
            buf_sz = MiB2B(Global::rdma_buf_size_mb);
            rbf_sz = MiB2B(Global::rdma_rbf_size_mb);
        } else {
            buf_sz = rbf_sz = 0;
        }

        lrbf_hd_sz = rrbf_hd_sz = sizeof(uint64_t);

        // allocate memory and zeroing
        mem_sz = kvs_sz
                 + buf_sz * num_threads
                 + rbf_sz * num_servers * num_threads
                 + lrbf_hd_sz * num_servers * num_threads
                 + rrbf_hd_sz * num_servers * num_threads;
        for (int i = 0; i < bc_mems.size(); i++)
            mem_sz += bc_mems[i]->mem_size();
        // allocate mem_sz bytes.
        mem = new char[mem_sz / sizeof(char)];
        memset(mem, 0, mem_sz);

        // kvstore
        kvs_off = 0;
        kvs = mem + kvs_off;

        // RDMA buffer
        buf_off = kvs_off + kvs_sz;
        buf = mem + buf_off;

        // ring buffer
        rbf_off = buf_off + buf_sz * num_threads;
        rbf = mem + rbf_off;

        lrbf_hd_off = rbf_off + rbf_sz * num_servers * num_threads;
        lrbf_hd = mem + lrbf_hd_off;

        rrbf_hd_off = lrbf_hd_off + lrbf_hd_sz * num_servers * num_threads;
        rrbf_hd =  mem + rrbf_hd_off;

        uint64_t off = rrbf_hd_off + rrbf_hd_sz * num_servers * num_threads;

        for (int i = 0; i < bc_mems.size(); i++) {
            bc_mems[i]->init(mem + off, off);
            off += bc_mems[i]->mem_size();
        }
    }

    ~Mem() { free(mem); }

    inline char *address() { return mem; }
    inline uint64_t size() { return mem_sz; }

    // kvstore
    inline char *kvstore() { return kvs; }
    inline uint64_t kvstore_offset() { return kvs_off; }
    inline uint64_t kvstore_size() { return kvs_sz; }

    // RDMA buffer
    inline char *buffer(int tid) { return buf + buf_sz * tid; }
    inline uint64_t buffer_offset(int tid) { return buf_off + buf_sz * tid; }
    inline uint64_t buffer_size() { return buf_sz; }

    // ring buffer (task queue)
    // data: address/offset and size)
    inline char *ring(int tid, int sid) { return rbf + (rbf_sz * num_servers) * tid + rbf_sz * sid; }
    inline uint64_t ring_offset(int tid, int sid) { return rbf_off + (rbf_sz * num_servers) * tid + rbf_sz * sid; }
    inline uint64_t ring_size() { return rbf_sz; }

    // metadata: recieve-side (local) head
    inline char *local_ring_head(int tid, int sid) { return lrbf_hd + (lrbf_hd_sz * num_servers) * tid + lrbf_hd_sz * sid; }
    inline uint64_t local_ring_head_offset(int tid, int sid) { return lrbf_hd_off + (lrbf_hd_sz * num_servers) * tid + lrbf_hd_sz * sid; }
    inline uint64_t local_ring_head_size() { return lrbf_hd_sz; }

    // metadata: send-side (remote) head
    inline char *remote_ring_head(int tid, int sid) { return rrbf_hd + (rrbf_hd_sz * num_servers) * tid + rrbf_hd_sz * sid; }
    inline uint64_t remote_ring_head_offset(int tid, int sid) { return rrbf_hd_off + (rrbf_hd_sz * num_servers) * tid + rrbf_hd_sz * sid; }
    inline uint64_t remote_ring_head_size() { return rrbf_hd_sz; }

}; // end of class Mem
