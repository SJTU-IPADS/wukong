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

#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>

#include <iostream>
#include "utils.h"
#include "global_cfg.h"
#include "thread_cfg.h"
#include "string_server.h"
#include "distributed_graph.h"
#include "server.h"
#include "client.h"
#include "builtin_console.h"

using namespace std;

/* configure of Cube0-5 */
int socket_0[] = {
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18
};

int socket_1[] = {
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18
};

void
pin_to_core(size_t core)
{
    cpu_set_t  mask;
    CPU_ZERO(&mask);
    CPU_SET(core , &mask);
    int result = sched_setaffinity(0, sizeof(mask), &mask);
}


const int start_sz = 4;
const int end_sz = 8192 * 256;
//nops=500;

void
record_result(struct thread_cfg *cfg)
{
    int sz = start_sz;
    uint64_t t1 = timer::get_usec();
    uint64_t t2 = timer::get_usec();
    while (true) {
        uint64_t throughput = 0;
        for (int m = 1; m < cfg->m_num; m++) {
            for (int t = 1; t < cfg->t_num; t++) {
                throughput += RecvObject<uint64_t>(cfg);
            }
        }
        t2 = timer::get_usec();
        //cout<<"throughput of sz="<<sz <<" is "<<throughput<<endl;
        cout << "throughput of sz=" << sz << " is " << 500L * (cfg->m_num - 1)*(cfg->t_num - 1) * 1000000 / ((t2 - t1) ) << endl;
        for (int m = 1; m < cfg->m_num; m++) {
            for (int t = 1; t < cfg->t_num; t++) {
                SendObject<uint64_t>(cfg, m, t, throughput); // next-round
            }
        }
        t1 = timer::get_usec();
        sz = sz * 2;
    }
}

void*
Run(void *ptr)
{
    struct thread_cfg *cfg = (struct thread_cfg*) ptr;
    pin_to_core(socket_1[cfg->t_id]);
    string buffer;
    string data;
    if (cfg->t_id == 0) {
        if (cfg->m_id == 0) {
            record_result(cfg);
        } else {
            return NULL;
        }
    }

    if (cfg->m_id == 0) { // poll-message
        vector<string> ret_data;
        int sz = start_sz;
        while (sz <= end_sz) {
            string tmp;
            tmp.resize(sz);
            for (int i = 0; i < tmp.size(); i++) {
                tmp[i] = i * i * i;
            }
            ret_data.push_back(tmp);
            sz *= 2;
        }
        while (true) {
            buffer = cfg->node->Recv();
            cfg->node->Send(buffer[0], buffer[1], ret_data[buffer[2]]);
        }
    } else {
        int curr_index = 0;
        int sz = start_sz;
        while (sz <= end_sz) {
            uint64_t nops = 500;
            buffer.resize(3);
            buffer[0] = cfg->m_id;
            buffer[1] = cfg->t_id;
            buffer[2] = curr_index;
            uint64_t t1 = timer::get_usec();

            if (!global_use_rbf) { // zero-MQ
                for (int i = 0; i < global_batch_factor; i++) {
                    cfg->node->Send(0, cfg->t_id, buffer);
                }
                for (int i = 0; i < nops - global_batch_factor; i++) {
                    string tmp = cfg->node->Recv();
                    cfg->node->Send(0, cfg->t_id, buffer);
                }
                for (int i = 0; i < global_batch_factor; i++) {
                    string tmp = cfg->node->Recv();
                }
            } else {  // rdma-read
                for (int i = 0; i < global_batch_factor; i++) {
                    char* local_buffer = cfg->rdma->GetMsgAddr(cfg->t_id);
                    cfg->rdma->post(cfg->t_id, 0, (char *)local_buffer, sz, 0, IBV_WR_RDMA_READ);
                }
                for (int i = 0; i < nops - global_batch_factor; i++) {
                    cfg->rdma->poll(cfg->t_id, 0);
                    char* local_buffer = cfg->rdma->GetMsgAddr(cfg->t_id);
                    cfg->rdma->post(cfg->t_id, 0, (char *)local_buffer, sz, 0, IBV_WR_RDMA_READ);
                }
                for (int i = 0; i < global_batch_factor; i++) {
                    cfg->rdma->poll(cfg->t_id, 0);
                }
            }

            uint64_t t2 = timer::get_usec();
            uint64_t throughput = nops * 1000 * 1000 / (t2 - t1);
            SendObject<uint64_t>(cfg, 0, 0, throughput);
            //cout<<throughput<<endl;
            RecvObject<uint64_t>(cfg);
            //sleep(1);
            curr_index++;
            sz *= 2;
        }

        /// test latency
        if (cfg->m_id == 1 && cfg->t_id == 1) {

            int curr_index = 0;
            int sz = start_sz;
            while (sz <= end_sz) {
                uint64_t nops = 5000;
                buffer.resize(3);
                buffer[0] = cfg->m_id;
                buffer[1] = cfg->t_id;
                buffer[2] = curr_index;
                uint64_t t1 = timer::get_usec();

                if (!global_use_rbf) { // zero-MQ
                    for (int i = 0; i < nops; i++) {
                        cfg->node->Send(0, cfg->t_id, buffer);
                        string tmp = cfg->node->Recv();
                    }
                } else {  // rdma-read
                    for (int i = 0; i < nops; i++) {
                        char* local_buffer = cfg->rdma->GetMsgAddr(cfg->t_id);
                        cfg->rdma->post(cfg->t_id, 0, (char *)local_buffer, sz, 0, IBV_WR_RDMA_READ);
                        cfg->rdma->poll(cfg->t_id, 0);
                    }
                }
                uint64_t t2 = timer::get_usec();
                cout << "latency of sz=" << sz << " is " << (t2 - t1) * 1.0 / nops << endl;
                curr_index++;
                sz *= 2;
            }
        }
    }
}

int
main(int argc, char * argv[])
{
    if (argc != 3) {
        cout << "usage:./test_rdma config_file hostfile" << endl;
        return -1;
    }

    load_global_cfg(argv[1]);

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    int world_size = world.size();
    int world_rank = world.rank();

    /*
        if(argc !=5) {
            cout<<"usage:./test_rdma config_file hostfile world_size world_rank"<<endl;
            return -1;
        }
        load_global_cfg(argv[1]);
        int world_size=atoi(argv[3]);
        int world_rank=atoi(argv[4]);
    */
    uint64_t rdma_size = GiB2B(1);
    rdma_size = rdma_size * global_total_memory_gb;
    uint64_t msg_slot_per_thread = MiB2B(global_perslot_msg_mb);
    uint64_t rdma_slot_per_thread = MiB2B(global_perslot_rdma_mb);
    uint64_t total_size = rdma_size
                          + rdma_slot_per_thread * global_nthrs
                          + msg_slot_per_thread * global_nthrs;
    //[0-thread_num-1] are used
    Network_Node *node = new Network_Node(world_rank, global_nthrs, string(argv[2]));
    char *buffer = (char*) malloc(total_size);
    memset(buffer, 0, total_size);
    RdmaResource *rdma = new RdmaResource(world_size, global_nthrs,
                                          world_rank, buffer, total_size,
                                          rdma_slot_per_thread, msg_slot_per_thread, rdma_size);
    rdma->node = node;
    rdma->Servicing();
    rdma->Connect();
    thread_cfg* cfg_array = new thread_cfg[global_nthrs];
    for (int i = 0; i < global_nthrs; i++) {
        cfg_array[i].t_id = i;
        cfg_array[i].t_num = global_nthrs;
        cfg_array[i].m_id = world_rank;
        cfg_array[i].m_num = world_size;
        cfg_array[i].client_num = global_nfewkrs;
        cfg_array[i].server_num = global_nbewkrs;
        cfg_array[i].rdma = rdma;
        cfg_array[i].node = new Network_Node(cfg_array[i].m_id, cfg_array[i].t_id, string(argv[2]));
        cfg_array[i].init();
    }

    pthread_t     *thread  = new pthread_t[global_nthrs];
    for (size_t id = 0; id < global_nthrs; ++id) {
        pthread_create (&(thread[id]), NULL, Run, (void *) & (cfg_array[id]));
    }
    for (size_t t = 0 ; t < global_nthrs; t++) {
        int rc = pthread_join(thread[t], NULL);
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

    return 0;
}
