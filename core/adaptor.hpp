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

#include "config.hpp"
#include "query.hpp"
#include "tcp_adaptor.hpp"
#include "rdma_adaptor.hpp"

/// TODO: define adaptor as a C++ interface and make tcp and rdma implement it
class Adaptor {
public:
    int tid; // thread id

    TCP_Adaptor *tcp;   // communicaiton by TCP/IP
    RDMA_Adaptor *rdma; // communicaiton by RDMA

    Adaptor(int tid, TCP_Adaptor *tcp, RDMA_Adaptor *rdma)
        : tid(tid), tcp(tcp), rdma(rdma) { }

    ~Adaptor() { }

    bool send(int dst_sid, int dst_tid, Bundle &bundle) {
        if (global_use_rdma && rdma->init)
            return rdma->send(tid, dst_sid, dst_tid, bundle.get_type() + bundle.data);
        else
            return tcp->send(dst_sid, dst_tid, bundle.get_type() + bundle.data);
    }

    Bundle recv() {
        std::string str;
        if (global_use_rdma && rdma->init)
            str = rdma->recv(tid);
        else
            str = tcp->recv(tid);

        Bundle bundle(str);
        return bundle;
    }

    bool tryrecv(Bundle &bundle) {
        std::string str;
        if (global_use_rdma && rdma->init) {
            if (!rdma->tryrecv(tid, str)) return false;
        } else {
            if (!tcp->tryrecv(tid, str)) return false;
        }

        bundle.set_type(str.at(0));
        bundle.data = str.substr(1);
        return true;
    }

    #ifdef USE_GPU
    /* send the forked subquery(in CPU mem) and partial history(in GPU mem) to remote server
     * table_size refers to the number of elements in history, not n_rows
     */
    bool send_split(int dst_sid, int dst_tid, SPARQLQuery &r, char *history_ptr, uint64_t table_size) {
        ASSERT(tid < global_num_threads);
        ASSERT(r.subquery_type == SPARQLQuery::SubQueryType::SPLIT);
        Bundle bundle(r);
        string ctrl_msg = bundle.get_type() + bundle.data;
        return rdma->send_split(tid, dst_sid, dst_tid, ctrl_msg.c_str(), history_ptr, ctrl_msg.length(), table_size * sizeof(sid_t));
    }

    /* first receive the forked subquery, then receive the partial history and copy it to local gpu mem
     * receive does not need acquire lock since there are only one reader on ring buffer
     */
    bool tryrecv_split(SPARQLQuery &r) {
        std::string str;
        int sender_sid = 0;

        if (!rdma->tryrecv(tid, str))
            return false;

        Bundle b;
        b.set_type(str.at(0));
        b.data = str.substr(1);

        r = b.get_sparql_query();

        // continue receive history of query
        if (r.subquery_type == SPARQLQuery::SubQueryType::SPLIT) {
            int ret;
            std::string dumb_str;

            ret = rdma->recv_by_gpu(tid, sender_sid, dumb_str);
            ASSERT(ret > 0);
            GPU &gpu = GPU::instance();
            // hint: history has been copied to gpu mem(by recv_by_gpu->fetch), update r.result.gpu_history_ptr & r.result.gpu_history_table_size here
            r.result.gpu_history_ptr = gpu.history_inbuf();
            r.result.gpu_history_table_size = gpu.history_size();
        }
        return true;
    }
    #endif
};
