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

#ifdef USE_GPU

#include <tbb/concurrent_queue.h>
#include <vector>

#include "global.hpp"
#include "comm/adaptor.hpp"
#include "query.hpp"
#include "coder.hpp"
#include "assertion.hpp"
#include "rmap.hpp"
#include "gpu_utils.hpp"

#ifdef USE_GPU
#include "gpu_engine.hpp"
#endif

using namespace std;

// a vector of pointers of all local GPU agents
// class GPUEngine;
// std::vector<GPUAgent *> gpu_agents;

/**
 * An agent thread will assist the GPU with query handling
 */
class GPUAgent {
private:
    class Message {
    public:
        int sid;
        int tid;
        Bundle bundle;

        Message(int sid, int tid, Bundle &bundle)
            : sid(sid), tid(tid), bundle(bundle) { }
    };


    GPUEngine *gpu_engine;
    Adaptor *adaptor;

    Coder coder;
    RMap rmap; // a map of replies for pending (fork-join) queries
    pthread_spinlock_t rmap_lock;

    tbb::concurrent_queue<SPARQLQuery> runqueue;
    vector<Message> pending_msgs;


public:
    int sid;    // server id
    int tid;    // thread id

    GPUAgent(int sid, int tid, Adaptor* adaptor, GPUEngine* gpu_engine)
        : sid(sid), tid(tid), adaptor(adaptor), gpu_engine(gpu_engine), coder(sid, tid) {

    }

    ~GPUAgent() { }

    bool send_request(Bundle& bundle, int dst_sid, int dst_tid) {
        if (adaptor->send(dst_sid, dst_tid, bundle)) {
            return true;
        }

        // failed to send, then stash the msg to avoid deadlock
        pending_msgs.push_back(Message(dst_sid, dst_tid, bundle));
        return false;
    }

    void sweep_msgs() {
    }

    void collect_reply(SPARQLQuery& r) {
        pthread_spin_lock(&rmap_lock);
        rmap.put_reply(r);

        if (!rmap.is_ready(r.pid)) {
            pthread_spin_unlock(&rmap_lock);
            return; // not ready (waiting for the rest)
        }

        // all sub-queries have done, continue to execute
        r = rmap.get_merged_reply(r.pid);
        pthread_spin_unlock(&rmap_lock);
    }

    bool need_parallel(const SPARQLQuery& r) {
        return (r.pattern_step == 0
                && r.pattern_group.parallel == false
                && r.start_from_index()
                && (global_num_servers * r.mt_factor > 1));
    }

    void send_to_workers(SPARQLQuery& req) {
        // The mt_factor can be set on proxy side before sending to engine,
        // but must smaller than global_mt_threshold (Default: mt_factor == 1)
        // Normally, we will NOT let global_mt_threshold == #gpu, which will cause HANG
        int sub_reqs_size = global_num_servers * req.mt_factor;
        rmap.put_parent_request(req, sub_reqs_size);
        SPARQLQuery sub_query = req;
        for (int i = 0; i < global_num_servers; i++) {
            for (int j = 0; j < req.mt_factor; j++) {
                sub_query.id = -1;
                sub_query.pid = req.id;
                // start from the next engine thread
                int dst_tid = (tid + j + 1 - WUKONG_GPU_AGENT_TID) % global_num_gpus
                              + WUKONG_GPU_AGENT_TID;
                sub_query.tid = j;
                sub_query.mt_factor = 1;
                sub_query.pattern_group.parallel = true;

                Bundle bundle(sub_query);
                send_request(bundle, i, dst_tid);
            }
        }
    }


    // fork-join or in-place execution
    bool need_fork_join(SPARQLQuery &req) {
        // always need NOT fork-join when executing on single machine
        if (global_num_servers == 1) return false;

        // always need fork-join mode w/o RDMA
        if (!global_use_rdma) return true;

        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ASSERT(req.result.variable_type(pattern.subject) == known_var);
        //ssid_t start = pattern.subject;
        //return ((req.local_var != start)
        //        && (req.result.get_row_num() >= global_rdma_threshold));
        return (req.result.get_row_num() >= 0); // FIXME: not consider dedup
    }

    void execute_sparql_query(SPARQLQuery &req) {
        // encode the lineage of the query (server & thread)
        if (req.id == -1) req.id = coder.get_and_inc_qid();


        logstream(LOG_DEBUG) << "GPUAgent: " << "[" << sid << "-" << tid << "]"
                             << " got a req: r.id=" << req.id << ", r.state="
                             << (req.state == SPARQLQuery::SQState::SQ_REPLY ? "SQ_REPLY" : "Request") << LOG_endl;

        if (req.state == SPARQLQuery::SQState::SQ_REPLY) {
            collect_reply(req);
        }

        // execute_patterns
        while (true) {
            ASSERT(req.dev_type == SPARQLQuery::DeviceType::GPU);

            if (!gpu_engine->result_buf_ready(req))
                gpu_engine->load_result_buf(req);

            gpu_engine->execute_one_pattern(req);

            if (req.done(SPARQLQuery::SQState::SQ_PATTERN)) {
                // only send back row_num in blind mode
                req.result.row_num = req.result.get_row_num();
                logstream(LOG_DEBUG) << "GPUAgent: finished query r.id=" << req.id << LOG_endl;
                req.state = SPARQLQuery::SQState::SQ_REPLY;
                Bundle bundle(req);
                send_request(bundle, coder.sid_of(req.pid), coder.tid_of(req.pid));
                break;
            }

            // TODO
            if (need_fork_join(req)) {
                ASSERT_MSG(false, "doesn't support fork-join mode now");
                vector<SPARQLQuery> sub_reqs = gpu_engine->generate_sub_query(req);
                rmap.put_parent_request(req, sub_reqs.size());
                for (int i = 0; i < sub_reqs.size(); i++) {
                    if (i != sid) {
                        Bundle bundle(sub_reqs[i]);
                        send_request(bundle, i, tid);
                    } else {
                        runqueue.push(sub_reqs[i]);
                    }
                }
                break;
            }
        }
    }

    void run() {

        bool has_job;
        while (true) {
            has_job = false;

            // check and send pending messages first
            sweep_msgs();

            // priority path: sparql stage (FIXME: only for SPARQL queries)
            SPARQLQuery req;
            if (runqueue.try_pop(req)) {
                if (need_parallel(req)) {
                    send_to_workers(req);
                    continue; // exhaust all queries
                }
                execute_sparql_query(req);
                continue; // exhaust all queries
            }

            Bundle bundle;
            while (adaptor->tryrecv(bundle)) {
                if (bundle.type == SPARQL_QUERY) {
                    // To be fair, agent will handle sub-queries first, instead of a new job.
                    SPARQLQuery req = bundle.get_sparql_query();
                    ASSERT(req.dev_type == SPARQLQuery::DeviceType::GPU);

                    if (req.priority != 0) {
                        execute_sparql_query(req);
                        break;
                    }

                    runqueue.push(req);
                } else {
                    ASSERT(false);
                }
            }

            // if (runqueue.try_pop(req)) {
                // // process a new SPARQL query
                // execute_sparql_query(req);
            // }
        }
    }

};

#endif  // USE_GPU
