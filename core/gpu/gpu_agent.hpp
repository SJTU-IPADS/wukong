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
#include "query.hpp"
#include "coder.hpp"

// gpu
#include "gpu_engine.hpp"

#include "engine/rmap.hpp"
#include "comm/adaptor.hpp"

// utils
#include "assertion.hpp"
#include "gpu.hpp"

using namespace std;

/**
 * An agent thread will assist query processing on GPUs
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

        pthread_spin_init(&rmap_lock, 0);
    }

    ~GPUAgent() { }

    bool send_request(Bundle& bundle, int dst_sid, int dst_tid) {
        if (adaptor->send(dst_sid, dst_tid, bundle))
            return true;

        // failed to send, then stash the msg to avoid deadlock
        pending_msgs.push_back(Message(dst_sid, dst_tid, bundle));
        return false;
    }

    bool send_sub_query(SPARQLQuery &req, int dst_sid, int dst_tid) {
        // #1 send query
        Bundle b(req);
        if (adaptor->send(dst_sid, dst_tid, b)) {
            // #2 send result buffer
            if (req.result.gpu.rbuf_num_elems() > 0) {
                adaptor->send_dev2host(dst_sid, dst_tid, req.result.gpu.rbuf(),
                                       WUKONG_GPU_ELEM_SIZE * req.result.gpu.rbuf_num_elems());
            }
            return true;
        }

        ASSERT(false);

        // TODO if send fail, copy history to CPU and save to a staging area
        return false;
    }

    void send_reply(SPARQLQuery &req, int dst_sid, int dst_tid) {
        if (req.result.blind) {
            req.shrink();
        }

        req.state = SPARQLQuery::SQState::SQ_REPLY;
        req.job_type = SPARQLQuery::SubJobType::FULL_JOB;
        Bundle bundle(req);
        bool ret = send_request(bundle, dst_sid, dst_tid);
        ASSERT(ret == true);
    }

    void sweep_msgs() { }

    bool need_parallel(const SPARQLQuery& r) {
        return (coder.tid_of((r).pqid) < Global::num_proxies
                && r.pattern_step == 0
                && r.start_from_index()
                && (Global::num_servers > 1)
                && (r.job_type != SPARQLQuery::SubJobType::SPLIT_JOB));
    }

    void send_to_workers(SPARQLQuery& req) {
        rmap.put_parent_request(req, Global::num_servers);
        SPARQLQuery sub_query = req;
        ASSERT(req.mt_factor == 1);
        for (int i = 0; i < Global::num_servers; i++) {
            sub_query.qid = -1;
            sub_query.pqid = req.qid;
            // start from the next engine thread
            int dst_tid = (tid + 1 - WUKONG_GPU_AGENT_TID) % Global::num_gpus
                          + WUKONG_GPU_AGENT_TID;
            sub_query.mt_tid = 0;
            sub_query.mt_factor = 1;

            ASSERT(sub_query.job_type != SPARQLQuery::SubJobType::SPLIT_JOB);
            Bundle bundle(sub_query);
            send_request(bundle, i, dst_tid);
        }
    }


    // fork-join or in-place execution
    bool need_fork_join(SPARQLQuery &req) {
        // always need NOT fork-join when executing on single machine
        if (Global::num_servers == 1) return false;

        // always need fork-join mode w/o RDMA
        if (!Global::use_rdma) return true;

        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ASSERT(req.result.var_stat(pattern.subject) == KNOWN_VAR);
        ssid_t start = req.get_pattern().subject;

        return ((req.local_var != start)
                && (req.result.gpu.get_row_num() > 0));
    }

    void execute_sparql_query(SPARQLQuery &req) {
        // encode the lineage of the query (server & thread)
        if (req.qid == -1) req.qid = coder.get_and_inc_qid();

        logstream(LOG_DEBUG) << "#" << sid << " GPUAgent: "
                             << "[" << sid << "-" << tid << "]"
                             << " got a req: r.qid=" << req.qid
                             << ", pqid=" << req.pqid << ", r.state="
                             << (req.state == SPARQLQuery::SQState::SQ_REPLY ? "REPLY" : "REQUEST")
                             << LOG_endl;

        if (need_parallel(req)) {
            send_to_workers(req);
            return;
        }

        // if req is a reply
        if (req.state == SPARQLQuery::SQState::SQ_REPLY) {
            pthread_spin_lock(&rmap_lock);
            rmap.put_reply(req);

            if (!rmap.is_ready(req.pqid)) {
                pthread_spin_unlock(&rmap_lock);
                return; // not ready (waiting for the rest)
            }

            // all sub-queries have done, continue to execute
            req = rmap.get_reply(req.pqid);
            pthread_spin_unlock(&rmap_lock);

            send_reply(req, coder.sid_of(req.pqid), coder.tid_of(req.pqid));
            return;
        }

        // execute patterns of SPARQL query
        while (true) {
            ASSERT(req.dev_type == SPARQLQuery::DeviceType::GPU);
            ASSERT(req.has_pattern());

            if (!req.done(SPARQLQuery::SQState::SQ_PATTERN)) {
                if (!gpu_engine->result_buf_ready(req))
                    gpu_engine->load_result_buf(req);

                gpu_engine->execute_one_pattern(req);
            }

            // if req has been finished
            if (req.done(SPARQLQuery::SQState::SQ_PATTERN)) {
                // only send back row_num in blind mode
                req.result.row_num = req.result.get_row_num();
                send_reply(req, coder.sid_of(req.pqid), coder.tid_of(req.pqid));
                break;
            }

            if (need_fork_join(req)) {
                logstream(LOG_DEBUG) << "#" << sid << " GPUAgent: fork query r.qid="
                                     << req.qid << ", r.pqid="
                                     << req.pqid << LOG_endl;
                vector<SPARQLQuery> sub_reqs = gpu_engine->generate_sub_query(req);
                ASSERT(sub_reqs.size() == Global::num_servers);

                // clear parent's result buf after generating sub-jobs
                req.result.gpu.clear_rbuf();
                rmap.put_parent_request(req, sub_reqs.size());
                for (int i = 0; i < sub_reqs.size(); i++) {
                    if (i != sid)
                        send_sub_query(sub_reqs[i], i, tid);
                    else
                        runqueue.push(sub_reqs[i]);
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

            SPARQLQuery req;
            if (runqueue.try_pop(req)) {
                execute_sparql_query(req);
                continue; // exhaust all queries
            }

            Bundle bundle;
            int sender = 0;
            while (adaptor->tryrecv(bundle, sender)) {
                if (bundle.type == SPARQL_QUERY) {
                    // To be fair, agent will handle sub-queries first, instead of a new job.
                    SPARQLQuery req = bundle.get_sparql_query();
                    ASSERT(req.dev_type == SPARQLQuery::DeviceType::GPU);

                    // We need to wait for the result buffer if it is sent by GPUDirect RDMA.
                    if (req.job_type == SPARQLQuery::SubJobType::SPLIT_JOB
                            && req.result.gpu.rbuf_num_elems() > 0) {
                        // recv result buffer
                        std::string rbuf_str;
                        rbuf_str = adaptor->recv(sender);
                        ASSERT(rbuf_str.size() > 0);
                        ASSERT(rbuf_str.empty() == false);
                        gpu_engine->load_result_buf(req, rbuf_str);
                    }

                    if (req.priority != 0) {
                        execute_sparql_query(req);
                        break;
                    }

                    runqueue.push(req);
                } else {
                    ASSERT(false);
                }
            }
        }
    }

};

#endif  // USE_GPU
