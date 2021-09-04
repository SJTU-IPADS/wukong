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
#include <algorithm>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/coder.hpp"
#include "core/common/global.hpp"
#include "core/engine/rmap.hpp"
#include "core/network/adaptor.hpp"
#include "core/sparql/query.hpp"

// gpu
#include "gpu_engine.hpp"

// utils
#include "gpu_utils.hpp"
#include "utils/assertion.hpp"

namespace wukong {

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

        Message(int sid, int tid, Bundle& bundle)
            : sid(sid), tid(tid), bundle(bundle) {}
    };

    GPUEngine* gpu_engine = nullptr;
    GPUCache* gpu_cache = nullptr;
    Adaptor* adaptor = nullptr;
    SPARQLEngine* sparql = nullptr;  // for hybrid CPU/GPU execution

    Coder coder;
    RMap rmap;  // a map of replies for pending (fork-join) queries
    pthread_spinlock_t rmap_lock;

    // Task Queue
    std::deque<SPARQLQuery*> task_queue;

    GPUChannel* channels;
    uint32_t channel_rr_cnt = 0;

public:
    int sid;  // server id
    int tid;  // thread id

    GPUAgent(int sid, int tid, Adaptor* adaptor, StringServer* str_server, DGraph* graph, GPUEngine* gpu_engine, GPUCache* gpu_cache)
        : sid(sid), tid(tid), adaptor(adaptor), gpu_engine(gpu_engine), gpu_cache(gpu_cache), coder(sid, tid) {
        pthread_spin_init(&rmap_lock, 0);

        channels = new GPUChannel[Global::num_gpu_channels];
        sparql = new SPARQLEngine(sid, tid, str_server, graph, nullptr, nullptr);

        for (int i = 0; i < Global::num_gpu_channels; ++i) {
            channels[i].init(i, gpu_cache->get_vertex_gaddr(), gpu_cache->get_edge_gaddr(),
                             gpu_cache->get_num_key_blks(), gpu_cache->get_num_value_blks(),
                             gpu_cache->get_nbuckets_kblk(), gpu_cache->get_nentries_vblk());
        }
    }

    ~GPUAgent() {}

    bool send_request(Bundle& bundle, int dst_sid, int dst_tid) {
        if (adaptor->send(dst_sid, dst_tid, bundle))
            return true;
        return false;
    }

    bool send_sub_query(SPARQLQuery& req, int dst_sid, int dst_tid) {
        // #1 send query
        Bundle bundle(req);
        if (adaptor->send(dst_sid, dst_tid, bundle)) {
            // #2 send result buffer
            ASSERT(req.result.gpu.table_size()!=0);
            adaptor->send_dev2host(dst_sid, dst_tid, ((char*)req.result.gpu.rbuf()),
                                   WUKONG_VID_SIZE * req.result.gpu.table_size());
            return true;
        }
        ASSERT(false);
        return false;
    }

    void send_reply(SPARQLQuery& req, int dst_sid, int dst_tid) {
        // only send back meta data in blind mode
        if (req.result.blind) {
            req.shrink();
        }
        req.state = SPARQLQuery::SQState::SQ_REPLY;
        req.job_type = SPARQLQuery::SubJobType::FULL_JOB;
        Bundle bundle(req);
        bool ret = send_request(bundle, dst_sid, dst_tid);
        ASSERT(ret == true);
    }

    bool need_parallel(const SPARQLQuery& r) {
        return (coder.tid_of((r).pqid) < Global::num_proxies 
                && r.pattern_step == 0 && r.start_from_index() 
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
            bool ret = send_request(bundle, i, dst_tid);
            ASSERT(ret);
        }
    }

    // fork-join or in-place execution
    bool need_fork_join(SPARQLQuery& req) {
        ASSERT(!req.combined);
        // always need NOT fork-join when executing on single machine
        if (Global::num_servers == 1) return false;

        // always need fork-join mode w/o RDMA
        if (!Global::use_rdma) return true;

        SPARQLQuery::Pattern& pattern = req.get_pattern();
        ASSERT(req.result.var_stat(pattern.subject) == KNOWN_VAR);
        ssid_t start = req.get_pattern().subject;

        return ((req.local_var != start) 
               && (req.result.get_row_num() >= Global::gpu_threshold));
    }

    bool is_light_query(SPARQLQuery& req) {
        return req.result.get_row_num() < Global::gpu_threshold;
    }

    std::vector<SPARQLQuery*> generate_sub_query_cpu(SPARQLQuery& req) {
        SPARQLQuery::Pattern& pattern = req.get_pattern();
        ssid_t start = pattern.subject;

        // generate sub requests for all servers
        std::vector<SPARQLQuery*> sub_reqs(Global::num_servers);
        for (int i = 0; i < Global::num_servers; i++) {
            SPARQLQuery* req_ptr = new SPARQLQuery;
            req_ptr->pqid = req.qid;
            req_ptr->qid = -1;
            req_ptr->pg_type = (req.pg_type == SPARQLQuery::PGType::UNION) ? SPARQLQuery::PGType::BASIC : req.pg_type;
            req_ptr->pattern_group = req.pattern_group;
            req_ptr->pattern_step = req.pattern_step;
            req_ptr->corun_step = req.corun_step;
            req_ptr->fetch_step = req.fetch_step;
            req_ptr->local_var = start;
            req_ptr->priority = req.priority + 1;
            req_ptr->dev_type = SPARQLQuery::DeviceType::GPU;

            // metadata
            req_ptr->result.col_num = req.result.col_num;
            req_ptr->result.attr_col_num = req.result.attr_col_num;
            req_ptr->result.blind = req.result.blind;
            req_ptr->result.v2c_map = req.result.v2c_map;
            req_ptr->result.nvars = req.result.nvars;
            sub_reqs[i] = req_ptr;
        }

        // group intermediate results to servers
        int nrows = req.result.get_row_num();
        for (int i = 0; i < nrows; i++) {
            int dst_sid = wukong::math::hash_mod(req.result.get_row_col(i, req.result.var2col(start)),
                                                 Global::num_servers);
            req.result.append_row_to(i, sub_reqs[dst_sid]->result.result_table);
            req.result.append_attr_row_to(i, sub_reqs[dst_sid]->result.attr_res_table);

            if (req.pg_type == SPARQLQuery::PGType::OPTIONAL)
                sub_reqs[dst_sid]->result.optional_matched_rows.push_back(req.result.optional_matched_rows[i]);
        }

        for (int i = 0; i < Global::num_servers; i++)
            sub_reqs[i]->result.update_nrows();

        return sub_reqs;
    }

    // unload result buffer from GPU to CPU
    void unload_rbuf(SPARQLQuery* req_ptr) {
        ASSERT(req_ptr->result.gpu.valid());
        SPARQLQuery::Result& res = req_ptr->result;
        sid_t* rbuf_d = res.gpu.rbuf();
        uint32_t size = res.gpu.table_size();
        std::vector<sid_t>& table = res.result_table;

        table.clear();
        table.resize(size);
        thrust::device_ptr<sid_t> dptr(rbuf_d);
        thrust::copy(dptr, dptr + size, table.begin());

        res.set_device(SPARQLQuery::DeviceType::CPU);
        res.gpu.clear();
        res.check_and_sync();
        req_ptr->job_type = SPARQLQuery::SubJobType::FULL_JOB;
    }

    void push_into_waiting_queue(SPARQLQuery* req) {
        // encode the lineage of the query (server & thread)
        if (req->qid == -1) req->qid = coder.get_and_inc_qid();
        ASSERT(req->combined == false);

        // #1 large individual query: directly pushed into queue
        if (!req->result.is_medium()) {
            task_queue.push_back(req);
            return;
        }

        // #2 small individual query: directly pushed into queue
        if (is_light_query(*req)) {
            task_queue.push_back(req);
            return;
        }

        // #3 medium query: merged into a combined query
        SPARQLQuery::PatternType type = req->get_pattern_type();
        for (SPARQLQuery* req_ptr : task_queue) {
            // individual query
            if (!req_ptr->combined) continue;
            CombinedSPARQLQuery* combined_req_ptr = reinterpret_cast<CombinedSPARQLQuery*>(req_ptr);
            if (combined_req_ptr->type == type && combined_req_ptr->add_job(req)) {
                return;
            }
        }

        // #4 no suitable combined query: create a new combined query
        CombinedSPARQLQuery* combined_query = new CombinedSPARQLQuery(type);
        combined_query->add_job(req);
        task_queue.push_back(combined_query);
        return;
    }

    // Always return true when counter a combined query
    bool job_not_done(const SPARQLQuery& req) {
        return ((!req.combined && !req.done(SPARQLQuery::SQState::SQ_PATTERN)) || req.combined);
    }

    /*
    * After a pattern has been executed,
    * update query(on host) meta-data&state 
    */
    void update_query_states(GPUChannel& chnl) {
        ASSERT(chnl.occupier.valid);
        SPARQLQuery* req_ptr = chnl.occupier.job;

        size_t table_size = 0;
        SPARQLQuery::Result& res = req_ptr->result;

        // copy new row num to host
        if (req_ptr->combined) {
            table_size = req_ptr->result.gpu.table_size();
            chnl.para.reset();
        } else {
            int row_num = 0;
            CUDA_ASSERT(cudaMemcpy(&row_num,
                                   chnl.para.gpu.d_prefix_sum_list + chnl.para.query.row_num - 1,
                                   sizeof(int), cudaMemcpyDeviceToHost));
            if (req_ptr->get_pattern_type() == SPARQLQuery::PatternType::K2U) {
                table_size = row_num * (chnl.para.query.col_num + 1);
            } else {
                table_size = row_num * chnl.para.query.col_num;
            }
            res.gpu.set_rbuf(chnl.para.gpu.d_out_rbuf, table_size);
        }

        ASSERT_LT(WUKONG_GPU_RBUF_SIZE(table_size), MiB2B(Global::gpu_rbuf_size_mb));

        // update query states according to pattern type
        // combined.update_nrows() will update the pointer to gpu rbuf for each medium jobs.
        SPARQLQuery::PatternType patternType = req_ptr->get_pattern_type();
        switch (patternType) {
        case SPARQLQuery::PatternType::K2U:
            if (req_ptr->combined)
                req_ptr->update_var2col(0, 0);
            else
                req_ptr->update_var2col(req_ptr->get_pattern().object, res.get_col_num());

            req_ptr->update_ncols(res.get_col_num() + 1);
            req_ptr->update_nrows();
            req_ptr->update_pattern_step();
            break;
        case SPARQLQuery::PatternType::K2C:
        case SPARQLQuery::PatternType::K2K:
            req_ptr->update_nrows();
            req_ptr->update_pattern_step();
            break;
        default:
            ASSERT(false);
        }
    }

    void free_channel(GPUChannel& channel) {
        int qid = channel.occupier.job->qid;
        channel.reset();
    }

    // These two check functions has side effect: after check, the job is unloaded from GPU.
    bool need_suspend(SPARQLQuery& req, GPUChannel& channel) {
        if (req.combined)
            return check_and_unload_combined(req, channel);
        else
            return check_and_unload_single(req, channel);
    }

    // check if the combined job need suspend
    bool check_and_unload_combined(SPARQLQuery& req, GPUChannel& channel) {
        CombinedSPARQLQuery& combined = static_cast<CombinedSPARQLQuery&>(req);

        int k2c_rows = 0, k2k_rows = 0, k2u_rows = 0;
        auto& medium_jobs = combined.get_jobs();

        // first round, remove queries(done, need_fork_join), calculate rows of three patterns
        for (auto it = medium_jobs.begin(); it != medium_jobs.end();) {
            SPARQLQuery* req_ptr = it->req_ptr;
            if (req_ptr->done(SPARQLQuery::SQState::SQ_PATTERN)) {
                unload_rbuf(req_ptr);
                send_reply(*req_ptr, coder.sid_of(req_ptr->pqid), coder.tid_of(req_ptr->pqid));
                delete req_ptr;
                it = medium_jobs.erase(it);
            } else if (req_ptr->result.get_row_num() == 0) {
                req_ptr->result.set_device(SPARQLQuery::DeviceType::CPU);
                req_ptr->result.clear();
                req_ptr->pattern_step = req_ptr->pattern_group.patterns.size();
                req_ptr->result.check_and_sync();
                send_reply(*req_ptr, coder.sid_of(req_ptr->pqid), coder.tid_of(req_ptr->pqid));
                // remove from combined query
                delete req_ptr;
                it = medium_jobs.erase(it);
            } else if (need_fork_join(*req_ptr)) {
                std::vector<SPARQLQuery*> sub_reqs = gpu_engine->generate_sub_query(*req_ptr, channel, combined.qid);
                ASSERT(sub_reqs.size() == Global::num_servers);

                // clear parent's result buf after generating sub-jobs
                req_ptr->result.gpu.clear();
                req_ptr->result.set_device(SPARQLQuery::DeviceType::CPU);

                int sub_req_num = sub_reqs.size() - std::count(sub_reqs.begin(), sub_reqs.end(), nullptr);
                rmap.put_parent_request(*req_ptr, sub_req_num);
                for (int i = 0; i < sub_reqs.size(); i++) {
                    if (sub_reqs[i] == nullptr) continue;
                    if (i != sid) {
                        send_sub_query(*sub_reqs[i], i, tid);
                        delete sub_reqs[i];
                    } else {
                        unload_rbuf(sub_reqs[i]);
                        push_into_waiting_queue(sub_reqs[i]);
                    }
                }
                it = medium_jobs.erase(it);
            } else if (is_light_query(*req_ptr) 
                       && Global::num_servers != 1
                       && req_ptr->local_var != req_ptr->get_pattern().subject) {
                unload_rbuf(req_ptr);
                push_into_waiting_queue(req_ptr);
                it = medium_jobs.erase(it);
            } else {
                if (req_ptr->get_pattern_type() == SPARQLQuery::PatternType::K2C) {
                    k2c_rows += req_ptr->result.get_row_num();
                } else if (req_ptr->get_pattern_type() == SPARQLQuery::PatternType::K2K) {
                    k2k_rows += req_ptr->result.get_row_num();
                } else if (req_ptr->get_pattern_type() == SPARQLQuery::PatternType::K2U) {
                    k2u_rows += req_ptr->result.get_row_num();
                } else {
                    ASSERT(false);
                }
                it++;
            }
        }

        // determine how to do next
        SPARQLQuery::PatternType select_type = SPARQLQuery::PatternType::I2U;
        if (k2c_rows >= k2k_rows 
            && k2c_rows >= k2u_rows 
            && (k2c_rows > Global::pattern_combine_rows / 2 
                || medium_jobs.size() == Global::pattern_combine_window)) {
            select_type = SPARQLQuery::PatternType::K2C;
        } else if (k2u_rows >= k2k_rows 
                   && k2u_rows >= k2c_rows 
                   && (k2u_rows > Global::pattern_combine_rows / 2 
                       || medium_jobs.size() == Global::pattern_combine_window)) {
            select_type = SPARQLQuery::PatternType::K2U;
        } else if (k2k_rows >= k2c_rows 
                   && k2k_rows >= k2u_rows 
                   && (k2k_rows > Global::pattern_combine_rows / 2 
                       || medium_jobs.size() == Global::pattern_combine_window)) {
            select_type = SPARQLQuery::PatternType::K2K;
        }

        // second round, remove queries of other pattern type, remain queries of the same type in the job list
        for (auto it = medium_jobs.begin(); it != medium_jobs.end();) {
            SPARQLQuery* req_ptr = it->req_ptr;
            if (req_ptr->get_pattern_type() == select_type) {
                it++;
            } else {
                unload_rbuf(req_ptr);
                push_into_waiting_queue(req_ptr);
                it = medium_jobs.erase(it);
            }
        }

        if (medium_jobs.empty()) {
            return true;
        } else {
            combined.type = select_type;
            return false;
        }
    }

    // check if the normal job 1.need suspend and wait to be combined or 2.is finished
    bool check_and_unload_single(SPARQLQuery& req, GPUChannel& channel) {
        // if the query is done
        if (req.done(SPARQLQuery::SQState::SQ_PATTERN)) {
            if (req.result.get_row_num() != 0) {
                unload_rbuf(&req);
            } else {
                req.result.set_device(SPARQLQuery::DeviceType::CPU);
                req.result.clear();
            }
            send_reply(req, coder.sid_of(req.pqid), coder.tid_of(req.pqid));
            delete &req;
            return true;
        }

        // fast reply
        if (req.result.get_row_num() == 0) {
            req.result.set_device(SPARQLQuery::DeviceType::CPU);
            req.result.clear();
            req.result.check_and_sync();
            req.pattern_step = req.pattern_group.patterns.size();
            send_reply(req, coder.sid_of(req.pqid), coder.tid_of(req.pqid));
            delete &req;
            return true;
        }

        // check whether the single query need fork and join
        if (need_fork_join(req)) {
            logstream(LOG_INFO) << "#" << sid << " GPUAgent: fork query r.qid=" << req.qid << ", r.pqid=" << req.pqid << LOG_endl;
            std::vector<SPARQLQuery*> sub_reqs = gpu_engine->generate_sub_query(req, channel, req.qid);
            ASSERT(sub_reqs.size() == Global::num_servers);

            // clear parent's result buf after generating sub-jobs
            req.result.gpu.clear();
            req.result.set_device(SPARQLQuery::DeviceType::CPU);

            int sub_req_num = sub_reqs.size() - std::count(sub_reqs.begin(), sub_reqs.end(), nullptr);
            rmap.put_parent_request(req, sub_req_num);
            for (int i = 0; i < sub_reqs.size(); i++) {
                if (sub_reqs[i] == nullptr) continue;
                if (i != sid) {
                    send_sub_query(*sub_reqs[i], i, tid);
                    delete sub_reqs[i];
                } else {
                    unload_rbuf(sub_reqs[i]);
                    push_into_waiting_queue(sub_reqs[i]);
                }
            }
            return true;
        }

        // check whether next pattern is medium or large
        // If the query become a medium/2 query, push into waiting queue
        if (req.result.get_row_num() < Global::gpu_threshold || req.result.is_medium_for_single()) {
            unload_rbuf(&req);
            push_into_waiting_queue(&req);
            return true;
        }

        // otherwise we continue executing this query
        return false;
    }

    /* Get a new query to execute in a Channel */
    SPARQLQuery* get_next_query() {
        SPARQLQuery* query = nullptr;
        // when there is no task in queue, keep fetching new query
        if (task_queue.empty()) {
            fetch_jobs();
            if (task_queue.empty()) return nullptr;
        }
        query = task_queue.front();
        // encounter a not-full combined query, fetch some jobs
        if (query->combined 
            && reinterpret_cast<CombinedSPARQLQuery*>(query)->combined_row_num() < Global::pattern_combine_rows 
            && !reinterpret_cast<CombinedSPARQLQuery*>(query)->full()) {
            fetch_jobs();
        }
        query = task_queue.front();
        task_queue.pop_front();
        // if the combined query only contains a single query, execute it as a single query
        if (query->combined && reinterpret_cast<CombinedSPARQLQuery*>(query)->combined_job_size() == 1) {
            logstream(LOG_INFO) << "get a single query from a combined query!" << LOG_endl;
            CombinedSPARQLQuery* combined_query = reinterpret_cast<CombinedSPARQLQuery*>(query);
            query = combined_query->get_jobs().front().req_ptr;
            combined_query->get_jobs().pop_front();
            delete combined_query;
        }
        return query;
    }

    /* Begin a new round of fetching new jobs */
    void fetch_jobs() {
        Bundle bundle;
        int sender = 0;
        int fetched_jobs_cnt = 0;

        while (adaptor->tryrecv(bundle, sender)) {
            ASSERT(bundle.type == SPARQL_QUERY);
            SPARQLQuery* req = new SPARQLQuery;
            *req = bundle.get_sparql_query();
            // check this query is a GPU query
            ASSERT(req->dev_type == SPARQLQuery::DeviceType::GPU);

            /* If the query is a reply, handle reply immediately */
            if (req->state == SPARQLQuery::SQState::SQ_REPLY) {
                // handle reply from other servers
                pthread_spin_lock(&rmap_lock);
                rmap.put_reply(*req);

                // not ready (waiting for the rest)
                if (!rmap.is_ready(req->pqid)) {
                    pthread_spin_unlock(&rmap_lock);
                } else {  // all sub-queries have done, send back the merged reply
                    *req = rmap.get_reply(req->pqid);
                    pthread_spin_unlock(&rmap_lock);
                    ASSERT(req->done(SPARQLQuery::SQState::SQ_PATTERN));
                    send_reply(*req, coder.sid_of(req->pqid), coder.tid_of(req->pqid));
                }
                delete req;
                continue;
            }

            fetched_jobs_cnt++;

            // encode the lineage of the query (server & thread)
            if (req->qid == -1) req->qid = coder.get_and_inc_qid();
            // if recv a SPLIT_JOB, recv its result which is sent by GPUDirect
            if (req->job_type == SPARQLQuery::SubJobType::SPLIT_JOB) {
                // If the result table is empty, it should not be sent
                ASSERT(req->result.gpu.table_size() > 0);
                // We need to wait for the result buffer if it is sent by GPUDirect RDMA.
                std::string rbuf_str;
                rbuf_str = adaptor->recv(sender);
                ASSERT(!rbuf_str.empty());

                // store rbuf_str to result_table
                sid_t* rbuf = reinterpret_cast<sid_t*>(rbuf_str.data());
                req->result.result_table = std::vector<sid_t>(rbuf, rbuf + req->result.gpu.table_size());
                req->result.set_device(SPARQLQuery::DeviceType::CPU);
                req->result.gpu.clear();
                req->job_type = SPARQLQuery::SubJobType::FULL_JOB;  // set to FULL_JOB
            }
            // push the task into waiting queue
            push_into_waiting_queue(req);

            if (fetched_jobs_cnt >= Global::pattern_combine_window * Global::num_gpu_channels)
                break;
        }  // end of recv loop
    }

    void run() {
        uint32_t idx = 0;
        while (true) {
            if (!Global::use_rdma) {
                logstream(LOG_ERROR) << "For now, GPUAgent cannot run without RDMA, exited." << LOG_endl;
                break;
            }
            idx = (channel_rr_cnt++ % Global::num_gpu_channels);
            auto& channel = channels[idx];

            SPARQLQuery* req_ptr = nullptr;

            if (!channel.taken) {
                req_ptr = get_next_query();
                if (req_ptr == nullptr) continue;
                // encode the lineage of the query (server & thread)
                if (req_ptr->qid == -1) req_ptr->qid = coder.get_and_inc_qid();

                if (need_parallel(*req_ptr)) {
                    send_to_workers(*req_ptr);
                    continue;
                }

                if (req_ptr->pattern_step == 0 && req_ptr->start_from_index()) {
                    // execute the first step
                    gpu_engine->index_to_unknown(*req_ptr);
                    ASSERT(!req_ptr->done(SPARQLQuery::SQState::SQ_PATTERN));
                    ASSERT(!need_fork_join(*req_ptr));
                    push_into_waiting_queue(req_ptr);
                    continue;
                } else if (req_ptr->pattern_step == 0 && req_ptr->start_from_const()) {
                    // execute the first step
                    gpu_engine->const_to_unknown(*req_ptr);
                    ASSERT(!req_ptr->done(SPARQLQuery::SQState::SQ_PATTERN));
                    if (need_fork_join(*req_ptr)) {
                        std::vector<SPARQLQuery*> sub_reqs = generate_sub_query_cpu(*req_ptr);
                        ASSERT(sub_reqs.size() == Global::num_servers);
                        rmap.put_parent_request(*req_ptr, sub_reqs.size());
                        for (int i = 0; i < sub_reqs.size(); i++) {
                            if (i != sid) {
                                Bundle bundle(*sub_reqs[i]);
                                send_request(bundle, i, tid);
                                delete sub_reqs[i];
                            } else {
                                push_into_waiting_queue(sub_reqs[i]);
                            }
                        }
                    } else {
                        push_into_waiting_queue(req_ptr);
                    }
                    continue;
                } else if (!req_ptr->combined && is_light_query(*req_ptr)) {
                    sparql->execute_one_pattern(*req_ptr);
                    if (req_ptr->done(SPARQLQuery::SQState::SQ_PATTERN)) {
                        send_reply(*req_ptr, coder.sid_of(req_ptr->pqid), coder.tid_of(req_ptr->pqid));
                        delete req_ptr;
                    } else if (need_fork_join(*req_ptr)) {
                        std::vector<SPARQLQuery*> sub_reqs = generate_sub_query_cpu(*req_ptr);
                        ASSERT(sub_reqs.size() == Global::num_servers);
                        rmap.put_parent_request(*req_ptr, sub_reqs.size());
                        for (int i = 0; i < sub_reqs.size(); i++) {
                            if (i != sid) {
                                Bundle bundle(*sub_reqs[i]);
                                send_request(bundle, i, tid);
                                delete sub_reqs[i];
                            } else {
                                push_into_waiting_queue(sub_reqs[i]);
                            }
                        }
                    } else {
                        push_into_waiting_queue(req_ptr);
                    }
                    continue;
                } else {
                    channel.set_occupier(req_ptr);
                }
            } else {
                auto poll_result = channel.poll_finish_event();
                if (poll_result == cudaSuccess) {
                    req_ptr = channel.occupier.job;
                    // check for possible error code!
                    if (channel.error_code == GPUErrorCode::GIANT_TOTAL_RESULT_TABLE) {
                        ASSERT(!req_ptr->combined);
                        subquery_list_t split_queries = gpu_engine->split_giant_query(*req_ptr, channel);
                        // clear parent's result buf after generating sub-jobs
                        req_ptr->result.gpu.clear();
                        req_ptr->result.set_device(SPARQLQuery::DeviceType::CPU);

                        int split_query_num = split_queries.size() - std::count(split_queries.begin(), split_queries.end(), nullptr);
                        rmap.put_parent_request(*req_ptr, split_query_num);
                        for (int i = 0; i < split_queries.size(); i++) {
                            if (split_queries[i] == nullptr) continue;
                            unload_rbuf(split_queries[i]);
                            push_into_waiting_queue(split_queries[i]);
                        }
                        channel.error_code = GPUErrorCode::NORMAL;
                        delete channel.error_info;
                        channel.error_info = nullptr;
                        // make this channel available
                        free_channel(channel);
                        gpu_engine->free_result_buf(req_ptr->qid);
                        continue;
                    }

                    // update states of the query, which acts as a callback
                    update_query_states(channel);

                    // check whether the job is finished or should be suspended
                    if (need_suspend(*req_ptr, channel)) {
                        // release the channel
                        free_channel(channel);
                        gpu_engine->free_result_buf(req_ptr->qid);
                        continue;
                    }
                } else {
                    ASSERT(poll_result == cudaErrorNotReady);
                    continue;
                }
            }

            /* If control flow reach here, two case:
            *    1.Finsih one pattern in a channel, don't need to suspend, so we continue to execute
            *    2.A new query is put into a channel and begin to execute
            */

            /* Begin to process one pattern */
            SPARQLQuery& req = *req_ptr;
            ASSERT(job_not_done(req));

            int table_size = req.result.get_col_num() * req.result.get_row_num();
            ASSERT(WUKONG_GPU_RBUF_SIZE(table_size) < MiB2B(Global::gpu_rbuf_size_mb));

            /* gpu engine will allocate a gpu rbuf for the query if needed */
            if (!gpu_engine->result_buf_ready(req)) {
                gpu_engine->load_result_buf(req, channel);
            }

            /* begin to execute this query */
            ASSERT(req.result.gpu.valid());
            gpu_engine->execute_one_pattern(req, channel);

            /* push a finish event to channel */
            channel.add_finish_event();
        }
    }
};

}  // namespace wukong

#endif  // USE_GPU
