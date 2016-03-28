#pragma once
#include "message_wrap.h"
#include "distributed_graph.h"
#include "query_basic_types.h"
#include "global_cfg.h"
#include "thread_cfg.h"
#include "wait_queue.h"

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

class server{
    distributed_graph& g;
	thread_cfg* cfg;
    wait_queue wqueue;
    uint64_t last_time;
    pthread_spinlock_t recv_lock;
    pthread_spinlock_t wqueue_lock;
    vector<request_or_reply> msg_fast_path;

    // all of these means const predict
    void const_to_unknown(request_or_reply& req);
    void const_to_known(request_or_reply& req);
    void known_to_unknown(request_or_reply& req);
    void known_to_known(request_or_reply& req);
    void known_to_const(request_or_reply& req);
    void index_to_unknown(request_or_reply& req);

    // unknown_predict
    void const_unknown_unknown(request_or_reply& req);
    void known_unknown_unknown(request_or_reply& req);

    vector<request_or_reply> generate_sub_requests(request_or_reply& r);
    bool need_sub_requests(request_or_reply& req);
    bool execute_one_step(request_or_reply& req);
    void execute(request_or_reply& req);

    server** s_array;// array of server pointers
public:
    server(distributed_graph& _g,thread_cfg* _cfg);
    void set_server_array(server** array){
        s_array=array;
    };
    void run();
};
