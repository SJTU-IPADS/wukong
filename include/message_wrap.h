#pragma once

#include "query_basic_types.h"
#include "network_node.h"
#include "rdma_resource.h"
#include "thread_cfg.h"
#include "global_cfg.h"


void SendR(thread_cfg* cfg,int r_mid,int r_tid,request_or_reply& r);
request_or_reply RecvR(thread_cfg* cfg);

template<typename T>
void SendObject(thread_cfg* cfg,int r_mid,int r_tid,T& r){
    std::stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << r;
    cfg->node->Send(r_mid,r_tid,ss.str());
}

template<typename T>
T RecvObject(thread_cfg* cfg){
    std::string str;
    str=cfg->node->Recv();
    std::stringstream s;
    s << str;
    boost::archive::binary_iarchive ia(s);
    T r;
    ia >> r;
    return r;
}
