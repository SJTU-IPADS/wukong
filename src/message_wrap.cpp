#include "message_wrap.h"

void SendR(thread_cfg* cfg,int r_mid,int r_tid,request_or_reply& r){
    std::stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << r;
    if(global_use_rbf){
        cfg->rdma->rbfSend(cfg->t_id,r_mid, r_tid, ss.str().c_str(),ss.str().size());
    } else {
        cfg->node->Send(r_mid,r_tid,ss.str());
    }
}

request_or_reply RecvR(thread_cfg* cfg){
    std::string str;
    if(global_use_rbf){
        str=cfg->rdma->rbfRecv(cfg->t_id);
    } else {
        str=cfg->node->Recv();
    }
    std::stringstream s;
    s << str;
    boost::archive::binary_iarchive ia(s);
    request_or_reply r;
    ia >> r;
    return r;
}

bool TryRecvR(thread_cfg* cfg,request_or_reply& r){
    std::string str;
    if(global_use_rbf){
        bool ret=cfg->rdma->rbfTryRecv(cfg->t_id,str);
        if(!ret) {
            return false;
        }
    } else {
        str=cfg->node->tryRecv();
        if(str==""){
            return false;
        }
    }
    std::stringstream s;
    s << str;
    boost::archive::binary_iarchive ia(s);
    ia >> r;
    return true;
};
