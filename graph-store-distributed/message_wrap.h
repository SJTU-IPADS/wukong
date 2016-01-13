#pragma once 

#include "request.h"
#include "network_node.h"
#include "profile.h"
#include "thread_cfg.h"

#include "global_cfg.h"

#define USE_RBF 1
#include <vector>

void SendReq(thread_cfg* cfg,int r_mid,int r_tid,request& r,profile* profile_ptr=NULL){
    std::stringstream ss;
    //boost::archive::text_oarchive oa(ss);
    boost::archive::binary_oarchive oa(ss);
    
    oa << r;
    if(profile_ptr!=NULL){
        profile_ptr->record_msgsize(ss.str().size());
    }

    if(global_use_rbf){
        if(ss.str().size() > (cfg->rdma->rbf_size /2) ){
            cout<<"cfg->rdma->rbf_size= "<<  cfg->rdma->rbf_size<<endl;
            cout<<"Too large message size = "<<ss.str().size()*1.0/cfg->rdma->rbf_size<< " rbf_size" <<endl;
            cout<<"r.row_num()=" <<r.row_num()<<endl;
            r.result_table.clear();
            std::stringstream ss2;
            boost::archive::binary_oarchive oa2(ss2);
            oa2 << r;
            cfg->rdma->rbfSend(cfg->t_id,r_mid, r_tid, ss2.str().c_str(),ss2.str().size());   
            return ;
        }
        cfg->rdma->rbfSend(cfg->t_id,r_mid, r_tid, ss.str().c_str(),ss.str().size());    
    } else {
        cfg->node->Send(r_mid,r_tid,ss.str());
    }


// #if USE_RBF
//     cfg->rdma->rbfSend(cfg->t_id,r_mid, r_tid, ss.str().c_str(),ss.str().size());    
// #else 
//     cfg->node->Send(r_mid,r_tid,ss.str());
// #endif
}

request RecvReq(thread_cfg* cfg){

// #if USE_RBF
//     std::string str=cfg->rdma->rbfRecv(cfg->t_id);
// #else 
//     std::string str=cfg->node->Recv();
// #endif
    std::string str;
    if(global_use_rbf){
        str=cfg->rdma->rbfRecv(cfg->t_id);
    } else {
        str=cfg->node->Recv();
    }    

    std::stringstream s;
    s << str;
    //boost::archive::text_iarchive ia(s);
    boost::archive::binary_iarchive ia(s);
    request r;
    ia >> r;
    return r;
}


void SendStr(thread_cfg* cfg,int r_mid,int r_tid,std::string& str){

#if USE_RBF
    cfg->rdma->rbfSend(cfg->t_id,r_mid, r_tid, str.c_str(),str.size());    
#else 
    cfg->node->Send(r_mid,r_tid,str);
#endif
}

std::string RecvStr(thread_cfg* cfg){

#if USE_RBF
    std::string str=cfg->rdma->rbfRecv(cfg->t_id);
#else 
    std::string str=cfg->node->Recv();
#endif
    return str;
}







void SendVector(thread_cfg* cfg,int r_mid,int r_tid,vector<int>& r){
    std::stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << r;
    cfg->node->Send(r_mid,r_tid,ss.str());
}

vector<int> RecvVector(thread_cfg* cfg){
    std::string str;
    str=cfg->node->Recv();
    std::stringstream s;
    s << str;
    boost::archive::binary_iarchive ia(s);
    vector<int> r;
    ia >> r;
    return r;
}

void SendTables(thread_cfg* cfg,int r_mid,int r_tid,vector<vector<int> >& r){
    std::stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << r;
    cfg->node->Send(r_mid,r_tid,ss.str());
}

vector<vector<int> > RecvTables(thread_cfg* cfg){
    std::string str;
    str=cfg->node->Recv();
    std::stringstream s;
    s << str;
    boost::archive::binary_iarchive ia(s);
    vector<vector<int> > r;
    ia >> r;
    return r;
}
