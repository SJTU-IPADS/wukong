#pragma once 

#include "request.h"
#include "network_node.h"
#include "profile.h"

#define USE_RBF 0

#if USE_RBF

void SendReq(RdmaResource* rdma,Network_Node* node,int tid,int r_mid,int r_tid,request& r,profile* profile_ptr=NULL){
	std::stringstream ss;
    boost::archive::text_oarchive oa(ss);
    oa << r;
	if(profile_ptr!=NULL){
      profile_ptr->record(ss.str().size());
    }
    std::string str=ss.str();
	rdma->rbfSend(tid,r_mid, r_tid, str);    
}

request RecvReq(RdmaResource* rdma,Network_Node* node,int tid){
	std::string str=rdma->rbfRecv(tid);
    std::stringstream s;
    s << str;
    boost::archive::text_iarchive ia(s);
    request r;
    ia >> r;
    return r;
}

#else 


void SendReq(RdmaResource* rdma,Network_Node* node,int tid,int r_mid,int r_tid,request& r,profile* profile_ptr=NULL){
	node->SendReq(r_mid,r_tid,r,profile_ptr);
}

request RecvReq(RdmaResource* rdma,Network_Node* node,int tid){
	return node->RecvReq();
}

#endif