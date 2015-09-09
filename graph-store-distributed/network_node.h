#pragma once

#include <zmq.hpp>
#include <string>
#include <iostream>
#include <unistd.h>
#include <unordered_map>
#include <fstream>
#include <errno.h>

#include <sstream> 
#include "request.h"

class Network_Node {
  
public:
  int pid;
  int nid;
  zmq::context_t context;
  zmq::socket_t* receiver;
  
  std::vector<std::string> net_def;
  std::unordered_map<int,zmq::socket_t*> socket_map;
  inline int hash(int _pid,int _nid){
    return _pid*200+_nid;
  }
  
 Network_Node(int _pid,int _nid):nid(_nid),pid(_pid),context(1){
    
    net_def.push_back("10.0.0.100");
    net_def.push_back("10.0.0.101");
    net_def.push_back("10.0.0.102");
    net_def.push_back("10.0.0.103");
    net_def.push_back("10.0.0.104");
    net_def.push_back("10.0.0.105");
  
    receiver=new zmq::socket_t(context, ZMQ_PULL);
    char address[30]="";
    sprintf(address,"tcp://*:%d",5500+hash(pid,nid));
    fprintf(stdout,"tcp binding address %s\n",address);
    receiver->bind (address);
  }
  
  ~Network_Node(){
    for(auto iter:socket_map){
        if(iter.second!=NULL){
                delete iter.second;
                iter.second=NULL;
        }
    }
    delete receiver;
  }

  void Send(int _pid,int _nid,std::string msg){
    int id=hash(_pid,_nid);
    if(socket_map.find(id)== socket_map.end()){
      socket_map[id]=new zmq::socket_t(context, ZMQ_PUSH);
      char address[30]="";

      snprintf(address,30,"tcp://%s:%d",net_def[_pid].c_str(),5500 + id);
      //fprintf(stdout,"mul estalabish %s\n",address);

      socket_map[id]->connect (address);
    }
    zmq::message_t request(msg.length());
    memcpy ((void *) request.data(), msg.c_str(), msg.length());
    socket_map[id]->send(request);
  }
  
  std::string Recv(){
    zmq::message_t reply;
    if(receiver->recv(&reply) < 0) {
      fprintf(stderr,"recv with error %s\n",strerror(errno));
      exit(-1);
    }
    return std::string((char *)reply.data(),reply.size());
  }

  std::string tryRecv(){
    zmq::message_t reply;
    if (receiver->recv(&reply, ZMQ_NOBLOCK))
      return std::string((char *)reply.data(),reply.size());
    else
      return "";
  }

  void SendReq(int _pid,int _nid,request& r){
    stringstream ss;
    boost::archive::text_oarchive oa(ss);
    oa << r;
    Send(_pid,_nid,ss.str());
  }
  request RecvReq(){
    std::string str=Recv();
    stringstream s;
    s << str;
    boost::archive::text_iarchive ia(s);
    request r;
    ia >> r;
    return r;
  }
};

