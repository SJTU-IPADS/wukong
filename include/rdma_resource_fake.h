#pragma once

#ifdef USE_ZEROMQ
#include "network_node.h"

#pragma GCC diagnostic warning "-fpermissive"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>
#include <endian.h>
#include <byteswap.h>
#include <getopt.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include "timer.h"

#include <vector>
#include <pthread.h>


class RdmaResource {

  //site configuration settings
  int _total_partition = -1;
  int _total_threads = -1;
  int _current_partition = -1;


  uint64_t size;//The size of the rdma region,should be the same across machines!
  uint64_t off ;//The offset to send message
  char *buffer;

public:
  uint64_t get_memorystore_size(){
    //[0-off) can be used;
    //[off,size) should be reserve
    return off;
  }
  char * get_buffer(){
    return buffer;
  }
  uint64_t get_slotsize(){
    return rdma_slotsize;
  }
  //rdma location hashing
  uint64_t rdma_slotsize;
  uint64_t msg_slotsize;
  uint64_t rbf_size;
  Network_Node* node;

  //for testing
  RdmaResource(int t_partition,int t_threads,int current,char *_buffer,uint64_t _size,
                                        uint64_t rdma_slot,uint64_t msg_slot,uint64_t _off) {

    _total_threads = t_threads;
    _total_partition = t_partition;
    _current_partition = current;

    buffer = _buffer;
    size   = _size;

    off = _off;
    rdma_slotsize = rdma_slot;
    msg_slotsize = msg_slot;
    rbf_size=msg_slotsize/(_total_partition);
    rbf_size=rbf_size-(rbf_size%64);
  }
  void Connect(){assert(false);};
  void Servicing(){assert(false);};

  int RdmaRead(int t_id,int m_id,char *local,uint64_t size,uint64_t remote_offset){
      assert(false);
      return 0;
  };
  int RdmaWrite(int t_id,int m_id,char *local,uint64_t size,uint64_t remote_offset){
      assert(false);
      return 0;
  };
  int RdmaCmpSwap(int t_id,int m_id,char*local,uint64_t compare,uint64_t swap,uint64_t size,uint64_t off){
      assert(false);
      return 0;
  };
  // int post(int t_id,int machine_id,char* local,uint64_t size,uint64_t remote_offset,ibv_wr_opcode op){
  //     assert(false);
  //     return 0;
  // };
  // int poll(int t_id,int machine_id){
  //     assert(false);
  //     return 0;
  // };

  //TODO what if batched?
  inline char *GetMsgAddr(int t_id) {
      assert(false);
      return (char *)( buffer + off + t_id * rdma_slotsize);
  }


  void rbfSend(int local_tid,int remote_mid,int remote_tid,const char * str_ptr, uint64_t str_size){
      assert(false);
      return ;
  }

  std::string rbfRecv(int local_tid){
      assert(false);
      return std::string("");
  }
  bool rbfTryRecv(int local_tid, std::string& ret){
      assert(false);
      return false;
  }
};

#endif
