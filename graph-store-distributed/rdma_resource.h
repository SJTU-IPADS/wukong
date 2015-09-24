#ifndef RDMARESOURCE_H
#define RDMARESOURCE_H

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
#include <infiniband/verbs.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#include <vector>
struct config_t
{
  const char *dev_name;         /* IB device name */
  char *server_name;            /* server host name */
  u_int32_t tcp_port;           /* server TCP port */
  int ib_port;                  /* local IB port to work with */
  int gid_idx;                  /* gid index to use */
};

/* structure to exchange data which is needed to connect the QPs */
struct cm_con_data_t
{
  uint64_t addr;                /* Buffer address */
  uint32_t rkey;                /* Remote key */
  uint32_t qp_num;              /* QP number */
  uint16_t lid;                 /* LID of the IB port */
  uint8_t gid[16];              /* gid */
} __attribute__ ((packed));
/* structure of system resources */

struct dev_resource {
  struct ibv_device_attr device_attr;   /* Device attributes */
  struct ibv_port_attr port_attr;       /* IB port attributes */
  struct ibv_context *ib_ctx;   /* device handle */

  struct ibv_pd *pd;            /* PD handle */
  struct ibv_mr *mr;            /* MR handle for buf */
  char *buf;                    /* memory buffer pointer, used for RDMA and send*/

};

struct QP {
  struct cm_con_data_t remote_props;  /* values to connect to remote side */
  struct ibv_pd *pd;            /* PD handle */
  struct ibv_cq *cq;            /* CQ handle */
  struct ibv_qp *qp;            /* QP handle */
  struct ibv_mr *mr;            /* MR handle for buf */

  struct dev_resource *dev;
  
};


struct normal_op_req 
{
  ibv_wr_opcode opcode;
  char *local_buf;
  int size; //default set to sizeof(uint64_t)
  int remote_offset;
  
  //for atomicity operations
  uint64_t compare_and_add;
  uint64_t swap;
  
  //for internal usage!!
  struct ibv_send_wr sr;
  struct ibv_sge sge;
  
};

  
  class RdmaResource {

    //site configuration settings
    int _total_partition = -1;
    int _total_threads = -1;
    int _current_partition = -1;
  

    struct dev_resource *dev0;//for remote usage
    struct dev_resource *dev1;//for local usage
    
    struct QP **res;
    struct QP  *own_res;
    
    uint64_t size;//The size of the rdma region,should be the same across machines!
    uint64_t off ;//The offset to send message
    char *buffer;
    
    int rdmaOp(int t_id,int m_id,char*buf,uint64_t size,uint64_t off,int op) ;
    int batch_rdmaOp(int t_id,int m_id,char*buf,uint64_t size,uint64_t off,int op) ;
    
    void init();
    
  public:
    uint64_t get_size(){
      return size;
    }
    char * get_buffer(){
      return buffer;
    }
    //rdma location hashing
    uint64_t slotsize;
    uint64_t rbf_size;
    Network_Node* node;
    
    //for testing
    RdmaResource(int t_partition,int t_threads,int current,char *_buffer,uint64_t _size,uint64_t _slotsize,uint64_t _off = 0);
    
    void Connect();
    void Servicing();
    
    //0 on success,-1 otherwise
    int RdmaRead(int t_id,int m_id,char *local,uint64_t size,uint64_t remote_offset);
    int RdmaWrite(int t_id,int m_id,char *local,uint64_t size,uint64_t remote_offset);
    int RdmaCmpSwap(int t_id,int m_id,char*local,uint64_t compare,uint64_t swap,uint64_t size,uint64_t off);
    int post(int t_id,int machine_id,char* local,uint64_t size,uint64_t remote_offset,int op);
    int poll(int t_id,int machine_id);

    //TODO what if batched?    
    inline char *RdmaResource::GetMsgAddr(int t_id) {
      return (char *)( buffer + off + t_id * slotsize);
    }
    static void* RecvThread(void * arg);



    inline int global_tid(int mid,int tid){
      return mid*_total_threads+tid;
    }
    struct rbfMeta{
      uint64_t local_tail; // used for polling
      uint64_t remote_tail; // directly write to remote_tail of remote machine
      uint64_t local_head; 
      char padding1[64-3*sizeof(uint64_t)];
      uint64_t copy_of_remote_head;
      char padding2[64-1*sizeof(uint64_t)];
    };
    uint64_t rbfOffset(int src_mid,int src_tid,int dst_mid,int dst_tid){
      uint64_t result=off+(_total_threads+src_tid) * slotsize;
      result=result+rbf_size*(dst_mid*_total_threads+dst_tid);
      return result;
    }

    void rbfSend(int local_tid,int remote_mid,int remote_tid,std::string& str){
      char * rbf_ptr=buffer+rbfOffset(_current_partition,local_tid,remote_mid,remote_tid);
      struct rbfMeta* meta=(rbfMeta*) rbf_ptr;
      uint64_t remote_rbf_offset=rbfOffset(remote_mid,remote_tid,_current_partition,local_tid);
      //TODO check whether we can send

      //Send data 
      uint64_t rbf_datasize=rbf_size-sizeof(rbfMeta);
      uint64_t padding=str.size() % sizeof(uint64_t);
      if(padding!=0)
        padding=sizeof(uint64_t)-padding;
      uint64_t total_write_size=sizeof(uint64_t)*2+str.size()+padding;
      if(meta->remote_tail / rbf_datasize != (meta->remote_tail+total_write_size-1)/ rbf_datasize ){
        //printf("send too many message\n");
        //assert(false);
      }

      if(_current_partition==remote_mid){
        // directly write
        char * ptr=buffer+remote_rbf_offset+sizeof(rbfMeta);
        *((uint64_t*)(ptr+(meta->remote_tail)%rbf_datasize))=str.size();
        (meta->remote_tail)+=sizeof(uint64_t);
        for(uint64_t i=0;i<str.size();i++){
          *(ptr+(meta->remote_tail)%rbf_datasize)=str[i];
          meta->remote_tail++;
        }
        meta->remote_tail+=padding;
        *((uint64_t*)(ptr+(meta->remote_tail)%rbf_datasize))=str.size();
        (meta->remote_tail)+=sizeof(uint64_t);
        //printf("tid=%d write to (%d,%d),tail=%ld\n",local_tid,remote_mid,remote_tid,meta->remote_tail);
      } else {
          char* local_buffer=GetMsgAddr(local_tid);
          *((uint64_t*)local_buffer)=str.size();
          local_buffer+=sizeof(uint64_t);
          memcpy(local_buffer,str.c_str(),str.size());
          local_buffer+=str.size()+padding;
          *((uint64_t*)local_buffer)=str.size();
          if(meta->remote_tail / rbf_datasize == (meta->remote_tail+total_write_size-1)/ rbf_datasize ){
            uint64_t remote_msg_offset=remote_rbf_offset+sizeof(rbfMeta)+meta->remote_tail;
            RdmaWrite(local_tid,remote_mid,GetMsgAddr(local_tid),total_write_size,remote_msg_offset);
          } else {
            // we need to post 2 RdmaWrite
            uint64_t length1=rbf_datasize - (meta->remote_tail % rbf_datasize);
            uint64_t length2=total_write_size-length1;
            uint64_t remote_msg_offset1=remote_rbf_offset+sizeof(rbfMeta)+meta->remote_tail;
            uint64_t remote_msg_offset2=remote_rbf_offset+sizeof(rbfMeta);
            RdmaWrite(local_tid,remote_mid,GetMsgAddr(local_tid),length1,remote_msg_offset1);
            RdmaWrite(local_tid,remote_mid,GetMsgAddr(local_tid)+length1,length2,remote_msg_offset2);
          }
          meta->remote_tail =meta->remote_tail+total_write_size;
      }
    }
    std::string rbfRecv(int local_tid){
      while(true){
        for(int mid=0;mid<_total_partition;mid++){
          for(int tid=0;tid<_total_threads;tid++){
            char * rbf_ptr=buffer+rbfOffset(_current_partition,local_tid,mid,tid);
            char * rbf_data_ptr=rbf_ptr+ sizeof(rbfMeta);
            uint64_t rbf_datasize=rbf_size-sizeof(rbfMeta);
            struct rbfMeta* meta=(rbfMeta*) rbf_ptr;
            uint64_t msg_size=*(volatile uint64_t*)(rbf_data_ptr+meta->local_tail%rbf_datasize );
            if(msg_size==0)
              continue;
            *(uint64_t*)(rbf_data_ptr+meta->local_tail%rbf_datasize)=0;
            //have message
            uint64_t padding=msg_size % sizeof(uint64_t);
            if(padding!=0)
              padding=sizeof(uint64_t)-padding;
            volatile uint64_t * msg_end_ptr=(uint64_t*)(rbf_data_ptr+ (meta->local_tail+msg_size+padding+sizeof(uint64_t))%rbf_datasize);
            while(*msg_end_ptr !=msg_size){};
            *msg_end_ptr=0;
            std::string result;
            for(uint64_t i=0;i<msg_size;i++){
              char * tmp=rbf_data_ptr+(meta->local_tail+sizeof(uint64_t)+i)%rbf_datasize;
              result.push_back(*tmp);
              *tmp=0;
            }
            meta->local_tail+=msg_size+padding+2*sizeof(uint64_t);
            return result;
          }
        }
      }
    }
  };


#endif

