#pragma once

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
#include "timer.h"

#include <vector>
#include <pthread.h>
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

  struct per_thread_metadata{
    int prev_recv_tid;
    int prev_recv_mid;
    uint64_t own_count;
    uint64_t recv_count;
    uint64_t steal_count[40];
    pthread_spinlock_t recv_lock;
    char padding1[64];
    volatile bool need_help;
    char padding2[64];
    void lock(){
      pthread_spin_lock(&recv_lock);
    }
    bool trylock(){
      return pthread_spin_trylock(&recv_lock);
    }
    void unlock(){
      pthread_spin_unlock(&recv_lock);
    }
    per_thread_metadata(){
      need_help=false;
      pthread_spin_init(&recv_lock,0);
      own_count=0;
      recv_count=0;
      for(int i=0;i<40;i++){
        steal_count[i]=0;
      }
    }
  };
  class RdmaResource {

    //site configuration settings
    int _total_partition = -1;
    int _total_threads = -1;
    int _current_partition = -1;

    per_thread_metadata local_meta[40];
    struct dev_resource *dev0;//for remote usage
    struct dev_resource *dev1;//for local usage

    struct QP **res;
    struct QP  *own_res;

    uint64_t size;//The size of the rdma region,should be the same across machines!
    uint64_t off ;//The offset to send message
    char *buffer;

    int rdmaOp(int t_id,int m_id,char*buf,uint64_t size,uint64_t off,ibv_wr_opcode op) ;
    int batch_rdmaOp(int t_id,int m_id,char*buf,uint64_t size,uint64_t off,ibv_wr_opcode op) ;

    void init();

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
      return slotsize;
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
    int post(int t_id,int machine_id,char* local,uint64_t size,uint64_t remote_offset,ibv_wr_opcode op);
    int poll(int t_id,int machine_id);

    //TODO what if batched?
    inline char *GetMsgAddr(int t_id) {
      return (char *)( buffer + off + t_id * slotsize);
    }
    static void* RecvThread(void * arg);



    inline int global_tid(int mid,int tid){
      return mid*_total_threads+tid;
    }

    struct RemoteQueueMeta{ //used to send message to remote queue
      uint64_t remote_tail; // directly write to remote_tail of remote machine
      pthread_spinlock_t remote_lock;
      char padding1[64];
      RemoteQueueMeta(){
        remote_tail=0;
        pthread_spin_init(&remote_lock,0);
      }
      void lock(){
        pthread_spin_lock(&remote_lock);
      }
      bool trylock(){
        return pthread_spin_trylock(&remote_lock);
      }
      void unlock(){
        pthread_spin_unlock(&remote_lock);
      }
    };
    struct LocalQueueMeta{
      uint64_t local_tail; // recv from here
      pthread_spinlock_t local_lock;
      char padding1[64];
      LocalQueueMeta(){
        local_tail=0;
        pthread_spin_init(&local_lock,0);
      }
      void lock(){
        pthread_spin_lock(&local_lock);
      }
      bool trylock(){
        return pthread_spin_trylock(&local_lock);
      }
      void unlock(){
        pthread_spin_unlock(&local_lock);
      }
    };
    std::vector<std::vector<RemoteQueueMeta> > RemoteMeta; //RemoteMeta[0..m-1][0..t-1]
    std::vector<std::vector< LocalQueueMeta> > LocalMeta;  //LocalMeta[0..t-1][0..m-1]
    uint64_t inline ceil(uint64_t original,uint64_t n){
      if(n==0){
        assert(false);
      }
      if(original%n == 0){
        return original;
      }
      return original - original%n +n;
    }
    uint64_t start_of_recv_queue(int local_tid,int remote_mid){
      //[t0,m0][t0,m1] [t0,m5], [t1,m0],...
      uint64_t result=off+(_total_threads) * slotsize; //skip data-region and rdma_read-region
      result=result+rbf_size*(local_tid*_total_partition+remote_mid);
      return result;
    }


    void rbfSend(int local_tid,int remote_mid,int remote_tid,const char * str_ptr, uint64_t str_size){
      RemoteQueueMeta * meta=&RemoteMeta[remote_mid][remote_tid];
      meta->lock();
      uint64_t remote_start=start_of_recv_queue(remote_tid,_current_partition);
      if(_current_partition==remote_mid){
        char * ptr=buffer+remote_start;
        uint64_t tail=meta->remote_tail;
        (meta->remote_tail)+=sizeof(uint64_t)*2+ceil(str_size,sizeof(uint64_t));
        meta->unlock();

        *((uint64_t*)(ptr+ (tail)%rbf_size )) = str_size;
        tail+=sizeof(uint64_t);
        for(uint64_t i=0;i<str_size;i++){
          *(ptr+(tail+i)%rbf_size)=str_ptr[i];
        }
        tail+=ceil(str_size,sizeof(uint64_t));
        *((uint64_t*)(ptr+(tail)%rbf_size))=str_size;

      } else {
        uint64_t total_write_size=sizeof(uint64_t)*2+ceil(str_size,sizeof(uint64_t));
        char* local_buffer=GetMsgAddr(local_tid);
        *((uint64_t*)local_buffer)=str_size;
        local_buffer+=sizeof(uint64_t);
        memcpy(local_buffer,str_ptr,str_size);
        local_buffer+=ceil(str_size,sizeof(uint64_t));
        *((uint64_t*)local_buffer)=str_size;
        uint64_t tail=meta->remote_tail;
        meta->remote_tail =meta->remote_tail+total_write_size;
        meta->unlock();
        if(tail/ rbf_size == (tail+total_write_size-1)/ rbf_size ){
          uint64_t remote_msg_offset=remote_start+(tail% rbf_size);
         RdmaWrite(local_tid,remote_mid,GetMsgAddr(local_tid),total_write_size,remote_msg_offset);
        } else {
          uint64_t length1=rbf_size - (tail % rbf_size);
          uint64_t length2=total_write_size-length1;
          uint64_t remote_msg_offset1=remote_start+(tail% rbf_size);
          uint64_t remote_msg_offset2=remote_start;
          RdmaWrite(local_tid,remote_mid,GetMsgAddr(local_tid),length1,remote_msg_offset1);
          RdmaWrite(local_tid,remote_mid,GetMsgAddr(local_tid)+length1,length2,remote_msg_offset2);
        }
      }
    }
    bool check_rbf_msg(int local_tid,int mid){
      LocalQueueMeta * meta=&LocalMeta[local_tid][mid];
      char * rbf_ptr=buffer+start_of_recv_queue(local_tid,mid);
      uint64_t msg_size=*(volatile uint64_t*)(rbf_ptr+meta->local_tail%rbf_size );
      if(msg_size==0){
        return false;
      }
      return true;
    }
    std::string fetch_rbf_msg(int local_tid,int mid){
      LocalQueueMeta * meta=&LocalMeta[local_tid][mid];

      char * rbf_ptr=buffer+start_of_recv_queue(local_tid,mid);
      uint64_t msg_size=*(volatile uint64_t*)(rbf_ptr+meta->local_tail%rbf_size );

      //clear head
      *(uint64_t*)(rbf_ptr+(meta->local_tail)%rbf_size)=0;

      uint64_t skip_size=sizeof(uint64_t)+ceil(msg_size,sizeof(uint64_t));
      volatile uint64_t * msg_end_ptr=(uint64_t*)(rbf_ptr+ (meta->local_tail+skip_size)%rbf_size);
      while(*msg_end_ptr !=msg_size){
        uint64_t tmp=*msg_end_ptr;
        if(tmp!=0 && tmp!=msg_size){
          printf("waiting for %ld,but actually %ld\n",msg_size,tmp);
          exit(0);
        }
      }
      //clear tail
      *msg_end_ptr=0;

      std::string result;
      for(uint64_t i=0;i<ceil(msg_size,sizeof(uint64_t));i++){
        char * tmp=rbf_ptr+(meta->local_tail+sizeof(uint64_t)+i)%rbf_size;
        if(i<msg_size)
          result.push_back(*tmp);
        //clear data
        *tmp=0;
      }
      meta->local_tail+=2*sizeof(uint64_t)+ceil(msg_size,sizeof(uint64_t));
      return result;
    }
    std::string rbfRecv(int local_tid){
      while(true){
        int mid=local_meta[local_tid].own_count % _total_partition;
        local_meta[local_tid].own_count++;
        //for(int mid=0;mid<_total_partition;mid++){
          if(check_rbf_msg(local_tid,mid)){
            return fetch_rbf_msg(local_tid,mid);
          }
        //}
      }
    }
    bool rbfTryRecv(int local_tid, std::string& ret){
        for(int mid=0;mid<_total_partition;mid++){
          if(check_rbf_msg(local_tid,mid)){
            ret= fetch_rbf_msg(local_tid,mid);
            return true;
          }
        }
        return false;
    }
};
