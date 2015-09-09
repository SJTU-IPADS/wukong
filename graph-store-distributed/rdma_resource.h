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

    //rdma location hashing
    uint64_t bufferSize;
    uint64_t slotsize;
    uint64_t bufferEntrySize;
    uint64_t rdma_id;
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
      return (char *)( buffer + off + t_id * bufferEntrySize);
    }
    
    static void* RecvThread(void * arg);
  };


#endif

