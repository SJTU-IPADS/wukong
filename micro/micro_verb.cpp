#include <string>
#include <vector>
#include <iostream>

#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <iostream>
#include <pthread.h>

#include "../graph-store-distributed/timer.h"
#include "../graph-store-distributed/network_node.h"
#include "../graph-store-distributed/rdma_resource.h"
#include "../graph-store-distributed/message_wrap.h"

using namespace std;
struct pthread_parameter{
  int id;
  boost::mpi::communicator* world;
};
int socket_0[] = {
  0,2,4,6,8,10,12,14,16,18
};
void pin_to_core(size_t core) {
  cpu_set_t  mask;
  CPU_ZERO(&mask);
  CPU_SET(core , &mask);
  int result=sched_setaffinity(0, sizeof(mask), &mask);  
}
RdmaResource *rdma;

int num_sender;
int num_recver;
int num_thread;
int batch_factor;
int extra_work;
void* Run(void *ptr) {
  struct thread_cfg *cfg = (struct thread_cfg*) ptr;
  pin_to_core(socket_0[cfg->t_id]);
  
  //str[0]-'0' is machine id;
  //str[1]-'0' is thread id;
  string str="";
  str.push_back(cfg->m_id);
  str.push_back(cfg->t_id);
  str.push_back(0);
  //for(int i=0;i<125;i++){
  for(int i=0;i<256;i++){
  	str.push_back('0');
  }
  if(cfg->t_id < num_recver){
    
    timer t1;
    timer t2;
    int count=0;
    while(true){
      string tmp=RecvStr(cfg);
      int mid=tmp[0];
      int tid=tmp[1];
      SendStr(cfg,mid,tid,str);
      count++;
      if(count==100000){
        t2.reset();
        cout<<"(S="<<num_sender<<",R="<<num_recver <<")\t"
                <<"batch="<<batch_factor<<"\t"<<"extra="<<extra_work <<"\t"
                <<t2.diff(t1)<<" ms"<<endl;
        t1.reset();
        count=0;
      }
    }

  } else {
  	//send
  	sleep(1);	
    timer t1;

    for(int i=0;i<batch_factor;i++){
      SendStr(cfg,rand()%cfg->m_num, rand()%num_recver, str);
    }
    for(int i=0;i<1000*10000;i++){
      RecvStr(cfg);
      SendStr(cfg,rand()%cfg->m_num, rand()%num_recver, str);
    }
    for(int i=0;i<batch_factor;i++){
      RecvStr(cfg);
    }

    timer t2;
    cout<<endl<<"requests finished in "<<t2.diff(t1)<<" ms"<<endl;
  }
}

int main(int argc, char * argv[])
{
  if(argc!=5){
    printf("Wrong parameters\n");
    exit(0);
  }
  num_sender=atoi(argv[1]);
  num_recver=atoi(argv[2]);
  batch_factor=atoi(argv[3]);
  extra_work=atoi(argv[4]);
  num_thread=num_sender+num_recver;

	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	uint64_t rdma_size = 1024*1024*1024;  //1G
  	uint64_t slot_per_thread= 1024*1024*128;
  	//rdma_size = rdma_size*20; //20G 
  	uint64_t total_size=rdma_size+slot_per_thread*num_thread*2;
	Network_Node *node = new Network_Node(world.rank(),100);
	char *buffer= (char*) malloc(total_size);
	rdma=new RdmaResource(world.size(),num_thread,world.rank(),buffer,total_size,slot_per_thread,rdma_size);
	rdma->node = node;
	rdma->Servicing();
	rdma->Connect();

	uint64_t *local_buffer = (uint64_t *)rdma->GetMsgAddr(0);
  uint64_t start_addr=0;
	//rdma->RdmaRead(0,(world.rank()+1)%world.size() ,(char *)local_buffer,100,start_addr);
  //cout<<"Fucking OK"<<endl;


  thread_cfg* cfg_array= new thread_cfg[num_thread];
  for(int i=0;i<num_thread;i++){
    cfg_array[i].t_id=i;
    cfg_array[i].t_num=num_thread;
    cfg_array[i].m_id=world.rank();
    cfg_array[i].m_num=world.size();
    cfg_array[i].rdma=rdma;
    cfg_array[i].node=new Network_Node(cfg_array[i].m_id,cfg_array[i].t_id);
  }  
  pthread_t     *thread  = new pthread_t[num_thread];
  for(size_t id = 0;id < num_thread;++id) {
    cfg_array[id].ptr=NULL;
    pthread_create (&(thread[id]), NULL, Run, (void *) &(cfg_array[id]));
  }
  for(size_t t = 0 ; t < num_thread; t++) {
    int rc = pthread_join(thread[t], NULL);
    if (rc) {
      printf("ERROR; return code from pthread_join() is %d\n", rc);
      exit(-1);
    }
  }

    return 0;
}
