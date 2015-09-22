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
using namespace std;
struct Thread_config{
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
// #define NUM_SENDER 4
// #define NUM_RECVER 1
// #define NUM_THREAD (NUM_RECVER+NUM_SENDER)

int NUM_SENDER;
int NUM_RECVER;
int NUM_THREAD;
int batch_factor;
int extra_work;
void* Run(void *ptr) {
  struct Thread_config *config = (struct Thread_config*) ptr;
  pin_to_core(socket_0[config->id]);
  
  Network_Node *node = new Network_Node(config->world->rank(),config->id);
  //str[0]-'0' is machine id;
  //str[1]-'0' is thread id;
  string str="";
  str.push_back(config->world->rank());
  str.push_back(config->id);
  str.push_back(0);
  //for(int i=0;i<125;i++){
  for(int i=0;i<256;i++){
  	str.push_back('0');
  }
  if(config->id < NUM_RECVER){
  	//recv
    // while(true){
    //   string tmp=node->Recv();
    //   int mid=tmp[0];
    //   int tid=tmp[1];
    //   node->Send(mid,tid,str);
    // }
    
    while(true){
      string tmp=rdma->rbfRecv(config->id);
      int mid=tmp[0];
      int tid=tmp[1];
      rdma->rbfSend(config->id,mid,tid,str);
    }

  } else {
  	//send
  	sleep(1);	
    timer t1;
    // for(int i=0;i<1000*10;i++){
    //   node->Send(rand()%config->world->size(), rand()%NUM_RECVER, str);
    //   node->Recv();
    // }

    for(int i=0;i<1000*10;i++){
      rdma->rbfSend(config->id,rand()%config->world->size(), rand()%NUM_RECVER, str);
      //rdma->rbfSend(config->id,config->world->rank(), rand()%NUM_RECVER, str);
      rdma->rbfRecv(config->id);
    }

    // for(int i=0;i<1000*10;i++){
    //   uint64_t *local_buffer = (uint64_t *)rdma->GetMsgAddr(config->id);
    //   uint64_t start_addr=0;
    //   uint64_t read_length=100;
    //   rdma->RdmaRead(config->id,1-config->id ,(char *)local_buffer,read_length,start_addr);
    // }
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
  NUM_SENDER=atoi(argv[1]);
  NUM_RECVER=atoi(argv[2]);
  batch_factor=atoi(argv[3]);
  extra_work=atoi(argv[4]);
  NUM_THREAD=NUM_SENDER+NUM_RECVER;

	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	uint64_t rdma_size = 1024*1024*1024;  //1G
  	uint64_t slot_per_thread= 1024*1024*128;
  	//rdma_size = rdma_size*20; //20G 
  	uint64_t total_size=rdma_size+slot_per_thread*NUM_THREAD*2;
	Network_Node *node = new Network_Node(world.rank(),NUM_THREAD);
	char *buffer= (char*) malloc(total_size);
	rdma=new RdmaResource(world.size(),NUM_THREAD,world.rank(),buffer,total_size,slot_per_thread,rdma_size);
	rdma->node = node;
	rdma->Servicing();
	rdma->Connect();

	uint64_t *local_buffer = (uint64_t *)rdma->GetMsgAddr(0);
  	uint64_t start_addr=0;
  	//rdma->RdmaRead(0,(world.rank()+1)%world.size() ,(char *)local_buffer,100,start_addr);
    //cout<<"Fucking OK"<<endl;

	Thread_config *configs = new Thread_config[NUM_THREAD];
  	pthread_t     *thread  = new pthread_t[NUM_THREAD];
	for(size_t id = 0;id < NUM_THREAD;++id) {
      configs[id].id = id;
      configs[id].world = &world;
      pthread_create (&(thread[id]), NULL, Run, (void *) &(configs[id]));
    }
    for(size_t t = 0 ; t < NUM_THREAD; t++) {
      int rc = pthread_join(thread[t], NULL);
      if (rc) {
        printf("ERROR; return code from pthread_join() is %d\n", rc);
        exit(-1);
      }
    }
    return 0;
}
