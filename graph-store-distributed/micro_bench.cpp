#include <string>
#include <vector>
#include <iostream>
#include "timer.h"

#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <iostream>

#include "network_node.h"
#include "rdma_resource.h"
#include <pthread.h>
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
  for(int i=0;i<97;i++){
  	str.push_back('0');
  }
  unsigned int seed=config->world->rank()*100+config->id;
  if(config->id < NUM_RECVER){
  	//recv
  	int end_num=0;
  	timer t1;
  	timer t2;
  	while(true){
  		string tmp=node->Recv();
      for(int times=0;times<extra_work;times++){
        for(int i=3;i<tmp.size();i++){
          tmp[i]++;
        }
      }
  		int mid=tmp[0];
  		int tid=tmp[1];
  		int count=tmp[2];
  		if(count==10){
        node->Send(mid, tid, tmp);
  			end_num++;
  			if(end_num==10000){
  				end_num=0;
	  			t2.reset();
          if(config->world->rank()==0 && config->id==0)
	  			  cout<<NUM_SENDER<<"\t"<<NUM_RECVER <<"\t"
                <<batch_factor<<"\t"<<extra_work <<"\t"
                <<t2.diff(t1)<<" ms"<<endl;
          //cout<<"("<<config->world->rank()<<","<<config->id<<") finished in "<<t2.diff(t1)<<" ms"<<endl;
	  			t1.reset();
  			}
  		}
  		else{
  			tmp[2]=tmp[2]+1;
  			node->Send(rand_r(&seed)%config->world->size(), rand_r(&seed)%NUM_RECVER, tmp);
  		}
  	}
  } else {
  	//send
  	sleep(1);	
	timer t1;

  for(int i=0;i<batch_factor;i++){
    node->Send(rand_r(&seed)%config->world->size(), rand_r(&seed)%NUM_RECVER, str);
  }
	for(int i=0;i<1000*10000;i++){
    node->Recv();
		node->Send(rand_r(&seed)%config->world->size(), rand_r(&seed)%NUM_RECVER, str);
		//usleep(1);
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
  	uint64_t total_size=rdma_size+slot_per_thread*NUM_THREAD;
	Network_Node *node = new Network_Node(world.rank(),NUM_THREAD);
	char *buffer= (char*) malloc(total_size);
	RdmaResource *rdma=new RdmaResource(world.size(),NUM_THREAD,world.rank(),buffer,total_size,slot_per_thread,rdma_size);
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
