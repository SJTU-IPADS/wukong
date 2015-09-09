
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <iostream>

#include "graph.h"
#include "traverser.h"
#include "index_server.h"
#include "network_node.h"
#include <pthread.h>
struct Thread_config{
  int id;
  boost::mpi::communicator* world_ptr;
  graph* graph_ptr;
  index_server* index_server_ptr;
};


void* Run(void *ptr) {
  struct Thread_config *config = (struct Thread_config*) ptr;
  //Network_Node node(config->world_ptr->rank(),config->id);
  if(config->id!=0){
  	//node.Send(config->world_ptr->rank(),1-config->id,"fuck");
	traverser t(*(config->world_ptr),*(config->graph_ptr),config->id);
	t.run();
  }else {
  	// 	//query 1
		
  	config->index_server_ptr->lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
			.neighbors("in","<ub#takesCourse>")
			.subclass_of("<ub#GraduateStudent>")
			.execute();

  	//string result=node.Recv();
  	//cout<<result<<endl;
  	timer t1;
  	for(int i=0;i<100;i++){
  		config->index_server_ptr->lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
			.neighbors("in","<ub#takesCourse>")
			.subclass_of("<ub#GraduateStudent>")
			.execute();
			//.print_count();

		config->index_server_ptr->get_subtype("<ub#Student>")
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
			.execute();
			//.print_count();
  	}
	//Query1(config->index_server_ptr);
	timer t2;
	cout<<endl<<"requests finished in "<<t2.diff(t1)<<" ms"<<endl;

  }
}

// void* Run(void *ptr) {
//   struct Thread_config *config = (struct Thread_config*) ptr;
//   if(config->id==0){
//   	int r=100;
//   	for(int i=0;i<1000;i++){
// 		config->world_ptr->send(i% config->world_ptr->size() , 1, r);
// 	}
//   }else {
//   	int r=1;
//   	for(int i=0;i<1000;i++){
// 		config->world_ptr->recv(boost::mpi::any_source, 1, r);
// 		cout<<"id="<<config->id<<"\t"<<i<<endl;
//   	}
//   }
// }
#define THREAD_NUM 2
int main(int argc, char * argv[])
{
	int provided;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE)
	{
	    printf("ERROR: The MPI library does not have full thread support\n");
	    MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if(argc !=2)
	{
		printf("usage:./test_graph dir\n");
		return -1;
	}
	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;
	graph g(world,argv[1]);
	index_server is(world,argv[1],0);
	//id 0 is used for index_server

	Thread_config *configs = new Thread_config[THREAD_NUM];
  	pthread_t     *thread  = new pthread_t[THREAD_NUM];
	for(size_t id = 0;id < THREAD_NUM;++id) {
      configs[id].id = id;
      configs[id].world_ptr=&world;
      configs[id].graph_ptr=&g;
      configs[id].index_server_ptr=&is;
      
      pthread_create (&(thread[id]), NULL, Run, (void *) &(configs[id]));
      sleep(1);
    }
    for(size_t t = 0 ; t < THREAD_NUM; t++) {
      int rc = pthread_join(thread[t], NULL);
      if (rc) {
        printf("ERROR; return code from pthread_join() is %d\n", rc);
        exit(-1);
      }
    }


	// if(world.rank() < world.size()-1){
	// 	graph g(world,argv[1]);
	// 	traverser t(world,g);
	// 	t.run();
	// } else {
	// 	index_server is(world,argv[1]);

	// 	//query 1
	// 	is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
	// 		.neighbors("in","<ub#takesCourse>")
	// 		.subclass_of("<ub#GraduateStudent>")
	// 		.execute()
	// 		.print_count();

	// 	//query 1
	// 	is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
	// 		.neighbors("in","<ub#takesCourse>")
	// 		.subclass_of("<ub#GraduateStudent>")
	// 		.execute()
	// 		.print_count();
	// 	//query 1
	// 	is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
	// 		.neighbors("in","<ub#takesCourse>")
	// 		//.subclass_of("<ub#GraduateStudent>")
	// 		.execute()
	// 		.print_count();
	// 	//query 1
	// 	is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
	// 		.neighbors("in","<ub#takesCourse>")
	// 		//.subclass_of("<ub#GraduateStudent>")
	// 		.execute()
	// 		.print_count();

		//query 2
		//TODO
	// 	{
	// 		request r1=	is.get_subtype("<ub#Department>")
	// 						.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
	// 						.neighbors("out","<ub#subOrganizationOf>")
	// 						.neighbors("in","<ub#undergraduateDegreeFrom>")
	// 						.execute()
	// 						.req;
	// 		request r2=	is.get_subtype("<ub#Department>")
	// 						.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
	// 						.neighbors("in","<ub#memberOf>")
	// 						.execute()
	// 						.req;
	// 		int c1=r1.path_num();
	// 		int c2=r2.path_num();
	// 		r1.merge(r2,2);
			
	// 		cout<<c1<<"*"<<c2<<"="<<r1.path_num()<<endl;
	// 	}
	// 	//query 3
	// 	is.lookup("<http://www.Department0.University0.edu/AssistantProfessor0>")
	// 		.neighbors("in","<ub#publicationAuthor>")
	// 		.subclass_of("<ub#Publication>")
	// 		.execute()
	// 		.print_count();

	// 	//query 4
	// 	is.lookup("<http://www.Department0.University0.edu>")
	// 		.neighbors("in","<ub#worksFor>")
	// 		.subclass_of("<ub#Professor>")
	// 		.execute()
	// 		.print_count();
		
	// 	//query 5
	// 	is.lookup("<http://www.Department0.University0.edu>")
	// 		.neighbors("in","<ub#memberOf>")
	// 		.subclass_of("<ub#Person>")
	// 		.execute()
	// 		.print_count();

	// 	//query 6
	// 	is.get_subtype("<ub#Student>")
	// 		.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
	// 		.execute()
	// 		.print_count();

	// 	//query 7
	// 	is.lookup("<http://www.Department0.University0.edu/AssociateProfessor0>")
	// 			.neighbors("out","<ub#teacherOf>")
	// 			.subclass_of("<ub#Course>")
	// 			.neighbors("in","<ub#takesCourse>")
	// 			.subclass_of("<ub#Student>")
	// 			.execute()
	// 			.print_count();

	// 	//query 8
				
	// 	is.lookup("<http://www.University0.edu>")
	// 		.neighbors("in","<ub#subOrganizationOf>")
	// 		.subclass_of("<ub#Department>")	
	// 		.neighbors("in","<ub#memberOf>")
	// 		.subclass_of("<ub#Student>")
	// 		.execute()
	// 		.print_count();

	// 	//query 9
	// 	//TODO
	// 	{
	// 		cout<<"Query 9-1 :"<<endl;
	// 		cout<<"Faculty (teacherOf)-> Course <-(takesCourse) Student"<<endl;
	// 		cout<<"Faculty <-(advisor) Student"<<endl;
	// 		request r1=	is.get_subtype("<ub#Faculty>")
	// 						.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
	// 						.neighbors("out","<ub#teacherOf>")
	// 						.neighbors("in","<ub#takesCourse>")
	// 						.execute()
	// 						.req;
	// 		request r2=	is.get_subtype("<ub#Faculty>")
	// 						.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
	// 						.neighbors("in","<ub#advisor>")
	// 						.execute()
	// 						.req;
	// 		int c1=r1.path_num();
	// 		int c2=r2.path_num();
	// 		r1.merge(r2,2);
	// 		cout<<c1<<"*"<<c2<<"="<<r1.path_num()<<endl;
	// 	}
	// 	{
	// 		cout<<"Query 9-2 :"<<endl;
	// 		cout<<"Faculty <-(advisor) Student (takesCourse)-> Course"<<endl;
	// 		cout<<"Faculty (teacherOf)-> Course "<<endl;
	// 		request r1=	is.get_subtype("<ub#Faculty>")
	// 						.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
	// 						.neighbors("in","<ub#advisor>")
	// 						.neighbors("out","<ub#takesCourse>")
	// 						.execute()
	// 						.req;
	// 		request r2=	is.get_subtype("<ub#Faculty>")
	// 						.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
	// 						.neighbors("out","<ub#teacherOf>")
	// 						.execute()
	// 						.req;
	// 		int c1=r1.path_num();
	// 		int c2=r2.path_num();
	// 		r1.merge(r2,2);
	// 		cout<<c1<<"*"<<c2<<"="<<r1.path_num()<<endl;
	// 	}

	// 	//query 10
	// 	is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
	// 		.neighbors("in","<ub#takesCourse>")
	// 		.subclass_of("<ub#Student>")
	// 		.execute()
	// 		.print_count();

	// 	cout<<"finish"<<endl;

	// }

    return 0;
}