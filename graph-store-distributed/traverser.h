#pragma once
#include "graph.h"
#include "request.h"
//traverser will remember all the paths just like
//traverser_keeppath in single machine

class traverser{
	graph& g;
	boost::mpi::communicator& world;

	
public:
	traverser(boost::mpi::communicator& para_world,graph& gg):world(para_world),g(gg){
	} 
	void run(){
		while(true){
			request r;
			boost::mpi::status s =world.recv(boost::mpi::any_source, 1, r);
			cout<< world.rank()<<" recv a request from " <<s.source()<<endl;
			world.send(s.source(), 1, r);
		}

	}
};