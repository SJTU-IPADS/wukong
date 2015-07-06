
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <iostream>

#include "graph.h"
#include "index_server.h"

int main(int argc, char * argv[])
{
	if(argc !=2)
	{
		printf("usage:./test_graph dir\n");
		return -1;
	}
	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;
	graph g(world,argv[1]);
	
    return 0;
}