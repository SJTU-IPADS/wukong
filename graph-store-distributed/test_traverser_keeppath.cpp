
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <iostream>

#include "graph.h"
#include "traverser.h"
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
	
	if(world.rank() < world.size()-1){
		graph g(world,argv[1]);
		traverser t(world,g);
		t.run();
	} else {
		index_server is(world,argv[1]);
		is.lookup("<http://www.Department0.University0.edu/AssociateProfessor0>")
				.neighbors("out","<ub#teacherOf>")
				.subclass_of("<ub#Course>")
				.neighbors("in","<ub#takesCourse>")
				.subclass_of("<ub#Student>")
				.execute()
				.print_count();
	}

    return 0;
}