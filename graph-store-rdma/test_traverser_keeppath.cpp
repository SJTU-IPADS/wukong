
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

		//query 1
		is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
			.neighbors("in","<ub#takesCourse>")
			.subclass_of("<ub#GraduateStudent>")
			.execute()
			.print_count();

		//query 1
		is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
			.neighbors("in","<ub#takesCourse>")
			.subclass_of("<ub#GraduateStudent>")
			.execute()
			.print_count();
		//query 1
		is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
			.neighbors("in","<ub#takesCourse>")
			//.subclass_of("<ub#GraduateStudent>")
			.execute()
			.print_count();
		//query 1
		is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
			.neighbors("in","<ub#takesCourse>")
			//.subclass_of("<ub#GraduateStudent>")
			.execute()
			.print_count();

		//query 2
		//TODO
		{
			request r1=	is.get_subtype("<ub#Department>")
							.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
							.neighbors("out","<ub#subOrganizationOf>")
							.neighbors("in","<ub#undergraduateDegreeFrom>")
							.execute()
							.req;
			request r2=	is.get_subtype("<ub#Department>")
							.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
							.neighbors("in","<ub#memberOf>")
							.execute()
							.req;
			int c1=r1.path_num();
			int c2=r2.path_num();
			r1.merge(r2,2);
			
			cout<<c1<<"*"<<c2<<"="<<r1.path_num()<<endl;
		}
		//query 3
		is.lookup("<http://www.Department0.University0.edu/AssistantProfessor0>")
			.neighbors("in","<ub#publicationAuthor>")
			.subclass_of("<ub#Publication>")
			.execute()
			.print_count();

		//query 4
		is.lookup("<http://www.Department0.University0.edu>")
			.neighbors("in","<ub#worksFor>")
			.subclass_of("<ub#Professor>")
			.execute()
			.print_count();
		
		//query 5
		is.lookup("<http://www.Department0.University0.edu>")
			.neighbors("in","<ub#memberOf>")
			.subclass_of("<ub#Person>")
			.execute()
			.print_count();

		//query 6
		is.get_subtype("<ub#Student>")
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
			.execute()
			.print_count();

		//query 7
		is.lookup("<http://www.Department0.University0.edu/AssociateProfessor0>")
				.neighbors("out","<ub#teacherOf>")
				.subclass_of("<ub#Course>")
				.neighbors("in","<ub#takesCourse>")
				.subclass_of("<ub#Student>")
				.execute()
				.print_count();

		//query 8
				
		is.lookup("<http://www.University0.edu>")
			.neighbors("in","<ub#subOrganizationOf>")
			.subclass_of("<ub#Department>")	
			.neighbors("in","<ub#memberOf>")
			.subclass_of("<ub#Student>")
			.execute()
			.print_count();

		//query 9
		//TODO
		{
			cout<<"Query 9-1 :"<<endl;
			cout<<"Faculty (teacherOf)-> Course <-(takesCourse) Student"<<endl;
			cout<<"Faculty <-(advisor) Student"<<endl;
			request r1=	is.get_subtype("<ub#Faculty>")
							.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
							.neighbors("out","<ub#teacherOf>")
							.neighbors("in","<ub#takesCourse>")
							.execute()
							.req;
			request r2=	is.get_subtype("<ub#Faculty>")
							.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
							.neighbors("in","<ub#advisor>")
							.execute()
							.req;
			int c1=r1.path_num();
			int c2=r2.path_num();
			r1.merge(r2,2);
			cout<<c1<<"*"<<c2<<"="<<r1.path_num()<<endl;
		}
		{
			cout<<"Query 9-2 :"<<endl;
			cout<<"Faculty <-(advisor) Student (takesCourse)-> Course"<<endl;
			cout<<"Faculty (teacherOf)-> Course "<<endl;
			request r1=	is.get_subtype("<ub#Faculty>")
							.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
							.neighbors("in","<ub#advisor>")
							.neighbors("out","<ub#takesCourse>")
							.execute()
							.req;
			request r2=	is.get_subtype("<ub#Faculty>")
							.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
							.neighbors("out","<ub#teacherOf>")
							.execute()
							.req;
			int c1=r1.path_num();
			int c2=r2.path_num();
			r1.merge(r2,2);
			cout<<c1<<"*"<<c2<<"="<<r1.path_num()<<endl;
		}

		//query 10
		is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
			.neighbors("in","<ub#takesCourse>")
			.subclass_of("<ub#Student>")
			.execute()
			.print_count();

		cout<<"finish"<<endl;

	}

    return 0;
}