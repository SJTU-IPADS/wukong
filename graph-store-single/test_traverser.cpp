#include "graph.h"
#include "traverser.h"
#include "traverser_keeppath.h"
int main(int argc, char * argv[])
{
	if(argc !=2)
	{
		printf("usage:./test_graph dir\n");
		return -1;
	}
	graph g(argv[1]);
	//Current I cannot correct parse the ontology file.
	//and not all the subClass are list in the file.
	//So I manually insert it and make it reasonable

	//Course
	g.insert_subclass("<ub#GraduateCourse>","<ub#Course>");
	//student
	g.insert_subclass("<ub#GraduateStudent>","<ub#Student>");
	g.insert_subclass("<ub#UndergraduateStudent>","<ub#Student>");
	//professor
	g.insert_subclass("<ub#FullProfessor>","<ub#Professor>");
	g.insert_subclass("<ub#AssistantProfessor>","<ub#Professor>");
	g.insert_subclass("<ub#AssociateProfessor>","<ub#Professor>");
	//Faculty
	g.insert_subclass("<ub#Professor>","<ub#Faculty>");
	g.insert_subclass("<ub#Lecturer>","<ub#Faculty>");

	//Person
	g.insert_subclass("<ub#Student>","<ub#Person>");
	g.insert_subclass("<ub#Faculty>","<ub#Person>");

	g.insert_subclass("<ub#TeachingAssistant>","<ub#Person>");
	g.insert_subclass("<ub#ResearchAssistant>","<ub#Person>");



	g.print_ontology_tree();


	{
		// Query 1
		traverser t(g);
		t.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
			.LoadNeighbors("in","<ub#takesCourse>")
			.is_subclass_of("<ub#GraduateStudent>")
			.print_count();

	}

	{
		// Query 2 TODO
		cout<<"Query 2 :TODO"<<endl;
//		traverser t(g);
//		t.print_count();
	}
	
	{
		// Query 3
		traverser t(g);
		t.lookup("<http://www.Department0.University0.edu/AssistantProfessor0>")
			.LoadNeighbors("in","<ub#publicationAuthor>")
			.is_subclass_of("<ub#Publication>")
			.print_count();
	}
	
	{
		// Query 4
		traverser t(g);
		t.lookup("<http://www.Department0.University0.edu>")
			.LoadNeighbors("in","<ub#worksFor>")
			.is_subclass_of("<ub#Professor>")
			//.print_property()
			.print_count();
	}

	{
		// Query 5 
		traverser t(g);
		t.lookup("<http://www.Department0.University0.edu>")
			.LoadNeighbors("in","<ub#memberOf>")
			.print_count();

	}

	{
		// Query 6 
		traverser t(g);
		t.get_all_subtype("<ub#Student>")
			.LoadNeighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
			.print_count();

	}

	{
		// Query 7
		traverser t(g);
		t.lookup("<http://www.Department0.University0.edu/AssociateProfessor0>")
			.LoadNeighbors("out","<ub#teacherOf>")
			.is_subclass_of("<ub#Course>")
			.LoadNeighbors("in","<ub#takesCourse>")
			.is_subclass_of("<ub#Student>")
			//.print_property()			
			.print_count();
	}

	{
		// Query 8
		traverser t(g);
		t.lookup("<http://www.University0.edu>")
			.LoadNeighbors("in","<ub#subOrganizationOf>")
			.is_subclass_of("<ub#Department>")	
			.LoadNeighbors("in","<ub#memberOf>")
			.is_subclass_of("<ub#Student>")
			.print_count();
	}

	{
		// Query 9 TODO
//		traverser t(g);
//		t.print_count();
		cout<<"Query 9 :TODO"<<endl;
	}

	{
		// Query 10 
		traverser t(g);
		t.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
			.LoadNeighbors("in","<ub#takesCourse>")
			.is_subclass_of("<ub#Student>")
			.print_count();
	}

    return 0;
}