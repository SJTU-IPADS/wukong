#include "graph.h"
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
		traverser_keeppath t(g);
		t.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
			.LoadNeighbors("in","<ub#takesCourse>")
			.is_subclass_of("<ub#GraduateStudent>")
			.print_count();

	}

	{
		// Query 2 
		cout<<"Query 2 :"<<endl;
		cout<<"\tDepartment (subOrganizationOf)-> University <-(undergraduateDegreeFrom) GraduateStudent"<<endl;
		cout<<"\tDepartment <-(memberOf) GraduateStudent"<<endl;
		traverser_keeppath t1(g);
		t1.get_all_subtype("<ub#Department>")
			.LoadNeighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>");

		traverser_keeppath t2(t1);
		int split_length=t1.get_path_length();

		int c1=	t1.LoadNeighbors("out","<ub#subOrganizationOf>")
					.LoadNeighbors("in","<ub#undergraduateDegreeFrom>")
					.count();
		int c2= t2.LoadNeighbors("in","<ub#memberOf>")
					.count();
		int c3= t1.merge(t2,split_length)
					.count();
		cout<<"\t"<<c1<<" X "<< c2<<" -> "<<c3<<endl;

	}
	
	{
		// Query 3
		traverser_keeppath t(g);
		t.lookup("<http://www.Department0.University0.edu/AssistantProfessor0>")
			.LoadNeighbors("in","<ub#publicationAuthor>")
			.is_subclass_of("<ub#Publication>")
			.print_count();
	}
	
	{
		// Query 4
		traverser_keeppath t(g);
		t.lookup("<http://www.Department0.University0.edu>")
			.LoadNeighbors("in","<ub#worksFor>")
			.is_subclass_of("<ub#Professor>")
			//.print_property()
			.print_count();
	}

	{
		// Query 5 
		traverser_keeppath t(g);
		t.lookup("<http://www.Department0.University0.edu>")
			.LoadNeighbors("in","<ub#memberOf>")
			.print_count();

	}

	{
		// Query 6 
		traverser_keeppath t(g);
		t.get_all_subtype("<ub#Student>")
			.LoadNeighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
			.print_count();

	}

	{
		// Query 7
		traverser_keeppath t(g);
		t.lookup("<http://www.Department0.University0.edu/AssociateProfessor0>")
			.LoadNeighbors("out","<ub#teacherOf>")
			.is_subclass_of("<ub#Course>")
			.LoadNeighbors("in","<ub#takesCourse>")
			.is_subclass_of("<ub#Student>")
			//.sort()
			//.print_property()
			.print_count();

	}

	{
		// Query 8
		traverser_keeppath t(g);
		t.lookup("<http://www.University0.edu>")
			.LoadNeighbors("in","<ub#subOrganizationOf>")
			.is_subclass_of("<ub#Department>")	
			.LoadNeighbors("in","<ub#memberOf>")
			.is_subclass_of("<ub#Student>")
			.print_count();
	}

	{
		// Query 9 
		cout<<"Query 9 :"<<endl;
		cout<<"\tFaculty (teacherOf)-> Course <-(takesCourse) Student"<<endl;
		cout<<"\tFaculty <-(advisor) Student"<<endl;
		traverser_keeppath t1(g);
		t1.get_all_subtype("<ub#Faculty>")
			.LoadNeighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>");

		traverser_keeppath t2(t1);
		int split_length=t1.get_path_length();

		int c1=	t1.LoadNeighbors("out","<ub#teacherOf>")
					.LoadNeighbors("in","<ub#takesCourse>")
					.count();
		int c2= t2.LoadNeighbors("in","<ub#advisor>")
					.count();
		int c3= t1.merge(t2,split_length)
					.count();
		cout<<"\t"<<c1<<" X "<< c2<<" -> "<<c3<<endl;

	}

	{
		// Query 9 
		
		cout<<"\tFaculty <-(advisor) Student (takesCourse)-> Course"<<endl;
		cout<<"\tFaculty (teacherOf)-> Course "<<endl;
		traverser_keeppath t1(g);
		t1.get_all_subtype("<ub#Faculty>")
			.LoadNeighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>");

		traverser_keeppath t2(t1);
		int split_length=t1.get_path_length();

		int c1=	t1.LoadNeighbors("in","<ub#advisor>")
					.LoadNeighbors("out","<ub#takesCourse>")
					.count();
		int c2= t2.LoadNeighbors("out","<ub#teacherOf>")
					.count();
		int c3= t1.merge(t2,split_length)
					.count();
		cout<<"\t"<<c1<<" X "<< c2<<" -> "<<c3<<endl;		
	}

	{
		// Query 10 
		traverser_keeppath t(g);
		t.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
			.LoadNeighbors("in","<ub#takesCourse>")
			.is_subclass_of("<ub#Student>")
			.print_count();
	}

    return 0;
}