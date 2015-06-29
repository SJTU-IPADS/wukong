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
			.neighbors("in","<ub#takesCourse>")
			.subclass_of("<ub#GraduateStudent>")
			.print_count();

	}

	{
		// Query 2 
		cout<<"Query 2 :"<<endl;
		cout<<"\tDepartment (subOrganizationOf)-> University <-(undergraduateDegreeFrom) GraduateStudent"<<endl;
		cout<<"\tDepartment <-(memberOf) GraduateStudent"<<endl;
		traverser_keeppath t1(g);
		timer time1;

		t1.get_all_subtype("<ub#Department>")
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>");

		traverser_keeppath t2(t1);
		int split_length=t1.get_path_length();

		int c1=	t1.neighbors("out","<ub#subOrganizationOf>")
					.neighbors("in","<ub#undergraduateDegreeFrom>")
					.count();
		int c2= t2.neighbors("in","<ub#memberOf>")
					.count();
		int c3= t1.merge(t2,split_length)
					.count();
		
		timer time2;
		cout<<"query 2 spends "<<time2.diff(time1)<<" ms"<<endl;
		cout<<"\t"<<c1<<" X "<< c2<<" -> "<<c3<<endl;

	}
	
	{
		// Query 3
		traverser_keeppath t(g);
		t.lookup("<http://www.Department0.University0.edu/AssistantProfessor0>")
			.neighbors("in","<ub#publicationAuthor>")
			.subclass_of("<ub#Publication>")
			.print_count();
	}
	
	{
		// Query 4
		traverser_keeppath t(g);
		t.lookup("<http://www.Department0.University0.edu>")
			.neighbors("in","<ub#worksFor>")
			.subclass_of("<ub#Professor>")
			//.print_property()
			.print_count();
	}

	{
		// Query 5 
		traverser_keeppath t(g);
		t.lookup("<http://www.Department0.University0.edu>")
			.neighbors("in","<ub#memberOf>")
			.print_count();

	}

	{
		// Query 6 
		traverser_keeppath t(g);
		t.get_all_subtype("<ub#Student>")
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
			.print_count();

	}

	{
		// Query 7
		traverser_keeppath t(g);
		t.lookup("<http://www.Department0.University0.edu/AssociateProfessor0>")
			.neighbors("out","<ub#teacherOf>")
			.subclass_of("<ub#Course>")
			.neighbors("in","<ub#takesCourse>")
			.subclass_of("<ub#Student>")
			//.sort()
			//.print_property()
			.print_count();

	}

	{
		// Query 8
		traverser_keeppath t(g);
		timer t1;
		t.lookup("<http://www.University0.edu>")
			.neighbors("in","<ub#subOrganizationOf>")
			.subclass_of("<ub#Department>")	
			.neighbors("in","<ub#memberOf>")
			.subclass_of("<ub#Student>");
//			.print_count();

		timer t2;
		cout<<"query 8 spends "<<t2.diff(t1)<<" ms"<<endl;
	}

	{
		// Query 9 
		cout<<"Query 9 :"<<endl;
		cout<<"\tFaculty (teacherOf)-> Course <-(takesCourse) Student"<<endl;
		cout<<"\tFaculty <-(advisor) Student"<<endl;
		traverser_keeppath t1(g);
		t1.get_all_subtype("<ub#Faculty>")
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>");

		traverser_keeppath t2(t1);
		int split_length=t1.get_path_length();

		int c1=	t1.neighbors("out","<ub#teacherOf>")
					.neighbors("in","<ub#takesCourse>")
					.count();
		int c2= t2.neighbors("in","<ub#advisor>")
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
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>");

		traverser_keeppath t2(t1);
		int split_length=t1.get_path_length();

		int c1=	t1.neighbors("in","<ub#advisor>")
					.neighbors("out","<ub#takesCourse>")
					.count();
		int c2= t2.neighbors("out","<ub#teacherOf>")
					.count();
		int c3= t1.merge(t2,split_length)
					.count();
		cout<<"\t"<<c1<<" X "<< c2<<" -> "<<c3<<endl;		
	}

	{
		// Query 10 
		traverser_keeppath t(g);
		t.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
			.neighbors("in","<ub#takesCourse>")
			.subclass_of("<ub#Student>")
			.print_count();
	}

    return 0;
}