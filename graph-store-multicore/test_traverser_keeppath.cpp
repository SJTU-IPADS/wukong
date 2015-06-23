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
		// Query 2 
		cout<<"Query 2 :"<<endl;
		cout<<"\tDepartment (subOrganizationOf)-> University <-(undergraduateDegreeFrom) GraduateStudent"<<endl;
		cout<<"\tDepartment <-(memberOf) GraduateStudent"<<endl;
		timer time1;

		traverser_keeppath t1(g);
		t1.get_subtype("<ub#Department>")
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
			.execute();
		traverser_keeppath t2(t1);
		int split_length=t1.get_path_length();

		int c1=	t1.neighbors("out","<ub#subOrganizationOf>")
					.neighbors("in","<ub#undergraduateDegreeFrom>")
					.execute()
					.get_path_num();
		int c2= t2.neighbors("in","<ub#memberOf>")
					.execute()
					.get_path_num();
		int c3= t1.merge(t2,split_length)
					.get_path_num();

		timer time2;
		cout<<"query 2 spends "<<time2.diff(time1)<<" ms"<<endl;
		cout<<"\t"<<c1<<" X "<< c2<<" -> "<<c3<<endl;

	}

	{
		// Query 7 
		traverser_keeppath t(g);
		t.lookup("<http://www.Department0.University0.edu/AssociateProfessor0>")
				.neighbors("out","<ub#teacherOf>")
				.subclass_of("<ub#Course>")
				.neighbors("in","<ub#takesCourse>")
				.subclass_of("<ub#Student>")
				.execute()
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
			.subclass_of("<ub#Student>")
			.execute();
		timer t2;
		cout<<"query 8 spends "<<t2.diff(t1)<<" ms"<<endl;
	}
    return 0;
}