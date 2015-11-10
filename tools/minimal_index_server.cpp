// this small program is used to generate string index files

#include <string>  
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <assert.h> 
#include <string>
using namespace std;



void get_university_str(unordered_set<string>& subject_set,int global_num_lubm_university){	
	for(int univ_id=0;univ_id<global_num_lubm_university;univ_id++){
		string s1= "<http://www.University";
		string s2= to_string(univ_id);	
		string s3= ".edu>";
		string str=s1+s2+s3;
		subject_set.insert(str);
	}
	return ;
}
void get_department_str(unordered_set<string>& subject_set,int global_num_lubm_university){	
	for(int univ_id=0;univ_id<global_num_lubm_university;univ_id++){
		for(int depart_id=0;depart_id<20;depart_id++){
			//<http://www.Department19.University4.edu>
			string s1= "<http://www.Department";
			string s2= to_string(depart_id);	
			string s3= ".University";
			string s4= to_string(univ_id);	
			string s5= ".edu>";
			string str=s1+s2+s3+s4+s5;
			subject_set.insert(str);
		}
	}
	return ;
}
void get_AssistantProfessor_str(unordered_set<string>& subject_set,int global_num_lubm_university){	
	for(int univ_id=0;univ_id<global_num_lubm_university;univ_id++){
		for(int depart_id=0;depart_id<20;depart_id++){
			for(int ap_id=0;ap_id<20;ap_id++){	
				string s1= "<http://www.Department";
				string s2= to_string(depart_id);	
				string s3= ".University";
				string s4= to_string(univ_id);	
				string s5= ".edu/AssistantProfessor";
				string s6= to_string(ap_id);
				string s7= ">";
				string str=s1+s2+s3+s4+s5+s6+s7;
				subject_set.insert(str);
			}
		}
	}
	return ;
}
void get_AssociateProfessor_str(unordered_set<string>& subject_set,int global_num_lubm_university){	
	for(int univ_id=0;univ_id<global_num_lubm_university;univ_id++){
		for(int depart_id=0;depart_id<20;depart_id++){
			for(int ap_id=0;ap_id<20;ap_id++){	
				string s1= "<http://www.Department";
				string s2= to_string(depart_id);	
				string s3= ".University";
				string s4= to_string(univ_id);	
				string s5= ".edu/AssociateProfessor";
				string s6= to_string(ap_id);
				string s7= ">";
				string str=s1+s2+s3+s4+s5+s6+s7;
				subject_set.insert(str);
			}
		}
	}
	return ;
}

void get_GraduateCourse_str(unordered_set<string>& subject_set,int global_num_lubm_university){	
	for(int univ_id=0;univ_id<global_num_lubm_university;univ_id++){
		for(int depart_id=0;depart_id<20;depart_id++){
			for(int course_id=0;course_id<20;course_id++){	
				string s1= "<http://www.Department";
				string s2= to_string(depart_id);	
				string s3= ".University";
				string s4= to_string(univ_id);	
				string s5= ".edu/GraduateCourse";
				string s6= to_string(course_id);
				string s7= ">";
				string str=s1+s2+s3+s4+s5+s6+s7;
				subject_set.insert(str);
			}
		}
	}
	return ;
}
int main(int argc,char** argv){

	if(argc<=1){
    	printf("usage: ./minimal_index_server num_university\n");
    	return 0;
    }
	int global_num_lubm_university=atoi(argv[1]);
	
	unordered_set<int> ontology_set;
	unordered_set<string> subject_set;
	{
		ifstream file("index_ontology");
		int child,parent;
		while(file>>child>>parent){
			ontology_set.insert(child);
			if(parent !=-1){
				ontology_set.insert(parent);
			}
		}
		file.close();
	}
	{
		get_university_str(subject_set,global_num_lubm_university);
		get_department_str(subject_set,global_num_lubm_university);
		get_AssistantProfessor_str(subject_set,global_num_lubm_university);
		get_AssociateProfessor_str(subject_set,global_num_lubm_university);
		get_GraduateCourse_str(subject_set,global_num_lubm_university);
	}

	{
		ifstream finput("index_subject");
		ofstream foutput("minimal_index_subject");
		string subject;
		int id=0;
		while(finput>>subject){
			if(ontology_set.find(id)!=ontology_set.end()){
				foutput<<subject<<" "<<id<<endl;
			} else if(subject_set.find(subject)!=subject_set.end()){
				foutput<<subject<<" "<<id<<endl;
			}
			id++;
		}
		finput.close();
		foutput.close();
	}    
}

