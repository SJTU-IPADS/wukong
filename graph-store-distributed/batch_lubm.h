#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <iostream>

#include "graph.h"
#include "traverser.h"
#include "index_server.h"
#include "client.h"
#include "network_node.h"
#include "rdma_resource.h"
#include "thread_cfg.h"
#include "global_cfg.h"
#include <pthread.h>
#include <string>
using namespace std;


//global_num_lubm_university
vector<uint64_t> get_university_id(client* clnt){	
	vector<uint64_t> result;
	for(int univ_id=0;univ_id<global_num_lubm_university;univ_id++){
		//<http://www.University0.edu>
		string s1= "<http://www.University";
		string s2= to_string(univ_id);	
		string s3= ".edu>";
		string str=s1+s2+s3;
		if(clnt->is->subject_to_id.find(str)!=clnt->is->subject_to_id.end()){
			uint64_t id=clnt->is->subject_to_id[str];
			result.push_back(id);
		} 
	}
	return result;
}
vector<uint64_t> get_department_id(client* clnt){	
	vector<uint64_t> result;
	for(int univ_id=0;univ_id<global_num_lubm_university;univ_id++){
		for(int depart_id=0;depart_id<20;depart_id++){
			//<http://www.Department19.University4.edu>
			string s1= "<http://www.Department";
			string s2= to_string(depart_id);	
			string s3= ".University";
			string s4= to_string(univ_id);	
			string s5= ".edu>";
			string str=s1+s2+s3+s4+s5;
			if(clnt->is->subject_to_id.find(str)!=clnt->is->subject_to_id.end()){
				uint64_t id=clnt->is->subject_to_id[str];
				result.push_back(id);
			}
		}
	}
	return result;
}
vector<uint64_t> get_AssistantProfessor_id(client* clnt){	
	vector<uint64_t> result;
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
				if(clnt->is->subject_to_id.find(str)!=clnt->is->subject_to_id.end()){
					uint64_t id=clnt->is->subject_to_id[str];
					result.push_back(id);
				}
			}
		}
	}
	return result;
}
vector<uint64_t> get_AssociateProfessor_id(client* clnt){	
	vector<uint64_t> result;
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
				if(clnt->is->subject_to_id.find(str)!=clnt->is->subject_to_id.end()){
					uint64_t id=clnt->is->subject_to_id[str];
					result.push_back(id);
				}
			}
		}
	}
	return result;
}

vector<uint64_t> get_GraduateCourse_id(client* clnt){	
	vector<uint64_t> result;
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
				if(clnt->is->subject_to_id.find(str)!=clnt->is->subject_to_id.end()){
					uint64_t id=clnt->is->subject_to_id[str];
					result.push_back(id);
				}
			}
		}
	}
	return result;
}
vector<uint64_t>& get_ids(client* clnt,string cmd){
	static vector<uint64_t> vec_university_id;
	static vector<uint64_t> vec_department_id;
	static vector<uint64_t> vec_AssistantProfessor_id;
	static vector<uint64_t> vec_AssociateProfessor_id;
	static vector<uint64_t> vec_GraduateCourse_id;
	if(cmd=="get_university_id"){
		if(vec_university_id.size()==0){
			vec_university_id=get_university_id(clnt);
		}
		return vec_university_id;
	}
	if(cmd=="get_department_id"){
		if(vec_department_id.size()==0){
			vec_department_id=get_department_id(clnt);
		}
		return vec_department_id;
	}
	if(cmd=="get_AssistantProfessor_id"){
		if(vec_AssistantProfessor_id.size()==0){
			vec_AssistantProfessor_id=get_AssistantProfessor_id(clnt);
		}
		return vec_AssistantProfessor_id;
	}
	if(cmd=="get_AssociateProfessor_id"){
		if(vec_AssociateProfessor_id.size()==0){
			vec_AssociateProfessor_id=get_AssociateProfessor_id(clnt);
		}
		return vec_AssociateProfessor_id;
	}
	if(cmd=="get_GraduateCourse_id"){
		if(vec_GraduateCourse_id.size()==0){
			vec_GraduateCourse_id=get_GraduateCourse_id(clnt);
		}
		return vec_GraduateCourse_id;
	}
	exit(0);
	//return vector<uint64_t>();
}



