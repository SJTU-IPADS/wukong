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

vector<uint64_t> get_ids(client* clnt,string cmd){
	if(cmd=="get_university_id")
		return get_university_id(clnt);
	if(cmd=="get_department_id")
		return get_department_id(clnt);
	if(cmd=="get_AssistantProfessor_id")
		return get_AssistantProfessor_id(clnt);
	if(cmd=="get_AssociateProfessor_id")
		return get_AssociateProfessor_id(clnt);
	if(cmd=="get_GraduateCourse_id")
		return get_GraduateCourse_id(clnt);
	exit(0);
	return vector<uint64_t>();
}

void send_using_cmd_chain(client* is,vector<string>& cmd_chain){
	int i=0;
	while(true){
		if(cmd_chain[i]=="neighbors"){
			is->neighbors(cmd_chain[i+1],cmd_chain[i+2]);
			i+=3;
		} else if(cmd_chain[i]=="subclass_of"){
			is->subclass_of(cmd_chain[i+1]);
			i+=2;
		} else if(cmd_chain[i]=="get_attr"){
			is->get_attr(cmd_chain[i+1]);
			i+=2;
		} else if(cmd_chain[i]=="execute"){
			is->Send();
			return ;
		}
	}
}
void batch_execute(client* clnt,struct thread_cfg *cfg,int total_request,vector<uint64_t>& ids,vector<string>& cmd_chain){
	unsigned int seed=cfg->m_id*cfg->t_num+cfg->t_id;
	for(int i=0;i<global_batch_factor;i++){
		clnt->lookup_id(ids[0]);
		clnt->req.timestamp=timer::get_usec();
		send_using_cmd_chain(clnt,cmd_chain);
	}
	uint64_t total_latency=0;
	uint64_t t1;
	uint64_t t2;
	for(int times=0;times<total_request;times++){
		clnt->Recv();
		if(times==total_request/4)
			t1=timer::get_usec();
		if(times==total_request/4 *3)
			t2=timer::get_usec();
		if(times>=total_request/4 && times<total_request/4*3){
			total_latency+=timer::get_usec()-clnt->req.timestamp;
		}
		int i=rand_r(&seed) % ids.size();
		clnt->lookup_id(ids[i]);
		clnt->req.timestamp=timer::get_usec();
		send_using_cmd_chain(clnt,cmd_chain);
	}
	for(int i=0;i<global_batch_factor;i++){
		clnt->Recv();
	}
	total_latency=total_latency/(total_request/2);
	cout<<total_latency<<" us"<<endl;
	cout<<(total_request/2)*1000.0/(t2-t1)<<" Kops"<<endl;
	cout<<"Total execution time "<<total_latency/1000 <<" ms"<<endl;
}



