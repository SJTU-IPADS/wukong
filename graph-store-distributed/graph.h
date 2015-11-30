#pragma once

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
#include "timer.h"
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>

#include "ontology.h"
#include "klist_store.h"
#include "rdma_resource.h"
#include "ingress.h"
#include "omp.h"
using namespace std;

class graph{
	boost::mpi::communicator& world;
	RdmaResource* rdma;
public:
	static const int num_vertex_table=100;
	unordered_map<uint64_t,vertex_row> vertex_table[num_vertex_table];
	pthread_spinlock_t vertex_table_lock[num_vertex_table];
	klist_store kstore;
	ontology ontology_table;
	void load_ontology(string filename){
		cout<<"loading "<<filename<<endl;
		ifstream file(filename.c_str());
		uint64_t child,parent;
		while(file>>child>>parent){
			ontology_table.insert_type(child);
			if(parent!=-1){
				ontology_table.insert_type(parent);	
				ontology_table.insert(child,parent);
			}
		}
		file.close();
	}

	void inline send_edge(int localtid,int mid,uint64_t s,uint64_t p,uint64_t o){
		uint64_t *local_buffer = (uint64_t *)rdma->GetMsgAddr(localtid);
		uint64_t subslot_size=rdma->get_slotsize()/sizeof(uint64_t)/world.size();
		local_buffer+=subslot_size*mid;
		if((*local_buffer)*3+3 >=subslot_size){
			cout<<"input file is too large, please split files into smaller files"<<endl;
			assert(false);
		} 
		*(local_buffer+(*local_buffer)*3+1)=s;
		*(local_buffer+(*local_buffer)*3+2)=p;
		*(local_buffer+(*local_buffer)*3+3)=o;
		*local_buffer=*local_buffer+1;
	}
	void flush_edge(int localtid,int nfile,int fileid){
		uint64_t subslot_size=rdma->get_slotsize()/sizeof(uint64_t)/world.size();
		uint64_t fileslot_size=rdma->get_memorystore_size()/sizeof(uint64_t)/nfile;
		for(int mid=0;mid<world.size();mid++){
			uint64_t *local_buffer = (uint64_t *)rdma->GetMsgAddr(localtid);//GetMsgAddr(tid);
			local_buffer+=subslot_size*mid;
			uint64_t send_size=((*local_buffer)*3+1)*sizeof(uint64_t);
			uint64_t remote_offset=	fileslot_size*sizeof(uint64_t)*fileid;
			if(send_size > fileslot_size*sizeof(uint64_t)){
				cout<<"fileslot_size is not large enough, please split files into smaller files"<<endl;
				assert(false);
			}
			if(mid!=world.rank()){
				rdma->RdmaWrite(localtid,mid,(char*)local_buffer,send_size, remote_offset);
			} else {
				memcpy(rdma->get_buffer()+remote_offset,(char*)local_buffer,send_size);
			}
			*local_buffer=0;
		}
	}
	bool inline recv_edge(int nfile,int fileid,uint64_t row,uint64_t* s,uint64_t* p,uint64_t* o){
		uint64_t fileslot_size=rdma->get_memorystore_size()/sizeof(uint64_t)/nfile;
		uint64_t offset=fileslot_size*sizeof(uint64_t)*fileid;
		uint64_t* buffer=(uint64_t*)(rdma->get_buffer()+offset);
		uint64_t size=*buffer;
		if(row>=size)
			return false;
		buffer++;
		*s=buffer[row*3];
		*p=buffer[row*3+1];
		*o=buffer[row*3+2];
		return true;
	}
	struct edge_tmp_for_sort{
		uint64_t s;
		uint64_t p;
		uint64_t o;
		bool operator < (edge_tmp_for_sort const& _A) const {  
			if(s < _A.s)  
				return true;  
			if(s == _A.s) 
				return o< _A.o;  
			return false;  
		}
	};
	//#vertex and #edge number of each machine 
	vector<double> balance_greedy;
	void vid2mid_greedy(const vector<edge_tmp_for_sort>& edge_vec,uint64_t start,uint64_t end){
		int m_num=world.size();
		vector<double> score_greedy(m_num);
		vector<int> degrees_greedy(m_num);
		for (size_t i = start; i < end; ++i) {
			uint64_t vid=edge_vec[i].o;
			if (global_mid_table[vid]!=-1)
				degrees_greedy[global_mid_table[vid]]++;
		}
		double gamma = 1.5;
		double alpha = alpha = sqrt(m_num) * double(global_estimate_enum) / pow(global_estimate_vnum, gamma);
		for (size_t i = 0; i < m_num; ++i) {
			score_greedy[i] = degrees_greedy[i] 
				- alpha * gamma * pow(balance_greedy[i], (gamma - 1));
		}
		double best_score = score_greedy[0];
		int best_mid = 0;
		for (size_t i = 1; i < m_num; ++i) {
			if (score_greedy[i] > best_score) {
				best_score = score_greedy[i];
				best_mid = i;
			}
		}
		global_mid_table[edge_vec[start].s]=best_mid;
		//update balance 
		balance_greedy[best_mid]++;
		balance_greedy[best_mid] +=
			((end-start) * float(global_estimate_vnum) / float(global_estimate_enum));

	}
	void merge_mid_table(int target_mid){
		uint64_t read_length=global_estimate_vnum*sizeof(int);
		rdma->RdmaRead(0,target_mid,rdma->get_buffer()+read_length,read_length,0);
		int *remote_mid_table=(int*)(rdma->get_buffer()+read_length);
		for(uint64_t i=0;i<global_estimate_vnum;i++){
			if(remote_mid_table[i]!=-1){
				global_mid_table[i]=remote_mid_table[i];
			}
		}
	}
	void load_and_sync_data_greedy(vector<string> file_vec){
		sort(file_vec.begin(),file_vec.end());
		int nfile=file_vec.size();
		#pragma omp parallel for num_threads(global_num_server)
		for(int i=0;i<nfile;i++){
			int localtid = omp_get_thread_num();
			//cout<<localtid<<endl;
			if(i%world.size()!=world.rank()){
				continue;
			}
			ifstream file(file_vec[i].c_str());
			uint64_t s,p,o;
			while(file>>s>>p>>o){
				//int s_mid=ingress::vid2mid(s,world.size());
				int s_mid=s%world.size();
				//only send to local buffer
				send_edge(localtid,s_mid,s,p,o);
			}
			//flush data using rdma_write
			flush_edge(localtid,nfile,i);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		vector<edge_tmp_for_sort> edge_vec;
		for(int fileid=0;fileid<file_vec.size();fileid++){
			uint64_t row=0;
			//uint64_t s,p,o;
			edge_tmp_for_sort e;
			while(recv_edge(nfile,fileid,row,&e.s,&e.p,&e.o)){
				edge_vec.push_back(e);
				row++;//try to read next row
			}
		}
		sort(edge_vec.begin(),edge_vec.end());
		balance_greedy.resize(world.size());
		global_mid_table=(int*)rdma->get_buffer();
		for(uint64_t i=0;i<global_estimate_vnum;i++){
			global_mid_table[i]=-1;
		}
		MPI_Barrier(MPI_COMM_WORLD);
		//start to calculate the mid of all vertex

		//ingress::create_table(global_estimate_vnum);
		uint64_t edge_start=0;
		uint64_t sync_point=edge_vec.size()/10;
		while(edge_start<edge_vec.size()){
			uint64_t edge_end=edge_start+1;
			while(edge_end<edge_vec.size() && edge_vec[edge_start].s == edge_vec[edge_end].s){
				edge_end++;
			}
			vid2mid_greedy(edge_vec,edge_start,edge_end);
			edge_start=edge_end;
			if(edge_start>sync_point){
				for(int i=0;i<world.size();i++){
					if(i==world.rank()){
						continue;
					}
					merge_mid_table(i);
				}
				sync_point+=edge_vec.size()/10;
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		for(int i=0;i<world.size();i++){
			if(i==world.rank()){
				continue;
			}
			merge_mid_table(i);
		}
		int* old_mid_table= global_mid_table;
		global_mid_table=new int[global_estimate_vnum];
		for(uint64_t i=0;i<global_estimate_vnum;i++){
			global_mid_table[i]=old_mid_table[i];
			if(global_mid_table[i]==-1){
				global_mid_table[i]=i%world.size();
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

		//we will dive the vector into a lot of logic files
		//since the interface is file based
		int nfile_per_machine=20;
		nfile=nfile_per_machine*world.size();
		#pragma omp parallel for num_threads(global_num_server)
		for(int i=0;i<nfile_per_machine;i++){
			uint64_t start=edge_vec.size()/nfile_per_machine*i;
			uint64_t end=edge_vec.size()/nfile_per_machine*(i+1);
			if(i== nfile_per_machine-1){
				end=edge_vec.size();
			}
			int localtid = omp_get_thread_num();
			for(uint64_t i=start;i<end;i++){
				uint64_t s,p,o;
				s=edge_vec[i].s;
				p=edge_vec[i].p;
				o=edge_vec[i].o;
				int s_mid=ingress::vid2mid(s,world.size());
				int o_mid=ingress::vid2mid(o,world.size());
				if(s_mid==o_mid){
					send_edge(localtid,s_mid,s,p,o);
				}
				else {
					send_edge(localtid,s_mid,s,p,o);
					send_edge(localtid,o_mid,s,p,o);
				}
			}
			flush_edge(localtid,nfile,nfile_per_machine*world.rank()+i);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		uint64_t t2=timer::get_usec();
		int parallel_factor=global_num_server;
		#pragma omp parallel for num_threads(parallel_factor)
		for(int t=0;t<parallel_factor;t++){
			for(int fileid=0;fileid<nfile;fileid++){
				uint64_t row=0;
				uint64_t s,p,o;
				while(recv_edge(nfile,fileid,row,&s,&p,&o)){
					if(ingress::vid2mid(s,world.size())==world.rank()){
						if((s/world.size())%parallel_factor==t){
							vertex_table[t][s].out_edges.push_back(edge_row(p,o));
						}
					}
					if(ingress::vid2mid(o,world.size())==world.rank()){
						if((o/world.size())%parallel_factor==t){
							vertex_table[t][o].in_edges.push_back(edge_row(p,s));
						}
					}
					row++;
				}
			}
		}
		uint64_t t3=timer::get_usec();
		cout<<(t3-t2)/1000<<" ms for aggregrate edges"<<endl;
	}
	void load_and_sync_data(vector<string> file_vec){
		sort(file_vec.begin(),file_vec.end());
		uint64_t t1=timer::get_usec();
		int nfile=file_vec.size();
		#pragma omp parallel for num_threads(global_num_server)
		for(int i=0;i<nfile;i++){
			int localtid = omp_get_thread_num();
			//cout<<localtid<<endl;
			if(i%world.size()!=world.rank()){
				continue;
			}
			ifstream file(file_vec[i].c_str());
			uint64_t s,p,o;
			while(file>>s>>p>>o){
				int s_mid=ingress::vid2mid(s,world.size());
				int o_mid=ingress::vid2mid(o,world.size());
				if(s_mid==o_mid){
					send_edge(localtid,s_mid,s,p,o);
				}
				else {
					send_edge(localtid,s_mid,s,p,o);
					send_edge(localtid,o_mid,s,p,o);
				}
			}
			flush_edge(localtid,nfile,i);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		uint64_t t2=timer::get_usec();
		int parallel_factor=global_num_server;
		#pragma omp parallel for num_threads(parallel_factor)
		for(int t=0;t<parallel_factor;t++){
			for(int fileid=0;fileid<file_vec.size();fileid++){
				uint64_t row=0;
				uint64_t s,p,o;
				while(recv_edge(nfile,fileid,row,&s,&p,&o)){
					if(ingress::vid2mid(s,world.size())==world.rank()){
						if((s/world.size())%parallel_factor==t){
							vertex_table[t][s].out_edges.push_back(edge_row(p,o));
						}
					}
					if(ingress::vid2mid(o,world.size())==world.rank()){
						if((o/world.size())%parallel_factor==t){
							vertex_table[t][o].in_edges.push_back(edge_row(p,s));
						}
					}
					row++;
				}
			}
		}
		uint64_t t3=timer::get_usec();
		cout<<(t2-t1)/1000<<" ms for loading files"<<endl;
		cout<<(t3-t2)/1000<<" ms for aggregrate edges"<<endl;
	}
	void load_data(string filename){
		//cout<<"loading "<<filename<<endl;
		ifstream file(filename.c_str());
		uint64_t s,p,o;
		while(file>>s>>p>>o){
			int vt_id;
			if(ingress::vid2mid(s,world.size())==world.rank()){
				vt_id=(s/world.size())%num_vertex_table;
				pthread_spin_lock(&vertex_table_lock[vt_id]);
				vertex_table[vt_id][s].out_edges.push_back(edge_row(p,o));
				pthread_spin_unlock(&vertex_table_lock[vt_id]);
			}
			if(ingress::vid2mid(o,world.size())==world.rank()){
				vt_id=(o/world.size())%num_vertex_table;
				pthread_spin_lock(&vertex_table_lock[vt_id]);
				vertex_table[vt_id][o].in_edges.push_back(edge_row(p,s));
				pthread_spin_unlock(&vertex_table_lock[vt_id]);
			}
		}
		file.close();
	}
	void load_convert_data(int i,string s_file,string o_file){
		ifstream file1(s_file.c_str());
		ifstream file2(o_file.c_str());
		uint64_t s,p,o;
		uint64_t count=0;
		while(file1>>s>>p>>o){
			vertex_table[i][s].out_edges.push_back(edge_row(p,o));
			count++;
		}
		while(file2>>s>>p>>o){
			vertex_table[i][o].in_edges.push_back(edge_row(p,s));
			count++;
		}
		uint64_t curr_edge_ptr=kstore.atomic_alloc_edges(count);
		unordered_map<uint64_t,vertex_row>::iterator iter;
		for(iter=vertex_table[i].begin();iter!=vertex_table[i].end();iter++){
			if(iter->second.out_edges.size()==0 && iter->second.in_edges.size()==0){
				continue;
			}
			kstore.insert_at(iter->first,iter->second,curr_edge_ptr);
			curr_edge_ptr+=iter->second.out_edges.size();
			curr_edge_ptr+=iter->second.in_edges.size();
		}
		cout<<"finished loading "<<s_file<<endl;
	}

	graph(boost::mpi::communicator& para_world,RdmaResource* _rdma,const char* dir_name)
			:world(para_world),rdma(_rdma){
		struct dirent *ptr;    
	    DIR *dir;
	    dir=opendir(dir_name);
	    vector<string> filenames;
	    vector<string> convert_s_files;
	    vector<string> convert_o_files;
	    for(int i=0;i<num_vertex_table;i++){
			pthread_spin_init(&vertex_table_lock[i],0);
		}
	    while((ptr=readdir(dir))!=NULL){
	        if(ptr->d_name[0] == '.')
	            continue;
	        string fname(ptr->d_name);
	        string index_prefix="index";
	        string data_prefix="id";
			string convert_s_prefix="s_";
			string convert_o_prefix="o_";
			if(equal(index_prefix.begin(), index_prefix.end(), fname.begin())){
				// we only need to load index_ontology
				string ontology_prefix="index_ontology";
				if(equal(ontology_prefix.begin(), ontology_prefix.end(), fname.begin())){
					load_ontology(string(dir_name)+"/"+fname);
				} else {
					continue;
				}
			} else if(equal(data_prefix.begin(), data_prefix.end(), fname.begin())){
				filenames.push_back(string(dir_name)+"/"+fname);
			} else if(equal(convert_s_prefix.begin(), convert_s_prefix.end(), fname.begin())){
				//filename example : s_1
				if(atoi(fname.c_str()+2)%world.size()==world.rank()){
					convert_s_files.push_back(string(dir_name)+"/"+fname);
				}
			} else if(equal(convert_o_prefix.begin(), convert_o_prefix.end(), fname.begin())){
				//filename example : o_1
				if(atoi(fname.c_str()+2)%world.size()==world.rank()){
					convert_o_files.push_back(string(dir_name)+"/"+fname);
				}
			} else{
				//cout<<"What's this file:"<<fname<<endl;
				//assert(false);
			}
	    }
	    uint64_t max_v_num=1000000*80;//80;
	    
	    
	    uint64_t t1=timer::get_usec();
		if(!global_load_convert_format){
			// #pragma omp parallel for num_threads(10)
			// for(int i=0;i<filenames.size();i++){
			// 	load_data(filenames[i]);
			// }
			load_and_sync_data(filenames);
			//load_and_sync_data_greedy(filenames);
			uint64_t t2=timer::get_usec();
			
			//load_and_sync_data will use the memory of rdma_region
			//so kstore should be init here
		    kstore.init(rdma,max_v_num,world.size(),world.rank());
		 	
		 	#pragma omp parallel for num_threads(10)
			for(int i=0;i<num_vertex_table;i++){
				uint64_t count=0;
				unordered_map<uint64_t,vertex_row>::iterator iter;
				for(iter=vertex_table[i].begin();iter!=vertex_table[i].end();iter++){
					count+=iter->second.in_edges.size() + iter->second.out_edges.size() ;
				}
				uint64_t curr_edge_ptr=kstore.atomic_alloc_edges(count);
				for(iter=vertex_table[i].begin();iter!=vertex_table[i].end();iter++){
					kstore.insert_at(iter->first,iter->second,curr_edge_ptr);
					curr_edge_ptr+=iter->second.out_edges.size();
					curr_edge_ptr+=iter->second.in_edges.size();
				}
			}
			
			uint64_t t3=timer::get_usec();
			cout<<"loading files in "<<(t2-t1)/1000.0/1000.0<<"s ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
		    cout<<"init rdma store in "<<(t3-t2)/1000.0/1000.0<<"s ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
		} else {
			kstore.init(rdma,max_v_num,world.size(),world.rank());
			if(convert_s_files.size()==0){
				cout<<"convert files not found!!!!!!!!!"<<endl;
				exit(0);
			}
			sort(convert_s_files.begin(),convert_s_files.end());
			sort(convert_o_files.begin(),convert_o_files.end());
			
			#pragma omp parallel for num_threads(5)
			for(int i=0;i<convert_s_files.size();i++){
		    	load_convert_data(i,convert_s_files[i],convert_o_files[i]);
		    }


		    uint64_t t2=timer::get_usec();
			cout<<"machine "<<world.rank()<<" load and init in "<<(t2-t1)/1000.0/1000.0<<"s ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
		}
	    kstore.calculate_edge_cut();
	    cout<<world.rank()<<" finished "<<endl;
		cout<<"graph-store use "<<max_v_num*sizeof(vertex) / 1024 / 1024<<" MB for vertex data"<<endl;
		cout<<"graph-store use "<<kstore.new_edge_ptr * sizeof(edge_row) / 1024 / 1024<<" MB for edge data"<<endl;
	
	}
};

