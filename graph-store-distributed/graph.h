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
			cout<<"input file is too large, please split it into smaller files"<<endl;
			assert(false);
		} 
		*(local_buffer+(*local_buffer)*3+1)=s;
		*(local_buffer+(*local_buffer)*3+2)=p;
		*(local_buffer+(*local_buffer)*3+3)=o;
		*local_buffer=*local_buffer+1;
	}
	void flush_edge(int localtid,int nfile,int fileid){
		uint64_t subslot_size=rdma->get_slotsize()/sizeof(uint64_t)/world.size();
		uint64_t fileslot_size=rdma->get_size()/sizeof(uint64_t)/nfile;
		for(int mid=0;mid<world.size();mid++){
			uint64_t *local_buffer = (uint64_t *)rdma->GetMsgAddr(localtid);//GetMsgAddr(tid);
			local_buffer+=subslot_size*mid;
			uint64_t send_size=((*local_buffer)*3+1)*sizeof(uint64_t);
			uint64_t remote_offset=	fileslot_size*sizeof(uint64_t)*fileid;
			if(mid!=world.rank()){
				rdma->RdmaWrite(localtid,mid,(char*)local_buffer,send_size, remote_offset);
			} else {
				memcpy(rdma->get_buffer()+remote_offset,(char*)local_buffer,send_size);
			}
			*local_buffer=0;
		}
	}
	void load_and_sync_data(vector<string> file_vec){
		sort(file_vec.begin(),file_vec.end());
		int nfile=file_vec.size();
		int total_edge=0;
		for(int i=0;i<nfile;i++){
			if(i%world.size()!=world.rank()){
				continue;
			}
			ifstream file(file_vec[i].c_str());
			uint64_t s,p,o;
			//rdma->get_slotsize();
			while(file>>s>>p>>o){
				int s_mid=ingress::vid2mid(s,world.size());
				int o_mid=ingress::vid2mid(o,world.size());
				if(s_mid==o_mid){
					send_edge(0,s_mid,s,p,o);
					total_edge++;
				}
				else {
					send_edge(0,s_mid,s,p,o);
					send_edge(0,o_mid,s,p,o);
					total_edge+=2;
				}
			}
			flush_edge(0,nfile,i);
		}
		total_edge=0;
		MPI_Barrier(MPI_COMM_WORLD);
		uint64_t fileslot_size=rdma->get_size()/sizeof(uint64_t)/nfile;
		for(int i=0;i<file_vec.size();i++){
			uint64_t offset=	fileslot_size*sizeof(uint64_t)*i;
			uint64_t* buffer=(uint64_t*)(rdma->get_buffer()+fileslot_size*sizeof(uint64_t)*i);
			uint64_t size=*buffer;
			uint64_t s,p,o;
			buffer++;
			while(size>0){
				total_edge++;
				s=buffer[0];
				p=buffer[1];
				o=buffer[2];
				buffer+=3;
				size-=1;
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
		}
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
			uint64_t t2=timer::get_usec();
			
			//load_and_sync_data will use the memory of rdma_region
			//so kstore should be init here
		    kstore.init(rdma,max_v_num,world.size(),world.rank());
		 	
		 	//  for(int i=0;i<num_vertex_table;i++){
			//     unordered_map<uint64_t,vertex_row>::iterator iter;
			//     for(iter=vertex_table[i].begin();iter!=vertex_table[i].end();iter++){
			// 		kstore.insert(iter->first,iter->second);
			// 	}
			// 	vertex_table[i].clear();
			// }

			//#pragma omp parallel for num_threads(10)
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
	    
	    cout<<world.rank()<<" finished "<<endl;
		cout<<"graph-store use "<<max_v_num*sizeof(vertex) / 1024 / 1024<<" MB for vertex data"<<endl;
		cout<<"graph-store use "<<kstore.new_edge_ptr * sizeof(edge_row) / 1024 / 1024<<" MB for edge data"<<endl;
	
	}
};

