#pragma once

#include <string>  
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <boost/unordered_map.hpp>
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
	boost::unordered_map<uint64_t,vertex_row> vertex_table[num_vertex_table];
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
	//first version will fail when we use semantic hash, so I implement v2
	//first version should be removed 
	vector<uint64_t> edge_num_per_machine;
	uint64_t inline floor(uint64_t original,uint64_t n){
		if(n==0){
			assert(false);
		}
		if(original%n == 0){
			return original;
		}
		return original - original%n; 
	}
	void inline send_edge_v2(int localtid,int mid,uint64_t s,uint64_t p,uint64_t o){
		uint64_t subslot_size=floor(rdma->get_slotsize()/world.size(),sizeof(uint64_t));
		uint64_t *local_buffer = (uint64_t *) (rdma->GetMsgAddr(localtid) + subslot_size*mid );
		*(local_buffer+(*local_buffer)*3+1)=s;
		*(local_buffer+(*local_buffer)*3+2)=p;
		*(local_buffer+(*local_buffer)*3+3)=o;
		*local_buffer=*local_buffer+1;
		if( ((*local_buffer)*3+10)*sizeof(uint64_t) >=subslot_size){
			//full , should be flush!
			flush_edge_v2(localtid,mid);
		}
	}
	void flush_edge_v2(int localtid,int mid){
		uint64_t subslot_size=floor(rdma->get_slotsize()/world.size(),sizeof(uint64_t));
		uint64_t *local_buffer = (uint64_t *) (rdma->GetMsgAddr(localtid) + subslot_size*mid );
		uint64_t num_edge_to_send=*local_buffer;
		//clear and skip the number infomation
		*local_buffer=0;
		local_buffer+=1;
		uint64_t max_size=floor(rdma->get_memorystore_size()/world.size(),sizeof(uint64_t));
		uint64_t old_num=__sync_fetch_and_add( &edge_num_per_machine[mid], num_edge_to_send);
		if((old_num+num_edge_to_send+1)*3*sizeof(uint64_t) >=max_size){
			cout<<"old ="<<old_num<<endl;
			cout<<"num_edge_to_send ="<<num_edge_to_send<<endl;
			cout<<"max_size ="<<max_size<<endl;
			cout<<"Don't have enough space to store data"<<endl;
			assert(false);
		}
		// we need to flush to the same offset of different machine 
		uint64_t remote_offset=	max_size * world.rank();
		remote_offset  +=	(old_num*3+1)*sizeof(uint64_t);
		uint64_t remote_length= num_edge_to_send*3*sizeof(uint64_t);
		if(mid!=world.rank()){
			rdma->RdmaWrite(localtid,mid,(char*)local_buffer,remote_length, remote_offset);
		} else {
			memcpy(rdma->get_buffer()+remote_offset,(char*)local_buffer,remote_length);
		}	
	}

	void load_and_sync_data_v2(vector<string> file_vec){
		sort(file_vec.begin(),file_vec.end());
		uint64_t t1=timer::get_usec();
		int nfile=file_vec.size();
		volatile int finished_count = 0;
		#pragma omp parallel for num_threads(global_num_server)
		for(int i=0;i<nfile;i++){
			int localtid = omp_get_thread_num();
			if(i%world.size()!=world.rank()){
				continue;
			}
			ifstream file(file_vec[i].c_str());
			uint64_t s,p,o;
			while(file>>s>>p>>o){
				int s_mid=ingress::vid2mid(s,world.size());
				int o_mid=ingress::vid2mid(o,world.size());
				if(s_mid==o_mid){
					send_edge_v2(localtid,s_mid,s,p,o);
				}
				else {
					send_edge_v2(localtid,s_mid,s,p,o);
					send_edge_v2(localtid,o_mid,s,p,o);
				}
			}
			int ret=__sync_fetch_and_add( &finished_count, 1 );
			if(ret%40==39){
				cout<<"already load "<<ret+1<<" files"<<endl;
			}
		}
		for(int mid=0;mid<world.size();mid++){
			for(int i=0;i<global_num_server;i++){
				flush_edge_v2(i,mid);
			}
		}
		for(int mid=0;mid<world.size();mid++){
			//after flush all data,we need to write the number of total edges;
			uint64_t *local_buffer = (uint64_t *) rdma->GetMsgAddr(0); 
			*local_buffer=edge_num_per_machine[mid];
			uint64_t max_size=floor(rdma->get_memorystore_size()/world.size(),sizeof(uint64_t));
			uint64_t remote_offset=	max_size * world.rank();
			if(mid!=world.rank()){
				rdma->RdmaWrite(0,mid,(char*)local_buffer,sizeof(uint64_t), remote_offset);
			} else {
				memcpy(rdma->get_buffer()+remote_offset,(char*)local_buffer,sizeof(uint64_t));
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		uint64_t t2=timer::get_usec();
		cout<<(t2-t1)/1000<<" ms for loading files"<<endl;
		finished_count=0;
		uint64_t total_count=0;
		//lock_free
		for(int mid=0;mid<world.size();mid++){
			uint64_t max_size=floor(rdma->get_memorystore_size()/world.size(),sizeof(uint64_t));
			uint64_t offset= max_size * mid;
			uint64_t* recv_buffer=(uint64_t*)(rdma->get_buffer()+offset);
			total_count+= *recv_buffer;
		}
		int parallel_factor=20;
		#pragma omp parallel for num_threads(parallel_factor)
		for(int t=0;t<parallel_factor;t++){
			int local_count=0;
			for(int mid=0;mid<world.size();mid++){
				//recv from different machine
				uint64_t max_size=floor(rdma->get_memorystore_size()/world.size(),sizeof(uint64_t));
				uint64_t offset= max_size * mid;
				uint64_t* recv_buffer=(uint64_t*)(rdma->get_buffer()+offset);
				uint64_t num_edge=*recv_buffer;
				for(uint64_t i=0;i< num_edge;i++){
					uint64_t s=recv_buffer[1+i*3];
					uint64_t p=recv_buffer[1+i*3+1];
					uint64_t o=recv_buffer[1+i*3+2];
					if(ingress::vid2mid(s,world.size())==world.rank()){
						int s_tableid=(s/world.size())%num_vertex_table;
						if( s_tableid %parallel_factor==t){
							vertex_table[s_tableid][s].out_edges.push_back(edge_row(p,o));
						}
					}
					if(ingress::vid2mid(o,world.size())==world.rank()){
						int o_tableid=(o/world.size())%num_vertex_table;
						if(o_tableid % parallel_factor==t){
							vertex_table[o_tableid][o].in_edges.push_back(edge_row(p,s));
						}
					}
					local_count++;
					if(local_count==total_count/100){
						local_count=0;
						int ret=__sync_fetch_and_add( &finished_count, 1 );
						if((ret+1)%parallel_factor==0){
							cout<<"already aggregrate "<<(ret+1)/parallel_factor<<" %"<<endl;
						}
					}
				}
			}
		}
		uint64_t t3=timer::get_usec();
		cout<<(t3-t2)/1000<<" ms for aggregrate edges"<<endl;

	}
	///////////////////
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
		volatile int finished_count = 0;
		#pragma omp parallel for num_threads(global_num_server)
		for(int i=0;i<nfile;i++){
			int localtid = omp_get_thread_num();
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
			int ret=__sync_fetch_and_add( &finished_count, 1 );
			if(ret%40==39){
				cout<<"already load "<<ret+1<<" files"<<endl;
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		uint64_t t2=timer::get_usec();
		cout<<(t2-t1)/1000<<" ms for loading files"<<endl;
		finished_count=0;


//lock_free
		int parallel_factor=20;
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
				if((fileid+1)%parallel_factor==0){
					int ret=__sync_fetch_and_add( &finished_count, 1 );
					if(ret%400==399){
						cout<<"already aggregrate "<<ret+1<<endl;
					}
				}
			}
		}
		uint64_t t3=timer::get_usec();
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
	int count_pivot(vector<edge_row>& edge_list){
		//should return number of pivot

		sort(edge_list.begin(),edge_list.end());
		//sort(edge_list.begin(),edge_list.end());
		//remove duplicate	
		if(edge_list.size()>1){
			int end=1;
			for(int i=1;i<edge_list.size();i++){
				if(edge_list[i].predict==edge_list[i-1].predict
						&&	edge_list[i].vid==edge_list[i-1].vid){
					continue;
				}
				edge_list[end]=edge_list[i];
				end++;
			}
			edge_list.resize(end);
		}
		uint64_t last;
		uint64_t old_size;
		uint64_t new_size;
		uint64_t count;
		int num_pivot=0;
		last=-1;
		old_size=edge_list.size();
		new_size=old_size;
		count=0;
		for(uint64_t i=0;i<old_size;i++){
			if(edge_list[i].predict!=last){
				new_size++;
				num_pivot++;
				last=edge_list[i].predict;
			}
		}
		// at first version , we modify the vector in place 
		// but it's not scale to multiple thread, due to calling edge_list.resize();
		return num_pivot;

		// edge_list.resize(new_size);
		// while(new_size>0){
		// 	edge_list[new_size-1]=edge_list[old_size-1];
		// 	count++;
		// 	new_size--;
		// 	old_size--;
		// 	if(old_size==0 || 
		// 		edge_list[old_size-1].predict != edge_list[old_size].predict){
		// 		edge_list[new_size-1].predict=-1;
		// 		edge_list[new_size-1].vid=count;
		// 		count=0;
		// 		new_size--;
		// 	}
		// }
	}

	graph(boost::mpi::communicator& para_world,RdmaResource* _rdma,const char* dir_name)
			:world(para_world),rdma(_rdma){
		struct dirent *ptr;    
	    DIR *dir;
	    dir=opendir(dir_name);
	    vector<string> filenames;
	    for(int i=0;i<num_vertex_table;i++){
			pthread_spin_init(&vertex_table_lock[i],0);
		}
	    while((ptr=readdir(dir))!=NULL){
	        if(ptr->d_name[0] == '.')
	            continue;
	        string fname(ptr->d_name);
	        string index_prefix="index";
	        string data_prefix="id";
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
			} else{
				//cout<<"What's this file:"<<fname<<endl;
				//assert(false);
			}
	    }
	    edge_num_per_machine.resize(world.size());
	    //uint64_t max_v_num=1000000*160;//80;
	    uint64_t max_v_num=1000000*240;//80;
	    
	    
	    uint64_t t1=timer::get_usec();
		if(!global_load_convert_format){
			// #pragma omp parallel for num_threads(10)
			// for(int i=0;i<filenames.size();i++){
			// 	load_data(filenames[i]);
			// }
			//load_and_sync_data(filenames);
			load_and_sync_data_v2(filenames);
			//load_and_sync_data_greedy(filenames);
			uint64_t t2=timer::get_usec();
			
			//load_and_sync_data will use the memory of rdma_region
			//so kstore should be init here
		    kstore.init(rdma,max_v_num,world.size(),world.rank());
			
			uint64_t pivot_time=0;
			uint64_t insert_time=0;
		 	#pragma omp parallel for num_threads(8)
			for(int i=0;i<num_vertex_table;i++){
				uint64_t pivot_start=timer::get_usec();
				uint64_t count=0;
				boost::unordered_map<uint64_t,vertex_row>::iterator iter;
				for(iter=vertex_table[i].begin();iter!=vertex_table[i].end();iter++){
					int num_pivot_in = count_pivot(iter->second.in_edges);
					int num_pivot_out= count_pivot(iter->second.out_edges);
					count+=iter->second.in_edges.size() + iter->second.out_edges.size() ;
					count+=num_pivot_in+num_pivot_out;
				}
				uint64_t pivot_end=timer::get_usec();

				uint64_t curr_edge_ptr=kstore.atomic_alloc_edges(count);
				for(iter=vertex_table[i].begin();iter!=vertex_table[i].end();iter++){
					uint64_t num_total_edge=kstore.insert_at(iter->first,iter->second,curr_edge_ptr);
					curr_edge_ptr+=num_total_edge;
				}

				uint64_t insert_end=timer::get_usec();
				__sync_fetch_and_add( &pivot_time, pivot_end - pivot_start);
				__sync_fetch_and_add( &insert_time, insert_end - pivot_end);
				//cout<<"[num_vertex_table] "<<i<<" , count = "<<count<<endl;
				//vertex_table[i].clear();
			}
			//cout<<"spend "<<pivot_time/1000.0<<" ms for pivot"<<endl;
			//cout<<"spend "<<insert_time/1000.0<<" ms for insert"<<endl;
			uint64_t t3=timer::get_usec();
			cout<<world.rank()<<" loading files in "<<(t2-t1)/1000.0/1000.0<<"s ~~~~~~~~~~~"<<endl;
		    cout<<world.rank()<<" init rdma store in "<<(t3-t2)/1000.0/1000.0<<"s ~~~~~~~~~~~"<<endl;
		    for(int i=0;i<num_vertex_table;i++){
				vertex_table[i].clear();
			}
			uint64_t t4=timer::get_usec();
			cout<<world.rank()<<" clear tmp memory in "<<(t4-t3)/1000.0/1000.0<<"s ~~~~~~~~~~~"<<endl;
		} else {
			assert(false);
		} 
	    
	    //kstore.calculate_edge_cut();
	    if(global_use_predict_index){
			kstore.init_predict_index();
		}
	    cout<<world.rank()<<" finished "<<endl;
		
		//cout<<"graph-store use "<<kstore.used_v_num*sizeof(vertex)/1048576<<"/"
		cout<<"graph-store use "<<kstore.used_indirect_num <<" / "<<(max_v_num/4)/5*1
								<<" indirect_num"<<endl;
		cout<<"graph-store use "<<max_v_num*sizeof(vertex) / 1048576<<" MB for vertex data"<<endl;
		cout<<"graph-store use "<<kstore.new_edge_ptr*sizeof(edge_row)/1048576<<"/"
								<<kstore.max_edge_ptr*sizeof(edge_row)/1048576<<" MB for edge data"<<endl;
	
	}

};

