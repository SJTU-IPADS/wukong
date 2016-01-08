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
	static const int nthread_parallel_load=20;
	vector<vector<edge_triple> > triple_spo;
	vector<vector<edge_triple> > triple_ops;

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
			file.close();
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

		////refine here
		triple_spo.clear();
		triple_ops.clear();
		triple_spo.resize(nthread_parallel_load);
		triple_ops.resize(nthread_parallel_load);
		for(int i=0;i<triple_spo.size();i++){
			triple_spo.reserve(total_count/world.size()*1.5);
			triple_ops.reserve(total_count/world.size()*1.5);
		}
		#pragma omp parallel for num_threads(nthread_parallel_load)
		for(int t=0;t<nthread_parallel_load;t++){
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
						int s_tableid=(s/world.size())%nthread_parallel_load;
						if( s_tableid ==t){
							triple_spo[t].push_back(edge_triple(s,p,o));
						}
					}
					local_count++;
					if(local_count==total_count/100){
						local_count=0;
						int ret=__sync_fetch_and_add( &finished_count, 1 );
						if((ret+1)%nthread_parallel_load==0){
							cout<<"already aggregrate by s "<<(ret+1)/nthread_parallel_load<<" %"<<endl;
						}
					}
				}
			}
			sort(triple_spo[t].begin(),triple_spo[t].end(),edge_sort_by_spo());
		}
		finished_count=0;
		#pragma omp parallel for num_threads(nthread_parallel_load)
		for(int t=0;t<nthread_parallel_load;t++){
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
					if(ingress::vid2mid(o,world.size())==world.rank()){
						int o_tableid=(o/world.size())%nthread_parallel_load;
						if( o_tableid ==t){
							triple_ops[t].push_back(edge_triple(s,p,o));
						}
					}
					local_count++;
					if(local_count==total_count/100){
						local_count=0;
						int ret=__sync_fetch_and_add( &finished_count, 1 );
						if((ret+1)%nthread_parallel_load==0){
							cout<<"already aggregrate by o "<<(ret+1)/nthread_parallel_load<<" %"<<endl;
						}
					}
				}
			}
			sort(triple_ops[t].begin(),triple_ops[t].end(),edge_sort_by_ops());
		}
		uint64_t t3=timer::get_usec();
		cout<<(t3-t2)/1000<<" ms for aggregrate edges"<<endl;

	}
	void remove_duplicate(vector<edge_triple>& elist){
		if(elist.size()>1){
			uint64_t end=1;
			for(uint64_t i=1;i<elist.size();i++){
				if(elist[i].s==elist[i-1].s &&
					elist[i].p==elist[i-1].p &&
					elist[i].o==elist[i-1].o){
					continue;
				}
				elist[end]=elist[i];
				end++;
			}
			elist.resize(end);
		}
	}
	uint64_t count_pivot(vector<edge_triple>& elist, bool order_by_s){
		//already_sorted , by spo or by ops
		//remove duplicate first
		if(elist.size()>1){
			uint64_t end=1;
			for(uint64_t i=1;i<elist.size();i++){
				if(elist[i].s==elist[i-1].s &&
					elist[i].p==elist[i-1].p &&
					elist[i].o==elist[i-1].o){
					continue;
				}
				elist[end]=elist[i];
				end++;
			}
			elist.resize(end);
		}
		uint64_t last_s=-1;
		uint64_t last_p=-1;
		uint64_t last_o=-1;
		
		uint64_t old_size=elist.size();
		uint64_t num_pivot=0;
		for(uint64_t i=0;i<old_size;i++){
			if(order_by_s){
				if(elist[i].s!=last_s || elist[i].p!=last_p ){
					last_s=elist[i].s;
					last_p=elist[i].p;
					num_pivot++;
				}
			} else {
				if(elist[i].o!=last_o || elist[i].p!=last_p ){
					last_o=elist[i].o;
					last_p=elist[i].p;
					num_pivot++;
				}
			}
		}
		return num_pivot;
	}

	graph(boost::mpi::communicator& para_world,RdmaResource* _rdma,const char* dir_name)
			:world(para_world),rdma(_rdma){
		struct dirent *ptr;    
	    DIR *dir;
	    dir=opendir(dir_name);
	    vector<string> filenames;

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
	    uint64_t max_v_num=1000000*240*3;//80;
	    
	    
	    uint64_t t1=timer::get_usec();
		load_and_sync_data_v2(filenames);
		uint64_t t2=timer::get_usec();
		
		//load_and_sync_data will use the memory of rdma_region
		//so kstore should be init here
	    kstore.init(rdma,max_v_num,world.size(),world.rank());
		
		uint64_t pivot_time=0;
		uint64_t insert_time=0;
		#pragma omp parallel for num_threads(nthread_parallel_load)
		for(int t=0;t<nthread_parallel_load;t++){
			uint64_t count=0;
			remove_duplicate(triple_spo[t]);
			remove_duplicate(triple_ops[t]);
			count+=triple_spo[t].size();
			count+=triple_ops[t].size();			
			uint64_t curr_edge_ptr=kstore.atomic_alloc_edges(count);
			kstore.batch_insert(triple_spo[t],triple_ops[t],curr_edge_ptr);
			triple_spo[t].clear();
			triple_ops[t].clear();
		}
		
		uint64_t t3=timer::get_usec();
		cout<<world.rank()<<" loading files in "<<(t2-t1)/1000.0/1000.0<<"s ~~~~~~~~~~~"<<endl;
	    cout<<world.rank()<<" init rdma store in "<<(t3-t2)/1000.0/1000.0<<"s ~~~~~~~~~~~"<<endl;
		
	    
	    //kstore.calculate_edge_cut();
	    if(global_use_index_table){
	    	kstore.init_index_table();
	    	uint64_t t4=timer::get_usec();
	    	cout<<world.rank()<<" init index_table in "<<(t4-t3)/1000.0/1000.0<<"s ~~~~~~~~~~~"<<endl;
		}
	    cout<<world.rank()<<" finished "<<endl;
		
		cout<<"graph-store use "<<kstore.used_indirect_num <<" / "<<(max_v_num/4)/5*1
								<<" indirect_num"<<endl;
		cout<<"graph-store use "<<max_v_num*sizeof(vertex_v2) / 1048576<<" MB for vertex data"<<endl;
		cout<<"graph-store use "<<kstore.new_edge_ptr*sizeof(edge_v2)/1048576<<"/"
								<<kstore.max_edge_ptr*sizeof(edge_v2)/1048576<<" MB for edge data"<<endl;
	
	}

};

