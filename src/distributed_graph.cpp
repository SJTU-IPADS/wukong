#include "distributed_graph.h"

distributed_graph::distributed_graph(boost::mpi::communicator& _world,
                RdmaResource* _rdma,string dir_name):world(_world),rdma(_rdma){
    struct dirent *ptr;
    DIR *dir;
    dir=opendir(dir_name.c_str());
    vector<string> filenames;
    if(dir==NULL){
        cout<<"Error folder: "<<dir_name<<" at node "<<world.rank()<<endl;
        exit(-1);
    }
    while((ptr=readdir(dir))!=NULL){
        if(ptr->d_name[0] == '.'){
            continue;
        }
        string fname(ptr->d_name);
        string index_prefix="index";
        string data_prefix="id";
		if(equal(data_prefix.begin(), data_prefix.end(), fname.begin())){
			filenames.push_back(dir_name+"/"+fname);
		} else{
			//skip the index files
		}
    }

    edge_num_per_machine.resize(world.size());
    load_and_sync_data(filenames);
    local_storage.init(rdma,world.size(),world.rank());
    #pragma omp parallel for num_threads(nthread_parallel_load)
	for(int t=0;t<nthread_parallel_load;t++){
		local_storage.atomic_batch_insert(triple_spo[t],triple_ops[t]);
		triple_spo[t].clear();
		triple_ops[t].clear();
	}
    local_storage.init_index_table();
    cout<<world.rank()<<" finished "<<endl;
    local_storage.print_memory_usage();
}
void distributed_graph::load_data(vector<string>& file_vec){
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
			int s_mid=mymath::hash_mod(s,world.size());
			int o_mid=mymath::hash_mod(o,world.size());
			if(s_mid==o_mid){
				send_edge(localtid,s_mid,s,p,o);
			}
			else {
				send_edge(localtid,s_mid,s,p,o);
				send_edge(localtid,o_mid,s,p,o);
			}
		}
		file.close();
		int ret=__sync_fetch_and_add( &finished_count, 1 );
		if(ret%40==39){
			cout<<"node "<<world.rank()<<" already load "<<ret+1<<" files"<<endl;
		}
	}
	for(int mid=0;mid<world.size();mid++){
		for(int i=0;i<global_num_server;i++){
			flush_edge(i,mid);
		}
	}
	for(int mid=0;mid<world.size();mid++){
		//after flush all data,we need to write the number of total edges;
		uint64_t *local_buffer = (uint64_t *) rdma->GetMsgAddr(0);
		*local_buffer=edge_num_per_machine[mid];
		uint64_t max_size=mymath::floor(rdma->get_memorystore_size()/world.size(),sizeof(uint64_t));
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
}
void distributed_graph::load_and_sync_data(vector<string>& file_vec){
    load_data(file_vec);
    uint64_t t1=timer::get_usec();
	volatile int finished_count=0;
	uint64_t total_count=0;
	for(int mid=0;mid<world.size();mid++){
		uint64_t max_size=mymath::floor(rdma->get_memorystore_size()/world.size(),sizeof(uint64_t));
		uint64_t offset= max_size * mid;
		uint64_t* recv_buffer=(uint64_t*)(rdma->get_buffer()+offset);
		total_count+= *recv_buffer;
	}
    triple_spo.clear();
	triple_ops.clear();
	triple_spo.resize(nthread_parallel_load);
	triple_ops.resize(nthread_parallel_load);
	for(int i=0;i<triple_spo.size();i++){
		triple_spo[i].reserve(total_count/nthread_parallel_load*1.5);
		triple_ops[i].reserve(total_count/nthread_parallel_load*1.5);
	}
    #pragma omp parallel for num_threads(nthread_parallel_load)
	for(int t=0;t<nthread_parallel_load;t++){
		int local_count=0;
		for(int mid=0;mid<world.size();mid++){
			//recv from different machine
			uint64_t max_size=mymath::floor(rdma->get_memorystore_size()/world.size(),sizeof(uint64_t));
			uint64_t offset= max_size * mid;
			uint64_t* recv_buffer=(uint64_t*)(rdma->get_buffer()+offset);
			uint64_t num_edge=*recv_buffer;
			for(uint64_t i=0;i< num_edge;i++){
				uint64_t s=recv_buffer[1+i*3];
				uint64_t p=recv_buffer[1+i*3+1];
				uint64_t o=recv_buffer[1+i*3+2];
				if(mymath::hash_mod(s,world.size())==world.rank()){
					int s_tableid=(s/world.size())%nthread_parallel_load;
					if( s_tableid ==t){
						triple_spo[t].push_back(edge_triple(s,p,o));
					}
				}
                if(mymath::hash_mod(o,world.size())==world.rank()){
					int o_tableid=(o/world.size())%nthread_parallel_load;
					if( o_tableid ==t){
						triple_ops[t].push_back(edge_triple(s,p,o));
					}
				}
				local_count++;
				if(local_count==total_count/100){
					local_count=0;
					int ret=__sync_fetch_and_add( &finished_count, 1 );
					if((ret+1)%(nthread_parallel_load*5)==0){
						cout<<"already aggregrate "<<(ret+1)/nthread_parallel_load<<" %"<<endl;
					}
				}
			}
		}
		sort(triple_spo[t].begin(),triple_spo[t].end(),edge_sort_by_spo());
        sort(triple_ops[t].begin(),triple_ops[t].end(),edge_sort_by_ops());
        remove_duplicate(triple_spo[t]);
        remove_duplicate(triple_ops[t]);
	}
	uint64_t t2=timer::get_usec();
	cout<<(t2-t1)/1000<<" ms for aggregrate edges"<<endl;
}

void distributed_graph::send_edge(int localtid,int mid,uint64_t s,uint64_t p,uint64_t o){
	uint64_t subslot_size=mymath::floor(rdma->get_slotsize()/world.size(),sizeof(uint64_t));
	uint64_t *local_buffer = (uint64_t *) (rdma->GetMsgAddr(localtid) + subslot_size*mid );
	*(local_buffer+(*local_buffer)*3+1)=s;
	*(local_buffer+(*local_buffer)*3+2)=p;
	*(local_buffer+(*local_buffer)*3+3)=o;
	*local_buffer=*local_buffer+1;
	if( ((*local_buffer)*3+10)*sizeof(uint64_t) >=subslot_size){
		//full , should be flush!
		flush_edge(localtid,mid);
	}
}
void distributed_graph::flush_edge(int localtid,int mid){
	uint64_t subslot_size=mymath::floor(rdma->get_slotsize()/world.size(),sizeof(uint64_t));
	uint64_t *local_buffer = (uint64_t *) (rdma->GetMsgAddr(localtid) + subslot_size*mid );
	uint64_t num_edge_to_send=*local_buffer;
	//clear and skip the number infomation
	*local_buffer=0;
	local_buffer+=1;
	uint64_t max_size=mymath::floor(rdma->get_memorystore_size()/world.size(),sizeof(uint64_t));
	uint64_t old_num=__sync_fetch_and_add( &edge_num_per_machine[mid], num_edge_to_send);
	if((old_num+num_edge_to_send+1)*3*sizeof(uint64_t) >=max_size){
		cout<<"old ="<<old_num<<endl;
		cout<<"num_edge_to_send ="<<num_edge_to_send<<endl;
		cout<<"max_size ="<<max_size<<endl;
		cout<<"Don't have enough space to store data"<<endl;
		exit(-1);
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

void distributed_graph::remove_duplicate(vector<edge_triple>& elist){
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
