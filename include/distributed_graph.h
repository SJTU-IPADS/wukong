#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <boost/mpi.hpp>
#include <boost/unordered_map.hpp>


#include "utils.h"
#include "global_cfg.h"
#include "graph_basic_types.h"
#include "rdma_resource.h"
#include "omp.h"
#include "graph_storage.h"
#include "old_graph_storage.h"

using namespace std;

class distributed_graph{
	boost::mpi::communicator& world;
	RdmaResource* rdma;
    static const int nthread_parallel_load=20;
	vector<vector<edge_triple> > triple_spo;
	vector<vector<edge_triple> > triple_ops;
    vector<uint64_t> edge_num_per_machine;
	void remove_duplicate(vector<edge_triple>& elist);
	void inline send_edge(int localtid,int mid,uint64_t s,uint64_t p,uint64_t o);
	void flush_edge(int localtid,int mid);
	void load_data(vector<string>& file_vec);
	void load_data_from_allfiles(vector<string>& file_vec);

    void load_and_sync_data(vector<string>& file_vec);
public:
	graph_storage local_storage;
	//old_graph_storage local_storage;
	distributed_graph(boost::mpi::communicator& para_world,RdmaResource* _rdma,string dir_name);
	edge* get_edges_global(int tid,uint64_t id,int direction,int predict,int* size){
		return local_storage.get_edges_global(tid,id,direction,predict,size);
	};
    edge* get_edges_local(int tid,uint64_t id,int direction,int predict,int* size){
		return local_storage.get_edges_local(tid,id,direction,predict,size);
	};
	edge* get_index_edges_local(int tid,uint64_t id,int direction,int* size){
		return local_storage.get_index_edges_local(tid,id,direction,size);
	};
};
