#pragma once

#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h> //atoi
using namespace std;
extern int global_rdftype_id;  // only a global variable, but not configurable
extern int global_num_thread;  //=global_num_server+global_num_client
extern int global_multithread_factor;


extern bool global_use_rbf;
extern bool global_use_rdma;
extern int global_rdma_threshold;
extern int global_num_server;
extern int global_num_client;
extern int global_batch_factor;
extern string global_input_folder;
extern int global_client_mode;
extern bool global_use_loc_cache;
extern bool global_load_minimal_index;
extern bool global_silent;
extern int global_max_print_row;
extern bool global_use_multithread;
extern bool global_use_index_table;
extern int global_total_memory_gb;
extern int global_perslot_msg_mb;
extern int global_hash_header_million;
extern int global_enable_workstealing;


extern int global_verbose;

void load_changeable_cfg();
void load_global_cfg(char* filename);
