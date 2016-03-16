#include <string>
#include "global_cfg.h"

bool global_use_rbf;
bool global_use_rdma;
int global_rdma_threshold;
int global_num_server;
int global_num_client;
int global_num_thread;
int global_batch_factor;
std::string global_input_folder;
int global_client_mode;
int global_rdftype_id;
bool global_use_loc_cache;
bool global_load_minimal_index;
bool global_silent;
int global_max_print_row;
bool global_use_multithread;
bool global_use_index_table;
int global_total_memory_gb;
int global_perslot_msg_mb;
int global_hash_header_million;
int global_enable_workstealing;

int global_verbose;

int* global_mid_table;

void load_global_cfg(char* filename){
	ifstream file(filename);
	global_rdftype_id=-1;
	string row;
	string val;
	if(!file){
		cout<<"Config file "<<filename<<" not exist"<<endl;
		exit(0);
	}
	map<string,string> config_map;
	while(file>>row>>val){
		config_map[row]=val;
	}
	global_use_rbf=atoi(config_map["global_use_rbf"].c_str());
	global_use_rdma=atoi(config_map["global_use_rdma"].c_str());
	global_rdma_threshold=atoi(config_map["global_rdma_threshold"].c_str());
	global_num_server=atoi(config_map["global_num_server"].c_str());
	global_num_client=atoi(config_map["global_num_client"].c_str());
	global_batch_factor=atoi(config_map["global_batch_factor"].c_str());
	global_input_folder=config_map["global_input_folder"];
	global_client_mode=atoi(config_map["global_client_mode"].c_str());
	global_use_loc_cache=atoi(config_map["global_use_loc_cache"].c_str());
	global_load_minimal_index=atoi(config_map["global_load_minimal_index"].c_str());
	global_silent=atoi(config_map["global_silent"].c_str());
	global_max_print_row=atoi(config_map["global_max_print_row"].c_str());
	global_use_multithread=atoi(config_map["global_use_multithread"].c_str());
	global_use_index_table=atoi(config_map["global_use_index_table"].c_str());
	global_total_memory_gb=atoi(config_map["global_total_memory_gb"].c_str());
	global_perslot_msg_mb=atoi(config_map["global_perslot_msg_mb"].c_str());
	global_hash_header_million=atoi(config_map["global_hash_header_million"].c_str());
	global_enable_workstealing=atoi(config_map["global_enable_workstealing"].c_str());

	global_verbose=atoi(config_map["global_verbose"].c_str());

	global_num_thread=global_num_server+global_num_client;
	global_mid_table=NULL;
}
