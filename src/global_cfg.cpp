#include <string>
#include "global_cfg.h"


/* non-configurable global variables */
int global_rdftype_id;
int global_num_thread;

/* configurable global variables */
bool global_use_rbf;
bool global_use_rdma;
int global_num_server;
int global_num_client;
std::string global_input_folder;
int global_client_mode;
bool global_load_minimal_index;
int global_max_print_row;
int global_total_memory_gb;
int global_perslot_msg_mb;
int global_perslot_rdma_mb;
int global_hash_header_million;
int global_enable_workstealing;
int global_enable_index_partition;
int global_verbose;

/* shared by client and server */
int global_batch_factor;
bool global_use_loc_cache;
bool global_silent;
int global_multithread_factor;
int global_rdma_threshold;

/* TODO: split the config file and related code
		 into two parts: client and server */

std::string config_filename;

/**
 * reconfig client
 */
void
client_reconfig()
{
	ifstream file(config_filename.c_str());
	if (!file) {
		cout << "ERROR: the config file ("
		     << config_filename
		     << ") does not exist."
		     << endl;
		exit(0);
	}

	map<string, string> config_map;
	string row;
	string val;
	// while(file>>row>>val){
	// 	config_map[row]=val;
	// }
	string line;
	while (std::getline(file, line)) {
		istringstream iss(line);
		iss >> row >> val;
		config_map[row] = val;
	}

	global_batch_factor = atoi(config_map["global_batch_factor"].c_str());
	global_use_loc_cache = atoi(config_map["global_use_loc_cache"].c_str());
	global_silent = atoi(config_map["global_silent"].c_str());
	global_multithread_factor = atoi(config_map["global_multithread_factor"].c_str());
	global_rdma_threshold = atoi(config_map["global_rdma_threshold"].c_str());

	return;
}

void
load_global_cfg(char* filename)
{
	config_filename = std::string(filename);
	ifstream file(config_filename.c_str());
	if (!file) {
		cout << "Config file " << config_filename << " not exist" << endl;
		exit(0);
	}

	map<string, string> config_map;
	string row;
	string val;
	// while(file>>row>>val){
	// 	config_map[row]=val;
	// }
	string line;
	while (std::getline(file, line)) {
		istringstream iss(line);
		iss >> row >> val;
		config_map[row] = val;
	}
	global_use_rbf = atoi(config_map["global_use_rbf"].c_str());
	global_use_rdma = atoi(config_map["global_use_rdma"].c_str());
	global_rdma_threshold = atoi(config_map["global_rdma_threshold"].c_str());
	global_num_server = atoi(config_map["global_num_server"].c_str());
	global_num_client = atoi(config_map["global_num_client"].c_str());
	global_batch_factor = atoi(config_map["global_batch_factor"].c_str());
	global_multithread_factor = atoi(config_map["global_multithread_factor"].c_str());
	global_input_folder = config_map["global_input_folder"];
	global_client_mode = atoi(config_map["global_client_mode"].c_str());
	global_use_loc_cache = atoi(config_map["global_use_loc_cache"].c_str());
	global_load_minimal_index = atoi(config_map["global_load_minimal_index"].c_str());
	global_silent = atoi(config_map["global_silent"].c_str());
	global_max_print_row = atoi(config_map["global_max_print_row"].c_str());
	global_total_memory_gb = atoi(config_map["global_total_memory_gb"].c_str());
	global_perslot_msg_mb = atoi(config_map["global_perslot_msg_mb"].c_str());
	global_perslot_rdma_mb = atoi(config_map["global_perslot_rdma_mb"].c_str());
	global_hash_header_million = atoi(config_map["global_hash_header_million"].c_str());
	global_enable_workstealing = atoi(config_map["global_enable_workstealing"].c_str());
	global_enable_index_partition = atoi(config_map["global_enable_index_partition"].c_str());
	global_verbose = atoi(config_map["global_verbose"].c_str());

	global_rdftype_id = -1;
	global_num_thread = global_num_server + global_num_client;

	return;
}
