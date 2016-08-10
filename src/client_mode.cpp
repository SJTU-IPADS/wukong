#include "client_mode.h"

void
translate_req_template(client* clnt, request_template& req_template)
{
	req_template.place_holder_vecptr.resize(req_template.place_holder_str.size());
	for (int i = 0; i < req_template.place_holder_str.size(); i++) {
		string type = req_template.place_holder_str[i];
		if (clnt->parser.type_to_idvec.find(type) != clnt->parser.type_to_idvec.end()) {
			// do nothing
		} else {
			request_or_reply type_request;
			assert(clnt->parser.find_type_of(type, type_request));
			request_or_reply reply;
			clnt->Send(type_request);
			reply = clnt->Recv();
			vector<int>* ptr = new vector<int>();
			*ptr = reply.result_table;
			clnt->parser.type_to_idvec[type] = ptr;
			cout << type << " has " << ptr->size() << " objects" << endl;
		}
		req_template.place_holder_vecptr[i] = clnt->parser.type_to_idvec[type];
	}
}

void
instantiate_request(client* clnt, request_template& req_template, request_or_reply& r)
{
	for (int i = 0; i < req_template.place_holder_position.size(); i++) {
		int pos = req_template.place_holder_position[i];
		vector<int>* vecptr = req_template.place_holder_vecptr[i];
		if (vecptr == NULL || vecptr->size() == 0) {
			assert(false);
		}
		r.cmd_chains[pos] = (*vecptr)[clnt->cfg->get_random() % (vecptr->size())];
	}
}

void
client_barrier(struct thread_cfg *cfg)
{
	static int _curr = 0;
	static __thread int _next = 1;

	// inter-node barrier
	if (cfg->wid == 0)
		MPI_Barrier(MPI_COMM_WORLD);

	// intra-node barrier
	__sync_fetch_and_add(&_curr, 1);
	while (_curr < _next)
		usleep(1); // wait
	_next += cfg->ncwkrs; // next barrier
}

void
single_execute(client* clnt, string filename, int execute_count)
{
	int sum = 0;
	int result_count;
	request_or_reply request;
	bool success = clnt->parser.parse(filename, request);
	if (!success) {
		cout << "sparql parse error" << endl;
		return ;
	}
	request.silent = global_silent;
	request_or_reply reply;
	for (int i = 0; i < execute_count; i++) {
		uint64_t t1 = timer::get_usec();
		clnt->Send(request);
		reply = clnt->Recv();
		uint64_t t2 = timer::get_usec();
		sum += t2 - t1;
	}
	cout << "result size:" << reply.silent_row_num << endl;
	int row_to_print = min(reply.row_num(), (uint64_t)global_max_print_row);
	if (row_to_print > 0) {
		clnt->print_result(reply, row_to_print);
	}
	cout << "average latency " << sum / execute_count << " us" << endl;
}

void
print_help(void)
{
	cout << "  Commands" << endl;
	cout << "\thelp:          \tdisplay help infomation" << endl;
	cout << "\tquit:          \tquit from client" << endl;
	cout << "\treconfig:      \treload config file" << endl;
	cout << "\tswitch_single: \trun single query (e.g., query [count])" << endl;
	cout << "\tswitch_batch:  \trun concurrent queries (e.g., queries)" << endl;
	cout << "\tswitch_mix:    \trun concurrent & single queries)" << endl;

}

enum { SINGLE_MODE = 0, BATCH_MODE, MIX_MODE, N_MODES };

string mode_str[N_MODES] = {
	"single mode (i.e., query [count]):",
	"batch mode (i.e., queries):",
	"mix mode: (i.e., queries query [count]):"
};

void
interactive_shell(client *clnt)
{
	struct thread_cfg *cfg = clnt->cfg;

	// the master client worker (i.e., sid == 0 and wid == 0)
	if (cfg->sid == 0 && cfg->wid == 0) {
		cout << "input help to get more infomation about the shell" << endl;
		cout << mode_str[global_client_mode] << endl;
	}

	while (true) {
		client_barrier(clnt->cfg);

		string input_str;
		if (cfg->sid == 0 && cfg->wid == 0) {
			cout << "> ";
			std::getline(std::cin, input_str);

			// send commands to all client workers
			for (int i = 0; i < cfg->nsrvs; i++) {
				for (int j = 0; j < cfg->ncwkrs; j++) {
					if (i == 0 && j == 0)
						continue;
					cfg->node->Send(i, j, input_str);
				}
			}
		} else {
			// recieve commands
			input_str = cfg->node->Recv();
		}

		//handle commands
		if (input_str == "help") {
			// TODO: support separate client workers
			if (cfg->sid == 0 && cfg->wid == 0)
				print_help();
		} else if (input_str == "quit") {
			if (cfg->wid == 0)
				exit(0);
		} else if (input_str == "reconfig") {
			if (cfg->wid == 0)
				reload_cfg();
		} else if (input_str == "switch_single") {
			if (cfg->wid == 0) {
				global_client_mode = SINGLE_MODE;
				if (cfg->sid == 0)
					cout << mode_str[global_client_mode] << endl;
			}
		} else if (input_str == "switch_batch") {
			if (cfg->wid == 0) {
				global_client_mode = BATCH_MODE;
				if (cfg->sid == 0)
					cout << mode_str[global_client_mode] << endl;
			}
		} else if (input_str == "switch_mix") {
			if (cfg->wid == 0) {
				global_client_mode = MIX_MODE;
				if (cfg->sid == 0)
					cout << mode_str[global_client_mode] << endl;
			}
		} else { //handle queries here
			if (global_client_mode == SINGLE_MODE) {
				if (cfg->sid == 0 && cfg->wid == 0) {
					// run single-command using the master client
					istringstream iss(input_str);
					string fname;
					int cnt = 1;

					iss >> fname >> cnt;
					if (cnt < 1) cnt = 1;

					single_execute(clnt, fname, cnt);
				}
			} else if (global_client_mode == BATCH_MODE) {
				batch_logger logger;

				// ISSUE: client vs. server
				// run batch-command on all clients
				istringstream iss(input_str);
				string fname;

				iss >> fname;

				logger.init();
				nonblocking_execute(clnt, fname, logger);
				//batch_execute(clnt,filename,logger);
				logger.finish();

				client_barrier(clnt->cfg);
				//MPI_Barrier(MPI_COMM_WORLD);

				// print results of batch command
				if (cfg->sid == 0 && cfg->wid == 0) {
					// collect logs from other clients
					for (int i = 0; i < cfg->nsrvs * cfg->ncwkrs - 1; i++) {
						batch_logger log = RecvObject<batch_logger>(clnt->cfg);
						logger.merge(log);
					}
					logger.print();
				} else {
					// transport logs to the master client
					SendObject<batch_logger>(clnt->cfg, 0, 0, logger);
				}
			} else if (global_client_mode == MIX_MODE) {
				batch_logger logger;

				istringstream iss(input_str);
				string b_fname;
				string s_fname;
				int cnt = 1;

				iss >> b_fname >> s_fname >> cnt;
				if (cnt < 1) cnt = 1;

				if (cfg->sid == 0 && cfg->wid == 0) {
					// dedicate the master client to run single-command
					single_execute(clnt, s_fname, cnt);
				} else {
					// run batch-command on other clients
					logger.init();
					nonblocking_execute(clnt, b_fname, logger);
					//batch_execute(clnt,batchfile,logger);
					logger.finish();
				}
				client_barrier(clnt->cfg);
				//MPI_Barrier(MPI_COMM_WORLD);

				if (cfg->sid == 0 && cfg->wid == 0) {
					// collect logs from other clients
					for (int i = 0; i < cfg->nsrvs * cfg->ncwkrs - 1 ; i++) {
						batch_logger log = RecvObject<batch_logger>(clnt->cfg);
						logger.merge(log);
					}
					logger.print();
				} else {
					// transport logs to the master client
					SendObject<batch_logger>(clnt->cfg, 0, 0, logger);
				}
			}
		}
	}
}

void
batch_execute(client* clnt, string mix_config, batch_logger& logger)
{
	ifstream configfile(mix_config);
	if (!configfile) {
		cout << "File " << mix_config << " not exist" << endl;
		return ;
	}

	int total_query_type;
	int total_request;
	int sleep_round = 1;
	configfile >> total_query_type >> total_request >> sleep_round;
	vector<int > distribution;
	vector<request_template > vec_template;
	vector<request_or_reply > vec_req;
	vec_template.resize(total_query_type);
	vec_req.resize(total_query_type);
	for (int i = 0; i < total_query_type; i++) {
		string filename;
		configfile >> filename;
		int current_dist;
		configfile >> current_dist;
		distribution.push_back(current_dist);
		bool success = clnt->parser.parse_template(filename, vec_template[i]);
		translate_req_template(clnt, vec_template[i]);
		vec_req[i].cmd_chains = vec_template[i].cmd_chains;
		if (!success) {
			cout << "sparql parse error" << endl;
			return ;
		}
		vec_req[i].silent = global_silent;
	}
	uint64_t start_time = timer::get_usec();
	for (int i = 0; i < global_batch_factor; i++) {
		int idx = mymath::get_distribution(clnt->cfg->get_random(), distribution);
		instantiate_request(clnt, vec_template[idx], vec_req[idx]);
		clnt->GetId(vec_req[idx]);
		logger.start_record(vec_req[idx].parent_id, idx);
		clnt->Send(vec_req[idx]);
	}
	for (int i = 0; i < total_request; i++) {
		request_or_reply reply = clnt->Recv();
		logger.end_record(reply.parent_id);
		int idx = mymath::get_distribution(clnt->cfg->get_random(), distribution);
		instantiate_request(clnt, vec_template[idx], vec_req[idx]);
		clnt->GetId(vec_req[idx]);
		logger.start_record(vec_req[idx].parent_id, idx);
		clnt->Send(vec_req[idx]);
	}
	for (int i = 0; i < global_batch_factor; i++) {
		request_or_reply reply = clnt->Recv();
		logger.end_record(reply.parent_id);
	}
	uint64_t end_time = timer::get_usec();
	cout << 1000.0 * (total_request + global_batch_factor) / (end_time - start_time) << " Kops" << endl;
};


void
nonblocking_execute(client* clnt, string mix_config, batch_logger& logger)
{
	ifstream configfile(mix_config);
	if (!configfile) {
		cout << "File " << mix_config << " not exist" << endl;
		return ;
	}

	int total_query_type;
	int total_request;
	int sleep_round = 1;
	configfile >> total_query_type >> total_request >> sleep_round;

	vector<int > distribution;
	vector<request_template > vec_template;
	vector<request_or_reply > vec_req;

	vec_template.resize(total_query_type);
	vec_req.resize(total_query_type);
	for (int i = 0; i < total_query_type; i++) {
		string filename;
		configfile >> filename;
		int current_dist;
		configfile >> current_dist;
		distribution.push_back(current_dist);
		bool success = clnt->parser.parse_template(filename, vec_template[i]);
		translate_req_template(clnt, vec_template[i]);
		vec_req[i].cmd_chains = vec_template[i].cmd_chains;
		if (!success) {
			cout << "sparql parse error" << endl;
			return ;
		}
		vec_req[i].silent = global_silent;
	}

	int send_request = 0;
	int recv_request = 0;
	while (recv_request != total_request) {
		for (int t = 0; t < 10; t++) {
			if (send_request < total_request) {
				send_request++;
				int idx = mymath::get_distribution(clnt->cfg->get_random(), distribution);
				instantiate_request(clnt, vec_template[idx], vec_req[idx]);
				clnt->GetId(vec_req[idx]);
				logger.start_record(vec_req[idx].parent_id, idx);
				clnt->Send(vec_req[idx]);
			}
		}

		for (int i = 0; i < sleep_round; i++) {
			timer::cpu_relax(100);
			request_or_reply reply;
			bool success = TryRecvR(clnt->cfg, reply);
			while (success) {
				recv_request++;
				logger.end_record(reply.parent_id);
				success = TryRecvR(clnt->cfg, reply);
			}
		}
	}
};
