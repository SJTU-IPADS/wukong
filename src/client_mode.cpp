#include "client_mode.h"

static void
translate_req_template(client *clnt, request_template &req_template)
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
			vector<int64_t> *ptr = new vector<int64_t>();
			*ptr = reply.result_table;
			clnt->parser.type_to_idvec[type] = ptr;
			cout << type << " has " << ptr->size() << " objects" << endl;
		}
		req_template.place_holder_vecptr[i] = clnt->parser.type_to_idvec[type];
	}
}

static void
instantiate_request(client *clnt, request_template &req_template, request_or_reply &r)
{
	for (int i = 0; i < req_template.place_holder_position.size(); i++) {
		int pos = req_template.place_holder_position[i];
		vector<int64_t> *vecptr = req_template.place_holder_vecptr[i];
		if (vecptr == NULL || vecptr->size() == 0) {
			assert(false);
		}
		r.cmd_chains[pos] = (*vecptr)[clnt->cfg->get_random() % (vecptr->size())];
	}
}

static void
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
single_execute(client* clnt, string fname, int cnt)
{
	request_or_reply request, reply;

	if (!clnt->parser.parse(fname, request)) {
		cout << "ERROR: parse SPARQL query failed!" << endl;
		return;
	}
	request.silent = global_silent;

	uint64_t t = timer::get_usec();
	for (int i = 0; i < cnt; i++) {
		clnt->Send(request);
		reply = clnt->Recv();
	}
	t = timer::get_usec() - t;

	cout << "(last) result size: " << reply.silent_row_num << endl;
	cout << "(average) latency: " << (t / cnt) << " usec" << endl;

	int row_to_print = min((uint64_t)reply.row_num(), (uint64_t)global_max_print_row);
	if (row_to_print > 0)
		clnt->print_result(reply, row_to_print);
}

void
print_help(void)
{
	cout << "Wukong's client commands: " << endl;
	cout << "    help            display help infomation" << endl;
	cout << "    quit            quit from client" << endl;
	cout << "    reconfig        reload config file" << endl;
	cout << "    switch_mode: single|batch|mix " << endl;
	cout << "        single      run a special query (e.g., query [count])" << endl;
	cout << "        batch       run a set of queries (e.g., queries)" << endl;
	cout << "        mix         run a set of queries with a special query" << endl;

}

enum { SINGLE_MODE = 0, BATCH_MODE, MIX_MODE, N_MODES };

string mode_str[N_MODES] = {
	"\tsingle mode (e.g., query [count]):",
	"\tbatch mode (e.g., queries):",
	"\tmix mode: (e.g., queries query [count]):"
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

		string cmd;
		if (cfg->sid == 0 && cfg->wid == 0) {
local_done:
			cout << "> ";
			std::getline(std::cin, cmd);

			// trim input
			size_t pos = cmd.find_first_not_of(" \t"); // trim blanks from head
			if (pos == string::npos) goto local_done;
			cmd.erase(0, pos);

			pos = cmd.find_last_not_of(" \t");  // trim blanks from tail
			cmd.erase(pos + 1, cmd.length() - (pos + 1));

			if (cmd == "help") {
				print_help();
				goto local_done;
			}

			// send commands to all client workers
			for (int i = 0; i < cfg->nsrvs; i++) {
				for (int j = 0; j < cfg->ncwkrs; j++) {
					if (i == 0 && j == 0)
						continue;
					cfg->node->Send(i, j, cmd);
				}
			}
		} else {
			// recieve commands
			cmd = cfg->node->Recv();
		}

		// handle a command
		istringstream cmd_stream(cmd);
		if (cmd_stream.str() == "quit") {
			if (cfg->wid == 0)
				exit(0); // each server exits once
		} else if (cmd_stream.str() == "reconfig") {
			if (cfg->wid == 0)
				reload_cfg(); // each server reconfigs once
		} else if (cmd_stream.str().find("switch_mode:") == 0) {
			if (cfg->wid == 0) {
				string skip, mode;
				cmd_stream >> skip >> mode;
				if (mode == "single")
					global_client_mode = SINGLE_MODE;
				else if (mode == "batch")
					global_client_mode = BATCH_MODE;
				else if (mode == "mix")
					global_client_mode = MIX_MODE;

				if (cfg->sid == 0)
					cout << mode_str[global_client_mode] << endl;
			}
		} else { // handle SPARQL queries
			batch_logger logger;
			string qfile;
			int cnt = 1;

			if (global_client_mode == SINGLE_MODE) {
				if (cfg->sid == 0 && cfg->wid == 0) {
					// run single-command using the master client
					cmd_stream >> qfile >> cnt;
					if (cnt < 1) cnt = 1;
					single_execute(clnt, qfile, cnt);
				}
			} else if (global_client_mode == BATCH_MODE) {
				// Q: run batch-command on all clients?
				cmd_stream >> qfile;

				logger.init();
				nonblocking_execute(clnt, qfile, logger);
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
				string qfile2;

				cmd_stream >> qfile >> qfile2 >> cnt;
				if (cnt < 1) cnt = 1;

				if (cfg->sid == 0 && cfg->wid == 0) {
					// dedicate the master client to run single-command
					single_execute(clnt, qfile2, cnt);
				} else {
					// run batch-command on other clients
					logger.init();
					nonblocking_execute(clnt, qfile, logger);
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
