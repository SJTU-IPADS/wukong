/*
 * Copyright (c) 2016 Shanghai Jiao Tong University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://ipads.se.sjtu.edu.cn/projects/wukong.html
 *
 */

#include "monitor.h"

void *
recv_cmd(void *ptr)
{
	cout << "star to receive commands from clients" << endl;

	monitor *d = (monitor *)ptr;
	while (true) {
		cout << "wait to new recv" << endl;
		CS_Request creq = d->recv_req();
		d->push(creq);
		cout << "recv a new request" << endl;
	}
}

void *
send_cmd(void *ptr)
{
	cout << "start to send commands to clients" << endl;

	monitor *d = (monitor *)ptr;
	while (true) {
		request_or_reply r = d->clnt->recv();
		cout << "(last) result size: " << r.row_num << endl;
		if (!global_silent)
			d->clnt->print_result(r, min(r.row_num, global_max_print_row));

		CS_Reply crep;
		crep.column = r.col_num;
		crep.result_table = r.result_table;
		crep.cid = d->get_cid(r.pid);
		d->send_rep(crep);
		d->remove_cid(r.pid);
	}
}

void
run_monitor(client *clnt, int port)
{
	monitor *d = new monitor(clnt, port);
	pthread_t tid[2];
	pthread_create(&(tid[0]), NULL, recv_cmd, (void *)d);
	pthread_create(&(tid[1]), NULL, send_cmd, (void *)d);

	while (true) {
		CS_Request creq = d->pop();
		string fname = creq.content;
		cout << fname << endl;
		request_or_reply r;

		ifstream ifs(fname);
		if (!ifs) {
			cout << "Query file not found: " << fname << endl;
			continue;
		}

		bool ok = clnt->parser.parse(ifs, r);
		if (!ok) {
			cout << "ERROR: SPARQL query parse error" << endl;
			CS_Reply crep;
			crep.type = "error";
			crep.content = "bad file";
			crep.cid = creq.cid;
			d->send_rep(crep);
			continue;
		}

		clnt->setpid(r);
		clnt->send(r);
		d->insert_cid(r.pid, creq.cid);
	}

	for (int i = 0; i < 2; i++) {
		int rc = pthread_join(tid[i], NULL);
		if (rc) {
			printf("ERROR: return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
	}
}
