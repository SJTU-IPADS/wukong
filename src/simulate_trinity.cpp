#include "client_mode.h"



bool simulate_execute_first_step(client* clnt, string cmd, request_or_reply& reply) {
	request_or_reply request;
	bool success = clnt->parser.parse_string(cmd, request);
	if (!success) {
		cout << "sparql parse_string error" << endl;
		return false;
	}
	request.silent = false;
	clnt->Send(request);
	reply = clnt->Recv();
	cout << "result size:" << reply.silent_row_num << endl;
	return true;
}
bool simulate_execute_other_step(client* clnt, string cmd, request_or_reply& reply, set<int>& s) {
	vector<vector<request_or_reply> > request_vec;
	int num_thread = global_num_server;

	request_vec.resize(clnt->cfg->m_num);
	for (int i = 0; i < request_vec.size(); i++) {
		request_vec[i].resize(num_thread);
		for (int j = 0; j < num_thread; j++) {
			bool success = clnt->parser.parse_string(cmd, request_vec[i][j]);
			request_vec[i][j].set_column_num(1);
			request_vec[i][j].silent = false;
			if (!success) {
				cout << "sparql parse_string error" << endl;
				return false;
			}
		}
	}

	for (set<int>::iterator iter = s.begin(); iter != s.end(); iter++) {
		int m_id = mymath::hash_mod(*iter, clnt->cfg->m_num);
		int t_id = mymath::hash_mod( (*iter) / clnt->cfg->m_num , num_thread);
		request_vec[m_id][t_id].result_table.push_back(*iter);
	}
	for (int i = 0; i < request_vec.size(); i++) {
		for (int j = 0; j < num_thread; j++) {
			clnt->GetId(request_vec[i][j]);
			SendR(clnt->cfg, i , j + clnt->cfg->client_num, request_vec[i][j]);
		}
	}
	reply = RecvR(clnt->cfg);
	for (int i = 0; i < clnt->cfg->m_num * num_thread - 1; i++) {
		request_or_reply r = RecvR(clnt->cfg);
		reply.silent_row_num += r.silent_row_num;
		int new_size = r.result_table.size() + reply.result_table.size();
		reply.result_table.reserve(new_size);
		reply.result_table.insert( reply.result_table.end(), r.result_table.begin(), r.result_table.end());
	}
	cout << "result size:" << reply.silent_row_num << endl;
	return true;
}
set<int> remove_dup(request_or_reply& reply, int col) {
	set<int> s;
	for (int i = 0; i < reply.row_num(); i++) {
		int id = reply.get_row_column(i, col);
		s.insert(id);
	}
	return s;
}

void simulate_trinity_q1(client* clnt) {
	if (clnt->cfg->m_id != 0 || clnt->cfg->t_id != 0) {
		return ;
	}
	string header = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "
	                "PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#> SELECT * WHERE { ";

	string part1 = "?Y  rdf:type ub:University <-      ?X  ub:undergraduateDegreeFrom ?Y <-";
	string part2 = "?X  rdf:type ub:GraduateStudent .  ?X  ub:memberOf ?Z .  ";
	string part3 = "?Z  ub:subOrganizationOf ?Y .      ?Z  rdf:type ub:Department .  ";
	request_or_reply r1;
	request_or_reply r2;
	request_or_reply r3;
	if (!simulate_execute_first_step(clnt, header + part1 + " }", r1)) {
		return ;
	}
	set<int> s1 = remove_dup(r1, 1);
	cout << "result size after remove_dup:" << s1.size() << endl;

	if (!simulate_execute_other_step(clnt, header + part2 + " }", r2, s1)) {
		return ;
	}
	set<int> s2 = remove_dup(r2, 1);
	cout << "result size after remove_dup:" << s2.size() << endl;

	if (!simulate_execute_other_step(clnt, header + part3 + " }", r3, s2)) {
		return ;
	}
	set<int> s3 = remove_dup(r3, 1);
	cout << "result size after remove_dup:" << s3.size() << endl;


	ofstream f1("q1_step1_yx");
	ofstream f2("q1_step2_xz");
	ofstream f3("q1_step3_zy");
	for (int i = 0; i < r1.row_num(); i++) {
		int v1 = r1.get_row_column(i, 0);
		int v2 = r1.get_row_column(i, 1);
		f1 << v1 << "\t" << v2 << endl;
	}
	for (int i = 0; i < r2.row_num(); i++) {
		int v1 = r2.get_row_column(i, 0);
		int v2 = r2.get_row_column(i, 1);
		f2 << v1 << "\t" << v2 << endl;
	}
	for (int i = 0; i < r3.row_num(); i++) {
		int v1 = r3.get_row_column(i, 0);
		int v2 = r3.get_row_column(i, 1);
		f3 << v1 << "\t" << v2 << endl;
	}

}

void simulate_trinity_q2(client* clnt) {
	if (clnt->cfg->m_id != 0 || clnt->cfg->t_id != 0) {
		return ;
	}
	string header = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "
	                "PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#> SELECT * WHERE { ";

	string part1 = "?X  rdf:type    ub:Course <- ";
	string part2 = "?X  ub:name     ?Y .  ";
	request_or_reply r1;
	request_or_reply r2;
	if (!simulate_execute_first_step(clnt, header + part1 + " }", r1)) {
		return ;
	}
	set<int> s1 = remove_dup(r1, 0);
	cout << "result size after remove_dup:" << s1.size() << endl;

	if (!simulate_execute_other_step(clnt, header + part2 + " }", r2, s1)) {
		return ;
	}
	set<int> s2 = remove_dup(r2, 0);
	cout << "result size after remove_dup:" << s2.size() << endl;


	ofstream f1("q2_step1_x");
	ofstream f2("q2_step2_x");
	for (int i = 0; i < r1.row_num(); i++) {
		int v1 = r1.get_row_column(i, 0);
		f1 << v1 << endl;
	}
	for (int i = 0; i < r2.row_num(); i++) {
		int v1 = r2.get_row_column(i, 0);
		f2 << v1 << endl;
	}

	// vector<int> vec;
	// vec.resize(11058812*2);
	// uint64_t t1=timer::get_usec();
	// vector<int> updated_result_table;
	// updated_result_table.reserve(11058812*2);
	// for(int i=0;i<vec.size();i++){
	// 	updated_result_table.push_back(vec[i]);
	// }
	// uint64_t t2=timer::get_usec();
	// cout<<"updated_result_table.size= "<<updated_result_table.size()<<endl;
	// cout<<"q2 join in "<< t2-t1<<" usec"<<endl;
}

void simulate_trinity_q3(client* clnt) {
	if (clnt->cfg->m_id != 0 || clnt->cfg->t_id != 0) {
		return ;
	}
	string header = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "
	                "PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#> SELECT * WHERE { ";

	string part1 = "?Y  rdf:type ub:University <-      ?X  ub:undergraduateDegreeFrom ?Y <-";
	string part2 = "?X  rdf:type ub:UndergraduateStudent .  ?X  ub:memberOf ?Z .  ";
	string part3 = "?Z  ub:subOrganizationOf ?Y .      ?Z  rdf:type ub:Department .  ";
	request_or_reply r1;
	request_or_reply r2;
	request_or_reply r3;
	if (!simulate_execute_first_step(clnt, header + part1 + " }", r1)) {
		return ;
	}
	set<int> s1 = remove_dup(r1, 1);
	cout << "result size after remove_dup:" << s1.size() << endl;

	if (!simulate_execute_other_step(clnt, header + part2 + " }", r2, s1)) {
		return ;
	}
	set<int> s2 = remove_dup(r2, 1);
	cout << "result size after remove_dup:" << s2.size() << endl;

	if (!simulate_execute_other_step(clnt, header + part3 + " }", r3, s2)) {
		return ;
	}
	set<int> s3 = remove_dup(r3, 1);
	cout << "result size after remove_dup:" << s3.size() << endl;


	ofstream f1("q3_step1_yx");
	ofstream f2("q3_step2_xz");
	ofstream f3("q3_step3_zy");
	for (int i = 0; i < r1.row_num(); i++) {
		int v1 = r1.get_row_column(i, 0);
		int v2 = r1.get_row_column(i, 1);
		f1 << v1 << "\t" << v2 << endl;
	}
	for (int i = 0; i < r2.row_num(); i++) {
		int v1 = r2.get_row_column(i, 0);
		int v2 = r2.get_row_column(i, 1);
		f2 << v1 << "\t" << v2 << endl;
	}
	for (int i = 0; i < r3.row_num(); i++) {
		int v1 = r3.get_row_column(i, 0);
		int v2 = r3.get_row_column(i, 1);
		f3 << v1 << "\t" << v2 << endl;
	}

}
void simulate_trinity_q4(client* clnt) {
	if (clnt->cfg->m_id != 0 || clnt->cfg->t_id != 0) {
		return ;
	}
	string header = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "
	                "PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#> SELECT * WHERE { ";

	string part1 = "?X  ub:worksFor   <http://www.Department0.University0.edu>    <-  ";
	string part2 = "?X  rdf:type   ub:FullProfessor . ";
	string part3 = "?X  ub:name ?Y1 . ";
	string part4 = "?X  ub:emailAddress ?Y2 . ";
	string part5 = "?X  ub:telephone ?Y3 . ";
	request_or_reply r1;
	request_or_reply r2;
	request_or_reply r3;
	request_or_reply r4;
	request_or_reply r5;
	if (!simulate_execute_first_step(clnt, header + part1 + " }", r1)) {
		return ;
	}
	set<int> s1 = remove_dup(r1, 0);
	cout << "result size after remove_dup:" << s1.size() << endl;

	if (!simulate_execute_other_step(clnt, header + part2 + " }", r2, s1)) {
		return ;
	}
	set<int> s2 = remove_dup(r2, 0);
	cout << "result size after remove_dup:" << s2.size() << endl;

	if (!simulate_execute_other_step(clnt, header + part3 + " }", r3, s2)) {
		return ;
	}
	set<int> s3 = remove_dup(r3, 0);
	cout << "result size after remove_dup:" << s3.size() << endl;

	if (!simulate_execute_other_step(clnt, header + part4 + " }", r4, s3)) {
		return ;
	}
	set<int> s4 = remove_dup(r4, 0);
	cout << "result size after remove_dup:" << s4.size() << endl;

	if (!simulate_execute_other_step(clnt, header + part5 + " }", r5, s4)) {
		return ;
	}
	set<int> s5 = remove_dup(r5, 0);
	cout << "result size after remove_dup:" << s2.size() << endl;


	ofstream f1("q4_step1_x");
	ofstream f2("q4_step2_x");
	ofstream f3("q4_step3_xy1");
	ofstream f4("q4_step4_xy2");
	ofstream f5("q4_step5_xy3");
	for (int i = 0; i < r1.row_num(); i++) {
		int v1 = r1.get_row_column(i, 0);
		f1 << v1 << endl;
	}
	for (int i = 0; i < r2.row_num(); i++) {
		int v1 = r2.get_row_column(i, 0);
		f2 << v1 << endl;
	}
	for (int i = 0; i < r3.row_num(); i++) {
		int v1 = r3.get_row_column(i, 0);
		int v2 = r3.get_row_column(i, 1);
		f3 << v1 << "\t" << v2 << endl;
	}
	for (int i = 0; i < r2.row_num(); i++) {
		int v1 = r4.get_row_column(i, 0);
		int v2 = r4.get_row_column(i, 1);
		f4 << v1 << "\t" << v2 << endl;
	}
	for (int i = 0; i < r2.row_num(); i++) {
		int v1 = r5.get_row_column(i, 0);
		int v2 = r5.get_row_column(i, 1);
		f5 << v1 << "\t" << v2 << endl;
	}
}
void simulate_trinity_q5(client* clnt) {
	if (clnt->cfg->m_id != 0 || clnt->cfg->t_id != 0) {
		return ;
	}
	string header = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "
	                "PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#> SELECT * WHERE { ";

	string part1 = "?X  ub:subOrganizationOf   <http://www.Department0.University0.edu>  <- "
	               ;
	string part2 = "?X  rdf:type   ub:ResearchGroup . "
	               ;
	request_or_reply r1;
	request_or_reply r2;
	if (!simulate_execute_first_step(clnt, header + part1 + " }", r1)) {
		return ;
	}
	set<int> s1 = remove_dup(r1, 0);
	cout << "result size after remove_dup:" << s1.size() << endl;

	if (!simulate_execute_other_step(clnt, header + part2 + " }", r2, s1)) {
		return ;
	}
	set<int> s2 = remove_dup(r2, 0);
	cout << "result size after remove_dup:" << s2.size() << endl;

	ofstream f1("q5_step1_x");
	ofstream f2("q5_step2_x");
	for (int i = 0; i < r1.row_num(); i++) {
		int v1 = r1.get_row_column(i, 0);
		f1 << v1 << endl;
	}
	for (int i = 0; i < r2.row_num(); i++) {
		int v1 = r2.get_row_column(i, 0);
		f2 << v1 << endl;
	}
}
void simulate_trinity_q6(client* clnt) {
	if (clnt->cfg->m_id != 0 || clnt->cfg->t_id != 0) {
		return ;
	}
	string header = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "
	                "PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#> SELECT * WHERE { ";

	string part1 = "?Y  ub:subOrganizationOf   <http://www.University0.edu>    <-  "
	               ;
	string part2 = "?Y  rdf:type   ub:Department . "
	               "?X  ub:worksFor ?Y    <- "
	               ;
	string part3 = "?X  rdf:type ub:FullProfessor . "
	               ;

	request_or_reply r1;
	request_or_reply r2;
	request_or_reply r3;

	if (!simulate_execute_first_step(clnt, header + part1 + " }", r1)) {
		return ;
	}
	set<int> s1 = remove_dup(r1, 0);
	cout << "result size after remove_dup:" << s1.size() << endl;

	if (!simulate_execute_other_step(clnt, header + part2 + " }", r2, s1)) {
		return ;
	}
	set<int> s2 = remove_dup(r2, 1);
	cout << "result size after remove_dup:" << s2.size() << endl;

	if (!simulate_execute_other_step(clnt, header + part3 + " }", r3, s2)) {
		return ;
	}
	set<int> s3 = remove_dup(r3, 0);
	cout << "result size after remove_dup:" << s3.size() << endl;
//two-step join

	ofstream f1("q6_step1_y");
	ofstream f2("q6_step2_yx");
	ofstream f3("q6_step3_x");
	for (int i = 0; i < r1.row_num(); i++) {
		int v1 = r1.get_row_column(i, 0);
		f1 << v1 << endl;
	}
	for (int i = 0; i < r2.row_num(); i++) {
		int v1 = r2.get_row_column(i, 0);
		int v2 = r2.get_row_column(i, 1);
		f2 << v1 << "\t" << v2 << endl;
	}
	for (int i = 0; i < r3.row_num(); i++) {
		int v1 = r3.get_row_column(i, 0);
		f3 << v1 << endl;
	}

	boost::unordered_map<int, vector<int> > hashtable1;
	for (int i = 0; i < r2.row_num(); i++) {
		int v1 = r2.get_row_column(i, 0);
		int v2 = r2.get_row_column(i, 1);
		hashtable1[v1].push_back(v2);
	}
	vector<int> updated_result_table;
	for (int i = 0; i < r1.row_num(); i++) {
		int vid = r1.get_row_column(i, 0);
		if (hashtable1.find(vid) != hashtable1.end()) {
			for (int k = 0; k < hashtable1[vid].size(); k++) {
				r1.append_row_to(i, updated_result_table);
				updated_result_table.push_back(hashtable1[vid][k]);
			}
		}
	}
	r1.set_column_num(r1.column_num() + 1);
	r1.result_table.swap(updated_result_table);
	updated_result_table.clear();

	boost::unordered_set<int > hashset2;
	for (int i = 0; i < r3.row_num(); i++) {
		int v1 = r3.get_row_column(i, 0);
		hashset2.insert(v1);
	}
	for (int i = 0; i < r1.row_num(); i++) {
		int v1 = r1.get_row_column(i, 0);
		int v2 = r1.get_row_column(i, 1);
		if (hashset2.find(v2) != hashset2.end()) {
			r1.append_row_to(i, updated_result_table);
		}
	}
	r1.result_table.swap(updated_result_table);
	cout << "final join result size:" << r1.row_num() << endl;

}
void simulate_trinity_q7(client* clnt) {
	if (clnt->cfg->m_id != 0 || clnt->cfg->t_id != 0) {
		return ;
	}
	/*
		string header="PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "
		"PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#> SELECT * WHERE { ";

		string part1="?Y  rdf:type ub:FullProfessor <-  "
					"?Y  ub:teacherOf ?Z . "
					;
		string part2=
					"?Z  rdf:type  ub:Course . "
					 "?X  ub:takesCourse ?Z <- "
					;
		string part3="?X  ub:advisor    ?Y . "
					 "?X  rdf:type ub:UndergraduateStudent . "
					;

		request_or_reply r1;
		request_or_reply r2;
		request_or_reply r3;
		uint64_t t[20];
		t[0]=timer::get_usec();
		if(!simulate_execute_first_step(clnt,header+part1+" }",r1)){
			return ;
		}
		t[1]=timer::get_usec();
		set<int> s1=remove_dup(r1,1);
		t[2]=timer::get_usec();
		cout<<"result size after remove_dup:"<<s1.size()<<endl;

		t[3]=timer::get_usec();
		if(!simulate_execute_other_step(clnt,header+part2+" }",r2,s1)){
			return ;
		}
		t[4]=timer::get_usec();
		set<int> s2=remove_dup(r2,1);
		t[5]=timer::get_usec();
		cout<<"result size after remove_dup:"<<s2.size()<<endl;

		t[6]=timer::get_usec();
		if(!simulate_execute_other_step(clnt,header+part3+" }",r3,s2)){
			return ;
		}
		t[7]=timer::get_usec();
		set<int> s3=remove_dup(r3,1);
		t[8]=timer::get_usec();
		cout<<"result size after remove_dup:"<<s3.size()<<endl;

		cout<<t[1]-t[0]<<"   "<<t[2]-t[1]<<" usec"<<endl;
		cout<<t[4]-t[3]<<"   "<<t[5]-t[4]<<" usec"<<endl;
		cout<<t[7]-t[6]<<"   "<<t[8]-t[7]<<" usec"<<endl;

		t[9]=timer::get_usec();

		ofstream f1("file_yz");
		ofstream f2("file_zx");
		ofstream f3("file_xy");
		for(int i=0;i<r1.row_num();i++){
			int v1=r1.get_row_column(i,0);
			int v2=r1.get_row_column(i,1);
			f1<<v1<<"\t"<<v2<<endl;
		}
		for(int i=0;i<r2.row_num();i++){
			int v1=r2.get_row_column(i,0);
			int v2=r2.get_row_column(i,1);
			f2<<v1<<"\t"<<v2<<endl;
		}
		for(int i=0;i<r3.row_num();i++){
			int v1=r3.get_row_column(i,0);
			int v2=r3.get_row_column(i,1);
			f3<<v1<<"\t"<<v2<<endl;
		}

	//two-step join
		boost::unordered_map<int,vector<int> > hashtable1;
		for(int i=0;i<r2.row_num();i++){
			int v1=r2.get_row_column(i,0);
			int v2=r2.get_row_column(i,1);
			hashtable1[v1].push_back(v2);
		}
		vector<int> updated_result_table;
		for(int i=0;i<r1.row_num();i++){
			int vid=r1.get_row_column(i,1);
			if(hashtable1.find(vid)!=hashtable1.end()){
				for(int k=0;k<hashtable1[vid].size();k++){
					r1.append_row_to(i,updated_result_table);
		            updated_result_table.push_back(hashtable1[vid][k]);
				}
			}
		}
		r1.set_column_num(r1.column_num()+1);
	    r1.result_table.swap(updated_result_table);
		updated_result_table.clear();

		boost::unordered_map<int,vector<int> > hashtable2;
		for(int i=0;i<r3.row_num();i++){
			int v1=r3.get_row_column(i,0);
			int v2=r3.get_row_column(i,1);
			hashtable2[v1].push_back(v2);
		}
		for(int i=0;i<r1.row_num();i++){
			int v1=r1.get_row_column(i,0);
			int v2=r1.get_row_column(i,2);
			if(hashtable2.find(v2)!=hashtable2.end()){
				for(int k=0;k<hashtable2[v2].size();k++){
					if(v1==hashtable2[v2][k]){
						r1.append_row_to(i,updated_result_table);
					}
				}
			}
		}
		r1.result_table.swap(updated_result_table);
		t[10]=timer::get_usec();
		cout<<"final join result size:"<<r1.row_num()<<endl;
		cout<<t[10]-t[9]<<" usec for join-time"<<endl;
	*/


	string header = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "
	                "PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#> SELECT * WHERE { ";

	string part1 = "?Y  rdf:type ub:FullProfessor <-  ?X  ub:advisor ?Y <-  " ;
	string part2 = "?X  rdf:type ub:UndergraduateStudent .  ?X  ub:takesCourse ?Z . ";
	string part3 = "?Z  rdf:type ub:Course .  ?Y  ub:teacherOf ?Z <- ";


	request_or_reply r1;
	request_or_reply r2;
	request_or_reply r3;
	if (!simulate_execute_first_step(clnt, header + part1 + " }", r1)) {
		return ;
	}
	set<int> s1 = remove_dup(r1, 1);
	cout << "result size after remove_dup:" << s1.size() << endl;

	if (!simulate_execute_other_step(clnt, header + part2 + " }", r2, s1)) {
		return ;
	}
	set<int> s2 = remove_dup(r2, 1);
	cout << "result size after remove_dup:" << s2.size() << endl;

	if (!simulate_execute_other_step(clnt, header + part3 + " }", r3, s2)) {
		return ;
	}
	set<int> s3 = remove_dup(r3, 1);
	cout << "result size after remove_dup:" << s3.size() << endl;

	ofstream f1("q7_step1_yx");
	ofstream f2("q7_step2_xz");
	ofstream f3("q7_step2_zy");
	for (int i = 0; i < r1.row_num(); i++) {
		int v1 = r1.get_row_column(i, 0);
		int v2 = r1.get_row_column(i, 1);
		f1 << v1 << "\t" << v2 << endl;
	}
	for (int i = 0; i < r2.row_num(); i++) {
		int v1 = r2.get_row_column(i, 0);
		int v2 = r2.get_row_column(i, 1);
		f2 << v1 << "\t" << v2 << endl;
	}
	for (int i = 0; i < r3.row_num(); i++) {
		int v1 = r3.get_row_column(i, 0);
		int v2 = r3.get_row_column(i, 1);
		f3 << v1 << "\t" << v2 << endl;
	}

	uint64_t t1 = timer::get_usec();

	boost::unordered_map<int, vector<int> > hashtable1;
	for (int i = 0; i < r2.row_num(); i++) {
		int v1 = r2.get_row_column(i, 0);
		int v2 = r2.get_row_column(i, 1);
		hashtable1[v1].push_back(v2);
	}
	vector<int> updated_result_table;
	for (int i = 0; i < r1.row_num(); i++) {
		int vid = r1.get_row_column(i, 1);
		if (hashtable1.find(vid) != hashtable1.end()) {
			for (int k = 0; k < hashtable1[vid].size(); k++) {
				r1.append_row_to(i, updated_result_table);
				updated_result_table.push_back(hashtable1[vid][k]);
			}
		}
	}
	r1.set_column_num(r1.column_num() + 1);
	r1.result_table.swap(updated_result_table);
	updated_result_table.clear();

	boost::unordered_map<int, vector<int> > hashtable2;
	for (int i = 0; i < r3.row_num(); i++) {
		int v1 = r3.get_row_column(i, 0);
		int v2 = r3.get_row_column(i, 1);
		hashtable2[v1].push_back(v2);
	}
	for (int i = 0; i < r1.row_num(); i++) {
		int v1 = r1.get_row_column(i, 0);
		int v2 = r1.get_row_column(i, 2);
		if (hashtable2.find(v2) != hashtable2.end()) {
			for (int k = 0; k < hashtable2[v2].size(); k++) {
				if (v1 == hashtable2[v2][k]) {
					r1.append_row_to(i, updated_result_table);
				}
			}
		}
	}
	r1.result_table.swap(updated_result_table);
	uint64_t t2 = timer::get_usec();
	cout << "final join result size:" << r1.row_num() << endl;
	cout << t2 - t1 << " usec for join-time" << endl;
}
