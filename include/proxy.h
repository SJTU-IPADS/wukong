#pragma once
#include "client.h"
#include <zmq.hpp>
#include <zhelpers.hpp>
#include <pthread.h>
#include <iostream>
#include <sstream>
#include "cs_basic_type.h"
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/unordered_map.hpp>
#include <tbb/concurrent_hash_map.h>


class Proxy {
public:
	client *clnt;
	zmq::context_t context;
	zmq::socket_t *router;
	boost::lockfree::spsc_queue<CS_Request,
	      boost::lockfree::capacity<1024>> que; //it may need a lock?
	pthread_spinlock_t send_lock;
	tbb::concurrent_hash_map<int, string> identity_table;



	Proxy(client *_clnt): clnt(_clnt) {
		pthread_spin_init(&send_lock, 0);

		router = new zmq::socket_t(context, ZMQ_ROUTER);
		s_set_id(*router);
		char address[30] = "";
		sprintf(address, "tcp://*:%d", 5450 + clnt->cfg->wid);
		cout << "port " << 5450 + clnt->cfg->wid << endl;
		router->bind (address);
	}

	~Proxy() {
		delete router;
	}

	string Recv() {
		string identity = s_recv(*router);
		s_recv(*router);
		string msg = s_recv(*router);
//		que.push(make_pair(identity, msg));
		return msg;
	}

	void Send(std::string identity, std::string msg) {
		s_sendmore(*router, identity);
		s_sendmore(*router, "");
		s_send(*router, msg);
	}

	CS_Request PopRequest() {
		while (que.empty())s_sleep(1000);
		CS_Request request = que.front();
		que.pop();
		return request;
	}

	void SendReply(string identity, CS_Reply &reply) {
		pthread_spin_lock(&send_lock);
		stringstream ss;
		boost::archive::binary_oarchive oa(ss);
		oa << reply;
		s_sendmore(*router, identity);
		s_sendmore(*router, "");
		s_send(*router, ss.str());
		pthread_spin_unlock(&send_lock);
	}

	CS_Request RecvRequest() {
		string identity = s_recv(*router);
		s_recv(*router);
		string result = s_recv(*router);
		stringstream s;
		s << result;
		boost::archive::binary_iarchive ia(s);
		CS_Request request;
		ia >> request;
		request.identity = identity;
		que.push(request);
		return request;
	}

	void InsertID(int id, string identity) {
		tbb::concurrent_hash_map<int, string>::accessor a;
		identity_table.insert(a, id);
		a->second = identity;
	}

	string GetIdentity(int id) {
		tbb::concurrent_hash_map<int, string>::const_accessor a;
		if (identity_table.find(a, id))
			return (string)(a->second);
		else
			return "";
	}

	int RemoveID(int id) {
		return identity_table.erase(id);
	}
};