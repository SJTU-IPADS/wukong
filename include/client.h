#pragma once

#include "query_basic_types.h"
#include "global_cfg.h"
#include "thread_cfg.h"
#include "message_wrap.h"
#include "sparql_parser.h"
#include "string_server.h"

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

class client{
public:
	thread_cfg* cfg;
	string_server* str_server;
	sparql_parser parser;
    client(thread_cfg* _cfg,string_server* str_server);
	void GetId(request_or_reply& req);
	void Send(request_or_reply& req);
	request_or_reply Recv();
	void print_result(request_or_reply& reply,int row_to_print);
};
