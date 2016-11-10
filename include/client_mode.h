#include <iostream>
#include "utils.h"
#include "global_cfg.h"
#include "thread_cfg.h"
#include "client.h"
#include "batch_logger.h"
#include "proxy.h"

#include <boost/unordered_map.hpp>
#include <set>

using namespace std;

void single_execute(client *clnt, string filename, int execute_count);
void batch_execute(client *clnt, string mix_config, batch_logger &logger);
void nonblocking_execute(client *clnt, string mix_config, batch_logger &logger);

void interactive_shell(client *clnt);

void proxy(client *clnt, int port);
void *recv_cmd(void *proxy);
void *resp_cmd(void *proxy);