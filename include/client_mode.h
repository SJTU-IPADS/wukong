#include <iostream>
#include "utils.h"
#include "global_cfg.h"
#include "thread_cfg.h"
#include "client.h"

#include <boost/unordered_map.hpp>

using namespace std;

void interactive_execute(client* clnt,string filename,int execute_count);
void interactive_mode(client* clnt);

void batch_execute(client* clnt,string filename,int execute_count);
