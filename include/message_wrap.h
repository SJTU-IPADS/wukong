#pragma once

#include "query_basic_types.h"
#include "network_node.h"
#include "rdma_resource.h"
#include "thread_cfg.h"
#include "global_cfg.h"


void SendR(thread_cfg* cfg,int r_mid,int r_tid,request_or_reply& r);
request_or_reply RecvR(thread_cfg* cfg);
