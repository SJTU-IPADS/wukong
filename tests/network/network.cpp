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
 *      http://ipads.se.sjtu.edu.cn/projects/wukong
 *
 */
#include "core/network/tcp_adaptor.hpp"
#include <boost/mpi.hpp>
#include <iostream>
#include <map>
#include "utils/timer.hpp"
#include <boost/program_options.hpp>
//#include <nng/nng.h>
//#include <nng/protocol/pipeline0/push.h>
//#include <nng/protocol/pipeline0/pull.h>
//#include <nng/protocol/reqrep0/rep.h>
//#include <nng/protocol/reqrep0/req.h>
using namespace std;
using namespace boost::program_options;


string send_msg;
TCP_Adaptor * test_adaptor;
int sid;

#define thread_id (0)
#define master_sid (0)
#define slave_sid (1)
#define IS_MASTER(_p) ((_p) == 0 )
#define thread_num (1)
#define server_num (2)
#define global_port_base (9576)

// init socket
void init_socket(string host_fname)
{
    // init the tcp TCP_Adaptor
    test_adaptor = new TCP_Adaptor(sid, host_fname, global_port_base, server_num, thread_num);
}

//implement the send method
void * send (bool to_master)
{
    if (to_master) {
        // from slave to master
        test_adaptor->send(master_sid, thread_id, send_msg);
    } else {
        //from master to slave
        test_adaptor->send(slave_sid, thread_id, send_msg);
    }
}

// implent the recv method
void * recv()
{
    string msg;
    msg = test_adaptor->recv(thread_id);
}

// now only support two server to run the test
int main(int argc, char *argv[])
{
    options_description     network_desc("network test:");
    network_desc.add_options()
    ("help,h", "help message about network test")
    ("num,n", value<int>()->default_value(100)->value_name("<num>"), "run <num> times")
    ("size,s", value<int>()->default_value(1000)->value_name("<size>"), "set sending message <size>");

    variables_map network_vm;
    try {
        store(parse_command_line(argc, argv, network_desc), network_vm);
    } catch (...) { // something go wrong
        cout << "Error: error to run" << endl;
        cout << network_desc;
        return -1;
    }
    notify(network_vm);

    // parse options
    if (network_vm.count("help")) {
        if (IS_MASTER(sid))
            cout << network_desc;
        return -1;
    }

    int num = network_vm["num"].as<int>();
    int size = network_vm["size"].as<int>();
    // construct the send message
    for (int i = 0; i < size; i++ ) {
        send_msg += "a";
    }

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    sid = world.rank(); // server ID

    // set the address file of host/cluster
    string host_fname = std::string(argv[1]);
    init_socket(host_fname);

    int sum = 0;

    // begin to elvation the latency
    for (int i = 0; i < num; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        uint64_t start = timer::get_usec();
        if (IS_MASTER(sid)) {
            send(false);
            recv();
        } else {
            recv();
            send(true);
        }
        uint64_t end = timer::get_usec();
        sum += (end - start);
    }

    if (IS_MASTER(sid))
        cout << " the average sending time " << sum / num << endl;

    return 0;
}
