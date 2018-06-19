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

#pragma once

#include <iostream>
#include <string>
#include <set>

#include <boost/unordered_map.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include "config.hpp"
#include "proxy.hpp"
#include "logger.hpp"

using namespace std;
using namespace boost;
using namespace boost::program_options;
// communicate between proxy threads
TCP_Adaptor *con_adaptor;

bool enable_oneshot = false;
string oneshot_cmd = "";

template<typename T>
static void console_send(int sid, int tid, T &r) {
    stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << r;
    con_adaptor->send(sid, tid, ss.str());
}

    template<typename T>
static T console_recv(int tid)
{
    string str;
    str = con_adaptor->recv(tid);

    stringstream ss;
    ss << str;

    boost::archive::binary_iarchive ia(ss);
    T r;
    ia >> r;
    return r;
}

static void console_barrier(int tid)
{
    static int _curr = 0;
    static __thread int _next = 1;

    // inter-server barrier
    if (tid == 0)
        MPI_Barrier(MPI_COMM_WORLD);

    // intra-server barrier
    __sync_fetch_and_add(&_curr, 1);
    while (_curr < _next)
        usleep(1); // wait
    _next += global_num_proxies; // next barrier
}

// the master proxy is the 1st proxy of the 1st server (i.e., sid == 0 and tid == 0)
#define IS_MASTER(_p) ((_p)->sid == 0 && (_p)->tid == 0)
#define PRINT_ID(_p) (cout << "[" << (_p)->sid << "-" << (_p)->tid << "]$ ")

static void file2str(string fname, string &str)
{
    ifstream file(fname.c_str());
    if (!file) {
        logstream(LOG_ERROR) << fname << " does not exist." << LOG_endl;
        return;
    }

    string line;
    while (getline(file, line))
        str += line + " ";
}

static void args2str(string &str)
{
    size_t found = str.find_first_of("=&");
    while (found != string::npos) {
        str[found] = ' ';
        found = str.find_first_of("=&", found + 1);
    }
}



// options define
// the all commmand 
options_description combine_desc("These are common Wukong commands: ");
// define the commmand
options_description help_desc("help                display help infomation");
options_description quit_desc("quit                quit from console");
options_description config_desc("config <args>       run commands on config");
options_description sparql_desc("sparql <args>       run a single SPARQL query");
options_description sparql_emu_desc("sparql-emu <args>   emulate clients to continuously send SPARQL queries");
options_description load_desc("load <args>         load linked data into dynamic (in-memmory) graph store");
options_description gsck_desc("gsck <args>         check the graph storage integrity");
options_description load_stat_desc("load-stat           load statistics from a file");
options_description store_stat_desc("store-stat           store statistics to a file");



/*
 * init the options_description
 * it should add the option to different options_description
 * all should be added to combine_desc
 */
void init_options_desc() {
    combine_desc.add(help_desc); // add it to combine_desc 
    combine_desc.add(quit_desc); // add it to combine_desc 

    // add the options that should be parsed
    config_desc.add_options() 
        ("help,h", "help message about config")
        (",v", "print current config")
        (",l", value<string>()->value_name("<file>"), "load config items from <file>")
        (",s", value<string>()->value_name("<string>"), "set config items by <str> (format: item1=val1&item2=...)")
        ;
    combine_desc.add(config_desc); // add it to combine_desc 

    sparql_desc.add_options() 
        ("help,h", "help message about sparql")
        (",f", value<string>()->value_name("<file>"), "run a single query from <file>")
        (",m", value<int>()->default_value(1)->value_name("<factor>"), "set multi-threading factor <factor> for heavy queries")
        (",n", value<int>()->default_value(1)->value_name("<num>"), "run <num> times")
        (",v", value<int>()->default_value(0)->value_name("<num>"), "print at most <num> lines of results")
        (",o", value<string>()->value_name("<file>"), "output results into <file>")
        ("batch,b", value<string>()->value_name("<file>"), "run a batch of queries configured by <file>")
        ;
    combine_desc.add(sparql_desc); // add it to combine_desc 

    sparql_emu_desc.add_options() 
        ("help,h", "help message about sparql-emu")
        (",f", value<string>()->value_name("<file>"), "run queries generated from temples configured by <file>")
        (",d", value<int>()->default_value(10)->value_name("<sec>"), "eval <sec> seconds (default: 10)")
        (",w", value<int>()->default_value(5)->value_name("<sec>"), "warmup <sec> seconds (default: 5)")
        (",p", value<int>()->default_value(20)->value_name("<num>"), "send <num> queries in parallel (default: 20)")
        ;
    combine_desc.add(sparql_emu_desc); // add it to combine_desc 

    load_desc.add_options() 
        ("help,h", "help message about load")
        ("directory,d", value<string>()->value_name("<dname>"), "load data from directory <dname>")
        ("check,c",  "check and skip duplicate rdf triple")
        ;
    combine_desc.add(load_desc); // add it to combine_desc 

    gsck_desc.add_options() 
        ("help,h", "help message about gsck")
        ("index,i", "check from index key/value pair to normal key/value pair")
        ("normal,n","check from normal key/value pair to index key/value pair")
        ("all,a", "check all above")
        ;
    combine_desc.add(gsck_desc); // add it to combine_desc 

    load_stat_desc.add_options() 
        ("help,h", "help message about load-stat")
        (",f", value<string>()->value_name("<file>"), "load statistics from <file> located at data folder")
        ;
    combine_desc.add(load_stat_desc); // add it to combine_desc 

    store_stat_desc.add_options() 
        ("help,h", "help message about store-stat")
        (",f", value<string>()->value_name("<file>"), "store statistics to <file> located at data folder")
        ;
    combine_desc.add(store_stat_desc); // add it to combine_desc 

}

/*
 * print help of all command
 */
void print_help(void)
{
    cout << combine_desc <<endl;
}

/*
 * fail to parse the command
 */
void fail_to_parse(Proxy *proxy, int argc, char** argv){
    if (IS_MASTER(proxy)) {
        string cmd;
        for(int i=0; i<argc; i++) {
            cmd = cmd + argv[i] + " ";
        }
        logstream(LOG_ERROR) << "Failed to run the command: " << cmd << LOG_endl;
        print_help();
    }
}

/**
 * split the string into  char** by the space
 * the argc is the name of char*
 * the return value argv is  char** 
 */
char**  str2command_args(string str, int& argc){
    vector<string> fields;

    split( fields, str,  is_any_of( " " ) );
    // set the argc as number of argument 
    argc = fields.size();
    char** argv = new char*[argc + 1];
    for(int i=0; i< argc; i++) {
        argv[i] = new char[fields[i].length() + 1];
        strcpy(argv[i],fields[i].c_str());
    }
    argv[argc] = NULL;
    return argv;
}

/*
 * execute the config command
 */
void run_config(Proxy *proxy, int argc, char** argv) {
    // only the main proxy do it 
    if (proxy->tid != 0) 
        return;

    string fname, str;

    // variables map that store the mapping of options and its value
    variables_map config_vm;
    // begin to parse
    try {
        store(parse_command_line(argc, argv, config_desc), config_vm);
    }
    catch (...){ // something go wrong
        fail_to_parse(proxy, argc, argv);
        return;
    }
    // check and refine the variables_map
    notify(config_vm);

    // different flag
    if (config_vm.count("help")) {
        if (IS_MASTER(proxy)) 
            cout << config_desc;
        return;
    }

    if(config_vm.count("-f"))
        fname = config_vm["-f"].as<string>();

    if(config_vm.count("-s"))
        str = config_vm["-s"].as<string>();

    if (config_vm.count("-v")) { // -v
        if (IS_MASTER(proxy))
            print_config();
    } else if (config_vm.count("-l") || config_vm.count("-s")) { // -l <file> or -s <str>
        if (IS_MASTER(proxy)) {
            if (config_vm.count("-l")) // -l
                file2str(fname, str);
            else if (config_vm.count("-s")) // -s
                args2str(str);

            // send <str> to all consoles
            for (int i = 1; i < global_num_servers; i++)
                console_send<string>(i, 0, str);
        } else {
            // recieve <str>
            str = console_recv<string>(proxy->tid);
        }

        if (!str.empty()) {
            reload_config(str);
        } else {
            if (IS_MASTER(proxy))
                logstream(LOG_ERROR) << "Failed to load config file: " << fname << LOG_endl;
        }
    }

}

/*
 * run the sparql command
 * usage: sparql -f <file> [<args>]
 *               -n <num>
 *               -v <num>
 *               -o <fname>
 *               -m <factor>
 *               -b <file>
 */
void run_sparql_cmd(Proxy *proxy, int argc, char** argv) {
    // use the main proxy thread to run a single query
    if (!IS_MASTER(proxy)) 
        return;

    // file name of query
    string fname;
    int cnt = 1, nlines = 0, mt_factor = 1;
    // the variables_map 
    variables_map sparql_vm;
    // parse command
    try {
        store(parse_command_line(argc, argv, sparql_desc), sparql_vm);
    }
    catch (...){// something go wrong
        fail_to_parse(proxy, argc, argv);
        return;
    }

    notify(sparql_vm);

    // different flag
    if (sparql_vm.count("help")) {
        if (IS_MASTER(proxy)) 
            cout << sparql_desc;
        return;
    }

    if (!(sparql_vm.count("-f") ^ sparql_vm.count("batch"))){ 
        fail_to_parse(proxy, argc, argv); // invalid cmd
        return;
    }
    if (sparql_vm.count("-m")) {
        mt_factor = sparql_vm["-m"].as<int>();
    }
    if (sparql_vm.count("-n")) {
        cnt = sparql_vm["-n"].as<int>();
    }
    if (sparql_vm.count("-v")) {
        nlines = sparql_vm["-v"].as<int>();
    }
    // [single mode]
    //  usage: sparql -f <file> [<args>]
    //  args:
    //    -n <num>
    //    -v <num>
    //    -o <fname>
    if (sparql_vm.count("-f")) {
        fname = sparql_vm["-f"].as<string>();
        // read a SPARQL query from a file
        ifstream ifs(fname);
        if (!ifs.good()) {
            logstream(LOG_ERROR) << "Query file not found: " << fname << LOG_endl;
            fail_to_parse(proxy, argc, argv); // invalid cmd
            return;
        }

        if (global_silent) { // not retrieve the query results
            if (nlines > 0) {
                logstream(LOG_ERROR) << "Can't print results (-v) with global_silent." << LOG_endl;
                fail_to_parse(proxy, argc, argv); // invalid cmd
                return;
            }

            if (sparql_vm.count("-o")) {
                logstream(LOG_ERROR) << "Can't output results (-o) with global_silent." << LOG_endl;
                fail_to_parse(proxy, argc, argv); // invalid cmd
                return;
            }
        }

        SPARQLQuery reply;
        SPARQLQuery::Result &result = reply.result;
        Logger logger;
        int ret = proxy->run_single_query(ifs, mt_factor, cnt, reply, logger);
        if (ret != 0) {
            logstream(LOG_ERROR) << "Failed to run the query (ERRNO: " << ret << ")!" << LOG_endl;
            fail_to_parse(proxy, argc, argv); // invalid cmd
            return;
        }

        logger.print_latency(cnt);
        logstream(LOG_INFO) << "(last) result size: " << result.row_num << LOG_endl;

        // print or dump results
        if (!global_silent && !result.blind && (nlines > 0 || sparql_vm.count("output"))) {
            if (nlines > 0)
                result.print_result(min(result.get_row_num(), nlines), proxy->str_server);

            if (sparql_vm.count("-o"))
                result.dump_result(sparql_vm["-o"].as<string>(), result.get_row_num(), proxy->str_server);
        }
    }

    // [batch mode]
    //  usage: sparql -b <fname>
    //  file-format:
    //    sparql -f <fname> [<args>]
    //    sparql -f <fname> [<args>]
    if (sparql_vm.count("batch")) {
        fname = sparql_vm["batch"].as<string>();
        ifstream ifs(fname);
        if (!ifs.good()) {
            logstream(LOG_ERROR) << "Query file not found: " << fname << LOG_endl;
            fail_to_parse(proxy, argc, argv); // invalid cmd
            return;
        }

        logstream(LOG_INFO) << "Batch-mode starting ..." << LOG_endl;

        string sg_cmd; // a single command line
        while (getline(ifs, sg_cmd)) {
            int sg_argc = 0;
            char** sg_argv = str2command_args(sg_cmd, sg_argc);

            string tk1, tk2;
            tk1 = string(sg_argv[0]);
            tk2 = string(sg_argv[1]);
            if (tk1 == "sparql" && tk2 == "-f") {
                cout << "Run the command: " << sg_cmd << endl;
                run_sparql_cmd(proxy, sg_argc, sg_argv);
                cout << endl;
            } else {
                // skip and go on
                logstream(LOG_ERROR) << "Failed to run the command: " << sg_cmd << LOG_endl;
                logstream(LOG_ERROR) << "only support single sparql query in batch mode "
                    << "(e.g., sparql -f ...)" << LOG_endl;
            }
        }

        logstream(LOG_INFO) << "Batch-mode end." << LOG_endl;
    }

}

// usage: sparql-emu -f <file> [<args>]
//   -d <sec>
//   -w <sec>
//   -p <num>
void run_sparql_emu(Proxy *proxy, int argc, char** argv) {
    /// file name of query
    string fname;
    int duration = 10, warmup = 5, parallel_factor = 20;
    // the variables_map 
    variables_map sparql_emu_vm;
    // parse command
    try {
        store(parse_command_line(argc, argv, sparql_emu_desc), sparql_emu_vm);
    }
    catch (...){// something go wrong
        fail_to_parse(proxy, argc, argv);
        return;
    }

    notify(sparql_emu_vm);

    // different flag
    if (sparql_emu_vm.count("help")) {
        if (IS_MASTER(proxy)) 
            cout << sparql_emu_desc;
        return;
    }

    if (!sparql_emu_vm.count("-f")) { 
        fail_to_parse(proxy, argc, argv); // invalid cmd
        return;
    } else {
        fname = sparql_emu_vm["-f"].as<string>();
    }
    if (!sparql_emu_vm.count("-d")) { 
        duration = sparql_emu_vm["-d"].as<int>();
    }
    if (!sparql_emu_vm.count("-w")) { 
        warmup = sparql_emu_vm["-w"].as<int>();
    }
    if (!sparql_emu_vm.count("-p")) { 
        parallel_factor = sparql_emu_vm["-p"].as<int>();
    }


    // read config file of a SPARQL emulator
    ifstream ifs(fname);
    if (!ifs.good()) {
        logstream(LOG_ERROR) << "Configure file not found: " << fname << LOG_endl;
        fail_to_parse(proxy, argc, argv);
        return;
    }

    if (duration <= 0 || warmup < 0 || parallel_factor <= 0) {
        logstream(LOG_ERROR) << "invalid parameters for SPARQL emulator! "
            << "(duration=" << duration << ", warmup=" << warmup
            << ", parallel_factor=" << parallel_factor << ")" << LOG_endl;
        fail_to_parse(proxy, argc, argv); // invalid cmd
        return;
    }

    if (duration <= warmup) {
        logstream(LOG_INFO) << "Duration time (" << duration
            << "sec) is less than warmup time ("
            << warmup << "sec)." << LOG_endl;
        fail_to_parse(proxy, argc, argv); // invalid cmd
        return;
    }

    Logger logger;
    proxy->run_query_emu(ifs, duration, warmup, parallel_factor, logger);

    // FIXME: maybe hang in here if the input file misses in some machines
    //        or inconsistent global variables (e.g., global_enable_planner)
    console_barrier(proxy->tid);

    // print a performance statistic for running emulator on all servers
    if (IS_MASTER(proxy)) {
        for (int i = 0; i < global_num_servers * global_num_proxies - 1; i++) {
            Logger other = console_recv<Logger>(proxy->tid);
            logger.merge(other);
        }
        logger.aggregate();
        logger.print_cdf();
        logger.print_thpt();
    } else {
        // send logs to the master proxy
        console_send<Logger>(0, 0, logger);
    }
}



void run_load(Proxy *proxy, int argc, char**argv) {
    // use the main proxy thread to run gstore checker
    if (!IS_MASTER(proxy)) 
        return;

#if DYNAMIC_GSTORE
    string dname;
    // the variables_map 
    variables_map load_vm;
    // parse command
    try {
        store(parse_command_line(argc, argv, load_desc), load_vm);
    }
    catch (...){// something go wrong
        fail_to_parse(proxy, argc, argv);
        return;
    }

    notify(load_vm);

    // different flag
    if (load_vm.count("help")) {
        if (IS_MASTER(proxy)) 
            cout << load_desc;
        return;
    }

    if (!load_vm.cout("directory"))
    {
        // invalid cmd
        fail_to_parse(proxy, argc, argv);
        return ;
    } else {
        dname = load_vm["directory"].ad<string>();
    }

    // force a "/" at the end of dname.
    if (dname[dname.length() - 1] != '/')
        dname = dname + "/";

    bool c_enable =false;
    if(load_vm.count("check")) {
        c_enable = true;
    }
    Logger logger;
    RDFLoad reply;
    int ret = proxy->dynamic_load_data(dname, reply, logger, c_enable);
    if (ret != 0) {
        logstream(LOG_ERROR) << "Failed to load dynamic data from directory " << dname
            << " (ERRNO: " << ret << ")!" << LOG_endl;
        return;
    }
    logger.print_latency();
#else
    logstream(LOG_ERROR) << "Can't load linked data into static graph-store." << LOG_endl;
    logstream(LOG_ERROR) << "You can enable it by building Wukong with -DUSE_DYNAMIC_GSTORE=ON." << LOG_endl;
#endif

}

void run_gsck(Proxy *proxy, int argc, char**argv) {
    // use the main proxy thread to run gstore checker
    if (!IS_MASTER(proxy)) 
        return;

    bool i_enable = false;
    bool n_enable = false;

    // the variables_map 
    variables_map gsck_vm;
    // parse command
    try {
        store(parse_command_line(argc, argv, gsck_desc), gsck_vm);
    }
    catch (...){
        fail_to_parse(proxy, argc, argv);
        return;
    }

    notify(gsck_vm);

    // different flag
    if (gsck_vm.count("help")) {
        if (IS_MASTER(proxy)) 
            cout << gsck_desc;
        return;
    }
    if (gsck_vm.count("index")) {
        i_enable = true;
    }
    if (gsck_vm.count("normal")) {
        n_enable = true;
    }
    if (gsck_vm.count("all")) {
        i_enable = true;
        n_enable = true;
    }

    Logger logger;
    GStoreCheck reply;
    int ret = proxy->gstore_check(reply, logger, i_enable, n_enable);
    if (ret != 0) {
        logstream(LOG_ERROR) << "Some error found in gstore "
            << " (ERRNO: " << ret << ")!" << LOG_endl;
        return;
    }
    logger.print_latency();

}

void run_load_stat(Proxy *proxy, int argc, char**argv) {
    // use the main proxy thread to load statistics
    if (!IS_MASTER(proxy)) 
        return;

    string fname;
    // the variables_map 
    variables_map load_stat_vm;
    // parse command
    try {
        store(parse_command_line(argc, argv, load_stat_desc), load_stat_vm);
    }
    catch (...){
        fail_to_parse(proxy, argc, argv);
        return;
    }

    notify(load_stat_vm);

    // different flag
    if (load_stat_vm.count("help")) {
        if (IS_MASTER(proxy)) 
            cout << load_stat_desc;
        return;
    }

    // if fname is not given, try the dataset name by default
    if (!load_stat_vm.count("-f")) {
        vector<string> strs;
        boost::split(strs, global_input_folder, boost::is_any_of("/"));
        fname = strs[strs.size() - 2] + ".statfile";
    } else {
        fname = load_stat_vm["-f"].as<string>();
    }

    proxy->statistic->load_stat_from_file(fname);

}

void run_store_stat(Proxy *proxy, int argc, char**argv) {
    // use the main proxy thread to save statistics
    if (!IS_MASTER(proxy)) 
        return;

    string fname;
    // the variables_map 
    variables_map store_stat_vm;
    // parse command
    try {
        store(parse_command_line(argc, argv, store_stat_desc), store_stat_vm);
    }
    catch (...){
        fail_to_parse(proxy, argc, argv);
        return;
    }

    notify(store_stat_vm);

    // different flag
    if (store_stat_vm.count("help")) {
        if (IS_MASTER(proxy)) 
            cout << store_stat_desc;
        return;
    }

    // if fname is not given, use the dataset name by default
    if (!store_stat_vm.count("-f")) {
        vector<string> strs;
        boost::split(strs, global_input_folder, boost::is_any_of("/"));
        fname = strs[strs.size() - 2] + ".statfile";
    } else {
        fname = store_stat_vm["-f"].as<string>();
    }

    proxy->statistic->store_stat_to_file(fname);

}


/**
 * The Wukong's console is co-located with the main proxy (the 1st proxy thread on the 1st server)
 * and provide a simple interactive cmdline to tester
 */
void run_console(Proxy *proxy)
{
    // init the option desc, 
    // only need to init by the master proxy
    if (IS_MASTER(proxy))
        init_options_desc();

    console_barrier(proxy->tid);
    if (IS_MASTER(proxy))
        cout << endl
            << "Input \'help\' command to get more information"
            << endl
            << endl;

    bool once = true;
    while (true) {
        console_barrier(proxy->tid);
next:
        string cmd;
        if (IS_MASTER(proxy)) {
            if (enable_oneshot) {
                // one-shot command mode: run the command once
                if (once) {
                    logstream(LOG_INFO) << "Run one-shot command: " << oneshot_cmd << LOG_endl;
                    cmd = oneshot_cmd;

                    once = false;
                } else {
                    logstream(LOG_INFO) << "Done" << LOG_endl;
                    cmd = "quit";
                }
            } else {
                // interactive mode: print a prompt and retrieve the command
                cout << "wukong> ";
                getline(cin, cmd);
            }

            // trim input
            size_t pos = cmd.find_first_not_of(" \t"); // trim blanks from head
            if (pos == string::npos) goto next;
            cmd.erase(0, pos);

            pos = cmd.find_last_not_of(" \t");  // trim blanks from tail
            cmd.erase(pos + 1, cmd.length() - (pos + 1));


            // send <cmd> to all consoles
            for (int i = 0; i < global_num_servers; i++) {
                for (int j = 0; j < global_num_proxies; j++) {
                    if (i == 0 && j == 0) continue ;
                    console_send<string>(i, j, cmd);
                }
            }
        } else {
            // recieve <cmd>
            cmd = console_recv<string>(proxy->tid);
        }

        // transform the string to argv and argc format
        int argc = 0;
        char** argv = str2command_args(cmd, argc);
        string cmd_type = argv[0];

        // get keyword of command and run with different way

        // only run <cmd> on the master console
        if (cmd_type == "help") {
            if( IS_MASTER(proxy)) {
                print_help();
            }
            continue;
        }

        if (cmd_type == "quit" || cmd_type == "q") {
            if (proxy->tid == 0)
                exit(0); // each server exits once by the 1st console
        }

        if (cmd_type == "config") {
            run_config(proxy, argc, argv);
            continue;
        }

        if (cmd_type == "sparql") { // handle SPARQL queries
            run_sparql_cmd(proxy, argc, argv);
            continue;
        }

        if (cmd_type == "sparql-emu") { // run a SPARQL emulator on each proxy
            run_sparql_emu(proxy, argc, argv);
            continue;
        }

        if (cmd_type == "load") {
            run_load(proxy, argc, argv);
            continue;
        }

        if (cmd_type == "gsck") {
            run_gsck(proxy, argc, argv);
            continue;
        }

        if (cmd_type == "load-stat") {
            run_load_stat(proxy, argc, argv);
            continue;
        }

        if (cmd_type == "store-stat") {
            run_store_stat(proxy, argc, argv);
            continue;
        }

        // fail to parse the command
        fail_to_parse(proxy, argc, argv); 
    }
}
