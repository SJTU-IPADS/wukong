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
static void console_send(int sid, int tid, T &r)
{
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


// options
options_description all_desc("These are common Wukong commands: ");
options_description       help_desc("help                display help infomation");
options_description       quit_desc("quit                quit from the console");
options_description     config_desc("config <args>       run commands for configueration");
options_description     sparql_desc("sparql <args>       run SPARQL queries in single or batch mode");
options_description sparql_emu_desc("sparql-emu <args>   emulate clients to continuously send SPARQL queries");
options_description       load_desc("load <args>         load RDF data into dynamic (in-memmory) graph store");
options_description       gsck_desc("gsck <args>         check the integrity of (in-memmory) graph storage");
options_description  load_stat_desc("load-stat           load statistics of SPARQL query optimizer");
options_description store_stat_desc("store-stat          store statistics of SPARQL query optimizer");


/*
 * init the options_description
 * it should add the option to different options_description
 * all should be added to all_desc
 */
void init_options_desc()
{
    // e.g., wukong> help
    all_desc.add(help_desc);

    // e.g., wukong> quit
    all_desc.add(quit_desc);

    // e.g., wukong> config <args>
    config_desc.add_options()
    (",v", "print current config")
    (",l", value<string>()->value_name("<fname>"), "load config items from <fname>")
    (",s", value<string>()->value_name("<string>"), "set config items by <str> (format: item1=val1&item2=...)")
    ("help,h", "help message about config")
    ;
    all_desc.add(config_desc);

    // e.g., wukong> sparql <args>
    sparql_desc.add_options()
    (",f", value<string>()->value_name("<fname>"), "run a single query from <fname>")
    (",m", value<int>()->default_value(1)->value_name("<factor>"), "set multi-threading <factor> for heavy query")
    (",n", value<int>()->default_value(1)->value_name("<num>"), "run <num> times")
    (",v", value<int>()->default_value(0)->value_name("<lines>"), "print at most <lines> of results")
    (",o", value<string>()->value_name("<fname>"), "output results into <fname>")
    ("batch,b", value<string>()->value_name("<fname>"), "run a batch of queries configured by <fname>")
    ("help,h", "help message about sparql")
    ;
    all_desc.add(sparql_desc);

    // e.g., wukong> sparql-emu <args>
    sparql_emu_desc.add_options()
    (",f", value<string>()->value_name("<fname>"), "run queries generated from temples configured by <fname>")
    (",d", value<int>()->default_value(10)->value_name("<sec>"), "eval <sec> seconds (default: 10)")
    (",w", value<int>()->default_value(5)->value_name("<sec>"), "warmup <sec> seconds (default: 5)")
    (",p", value<int>()->default_value(20)->value_name("<num>"), "send <num> queries in parallel (default: 20)")
    ("help,h", "help message about sparql-emu")
    ;
    all_desc.add(sparql_emu_desc);

    // e.g., wukong> load <args>
    load_desc.add_options()
    ("directory,d", value<string>()->value_name("<dname>"), "load data from directory <dname>")
    ("check,c",  "check and skip duplicate rdf triple")
    ("help,h", "help message about load")
    ;
    all_desc.add(load_desc);

    // e.g., wukong> gsck <args>
    gsck_desc.add_options()
    ("index,i", "check from index key/value pair to normal key/value pair")
    ("normal,n", "check from normal key/value pair to index key/value pair")
    ("all,a", "check all above")
    ("help,h", "help message about gsck")
    ;
    all_desc.add(gsck_desc);

    // e.g., wukong> load-stat
    load_stat_desc.add_options()
    (",f", value<string>()->value_name("<fname>"), "load statistics from <fname> located at data folder")
    ("help,h", "help message about load-stat")
    ;
    all_desc.add(load_stat_desc);

    // e.g., wukong> store-stat
    store_stat_desc.add_options()
    (",f", value<string>()->value_name("<fname>"), "store statistics to <fname> located at data folder")
    ("help,h", "help message about store-stat")
    ;
    all_desc.add(store_stat_desc);
}


/**
 * fail to parse the command
 */
static void fail_to_parse(Proxy *proxy, int argc, char** argv)
{
    if (IS_MASTER(proxy)) {
        string cmd;
        for (int i = 0; i < argc; i++)
            cmd = cmd + argv[i] + " ";

        logstream(LOG_ERROR) << "Failed to run the command: " << cmd << LOG_endl;
        cout << endl
             << "Input \'help\' command to get more information." << endl;
    }
}

/**
 * split the string into char** by the space
 * the argc is the name of char*
 * the return value argv is the type of char**
 */
static char ** cmd2args(string str, int &argc)
{
    vector<string> fields;

    split(fields, str, is_any_of(" "));

    argc = fields.size(); // set the argc as number of arguments
    char** argv = new char*[argc + 1];
    for (int i = 0; i < argc; i++) {
        argv[i] = new char[fields[i].length() + 1];
        strcpy(argv[i], fields[i].c_str());
    }
    argv[argc] = NULL;

    return argv;
}

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


/**
 * print help of all commands
 */
void print_help(void)
{
    cout << all_desc << endl;
}

/**
 * run the 'config' command
 * usage:
 * config [options]
 *   -v          print current config
 *   -l <fname>  load config items from <file>
 *   -s <str>    set config items by <str> (format: item1=val1&item2=...)
 */
static void run_config(Proxy *proxy, int argc, char **argv)
{
    // use the main proxy thread on each server to configure system
    if (proxy->tid != 0)
        return;

    // parse command
    variables_map config_vm;
    try {
        store(parse_command_line(argc, argv, config_desc), config_vm);
    } catch (...) {
        fail_to_parse(proxy, argc, argv);
        return;
    }
    notify(config_vm);

    // parse options
    if (config_vm.count("help")) {
        if (IS_MASTER(proxy))
            cout << config_desc;
        return;
    }

    if (config_vm.count("-v")) {
        if (IS_MASTER(proxy))
            print_config();
        return;
    }

    // exclusive
    if (!(config_vm.count("-l") ^ config_vm.count("-s"))) {
        fail_to_parse(proxy, argc, argv); // invalid cmd
        return;
    }

    string fname, str;
    if (IS_MASTER(proxy)) {
        if (config_vm.count("-l")) {
            fname = config_vm["-l"].as<string>();
            file2str(fname, str);
        }

        if (config_vm.count("-s")) {
            str = config_vm["-s"].as<string>();
            args2str(str);
        }

        // send <str> to all consoles
        for (int i = 1; i < global_num_servers; i++)
            console_send<string>(i, 0, str);
    } else {
        // recieve <str>
        str = console_recv<string>(proxy->tid);
    }

    /// do config
    if (!str.empty()) {
        reload_config(str);
    } else {
        if (IS_MASTER(proxy))
            logstream(LOG_ERROR) << "Failed to load config file: " << fname << LOG_endl;
    }
}


/**
 * run the 'sparql' command
 * usage:
 * sparql -f <fname> [options]
 *   -m <factor>  set multi-threading factor <factor> for heavy queries
 *   -n <num>     run <num> times
 *   -v <lines>   print at most <lines> of results
 *   -o <fname>   output results into <fname>
 *
 * sparql -b <fname>
 */
void run_sparql(Proxy * proxy, int argc, char **argv)
{
    // use the master proxy thread to run SPARQL queries in single mode or batch mode
    if (!IS_MASTER(proxy))
        return;

    // parse command
    variables_map sparql_vm;
    try {
        store(parse_command_line(argc, argv, sparql_desc), sparql_vm);
    } catch (...) { // something go wrong
        fail_to_parse(proxy, argc, argv);
        return;
    }
    notify(sparql_vm);

    // parse options
    if (sparql_vm.count("help")) {
        if (IS_MASTER(proxy))
            cout << sparql_desc;
        return;
    }

    // exclusive
    if (!(sparql_vm.count("-f") ^ sparql_vm.count("-b"))) {
        fail_to_parse(proxy, argc, argv); // invalid cmd
        return;
    }

    /// [single mode]
    if (sparql_vm.count("-f")) {
        string fname = sparql_vm["-f"].as<string>();
        ifstream ifs(fname);
        if (!ifs.good()) {
            logstream(LOG_ERROR) << "Query file not found: " << fname << LOG_endl;
            fail_to_parse(proxy, argc, argv); // invalid cmd
            return;
        }

        int cnt = 1, nlines = 0, mt_factor = 1;
        string ofname;

        if (sparql_vm.count("-m"))
            mt_factor = sparql_vm["-m"].as<int>();
        if (sparql_vm.count("-n"))
            cnt = sparql_vm["-n"].as<int>();
        if (sparql_vm.count("-v"))
            nlines = sparql_vm["-v"].as<int>();
        if (sparql_vm.count("-o"))
            ofname = sparql_vm["-o"].as<string>();

        if (global_silent) { // not retrieve the query results
            if (nlines > 0 || sparql_vm.count("-o")) {
                logstream(LOG_ERROR) << "Can't print/output results (-v/-o) with global_silent."
                                     << LOG_endl;
                fail_to_parse(proxy, argc, argv); // invalid cmd
                return;
            }
        }

        /// do sparql
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
        if (!global_silent && !result.blind) {
            if (nlines > 0)
                result.print_result(min(nlines, result.get_row_num()), proxy->str_server);

            if (sparql_vm.count("-o"))
                result.dump_result(ofname, result.get_row_num(), proxy->str_server);
        }
    }

    /// [batch mode]
    if (sparql_vm.count("-b")) {
        string fname = sparql_vm["-b"].as<string>();
        ifstream ifs(fname);
        if (!ifs.good()) {
            logstream(LOG_ERROR) << "Query file not found: " << fname << LOG_endl;
            fail_to_parse(proxy, argc, argv); // invalid cmd
            return;
        }

        /// do sparql
        logstream(LOG_INFO) << "Batch-mode starting ..." << LOG_endl;

        string sg_cmd; // a single command line
        while (getline(ifs, sg_cmd)) {
            int sg_argc = 0;
            char** sg_argv = cmd2args(sg_cmd, sg_argc);

            string tk1, tk2;
            tk1 = string(sg_argv[0]);
            tk2 = string(sg_argv[1]);
            if (tk1 == "sparql" && tk2 == "-f") {
                cout << "Run the command: " << sg_cmd << endl;
                run_sparql(proxy, sg_argc, sg_argv);
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

/**
 * run the 'sparql-emu' command
 * usage:
 * sparql-emu -f <fname> [options]
 *   -d <sec>   eval <sec> seconds (default: 10)
 *   -w <sec>   warmup <sec> seconds (default: 5)
 *   -p <num>   send <num> queries in parallel (default: 20)
 */
void run_sparql_emu(Proxy * proxy, int argc, char **argv)
{
    // use all proxy threads to run SPARQL emulators

    // parse command
    variables_map sparql_emu_vm;
    try {
        store(parse_command_line(argc, argv, sparql_emu_desc), sparql_emu_vm);
    } catch (...) { // something go wrong
        fail_to_parse(proxy, argc, argv);
        return;
    }
    notify(sparql_emu_vm);

    // parse options
    if (sparql_emu_vm.count("help")) {
        if (IS_MASTER(proxy))
            cout << sparql_emu_desc;
        return;
    }

    string fname;
    if (!sparql_emu_vm.count("-f")) {
        fail_to_parse(proxy, argc, argv); // invalid cmd
        return;
    } else {
        fname = sparql_emu_vm["-f"].as<string>();
    }

    // config file for the SPARQL emulator
    ifstream ifs(fname);
    if (!ifs.good()) {
        logstream(LOG_ERROR) << "Configure file not found: " << fname << LOG_endl;
        fail_to_parse(proxy, argc, argv);
        return;
    }

    int duration = 10, warmup = 5, parallel_factor = 20;
    if (!sparql_emu_vm.count("-d"))
        duration = sparql_emu_vm["-d"].as<int>();

    if (!sparql_emu_vm.count("-w"))
        warmup = sparql_emu_vm["-w"].as<int>();

    if (!sparql_emu_vm.count("-p"))
        parallel_factor = sparql_emu_vm["-p"].as<int>();

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


    /// do sparql-emu
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

/**
 * run the 'load' command
 * usage:
 * load -d <dname> [options]
 *   -c    check duplication or not
 */
void run_load(Proxy * proxy, int argc, char **argv)
{
    // use the master proxy thread to dyanmically load RDF data
    if (!IS_MASTER(proxy))
        return;

#if DYNAMIC_GSTORE
    // parse command
    variables_map load_vm;
    try {
        store(parse_command_line(argc, argv, load_desc), load_vm);
    } catch (...) {
        fail_to_parse(proxy, argc, argv);
        return;
    }
    notify(load_vm);

    // parse options
    if (load_vm.count("help")) {
        if (IS_MASTER(proxy))
            cout << load_desc;
        return;
    }

    string dname;
    if (!load_vm.count("-d")) {
        fail_to_parse(proxy, argc, argv);
        return ; // invalid cmd
    } else {
        dname = load_vm["-d"].as<string>();
    }

    bool c_enable = false;
    if (load_vm.count("-c"))
        c_enable = true;


    /// do load
    if (dname[dname.length() - 1] != '/')
        dname = dname + "/"; // force a "/" at the end of dname.

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
    logstream(LOG_ERROR) << "Can't load data into static graph store." << LOG_endl;
    logstream(LOG_ERROR) << "You can enable it by building Wukong with -DUSE_DYNAMIC_GSTORE=ON." << LOG_endl;
#endif
}

/**
 * run the 'gsck' command
 * usage:
 * gsck [options]
 *   -i   check from index key/value pair to normal key/value pair
 *   -n   check from normal key/value pair to index key/value pair
 *   -a   check all above
 */
void run_gsck(Proxy * proxy, int argc, char **argv)
{
    // use the master proxy thread to run gstore checker
    if (!IS_MASTER(proxy))
        return;

    // parse command
    variables_map gsck_vm;
    try {
        store(parse_command_line(argc, argv, gsck_desc), gsck_vm);
    } catch (...) {
        fail_to_parse(proxy, argc, argv);
        return;
    }
    notify(gsck_vm);

    // parse options
    if (gsck_vm.count("help")) {
        cout << gsck_desc;
        return;
    }

    bool i_enable = false;
    if (gsck_vm.count("-i"))
        i_enable = true;

    bool n_enable = false;
    if (gsck_vm.count("-n"))
        n_enable = true;

    if (gsck_vm.count("-a")) {
        i_enable = true;
        n_enable = true;
    }

    /// FIXME: how to deal with command without any options?

    /// do gsck
    Logger logger;
    GStoreCheck reply;
    int ret = proxy->gstore_check(reply, logger, i_enable, n_enable);
    if (ret != 0) {
        logstream(LOG_ERROR) << "Some error found in gstore!"
                             << " (ERRNO: " << ret << ")" << LOG_endl;
        return;
    }
    logger.print_latency();
}

/**
 * run the 'load-stat' command
 * usage:
 * load-stat [options]
 *   -f <fname>    load statistics from <fname> located at data folder
 */
void run_load_stat(Proxy * proxy, int argc, char **argv)
{
    // use the main proxy thread on each server to configure system
    if (proxy->tid != 0)
        return;

    // parse command
    variables_map load_stat_vm;
    try {
        store(parse_command_line(argc, argv, load_stat_desc), load_stat_vm);
    } catch (...) {
        fail_to_parse(proxy, argc, argv);
        return;
    }
    notify(load_stat_vm);

    // parse options
    if (load_stat_vm.count("help")) {
        if (IS_MASTER(proxy))
            cout << load_stat_desc;
        return;
    }

    /// do load-stat
    string fname;
    if (!load_stat_vm.count("-f")) {
        vector<string> strs;
        boost::split(strs, global_input_folder, boost::is_any_of("/"));
        // if fname is not given, try the dataset name by default
        fname = strs[strs.size() - 2] + ".statfile";
    } else {
        fname = load_stat_vm["-f"].as<string>();
    }

    proxy->statistic->load_stat_from_file(fname);
}

/**
 * run the 'store-stat' command
 * usage:
 * load-stat [options]
 *   -f <fname>    store statistics to <fname> located at data folder
 */
void run_store_stat(Proxy * proxy, int argc, char **argv)
{
    // use the master proxy thread to store statistics
    if (!IS_MASTER(proxy))
        return;

    // parse command
    variables_map store_stat_vm;
    try {
        store(parse_command_line(argc, argv, store_stat_desc), store_stat_vm);
    } catch (...) {
        fail_to_parse(proxy, argc, argv);
        return;
    }
    notify(store_stat_vm);

    // parse options
    if (store_stat_vm.count("help")) {
        if (IS_MASTER(proxy))
            cout << store_stat_desc;
        return;
    }

    /// do store-stat
    string fname;
    if (!store_stat_vm.count("-f")) {
        vector<string> strs;
        boost::split(strs, global_input_folder, boost::is_any_of("/"));
        // if fname is not given, use the dataset name by default
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
void run_console(Proxy * proxy)
{
    // init option descriptions
    if (IS_MASTER(proxy))   // only master proxy
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
                // one-shot command mode: run the command once and then quit
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

        // transform the comnmand to (argc, argv)
        int argc = 0;
        char **argv = cmd2args(cmd, argc);


        // run commmand on all consoles according to the keyword
        string cmd_type = argv[0];
        if (cmd_type == "help" || cmd_type == "h") {
            if (IS_MASTER(proxy)) // FIXME: no need to run help on all consoles
                print_help();
        } else if (cmd_type == "quit" || cmd_type == "q") {
            if (proxy->tid == 0)
                exit(0); // each server exits once by the 1st console
        } else if (cmd_type == "config") {
            run_config(proxy, argc, argv);
        } else if (cmd_type == "sparql") { // handle SPARQL queries
            run_sparql(proxy, argc, argv);
        } else if (cmd_type == "sparql-emu") { // run a SPARQL emulator on each proxy
            run_sparql_emu(proxy, argc, argv);
        } else if (cmd_type == "load") {
            run_load(proxy, argc, argv);
        } else if (cmd_type == "gsck") {
            run_gsck(proxy, argc, argv);
        } else if (cmd_type == "load-stat") {
            run_load_stat(proxy, argc, argv);
        } else if (cmd_type == "store-stat") {
            run_store_stat(proxy, argc, argv);
        } else {
            fail_to_parse(proxy, argc, argv);
        }
    }
}
