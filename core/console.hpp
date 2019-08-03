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

#include "global.hpp"
#include "config.hpp"
#include "errors.hpp"
#include "proxy.hpp"
#include "monitor.hpp"

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
    string str = con_adaptor->recv(tid);
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

    // inter-server barrier (by the leader proxy)
    if (tid == 0)
        MPI_Barrier(MPI_COMM_WORLD);

    // intra-server barrier
    __sync_fetch_and_add(&_curr, 1);
    while (_curr < _next)
        usleep(1); // wait
    _next += Global::num_proxies; // next barrier
}

// the master proxy is the 1st proxy of the 1st server (i.e., sid == 0 and tid == 0)
#define MASTER(_p) ((_p)->sid == 0 && (_p)->tid == 0)
// the leader proxy is the 1st proxy on each server (i.e., tid == 0)
#define LEADER(_p) ((_p)->tid == 0)

#define PRINT_ID(_p) "[" << (_p)->sid << "|" << (_p)->tid << "]"


// options
options_description        all_desc("These are common Wukong commands: ");
options_description       help_desc("help                display help infomation");
options_description       quit_desc("quit                quit from the console");
options_description     config_desc("config <args>       run commands for configueration");
options_description     logger_desc("logger <args>       run commands for logger");
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
    (",s", value<string>()->value_name("<string>"), "set config items by <str> (e.g., item1=val1&item2=...)")
    ("help,h", "help message about config")
    ;
    all_desc.add(config_desc);

    // e.g., wukong> logger <args>
    logger_desc.add_options()
    (",v", "print loglevel")
    (",s", value<int>()->value_name("<level>"), "set loglevel to <level> (e.g., DEBUG=1, INFO=2, WARNING=4, ERROR=5)")
    ("help,h", "help message about logger")
    ;
    all_desc.add(logger_desc);

    // e.g., wukong> sparql <args>
    sparql_desc.add_options()
    (",f", value<string>()->value_name("<fname>"), "run a [single] SPARQL query from <fname>")
    (",m", value<int>()->default_value(1)->value_name("<factor>"), "set multi-threading <factor> for heavy query processing")
    (",n", value<int>()->default_value(1)->value_name("<num>"), "repeat query processing <num> times")
    (",p", value<string>()->value_name("<fname>"), "adopt user-defined query plan from <fname> for running a single query")
    (",N", value<int>()->default_value(1)->value_name("<num>"), "do query optimization <num> times")
    (",v", value<int>()->default_value(0)->value_name("<lines>"), "print at most <lines> of results")
    (",o", value<string>()->value_name("<fname>"), "output results into <fname>")
    (",g", "leverage GPU to accelerate heavy query processing ")
    (",b", value<string>()->value_name("<fname>"), "run a [batch] of SPARQL queries configured by <fname>")
    ("help,h", "help message about sparql")
    ;
    all_desc.add(sparql_desc);

    // e.g., wukong> sparql-emu <args>
    sparql_emu_desc.add_options()
    (",f", value<string>()->value_name("<fname>"), "run queries generated from temples configured by <fname>")
    (",p", value<string>()->value_name("<fname>"), "adopt user-defined query plans from <fname> for running queries")
    (",d", value<int>()->default_value(10)->value_name("<sec>"), "eval <sec> seconds (default: 10)")
    (",w", value<int>()->default_value(5)->value_name("<sec>"), "warmup <sec> seconds (default: 5)")
    (",n", value<int>()->default_value(20)->value_name("<num>"), "keep <num> queries being processed (default: 20)")
    ("help,h", "help message about sparql-emu")
    ;
    all_desc.add(sparql_emu_desc);

    // e.g., wukong> load <args>
    load_desc.add_options()
    (",d", value<string>()->value_name("<dname>"), "load data from directory <dname>")
    (",c", "check and skip duplicate RDF triples")
    ("help,h", "help message about load")
    ;
    all_desc.add(load_desc);

    // e.g., wukong> gsck <args>
    gsck_desc.add_options()
    (",i", "check from index key/value pair to normal key/value pair")
    (",n", "check from normal key/value pair to index key/value pair")
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
    string cmd;
    for (int i = 0; i < argc; i++)
        cmd = cmd + argv[i] + " ";

    logstream(LOG_ERROR) << PRINT_ID(proxy)
                         << " Failed to run the command: " << cmd << LOG_endl;
    cout << endl
         << "Input \'help\' command to get more information." << endl;
}

/**
 * split the string into char** by the space
 * the argc is the name of char*
 * the return value argv is the type of char**
 */
static char **cmd2args(string str, int &argc)
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
static void print_help(void)
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
    // use the leader proxy thread (0) on each server to configure system
    if (!LEADER(proxy))
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
        if (MASTER(proxy))
            cout << config_desc;
        return;
    }

    if (config_vm.count("-v")) {
        if (MASTER(proxy))
            print_config();
        return;
    }

    // exclusive
    if (!(config_vm.count("-l") ^ config_vm.count("-s"))) {
        fail_to_parse(proxy, argc, argv); // invalid cmd
        return;
    }

    string fname, str;
    if (MASTER(proxy)) {
        if (config_vm.count("-l")) {
            fname = config_vm["-l"].as<string>();
            file2str(fname, str);
        }

        if (config_vm.count("-s")) {
            str = config_vm["-s"].as<string>();
            args2str(str);
        }

        // send <str> to all consoles
        for (int i = 1; i < Global::num_servers; i++)
            console_send<string>(i, 0, str);
    } else {
        // recieve <str>
        str = console_recv<string>(proxy->tid);
    }

    /// do config
    if (!str.empty()) {
        reload_config(str);
    } else {
        if (MASTER(proxy))
            logstream(LOG_ERROR) << "Failed to load config file: " << fname << LOG_endl;
    }
}

/**
 * run the 'logger' command
 * usage:
 * logger [options]
 *   -v          print current log level
 *   -s <level>  set log level to <level> (i.e., EVERYTHING=0, DEBUG=1, INFO=2, EMPH=3,
 *                                               WARNING=4, ERROR=5, FATAL=6, NONE=7)
 */
static void run_logger(Proxy *proxy, int argc, char **argv)
{
    // use the leader proxy thread (0) on each server to configure logger
    if (!LEADER(proxy))
        return;

    // parse command
    variables_map logger_vm;
    try {
        store(parse_command_line(argc, argv, logger_desc), logger_vm);
    } catch (...) {
        fail_to_parse(proxy, argc, argv);
        return;
    }
    notify(logger_vm);

    // parse options
    if (logger_vm.count("help")) {
        if (MASTER(proxy))
            cout << logger_desc;
        return;
    }

    if (logger_vm.count("-v")) {
        if (MASTER(proxy)) {
            int level = global_logger().get_log_level();
            cout << "loglevel: " << level
                 << " (" << levelname[level] << ")" << endl;
        }
        return;
    }

    if (logger_vm.count("-s")) {
        int level = global_logger().get_log_level();  // orginal loglevel
        if (MASTER(proxy)) {
            level = logger_vm["-s"].as<int>();

            // send <str> to all consoles
            for (int i = 1; i < Global::num_servers; i++)
                console_send<int>(i, proxy->tid, level);
        } else {
            // recieve <str>
            level = console_recv<int>(proxy->tid);
        }

        if (level >= LOG_EVERYTHING && level <= LOG_NONE) {
            global_logger().set_log_level(level);
            if (MASTER(proxy))
                cout << "set loglevel to " << level
                     << " (" << levelname[level] << ")" << endl;
        } else {
            if (MASTER(proxy))
                logstream(LOG_ERROR) << "failed to set loglevel: " << level
                                     << " (" << levelname[level] << ")" << endl;
        }
        return;
    }
}

/**
 * run the 'sparql' command
 * usage:
 * sparql -f <fname> [options]
 *   -m <factor>  set multi-threading factor <factor> for heavy query processing
 *   -n <num>     run <num> times
 *   -p <fname>   adopt user-defined query plan from <fname> for running a single query
 *   -N <num>     do query optimization <num> times
 *   -v <lines>   print at most <lines> of results
 *   -o <fname>   output results into <fname>
 *   -g           leverage GPU to accelerate heavy query processing
 *
 * sparql -b <fname>
 */
static void run_sparql(Proxy * proxy, int argc, char **argv)
{
    // use the master proxy thread to run SPARQL queries in single mode or batch mode
    if (!MASTER(proxy))
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
        if (MASTER(proxy))
            cout << sparql_desc;
        return;
    }

    // single mode (-f) and batch mode (-b) are exclusive
    if (!(sparql_vm.count("-f") ^ sparql_vm.count("-b"))) {
        logstream(LOG_ERROR) << "single mode (-f) and batch mode (-b) are exclusive!" << LOG_endl;
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

        // NOTE: the options with default_value are always available.
        //       default value: mfactor(1), cnt(1), nopts(1), nlines(0)

        // option: -m <factor>, -n <num>
        int mfactor = sparql_vm["-m"].as<int>(); // the number of multithreading
        int cnt = sparql_vm["-n"].as<int>();     // the number of executions

        // option: -p <fname>, -N <num>
        if (!(sparql_vm.count("-p") ^ Global::enable_planner)) {
            if (Global::enable_planner) {
                logstream(LOG_WARNING) << "Optimizer is enabled, "
                                       << "your plan file (-p) will be ignored." << LOG_endl;
            } else {
                logstream(LOG_ERROR) << "Optimizer is disabled, "
                                     << "you must provide a user-defined plan file (-p)." << LOG_endl;
                fail_to_parse(proxy, argc, argv); // invalid cmd
                return;
            }
        }

        ifstream fmt_stream;
        if (Global::enable_planner) {
            fmt_stream.setstate(std::ios::failbit);
        } else {
            string fmt_fname = sparql_vm["-p"].as<string>();
            fmt_stream.open(fmt_fname);
            if (!fmt_stream.good()) { // fail to load user-defined plan file
                logstream(LOG_ERROR) << "The plan file is not found: "
                                     << fmt_fname << LOG_endl;
                fail_to_parse(proxy, argc, argv); // invalid cmd
                return;
            }
        }

        int nopts = sparql_vm["-N"].as<int>();

        // option: -v <lines>, -o <fname>
        int nlines = sparql_vm["-v"].as<int>();  // the number of result lines

        string ofname = "";
        if (sparql_vm.count("-o"))
            ofname = sparql_vm["-o"].as<string>();

        if (Global::silent) { // not retrieve the query results
            if (nlines > 0 || sparql_vm.count("-o")) {
                logstream(LOG_ERROR) << "Can't print/output results (-v/-o) with global_silent."
                                     << LOG_endl;
                fail_to_parse(proxy, argc, argv); // invalid cmd
                return;
            }
        }

        // option: -g
        bool snd2gpu = sparql_vm.count("-g");
#ifndef USE_GPU
        if (snd2gpu) {
            logstream(LOG_WARNING) << "Please build Wukong with GPU support "
                                   << "to turn on the \"-g\" option." << LOG_endl;
            return;
        }
#endif

        /// do sparql
        SPARQLQuery reply;
        Monitor monitor;
        try {
            proxy->run_single_query(ifs, fmt_stream, nopts, mfactor, snd2gpu,
                                    cnt, nlines, ofname, reply, monitor);
        } catch (WukongException &ex) {
            logstream(LOG_ERROR) << "Query failed [ERRNO " << ex.code()
                                 << "]: " << ex.what() << LOG_endl;
            fail_to_parse(proxy, argc, argv);  // invalid cmd
            return;
        }
        monitor.print_latency(cnt);
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
        logstream(LOG_INFO) << "Batch-mode start ..." << LOG_endl;

        string sg_cmd; // a single command line
        while (getline(ifs, sg_cmd)) {
            int sg_argc = 0;
            char** sg_argv = cmd2args(sg_cmd, sg_argc);

            // only support single sparql query now (e.g., sparql -f ...)
            if (sg_argc > 2 && (string(sg_argv[0]) == "sparql" && string(sg_argv[1]) == "-f")) {
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
static void run_sparql_emu(Proxy * proxy, int argc, char **argv)
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
        if (MASTER(proxy))
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

    // option: -p <fname>
    if (!(sparql_emu_vm.count("-p") ^ Global::enable_planner)) {
        if (Global::enable_planner) {
            logstream(LOG_WARNING) << "Optimizer is enabled, "
                                   << "your config file of plans (-p) will be ignored." << LOG_endl;
        } else {
            logstream(LOG_ERROR) << "Optimizer is disabled, "
                                 << "you must provide a user-defined plan file (-p)." << LOG_endl;
            fail_to_parse(proxy, argc, argv); // invalid cmd
            return;
        }
    }

    ifstream fmt_stream;
    if (Global::enable_planner) {
        fmt_stream.setstate(std::ios::failbit);
    } else {
        string fmt_fname = sparql_emu_vm["-p"].as<string>();
        fmt_stream.open(fmt_fname);
        if (!fmt_stream.good()) { // fail to load user-defined plan file
            logstream(LOG_ERROR) << "The plan file is not found: "
                                 << fmt_fname << LOG_endl;
            fail_to_parse(proxy, argc, argv); // invalid cmd
            return;
        }
    }

    // config file for the SPARQL emulator
    ifstream ifs(fname);
    if (!ifs.good()) {
        logstream(LOG_ERROR) << "Configure file not found: " << fname << LOG_endl;
        fail_to_parse(proxy, argc, argv);
        return;
    }

    // NOTE: the option with default_value is always available
    // default value: duration(10), warmup(5), otf(20)
    int duration = sparql_emu_vm["-d"].as<int>();
    int warmup = sparql_emu_vm["-w"].as<int>();
    int otf = sparql_emu_vm["-n"].as<int>(); // the number of queries being processed (on the fly)

    if (duration <= 0 || warmup < 0 || otf <= 0) {
        logstream(LOG_ERROR) << "invalid parameters for SPARQL emulator! "
                             << "(duration=" << duration << ", warmup=" << warmup
                             << ", on-the-fly=" << otf << ")" << LOG_endl;
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
    Monitor monitor;
    int ret = proxy->run_query_emu(ifs, fmt_stream, duration, warmup, otf, monitor);
    if (ret != 0) {
        logstream(LOG_ERROR) << "Failed to run the query emulator (ERRNO: " << ret << ")!" << LOG_endl;
        fail_to_parse(proxy, argc, argv); // invalid cmd
        return;
    }

    // FIXME: maybe hang in here if the input file misses in some machines
    //        or inconsistent global variables (e.g., global_enable_planner)
    console_barrier(proxy->tid);

    // aggregate and print performance statistics for running emulators on all servers
    if (MASTER(proxy)) {
        for (int i = 1; i < Global::num_servers * Global::num_proxies; i++) {
            Monitor other = console_recv<Monitor>(proxy->tid);
            monitor.merge(other);
        }
        monitor.aggregate();
        monitor.print_cdf();
        monitor.print_thpt();
    } else {
        // send logs to the master proxy
        console_send<Monitor>(0, 0, monitor);
    }
}

/**
 * run the 'load' command
 * usage:
 * load -d <dname> [options]
 *   -c    check duplication or not
 */
static void run_load(Proxy * proxy, int argc, char **argv)
{
    // use the master proxy thread to dyanmically load RDF data
    if (!MASTER(proxy))
        return;

#ifdef DYNAMIC_GSTORE
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
        if (MASTER(proxy))
            cout << load_desc;
        return;
    }

    string dname;
    if (!load_vm.count("-d")) {
        fail_to_parse(proxy, argc, argv);
        return;
    } else {
        dname = load_vm["-d"].as<string>();
    }

    bool c_enable = load_vm.count("-c");

    /// do load
    if (dname[dname.length() - 1] != '/')
        dname = dname + "/"; // force a "/" at the end of dname.

    Monitor monitor;
    RDFLoad reply;
    //FIXME: the dynamic_load_data will exit if the directory is not exist
    int ret = proxy->dynamic_load_data(dname, reply, monitor, c_enable);
    if (ret != 0) {
        logstream(LOG_ERROR) << "Failed to load dynamic data from directory " << dname
                             << " (ERRNO: " << ret << ")!" << LOG_endl;
        return;
    }
    monitor.print_latency();
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
 */
static void run_gsck(Proxy *proxy, int argc, char **argv)
{
    // use the master proxy thread to run gstore checker
    if (!MASTER(proxy))
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

    bool i_enable = gsck_vm.count("-i");
    bool n_enable = gsck_vm.count("-n");

    // no option means enable all
    if (!i_enable && !n_enable) {
        i_enable = true;
        n_enable = true;
    }

    /// do gsck
    Monitor monitor;
    GStoreCheck reply;
    int ret = proxy->gstore_check(reply, monitor, i_enable, n_enable);
    if (ret != 0) {
        logstream(LOG_ERROR) << "Data integrity failed in graph store"
                             << " (" << ret << ")" << LOG_endl;
        return;
    }
    monitor.print_latency();
}

/**
 * run the 'load-stat' command
 * usage:
 * load-stat [options]
 *   -f <fname>    load statistics from <fname> located at data folder
 */
static void run_load_stat(Proxy *proxy, int argc, char **argv)
{
    // use the leader proxy thread on each server to configure system
    if (!LEADER(proxy))
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
        if (MASTER(proxy))
            cout << load_stat_desc;
        return;
    }

    /// do load-stat
    string fname;
    if (!load_stat_vm.count("-f")) {
        // if fname is not given, try the dataset name by default
        fname = Global::input_folder + "/statfile";
    } else {
        fname = load_stat_vm["-f"].as<string>();
    }

    proxy->stats->load_stat_from_file(fname, con_adaptor);
}

/**
 * run the 'store-stat' command
 * usage:
 * load-stat [options]
 *   -f <fname>    store statistics to <fname> located at data folder
 */
static void run_store_stat(Proxy *proxy, int argc, char **argv)
{
    // use the master proxy thread to store statistics
    if (!MASTER(proxy))
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
        if (MASTER(proxy))
            cout << store_stat_desc;
        return;
    }

    /// do store-stat
    string fname;
    if (!store_stat_vm.count("-f")) {
        vector<string> strs;
        boost::split(strs, Global::input_folder, boost::is_any_of("/"));
        // if fname is not given, use the dataset name by default
        fname = strs[strs.size() - 2] + ".statfile";
    } else {
        fname = store_stat_vm["-f"].as<string>();
    }

    proxy->stats->store_stat_to_file(fname);
}

/**
 * The Wukong's console is co-located with the main proxy (the 1st proxy thread on the 1st server)
 * and provide a simple interactive cmdline to tester
 */
void run_console(Proxy *proxy)
{
    // init option descriptions once on each server (by the leader proxy)
    if (LEADER(proxy))
        init_options_desc();

    console_barrier(proxy->tid);
    if (MASTER(proxy))
        cout << endl
             << "Input \'help\' command to get more information"
             << endl
             << endl;

    bool once = true;
    while (true) {
        console_barrier(proxy->tid);
        string cmd;
        if (MASTER(proxy)) {
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
                // skip input with blank
                size_t pos;
                do {
                    cout << "wukong> ";
                    getline(cin, cmd);
                    pos = cmd.find_first_not_of(" \t");
                } while (pos == string::npos); // if all are blanks, do again
            }

            // trim input
            size_t pos = cmd.find_first_not_of(" \t"); // trim blanks from head
            cmd.erase(0, pos);
            pos = cmd.find_last_not_of(" \t");  // trim blanks from tail
            cmd.erase(pos + 1, cmd.length() - (pos + 1));


            // send <cmd> to all proxies
            for (int i = 0; i < Global::num_servers; i++) {
                for (int j = 0; j < Global::num_proxies; j++) {
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


        // run commmand on all proxies according to the keyword
        string cmd_type = argv[0];
        try {
            if (cmd_type == "help" || cmd_type == "h") {
                if (MASTER(proxy)) print_help();
            } else if (cmd_type == "quit" || cmd_type == "q") {
                if (LEADER(proxy))
                    exit(0);  // each server exits once (by the leader proxy)
            } else if (cmd_type == "config") {
                run_config(proxy, argc, argv);
            } else if (cmd_type == "logger") {
                run_logger(proxy, argc, argv);
            } else if (cmd_type == "sparql") {  // handle SPARQL queries
                run_sparql(proxy, argc, argv);
            } else if (cmd_type == "sparql-emu") {  // run a SPARQL emulator on each proxy
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
                // the same invalid command dispatch to all proxies, print error
                // msg once
                if (MASTER(proxy)) fail_to_parse(proxy, argc, argv);
            }
        } catch (WukongException &ex) {
            logstream(LOG_ERROR)
                    << "ERRNO " << ex.code() << ": " << ex.what() << LOG_endl;
            fail_to_parse(proxy, argc, argv);
        }
    }
}
