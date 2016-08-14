#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <algorithm>
#include <assert.h>
#include <boost/mpi.hpp>
#include <boost/unordered_map.hpp>

#include "global_cfg.h"


using namespace std;

// reserved ID
enum { ID_PREDICATE = 0, ID_TYPE = 1 };

class string_server {
public:
	boost::unordered_map<string, int64_t> str2id;
	boost::unordered_map<int64_t, string> id2str;

	string_server(string dname);
private:
	void load_mapping(string fname);
};
