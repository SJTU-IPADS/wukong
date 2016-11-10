#pragma once
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

using namespace std;
using namespace boost::archive;

struct CS_Request {
	string type;
	bool use_file;
	string content;

	string identity;

	template <typename Archive>
	void serialize(Archive &ar, const unsigned int version) {
		ar &type;
		ar &use_file;
		ar &content;
	}
};

struct CS_Reply {
	string type;
	string content;
	int column;
	vector<int64_t> column_table;
	vector<int64_t> result_table;

	template <typename Archive>
	void serialize(Archive &ar, const unsigned int version) {
		ar &type;
		ar &content;
		ar &column;
		ar &column_table;
		ar &result_table;
	}
};