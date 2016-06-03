#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <assert.h>

#include <sys/stat.h>
#include <sys/types.h>

/**
 * A simple manual
 *  $g++ -std=c++11 generate_data.cpp -o generate_data
 *  $./generate_data lubm_raw_40 id_lubm_40
 */

/**
 * How to generate minimal_index file
 *  $grep "<http://www.University0.edu>" str_normal >> str_normal_minimal
 *  $grep "<http://www.Department0.University0.edu>" str_normal >> str_normal_minimal
 *  ...
 */

using namespace std;

enum { NBTITS_IDX = 17 };

int
main(int argc, char** argv)
{
    unordered_map<string, int> str_to_id;
    vector<string> normal_str;
    vector<string> index_str;

    if (argc != 3) {
        printf("usage: ./generate_data src_dir dst_dir\n");
        return -1;
    }

    if (mkdir(argv[2], S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) < 0) {
        cout << "Error: Creating dst_dir (" << argv[2] << ") failed." << endl;
        exit(-1);
    }

    DIR *src = opendir(argv[1]);
    if (!src) {
        cout << "Error: Opening src_dir (" << argv[1] << ") failed." << endl;
        exit(-1);
    }

    str_to_id["__PREDICT__"] = 0;
    index_str.push_back("__PREDICT__");

    str_to_id["<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"] = 1;
    index_str.push_back("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>");

    // reserve the first two ids for the class of index vertex (i.e, predicate or type)
    size_t next_index_id = 2;
    size_t next_normal_id = 1 << NBTITS_IDX; // reserve 2^NBTITS_IDX ids for index vertices
    int count = 0;

    struct dirent *dent;
    while ((dent = readdir(src)) != NULL) {
        if (dent->d_name[0] == '.')
            continue;

        ifstream file((string(argv[1]) + "/" + string(dent->d_name)).c_str());
        ofstream output((string(argv[2]) + "/id_" + string(dent->d_name)).c_str());
        cout << "Process No." << ++count << " file: " << dent->d_name << "." << endl;

        string subject, predict, object, dot;
        while (file >> subject >> predict >> object >> dot) {
            /* add a new normal vertex */
            if (str_to_id.find(subject) == str_to_id.end()) {
                str_to_id[subject] = next_normal_id;
                next_normal_id ++;
                normal_str.push_back(subject);
            }

            /* add a new predicate index vertex (i.e., p-idx) */
            if (str_to_id.find(predict) == str_to_id.end()) {
                str_to_id[predict] = next_index_id;
                next_index_id ++;
                index_str.push_back(predict);
            }

            /* treat different types as individual indexes */
            if (predict == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>") {
                /* add a new type index vertex (i.e., t-idx) */
                if (str_to_id.find(object) == str_to_id.end()) {
                    str_to_id[object] = next_index_id;
                    next_index_id ++;
                    index_str.push_back(object);
                }
            } else {
                /* add a new normal vertex */
                if (str_to_id.find(object) == str_to_id.end()) {
                    str_to_id[object] = next_normal_id;
                    next_normal_id ++;
                    normal_str.push_back(object);
                }
            }

            int id[3];
            id[0] = str_to_id[subject];
            id[1] = str_to_id[predict];
            id[2] = str_to_id[object];
            output << id[0] << "\t" << id[1] << "\t" << id[2] << endl;
        }
    }
    closedir(src);

    /* generate a file of the mapping between name and id for nomrmal vertices */
    {
        ofstream f_normal((string(argv[2]) + "/str_normal").c_str());
        for (int i = 0; i < normal_str.size(); i++)
            f_normal << normal_str[i] << "\t" << str_to_id[normal_str[i]] << endl;
    }

    /* generate a file of the mapping between name and id for index vertices */
    {
        ofstream f_index((string(argv[2]) + "/str_index").c_str());
        for (int i = 0; i < index_str.size(); i++)
            f_index << index_str[i] << "\t" << str_to_id[index_str[i]] << endl;
    }

    cout << "#total_vertex = " << str_to_id.size() << endl;
    cout << "#normal_vertex = " << normal_str.size() << endl;
    cout << "#index_vertex = " << index_str.size() << endl;

    return 0;
}
