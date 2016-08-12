#include "string_server.h"

void
string_server::load_mapping(string fname) {
    ifstream file(fname.c_str());
    string str;
    int id;

    while (file >> str >> id) {
        // both string and ID are unique
        assert(str2id.find(str) == str2id.end());
        assert(id2str.find(id) == id2str.end());

        str2id[str] = id;
        id2str[id] = str;
    }

    file.close();
}

string_server::string_server(string dname)
{
    global_rdftype_id = 1;

    DIR *dir = opendir(dname.c_str());
    if (dir == NULL) {
        cout << "ERROR: failed to open the directory of RDF data ("
             << dname << ")." << endl;
        exit(-1);
    }

    struct dirent *ptr;
    while ((ptr = readdir(dir)) != NULL) {
        if (ptr->d_name[0] == '.')
            continue;

        // load ID mapping table
        string fname(ptr->d_name);
        if ((fname == "str_index")
                || (fname == "str_normal" && !global_load_minimal_index)
                || (fname == "str_normal_minimal" && global_load_minimal_index)) {
            // NOTE: create str_normal_minimal file according to the queries of benchmark
            load_mapping(dname + "/" + fname); // normal vertex
        }
    }
}