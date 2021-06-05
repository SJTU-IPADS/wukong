/*
 * Author : LIN SHI (FDU)
 * This cpp file is used to add a timestamp for each triple.
 */
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <cstdlib>
#include <sys/types.h>
#include <dirent.h>
using namespace std;

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: ./add_timestamp id_triples_directory_name\n");
        return -1;
    }

    DIR *dir = opendir(argv[1]);
    if (dir == NULL) {
        cout << "failed to open directory" << argv[1];
        return -1;
    }

    unsigned seed = time(0);
    srand(seed);

    int64_t subject, predicate, object;

    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        string fname = string(argv[1]) + "/" + ent->d_name;
        if (string(ent->d_name).find("id") != string::npos && string(ent->d_name).find("nt") != string::npos) {
            cout << "Processing: " << ent->d_name << endl;
            ifstream infile;
            infile.open(fname);

            ofstream outfile;
            string dst = string(argv[1]) + "/" + ent->d_name + "1";
            outfile.open(dst, ios::out | ios::trunc);
            while (infile >> subject >> predicate >> object) {
                outfile << subject << "\t" << predicate << "\t" << object << "\t" << rand() % seed + 1 << "\t" << rand() % seed + 1 << endl;
            }
            infile.close();
            outfile.close();
            remove(fname.c_str());
            rename(dst.c_str(),fname.c_str());
        }
    }

    cout << "Finished!" << endl;
}