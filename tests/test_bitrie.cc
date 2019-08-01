#include <string>
#include <sstream>
#include <gtest/gtest.h>

#include "bitrie.hpp"

namespace test {
    
string string_dataset = "abc1 1 abc2 2 abc3 3 abc4 4 abc5 5 abc6 6 abc7 7 abc8 8 abc9 9 abc10 10 abc11 11 abc22 22 abc23 23 abc34 34 abc101 101 abc102 102 abc103 103 a 888 abc 999 ak777 777";

/*
            root
            |
            trie_node: a(888)
            |
            trie_node: b    | ...
            |
            trie_node: c(999)
            |                   | ...
            hash_node: 1(1)
            0(10), 1(11), 01(101), 02(102), 03(103)
*/

using namespace std;

#define TEST_ONLY_BUCKET 3
#define TEST_ONLY_ASSOCIATIVITY 2
#define TEST_ONLY_FAST_PATH_NODE 2

class Test_BiTrie : public ::testing::Test {
   protected:
    static void SetUpTestCase() {
        dataset_ptr = new vector<pair<string, unsigned int>>();

        string url = "";
        unsigned int v = 0;

        stringstream ss;
        ss << string_dataset;

        // Load data into bitrie
        while (ss >> url >> v) {
            dataset_ptr->push_back(pair<string, unsigned int>(url, v));
        }
    }

    static void TearDownTestCase() {
        delete dataset_ptr;
    }

    // Some expensive resource shared by all tests.
    static vector<pair<string, unsigned int>> *dataset_ptr;
};
vector<pair<string, unsigned int>>* Test_BiTrie::dataset_ptr = NULL;

// External interfaces(Insert, access, exist)
TEST_F(Test_BiTrie, External_interfaces) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    for(auto &it:dataset) {
        bt.insert_kv(it.first, it.second);
    }

    // Test testing dataset
    for (auto &it:dataset) {
        ASSERT_EQ(bt.exist(it.first), true);
        ASSERT_EQ(bt.exist(it.second), true);
        ASSERT_EQ(bt[it.first], it.second);
        ASSERT_EQ(bt[it.second], it.first);
    }
}

// Resize(page manager resizeing)
TEST_F(Test_BiTrie, Storage_resize) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    for(auto &it:dataset) {
        bt.insert_kv(it.first, it.second);
    }

    bt.storage_resize();

    // Test testing dataset
    for (auto &it:dataset) {
        ASSERT_EQ(bt.exist(it.first), true);
        ASSERT_EQ(bt.exist(it.second), true);
        ASSERT_EQ(bt[it.first], it.second);
        ASSERT_EQ(bt[it.second], it.first);
    }
}

// Test whether dynamic expand() will fail the test
TEST_F(Test_BiTrie, Dynamic_expand) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
        // It invokes expand at No.X  inserting
        // Success and continue: No.3, No.17 
        // Failed and pass it to cuckoo hash: No.6
        if (i == 2 || i== 15 || i == 6) {
            for (auto itt = dataset.begin(); itt!=it;itt++) {
                ASSERT_EQ(bt.exist(itt->first), true);
                ASSERT_EQ(bt.exist(itt->second), true);
                ASSERT_EQ(bt[itt->first], itt->second);
                ASSERT_EQ(bt[itt->second], itt->first);
            }
        }
        i++;
    }
}

// Test whether cuckoo hash() will fail the test
TEST_F(Test_BiTrie, Cuckoo_hashing) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
        // It invokes cuckoo at No.X  inserting
        // Success and continue: No.5
        // Failed and pass it to burst: No.6
        if (i == 5 || i == 6) {
            for (auto itt = dataset.begin(); itt!=it;itt++) {
                ASSERT_EQ(bt.exist(itt->first), true);
                ASSERT_EQ(bt.exist(itt->second), true);
                ASSERT_EQ(bt[itt->first], itt->second);
                ASSERT_EQ(bt[itt->second], itt->first);
            }
        }
        i++;
    }
}

// Test whether Burst() will fail the test
TEST_F(Test_BiTrie, Bursting) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
        // It invokes burst at No.X  inserting
        if (i == 6) {
            for (auto itt = dataset.begin(); itt!=it;itt++) {
                ASSERT_EQ(bt.exist(itt->first), true);
                ASSERT_EQ(bt.exist(itt->second), true);
                ASSERT_EQ(bt[itt->first], itt->second);
                ASSERT_EQ(bt[itt->second], itt->first);
            }
        }
        i++;
    }
}

// Test the element that locate on a trie node
TEST_F(Test_BiTrie, Search_element_on_trie_node) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    string key = "a";
    unsigned int value = 888;

    ASSERT_EQ(bt.exist(key), true);
    ASSERT_EQ(bt.exist(value), true);
    ASSERT_EQ(bt[key], value);
    ASSERT_EQ(bt[value], key);
}

// Test the element that locate on a hash node
TEST_F(Test_BiTrie, Search_element_on_hash_node) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }
    
    string key = "abc1";
    unsigned int value = 1;

    ASSERT_EQ(bt.exist(key), true);
    ASSERT_EQ(bt.exist(value), true);
    ASSERT_EQ(bt[key], value);
    ASSERT_EQ(bt[value], key);
}

// Test the element that locate in a hash node
TEST_F(Test_BiTrie, Search_element_in_hash_node) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }
    
    string key = "abc10";
    unsigned int value = 10;

    ASSERT_EQ(bt.exist(key), true);
    ASSERT_EQ(bt.exist(value), true);
    ASSERT_EQ(bt[key], value);
    ASSERT_EQ(bt[value], key);
}

// Test the duplicate element inserting on trie node
TEST_F(Test_BiTrie, Insert_duplicate_element_on_trie_node) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    string key = "a";
    unsigned int value = 123456;

    bt.insert_kv(key, value);

    ASSERT_EQ(bt.exist(key), true);
    ASSERT_EQ(bt.exist(value), true);
    ASSERT_EQ(bt[key], value);
    ASSERT_EQ(bt[value], key);
}

// Test the duplicate element inserting on hash node
TEST_F(Test_BiTrie, Insert_duplicate_element_on_hash_node) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    string key = "abc1";
    unsigned int value = 123456;

    bt.insert_kv(key, value);

    ASSERT_EQ(bt.exist(key), true);
    ASSERT_EQ(bt.exist(value), true);
    ASSERT_EQ(bt[key], value);
    ASSERT_EQ(bt[value], key);
}

// Test the duplicate element inserting in hash node
TEST_F(Test_BiTrie, Insert_duplicate_element_in_hash_node) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    string key = "abc10";
    unsigned int value = 123456;

    bt.insert_kv(key, value);

    ASSERT_EQ(bt.exist(key), true);
    ASSERT_EQ(bt.exist(value), true);
    ASSERT_EQ(bt[key], value);
    ASSERT_EQ(bt[value], key);
}

// Test the element that contains a fast path searching
TEST_F(Test_BiTrie, Using_normal_path) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    string key = "ak777";
    unsigned int value = 777;

    ASSERT_EQ(bt.exist(key), true);
    ASSERT_EQ(bt.exist(value), true);
    ASSERT_EQ(bt[key], value);
    ASSERT_EQ(bt[value], key);
}

// Test the element that contains a fast path searching
TEST_F(Test_BiTrie, Using_fast_path) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    string key = "abc101";
    unsigned int value = 101;

    ASSERT_EQ(bt.exist(key), true);
    ASSERT_EQ(bt.exist(value), true);
    ASSERT_EQ(bt[key], value);
    ASSERT_EQ(bt[value], key);
}

// Test the element that with a suffix in hash_node
TEST_F(Test_BiTrie, Get_string_with_suffix) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    string key = "abc101";
    unsigned int value = 101;

    ASSERT_EQ(bt.exist(key), true);
    ASSERT_EQ(bt.exist(value), true);
    ASSERT_EQ(bt[key], value);
    ASSERT_EQ(bt[value], key);
}

// Test the element that without a suffix in hash_node
TEST_F(Test_BiTrie, Get_string_without_suffix) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    string key = "abc1";
    unsigned int value = 1;

    ASSERT_EQ(bt.exist(key), true);
    ASSERT_EQ(bt.exist(value), true);
    ASSERT_EQ(bt[key], value);
    ASSERT_EQ(bt[value], key);
}

// Test the element that value doesn't exist
TEST_F(Test_BiTrie, Insert_empty_key_element) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    string key = "";
    unsigned int value = 555;

    bt.insert_kv(key, value);

    ASSERT_EQ(bt.exist(key), true);
    ASSERT_EQ(bt[key], value);
}

// Test the element that key is empty
TEST_F(Test_BiTrie, Get_value_with_empty_key) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    string key = "";

    ASSERT_EQ(bt.exist(key), false);
    ASSERT_EQ(bt[key], 0);
}

// Test the element that key is non-empty
TEST_F(Test_BiTrie, Get_value_with_nonempty_key) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    string key = "abc1";
    unsigned int value = 1;

    ASSERT_EQ(bt.exist(key), true);
    ASSERT_EQ(bt[key], value);
}


// Test the element that value doesn't exist
TEST_F(Test_BiTrie, Get_string_with_exist_value) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    unsigned int value = 102;
    string key = "abc102";

    ASSERT_EQ(bt.exist(value), true);
    ASSERT_EQ(bt[value], key);
}

// Test the element that value doesn't exist
TEST_F(Test_BiTrie, Get_string_with_unexist_value) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    unsigned int value = 1000000;

    ASSERT_EQ(bt.exist(value), false);
    ASSERT_EQ(bt[value], "");
}

// Test the element with long key
// In bitrie, the long key will be inserted into the special group
TEST_F(Test_BiTrie, Get_string_with_long_key) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    string key = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    unsigned int value = 123456;

    bt.insert_kv(key, value);

    ASSERT_EQ(bt.exist(value), true);
    ASSERT_EQ(bt[value], key);
}

// Test the element with long key
// In bitrie, the long key will be inserted into the special group
TEST_F(Test_BiTrie, Get_value_with_long_key) {
    bitrie<char, uint32_t, TEST_ONLY_BUCKET, TEST_ONLY_ASSOCIATIVITY, TEST_ONLY_FAST_PATH_NODE> bt;
    const vector<pair<string, unsigned int>> &dataset = *dataset_ptr;

    int i = 0;
    for (auto it = dataset.begin(); it!=dataset.end();it++) {
        bt.insert_kv(it->first, it->second);
    }

    string key = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    unsigned int value = 123456;

    bt.insert_kv(key, value);

    ASSERT_EQ(bt.exist(key), true);
    ASSERT_EQ(bt[key], value);
}

}