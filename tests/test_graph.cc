// #include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "core/common/mem.hpp"
#include "core/store/rdf_dgraph.hpp"

#define KV_SZ (1 << 15)
#define RBUF_SZ (1 << 15)
#define ENGINES 1
#define SID 0
#define TID 0
#define HTID_MIN 4
#define HID_MIN (1 << 6)
#define TID_MIN (1 << 8)
#define PID_MIN (1 << 10)
#define VID_MIN (1 << 17)

namespace test {
using namespace wukong;

// TestKVStore for testing StaticKVStore's protected functions
template <class KeyType, class PtrType, class ValueType>
class TestKVStore : public StaticKVStore<KeyType, PtrType, ValueType> {
public:
    TestKVStore(int sid, KVMem kv_mem) : StaticKVStore<KeyType, PtrType, ValueType>(sid, kv_mem) {}
    ~TestKVStore() {}
    uint64_t alloc_entries_test(uint64_t num_values, int tid = 0) {
        return this->alloc_entries(num_values, tid);
    }
    uint64_t insert_key_test(KeyType key, PtrType ptr) {
        return this->insert_key(key, ptr);
    }
};

using TestStore = TestKVStore<ikey_t, iptr_t, edge_t>;

struct slot_t {
    ikey_t key;
    iptr_t ptr;
};

TEST(Dgraph, Mem) {
    char* a = new char[10];
    printf("a = %p\n", a);

    delete a;
    char* b = new char[10];
    printf("b = %p\n", b);
    memset(b, 0, 10);
}

TEST(Dgraph, TestKVStore) {
    int DEBUG_TEST = 0;

    // alloc memory && mem load region
    char* kvs = new char[KV_SZ / sizeof(char)];
    char* rbuf = new char[RBUF_SZ / sizeof(char)];
    memset(kvs, 0, KV_SZ);
    memset(rbuf, 0, RBUF_SZ);
    KVMem kv_mem = {kvs, KV_SZ, rbuf, RBUF_SZ};

    // initialize KVStore
    TestStore* gstore = new TestStore(SID, kv_mem);
    slot_t* slot = reinterpret_cast<slot_t*>(gstore->get_slot_addr());
    edge_t* value = reinterpret_cast<edge_t*>(gstore->get_value_addr());
    printf("kvs = %p, rbuf = %p, gstore_ptr = %p\n", kvs, rbuf, gstore);
    if (DEBUG_TEST) printf("KVStore: slot_ptr = %p, value_ptr = %p\n", slot, value);
    if (DEBUG_TEST) gstore->print_mem_usage();

    // alloc entries
    uint64_t s1 = 1000, s2 = 1001, p = 20, o1 = 2000, o2 = 2001;
    uint64_t off1 = gstore->alloc_entries_test(1, 0);
    uint64_t off2 = gstore->alloc_entries_test(1, 0);
    if (DEBUG_TEST) printf("entries: off1 = %lu, off2 = %lu\n", off1, off2);
    ASSERT_NE(off1, off2);

    // write value
    value[off1] = edge_t(o1);
    value[off2] = edge_t(o2);

    // insert key
    uint64_t slot_id1 = gstore->insert_key_test(ikey_t(s1, p, OUT), iptr_t(1, off1));
    uint64_t slot_id2 = gstore->insert_key_test(ikey_t(s2, p, OUT), iptr_t(1, off2));
    if (DEBUG_TEST) printf("slots: \n");
    if (DEBUG_TEST) printf("slot_id = %lu, key.pid = %lu, key.vid = %lu, ptr.off = %lu\n", slot_id1, slot[slot_id1].key.pid, slot[slot_id1].key.vid, slot[slot_id1].ptr.off);
    if (DEBUG_TEST) printf("slot_id = %lu, key.pid = %lu, key.vid = %lu, ptr.off = %lu\n", slot_id2, slot[slot_id2].key.pid, slot[slot_id2].key.vid, slot[slot_id2].ptr.off);
    ASSERT_NE(slot_id1, slot_id2);

    // check the key existence
    ASSERT_EQ(gstore->check_key_exist(ikey_t(s1, p, OUT)), 1);
    ASSERT_EQ(gstore->check_key_exist(ikey_t(s2, p, OUT)), 1);
    ASSERT_EQ(gstore->check_key_exist(ikey_t(s2, p, IN)), 0);
    ASSERT_EQ(gstore->check_key_exist(ikey_t(s2, p, IN)), 0);

    // test get value
    uint64_t sz = 0;
    edge_t* edge = gstore->get_values(0, SID, ikey_t(s1, p, OUT), sz);
    ASSERT_EQ(sz, 1);
    ASSERT_NE(edge, NULL);
    ASSERT_EQ(edge->val, o1);
    edge = gstore->get_values(0, SID, ikey_t(s2, p, OUT), sz);
    ASSERT_EQ(sz, 1);
    ASSERT_NE(edge, NULL);
    ASSERT_EQ(edge->val, o2);

    // free memory
    delete gstore;
    delete kvs;
    delete rbuf;
}

TEST(Dgraph, TestRDFGraph) {
    int DEBUG_TEST = 0;

    // alloc memory && mem load region
    char* kvs = new char[KV_SZ / sizeof(char)];
    char* rbuf = new char[RBUF_SZ / sizeof(char)];
    memset(kvs, 0, KV_SZ);
    memset(rbuf, 0, RBUF_SZ);
    KVMem kv_mem = {kvs, KV_SZ, rbuf, RBUF_SZ};

    // initialize DGraph
    RDFGraph* dgraph = new RDFGraph(TID, kv_mem, false);
    printf("kvs = %p, rbuf = %p, dgraph_ptr = %p\n", kvs, rbuf, dgraph);
    if (DEBUG_TEST) dgraph->print_graph_stat();

    // init mock vectors
    std::vector<std::vector<triple_t>> triple_pso;
    std::vector<std::vector<triple_t>> triple_pos;
    std::vector<std::vector<triple_attr_t>> triple_sav;
    triple_pso.resize(ENGINES);
    triple_pos.resize(ENGINES);
    triple_sav.resize(ENGINES);

    /* create mock triples:
     * TYPE_ID = 1, PREDICATE_ID = 0
     * type range: [2, 10]
     * predicate range: [11, 20]
     * vid range: [MIN, MIN + 50], local vid range: [MIN, MIN + 20]
     * TODO: add attr
     */
    for (sid_t s = VID_MIN; s < VID_MIN + 20; s++) {
        // if (DEBUG_TEST) printf("creating mock triples for sid %u\n", s);
        /* insert spo triples */
        for (sid_t i = 1; i < 4; i++) {
            triple_pso[TID].push_back(triple_t(s, (s % 10) + 11, (s % VID_MIN) * 2 + VID_MIN + i));
            triple_pos[TID].push_back(triple_t((s % VID_MIN) * 3 + VID_MIN + i + 1, ((s + 1) % 10) + 11, s));
        }
        /* insert type triples */
        triple_pso[TID].push_back(triple_t(s, TYPE_ID, (s % 9) + 2));
    }

    // testing init_gstore
    printf("----------DGraph initializing gstore--------\n");
    dgraph->init_gstore(triple_pso, triple_pos, triple_sav);
    if (DEBUG_TEST) dgraph->print_graph_stat();

    // ----------testing get values---------
    uint64_t sz = 0;
    edge_t* edge = NULL;

    // testing vid's ngbrs w/ predicate
    printf("----------testing vid's ngbrs w/ predicate--------\n");
    for (sid_t s = VID_MIN; s < VID_MIN + 20; s++) {
        // test spo OUT triples
        edge = dgraph->get_triples(TID, s, (s % 10) + 11, OUT, sz);
        ASSERT_NE(edge, NULL);
        ASSERT_EQ(sz, 3);
        if (DEBUG_TEST) printf("vid = %u, pid = %u, dir = %d, sz = %lu, value: ", s, (s % 10) + 11, OUT, sz);
        for (size_t i = 0; i < sz; i++) {
            if (DEBUG_TEST) printf("%d, ", edge[i].val);
            ASSERT_EQ(edge[i].val, (s % VID_MIN) * 2 + VID_MIN + i + 1);
        }
        if (DEBUG_TEST) printf("\n");

        // test spo IN triples
        edge = dgraph->get_triples(TID, s, ((s + 1) % 10) + 11, IN, sz);
        ASSERT_NE(edge, NULL);
        ASSERT_EQ(sz, 3);
        if (DEBUG_TEST) printf("vid = %u, pid = %u, dir = %d, sz = %lu, value: ", s, ((s + 1) % 10) + 11, IN, sz);
        for (size_t i = 0; i < sz; i++) {
            if (DEBUG_TEST) printf("%d, ", edge[i].val);
            ASSERT_EQ(edge[i].val, (s % VID_MIN) * 3 + VID_MIN + i + 2);
        }
        if (DEBUG_TEST) printf("\n");
    }

    // testing vid's all types
    printf("----------testing vid's all types--------\n");
    for (sid_t s = VID_MIN; s < VID_MIN + 20; s++) {
        edge = dgraph->get_triples(TID, s, TYPE_ID, OUT, sz);
        ASSERT_NE(edge, NULL);
        ASSERT_EQ(sz, 1);
        if (DEBUG_TEST) printf("vid = %u, type = %u\n", s, edge[0].val);
    }

    // testing predicate-index
    printf("----------testing predicate-index--------\n");
    for (sid_t p = 11; p < 21; p++) {
        // TODO: test more cases
        // IN side
        edge = dgraph->get_triples(TID, 0, p, IN, sz);
        ASSERT_NE(edge, NULL);
        ASSERT_EQ(sz, 2);
        if (DEBUG_TEST) printf("predicate = %u, sz = %lu, value: ", p, sz);
        for (size_t i = 0; i < sz; i++)
            if (DEBUG_TEST) printf("%u, ", edge[i].val);
        if (DEBUG_TEST) printf("\n");
        // OUT side
        edge = dgraph->get_triples(TID, 0, p, OUT, sz);
        ASSERT_NE(edge, NULL);
        ASSERT_EQ(sz, 2);
        if (DEBUG_TEST) printf("predicate = %u, sz = %lu, value: ", p, sz);
        for (size_t i = 0; i < sz; i++)
            if (DEBUG_TEST) printf("%u, ", edge[i].val);
        if (DEBUG_TEST) printf("\n");
    }

    // testing type-index
    printf("----------testing type-index--------\n");
    for (sid_t t = 2; t < 11; t++) {
        edge = dgraph->get_triples(TID, 0, t, IN, sz);
        if (DEBUG_TEST) printf("type = %u, sz = %lu\n", t, sz);
        ASSERT_NE(edge, NULL);
        ASSERT_GE(sz, 2);
    }

    // ----------testing versatile KVs---------
    // testing all local objects/subjects
    printf("----------testing all local objects/subjects--------\n");
    edge = dgraph->get_triples(TID, 0, TYPE_ID, IN, sz);
    ASSERT_NE(edge, NULL);
    ASSERT_EQ(sz, 20);
    for (size_t i = 0; i < sz; i++) {
        ASSERT_GE(edge[i].val, VID_MIN);
        ASSERT_LT(edge[i].val, VID_MIN + 20);
        if (DEBUG_TEST) printf("%u, ", edge[i].val);
    }
    if (DEBUG_TEST) printf("\n");

    // testing all local types
    printf("----------testing all local types--------\n");
    edge = dgraph->get_triples(TID, 0, TYPE_ID, OUT, sz);
    ASSERT_NE(edge, NULL);
    ASSERT_EQ(sz, 9);
    for (size_t i = 0; i < sz; i++) {
        ASSERT_GE(edge[i].val, 2);
        ASSERT_LE(edge[i].val, 10);
        if (DEBUG_TEST) printf("%u, ", edge[i].val);
    }
    if (DEBUG_TEST) printf("\n");

    // testing all local predicates
    printf("----------testing all local predicates--------\n");
    edge = dgraph->get_triples(TID, 0, PREDICATE_ID, OUT, sz);
    ASSERT_NE(edge, NULL);
    ASSERT_EQ(sz, 10 + 1);
    for (size_t i = 0; i < sz; i++) {
        if (DEBUG_TEST) printf("%u, ", edge[i].val);
        if (edge[i].val == 1) continue;
        ASSERT_GE(edge[i].val, 11);
        ASSERT_LE(edge[i].val, 20);
    }
    if (DEBUG_TEST) printf("\n");

    // testing vid's all predicates
    printf("----------testing vid's all predicates--------\n");
    for (sid_t s = VID_MIN; s < VID_MIN + 20; s++) {
        // OUT edge
        edge = dgraph->get_triples(TID, s, PREDICATE_ID, OUT, sz);
        ASSERT_NE(edge, NULL);
        ASSERT_GE(sz, 1);
        if (DEBUG_TEST) printf("vid = %u, predicate: out = %u, ", s, edge[0].val);

        // IN edge
        edge = dgraph->get_triples(TID, s, PREDICATE_ID, IN, sz);
        ASSERT_NE(edge, NULL);
        ASSERT_GE(sz, 1);
        if (DEBUG_TEST) printf("in = %u\n", edge[0].val);
    }

    // free memory
    delete dgraph;
    delete kvs;
    delete rbuf;
}

}  // namespace test
