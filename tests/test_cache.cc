#include <gtest/gtest.h>

#include "core/store/rdma_cache.hpp"

namespace test {

using namespace wukong;
struct slot_t {
  ikey_t key;
  iptr_t ptr;
};

TEST(Store, Cache) {
  RDMA_Cache<ikey_t, slot_t> cache;
  cache.set_lease(SEC(2));

  ikey_t key(1, 1, 1);
  iptr_t ptr(10, 123, 0);
  slot_t v = {key, ptr};

  // test lookup, not found case
  slot_t lv;
  bool success = cache.lookup(key, lv);
  EXPECT_EQ(success, false);

  // test insert, first insert case
  // test lookup, found case
  cache.insert(v);
  success = cache.lookup(key, lv);
  EXPECT_EQ(success, true);
  EXPECT_EQ(ptr.size, lv.ptr.size);

  // test insert, re-insert case
  ptr = iptr_t(20, 456, 1);
  v.ptr = ptr;
  cache.insert(v);
  success = cache.lookup(key, lv);
  EXPECT_EQ(success, true);
  EXPECT_EQ(ptr.size, lv.ptr.size);

  // test invalidate
  cache.invalidate(key);
  success = cache.lookup(key, lv);
  EXPECT_EQ(success, false);

  // test lease
  // DUSE_DYNAMIC_GSTORE=ON
  cache.insert(v);
  success = cache.lookup(key, lv);
  EXPECT_EQ(success, true);
  EXPECT_EQ(ptr.size, lv.ptr.size);
  sleep(2);
  success = cache.lookup(key, lv);
  EXPECT_EQ(success, false);
}

}
