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
#include <vector>
#include <queue>
#include <string>

#include <boost/unordered_map.hpp>

// default configuration
static const unsigned int DEFAULT_ASSOCIATIVITY = 8;
static const unsigned int DEFAULT_BUCKET_NUM = 59;
static const unsigned int DEFAULT_NORMAL_PAGE_SIZE = 4096;
static const unsigned int DEFAULT_SPECIAL_PAGE_SIZE = (4096 * 4);
static const double DEFAULT_CUCKOO_HASH_RATIO = 0.5;

// slot use bits configuration
enum { NBITS_SPECIAL = 1, NBITS_SPECIAL_S = 1 };  // is special
enum { NBITS_LEN = 7, NBITS_LEN_S = 13 };         // length
enum { NBITS_PID = 12, NBITS_PID_S = 9 };         // page id
enum { NBITS_POS = 12, NBITS_POS_S = 9 };         // position in page

// normal/special bound
static const unsigned int MAX_NORMAL_LEN = (1 << NBITS_LEN);

// page manager configuration
static const unsigned int DEFAULT_NORMAL_PAGE_NUMBER = (1 << NBITS_PID);
static const unsigned int DEFAULT_SPECIAL_PAGE_NUMBER = (1 << NBITS_PID_S);
static const unsigned int DEFAULT_SPECIAL_ALIGNMENT = 32;
static const unsigned int DEFAULT_NORMAL_ALIGNMENT = 1;

// fast path configuration
static const unsigned int DEFAULT_FAST_PATH_NODE_NUM = 20;

using namespace std;

/**
 * @brief Bi-trie, a new data structure provides the bi-direction mapping 
 * between strings and integers in a memory-efficient way.
 * 
 * @tparam K_unit Key's unit type. Normally, it is char which is the unit of string.
 * @tparam T Value's type.
 * @tparam BUCKET_NUM Bucket number in a hash node.
 * @tparam ASSOCIATIVITY Associativity in a hash node.
 */
template <class K_unit, class T, size_t BUCKET_NUM = DEFAULT_BUCKET_NUM,
          size_t ASSOCIATIVITY = DEFAULT_ASSOCIATIVITY,
          size_t FAST_PATH_NODE_NUM = DEFAULT_FAST_PATH_NODE_NUM>
class bitrie {
private:
    static bool key_equal(const K_unit* key_lhs, const size_t key_size_lhs,
                const K_unit* key_rhs, const size_t key_size_rhs) {
        if (key_size_lhs == 0 && key_size_rhs == 0)
            return true;
        
        if (key_size_lhs != key_size_rhs)
            return false;
        else
            return (memcmp(key_lhs, key_rhs, key_size_lhs * sizeof(K_unit)) == 0);
    }

    // For fasthash64:
    static inline uint64_t mix(uint64_t h) {
        h ^= h >> 23;
        h *= 0x2127599bf4325c37ULL;
        h ^= h >> 47;
        return h;
    }

    // A default hash function:
    static uint64_t fasthash64(const char* buf, size_t len, uint64_t seed) {
        uint64_t const m = 0x880355f21e6d1965ULL;
        uint64_t const* pos = (uint64_t const*)buf;
        uint64_t const* end = pos + (len / 8);
        const unsigned char* pos2;
        uint64_t h = seed ^ (len * m);
        uint64_t v;

        while (pos != end) {
            v = *pos++;
            h ^= mix(v);
            h *= m;
        }

        pos2 = (const unsigned char*)pos;
        v = 0;

        switch (len & 7) {
            case 7:
                v ^= (uint64_t)pos2[6] << 48;
            case 6:
                v ^= (uint64_t)pos2[5] << 40;
            case 5:
                v ^= (uint64_t)pos2[4] << 32;
            case 4:
                v ^= (uint64_t)pos2[3] << 24;
            case 3:
                v ^= (uint64_t)pos2[2] << 16;
            case 2:
                v ^= (uint64_t)pos2[1] << 8;
            case 1:
                v ^= (uint64_t)pos2[0];
                h ^= mix(v);
                h *= m;
        }

        return mix(h);
    }

    /**
     * @brief Provide hash function for cuckoo hashing.
     * 
     * @param key The string's pointer being hashing.
     * @param key_size The string's size being hashing.
     * @param hash_type Chose a hash seed for hashing.
     * @return size_t The hash value of string(key, key_size).
     */
    static size_t hash(const K_unit* key, const size_t key_size, const size_t hash_type = 1) {
        if (hash_type == 1) 
            return fasthash64(key, key_size, 0xdeadbeefdeadbeefULL);
        else
            return fasthash64(key, key_size, 0xabcdefabcdef1234ULL);
    }

    /* node */
    class node;
    class trie_node;
    class hash_node;

    class found_result;

    class slot;

    /* page management */
    class page_manager;
    class page_manager_agent;
    class page_group;
    class page;

    /* group type divided by the key_size */
    enum class group_type : unsigned char { NORMAL_GROUP, SPECIAL_GROUP };

    /* node type definition */
    enum class node_type : unsigned char { HASH_NODE, TRIE_NODE };

    /* helper function */
    static inline const group_type get_group_type(const size_t key_size) {
        return (key_size < MAX_NORMAL_LEN) ? group_type::NORMAL_GROUP
                                           : group_type::SPECIAL_GROUP;
    }

    static inline const group_type get_group_type(const slot* s) {
        return (s->get_length() < MAX_NORMAL_LEN) ? group_type::NORMAL_GROUP
                                                  : group_type::SPECIAL_GROUP;
    }

private:
    /**
     * @brief Storing location recording class.
     * 
     * The encode denotes the storing location of a element.
     * 
     */
    class slot {
    private:
        uint32_t encode;

    public:
        slot() : encode(0) { }

        slot(bool is_special, uint64_t length, uint64_t pos, uint64_t page_id) : 
            encode(encode_slot(is_special, length, pos, page_id)) { }

        /**
         * @brief Encode the several parameters according to the NBITS configuration.
         * 
         * Encode slot as | special | length | position | page_id |.
         * 
         * @param is_special Whether the element is special.
         * @param length Length of the element.
         * @param pos Position of the element stored in a page.
         * @param page_id Page id of the element stored in a page group.
         * @return uint64_t Encoding value of the several parameters above.
         */
        uint64_t encode_slot(bool is_special, uint64_t length, uint64_t pos, uint64_t page_id) {
            encode = page_id;
            if (is_special) {
                assert(length < 1 << NBITS_LEN_S &&
                       (pos / DEFAULT_SPECIAL_ALIGNMENT) < 1 << NBITS_POS_S &&
                       page_id < 1 << NBITS_PID_S);
                encode += ((pos / DEFAULT_SPECIAL_ALIGNMENT) << NBITS_PID_S);
                encode += length << (NBITS_PID_S + NBITS_POS_S);
                encode += ((uint64_t)1)
                          << (NBITS_PID_S + NBITS_LEN_S + NBITS_POS_S);
            } else {
                assert(length < 1 << NBITS_LEN &&
                       (pos / DEFAULT_NORMAL_ALIGNMENT) < 1 << NBITS_POS &&
                       page_id < 1 << NBITS_PID);
                encode += (pos / DEFAULT_NORMAL_ALIGNMENT) << NBITS_PID;
                encode += length << (NBITS_PID + NBITS_POS);
                encode += ((uint64_t)0)  << (NBITS_PID + NBITS_LEN + NBITS_POS);
            }

            return encode;
        }

        bool is_empty() const { return get_length() == 0; }

        bool is_special() const { return get_special(); }

        void set_slot(bool is_special, uint64_t length, uint64_t pos, uint64_t page_id) {
            encode = encode_slot(is_special, length, pos, page_id);
        }

        void set_slot(slot s) { encode = s.encode; }

        bool get_special() const {
            return (encode >> (NBITS_PID + NBITS_LEN + NBITS_POS)) == 1;
        }

        const size_t get_length() const {
            return encode >> (NBITS_LEN + NBITS_POS + NBITS_PID)
                       ? ((encode >> (NBITS_PID_S + NBITS_POS_S)) %
                          (1 << NBITS_LEN_S))
                       : ((encode >> (NBITS_PID + NBITS_POS)) %
                          (1 << NBITS_LEN));
        }

        const size_t get_pos() const {
            return encode >> (NBITS_PID + NBITS_LEN + NBITS_POS)
                       ? ((encode >> NBITS_PID_S) % (1 << NBITS_POS_S) *
                          DEFAULT_SPECIAL_ALIGNMENT)
                       : ((encode >> NBITS_PID) % (1 << NBITS_POS) *
                          DEFAULT_NORMAL_ALIGNMENT);
        }

        const size_t get_page_id() const {
            return encode >> (NBITS_PID + NBITS_LEN + NBITS_POS)
                       ? encode % (1 << NBITS_PID_S)
                       : encode % (1 << NBITS_PID);
        }

        void swap(slot *sl) {
            uint32_t temp_encode = sl->encode;
            sl->encode = encode;
            encode = temp_encode;
        }

        void print_slot(page_manager_agent &pm_agent) {
            cout << get_special() << "," << get_length() << "," << get_pos()
                 << "," << get_page_id() << ","
                 << string(pm_agent.get_content_pointer(this), get_length()) << ","
                 << pm_agent.get_value(this) << endl;
        }
    };

    /**
     * @brief Node's base class.
     * 
     * We divide the node into two type, trie node and hash node that descriped above its class.
     * 
     */
    class node {
    private:
        node_type n_type_;
        trie_node* parent_;

        // The value of key that terminates in current node
        bool have_value_;
        T value_;

        string prefix_;

    public:
        node(const node_type n_type, trie_node* parent, const K_unit *key, const size_t key_size)
            : n_type_(n_type),
              parent_(parent),
              have_value_(false),
              value_(T()),
              prefix_("") {
            // If current node's layer level equals to multiple of FAST_PATH_NODE_NUM,
            // Set up a fast path in its fast-path parent
            if (key_size % FAST_PATH_NODE_NUM == 0 && key_size != 0) {              
                // Get the target parent who is going to add a fast path for destination node(this)
                trie_node* add_fast_path_parent = get_fast_path_parent();
                assert(add_fast_path_parent != nullptr);
                add_fast_path_parent->add_fast_path(key + key_size - FAST_PATH_NODE_NUM, FAST_PATH_NODE_NUM, this);
            }
        }

        /*---- Type predicate function ---*/
        bool is_hash_node() const { return n_type_ == node_type::HASH_NODE; }

        bool is_trie_node() const { return n_type_ == node_type::TRIE_NODE; }

        /*---- Set function ---*/
        void set_parent(trie_node* p) { parent_ = p; }

        void set_prefix(const string& prefix) { prefix_ = prefix; }

        /*---- Get function ---*/
        trie_node* get_parent() const { return parent_; }

        const string& get_prefix() const { return prefix_; }

        node_type get_node_type() const { return n_type_; }

        /**
         * @brief Get the fast-path parent for adding a fast path.
         * 
         * @return trie_node* (Grand)parent to add fast path for this node.
         */
        trie_node* get_fast_path_parent() const {
            trie_node* cur_parent = (trie_node*)this;
            for (int i = 0; i != FAST_PATH_NODE_NUM; i++) {
                cur_parent = cur_parent->node::get_parent();
                if (cur_parent == nullptr) return nullptr;
            }
            return cur_parent;
        }

        /**
         * @brief Virtual function for page_manager resize.
         * 
         * @param old_pm Source page manager for resizing.
         * @param new_pm Dest page manager for resizing.
         * @param resize_type Resizing type(special or normal).
         */
        virtual void traverse_for_pgm_resize(page_manager* old_pm,
                                             page_manager* new_pm,
                                             group_type resize_type) = 0;

        /**
         * @brief Insert element that terminates in 'this' node.
         * 
         * @param prefix Prefix of 'this' node.
         * @param v Inserting value.
         * @param bt Bi-trie that manages 'this' node.
         */
        void insert_value_in_node(const string& prefix, const T v,
                                  bitrie* const bt) {
            value_ = v;
            have_value_ = true;
            bt->set_v2k(v, this, -1);
            prefix_ = prefix;
            return;
        }

        /**
         * @brief Search element that terminates in 'this' node.
         * 
         * @return found_result The result that wether the target element exists.
         */
        found_result search_kv_in_node() const {
            return found_result(have_value_, value_, -1, -1);
        }

        /**
         * @brief Deleting 'this' node. (For bitrie deconstructor).
         * 
         */
        virtual void delete_me() = 0;

        // Deconstructor
        virtual ~node() {}
    };

    /**
     * @brief Burst-trie's trie node class.
     * 
     * Non-leaf node in burst trie,
     * Take in charge of leading the lookup function to reach the target hash
     * node, the leaf node, in normal trie searching way.
     */
    class trie_node : public node {
        friend class bitrie;

    private:
        /**
         * @brief Fast-path manager class.
         * 
         * Provide the trie_node a fast way to skip several trie traverse.
         * Using the original way only traverse burst-trie char by char, but
         * using the fast path traverse burst-trie string by string.
         * Fast-path manager store the strings in hash value order, so that the
         * searching can by transform to binary search.
         */
        class fast_path_manager {
        private:
            struct fast_path {
                unsigned int hash_val_;
                string fast_path_string_;
                node* dest_node_;

            public:
                fast_path(const unsigned int hash_val, const string &fast_path_string, node* dest_node)
                    : hash_val_(hash_val), fast_path_string_(fast_path_string), dest_node_(dest_node) { }

                inline void set_dest_node(node *dest_node) { dest_node_ = dest_node; }

                inline const unsigned int get_hash_val() const { return hash_val_; }
                inline const string& get_string() const { return fast_path_string_; }
                inline node* get_dest_node() const { return dest_node_; }
            };

            vector<fast_path> fast_paths_;

        public:
            fast_path_manager() {}

            /**
             * @brief Insert new fast-path into the fast_path_manager in hash value order.
             * 
             * @param key Pointer of inserting fast path's string.
             * @param key_size Size of inserting fast path's string.
             * @param node_ptr Destination node of inserting fast path.
             */
            void insert_fast_path(const char* key, size_t key_size, node* node_ptr) {
                fast_path new_fast_path(hash(key, key_size, 1), string(key, key_size), node_ptr);

                for (auto it = fast_paths_.begin(); it != fast_paths_.end();
                    it++) {
                    if (it->get_hash_val() >= new_fast_path.get_hash_val()) {
                        fast_paths_.insert(it, new_fast_path);
                        return;
                    }
                }
                fast_paths_.push_back(new_fast_path);
                return;
            }

            /**
             * @brief Lookup the target node in binary search.
             * 
             * @param key Pointer of lookuping fast path's string
             * @param key_size Size of lookuping fast path's string
             * @return node* Destination node of lookuping fast path
             */
            inline node* lookup_fast_path(const char* key, size_t key_size) const {
                unsigned int target_hash_val = hash(key, key_size);

                size_t node_size = fast_paths_.size();
                int low = 0;
                int high = node_size - 1;
                // The binary will deal with the same hash value fast-path
                // situation by skiping the "return mid" when
                // fast_paths_[mid].get_hash_val() == target_hash_val
                while (low < high) {
                    int mid = (low + high) >> 1;
                    if (fast_paths_[mid].get_hash_val() < target_hash_val)
                        low = mid + 1;
                    else
                        high = mid;
                }

                // Check the same hash value fast-path
                for (int i = low;
                    low != node_size &&
                    fast_paths_[i].get_hash_val() == target_hash_val;
                    i++) {
                    if (key_equal(
                            fast_paths_[i].get_string().data(),
                            fast_paths_[i].get_string().size(), key,
                            key_size)) {
                        return fast_paths_[i].get_dest_node();
                    }
                }
                return nullptr;
            }

            inline size_t size() const { return fast_paths_.size(); }

            unsigned int get_fpm_memory() const {
                return size() * (sizeof(fast_path) + FAST_PATH_NODE_NUM);
            }
        };

        /**
         * @brief Child_representation class.
         *
         * The child representation is used to save the relationship of nodes
         * in a trie.
         * 
         * Child representation can be implemented in several way:
         * Current version are implemented in list.
         *
         *      Implementation: | memory-efficiency | effectiveness |
         *      List:           |         10        |       8       |
         *      Array:          |          1        |      10       |
         *      std::map:       |          8        |       5       |
         */
        class child_representation {
        public:
            /* List node class */
            struct child_node {
            public:
                char child_node_char;
                node* current;
                child_node* next;

                child_node(char cnc, node* cur)
                    : child_node_char(cnc), current(cur), next(nullptr) { }

                inline bool have_next() const { return next != nullptr; }

                inline node* get_node() const { return current; }

                inline child_node* next_child() const { return next; }

                inline void add_next_child(char c) { next = new child_node(c, nullptr); }
            };

        private:
            child_node* first_child_;  // List header
            int size_;                 // List node number

        public:
            child_representation() : size_(0), first_child_(nullptr) {}

            /**
             * @brief Access specified element.
             * 
             * If find one, return the reference.
             * If not, add one and return the reference.
             * 
             * @param c Char of a child node's being lookuped.
             * @return node*& Reference to the requested element.
             */
            node *& operator[](const char c) {
                child_node* current_child_node = first_child_;
                child_node* last_child_node = nullptr;

                while (current_child_node) {
                    if (current_child_node->child_node_char == c)
                        return current_child_node->current;

                    last_child_node = current_child_node;
                    current_child_node = current_child_node->next;
                }

                // List is empty, add one
                if (first_child_ == nullptr) {
                    first_child_ = new child_node(c, nullptr);

                    size_++;
                    return first_child_->current;
                }

                // Find no target node, add one
                last_child_node->add_next_child(c);
                size_++;
                return last_child_node->next_child()->current;
            }

            /**
             * @brief Find specified element.
             * 
             * If find one, return the node*.
             * If not, return nullptr.
             * 
             * @param c Char of a child node's being lookuped.
             * @return node* Destination child node of char c.
             */
            node * find(const char c) const {
                child_node* current_child_node = first_child_;
                while(current_child_node != nullptr) {
                    if (current_child_node->child_node_char == c)
                        return current_child_node->current;

                    current_child_node = current_child_node->next_child();
                };
                return nullptr;
            }

            inline size_t size() const { return size_; }

            inline child_node* get_first_node() const { return first_child_; }

            ~child_representation() {
                // Release the list
                child_node* current_child_node = first_child_;
                child_node* previous_current_child_node = nullptr;

                while (current_child_node) {
                    previous_current_child_node = current_child_node;
                    current_child_node = current_child_node->next;
                    delete (previous_current_child_node);
                }
            }

            /* helper function: memory evaluation */
            size_t get_childs_representation_mem() const {
                return sizeof(child_representation) + size_ * sizeof(child_node);
            }
        };

    private:
        fast_path_manager *fpm_;       // Manage the fast-paths
        child_representation childs_;  // Store the suffix node of hash_node or trie_node

    public:
        trie_node(trie_node *p, const char *key, size_t key_size)
            : node(node_type::TRIE_NODE, p, key, key_size), fpm_(nullptr) { }

        /**
         * @brief Add a fast path of string(key, key_size) in fast path manager
         * 
         * @param key Pointer of string of the fast path adding to fast path manager.
         * @param key_size Size of string of the fast path adding to fast path manager.
         * @param node Destination node.
         */
        void add_fast_path(const char* key, size_t key_size, node* node) {
            if (fpm_ == nullptr)
                fpm_ = new fast_path_manager();
            fpm_->insert_fast_path(key, key_size, node);
            return;
        }

        // For bitrie deconstructor
        virtual void delete_me() {
            typename child_representation::child_node* cur_child = childs_.get_first_node();
            while(cur_child != nullptr) {
                node* cur_node = cur_child->get_node();
                cur_node->delete_me();
                cur_child = cur_child->next_child();
            }
            delete this;
        }

        // Deconstructor
        ~trie_node() { if (fpm_ != nullptr) delete fpm_; }

        /**
         * @brief Virtual function for page_manager resize.
         * 
         * The virtual function implementing for page_manager page resize.
         * 
         * @param old_pm Source page manager for resizing.
         * @param new_pm Dest page manager for resizing.
         * @param resize_type Resizing type(special or normal).
         */
        void traverse_for_pgm_resize(page_manager* old_pm, page_manager* new_pm,
                                     group_type resize_type) {
            typename child_representation::child_node* cur_child = childs_.get_first_node();
            while (cur_child != nullptr) {
                node* cur_node = cur_child->get_node();
                cur_node->traverse_for_pgm_resize(old_pm, new_pm, resize_type);
                cur_child = cur_child->next_child();
            }
        }

        /**
         * @brief Add node in child representation.
         * 
         * @param c Adding node's char.
         * @param adding_node Adding node.
         */
        void add_child(const K_unit c, node* adding_node) { childs_[c] = adding_node; }

        /**
         * @brief Finding target node.
         *
         * This function will use two ways to find a target node.
         * String by string: If 'this' node contains a fast path manager, then
         * find the target in fast path manager.(faster)
         * Char by char: If 'this' node doesn't contain a fast path manager,
         * then find the target in child representation.
         *
         * @note: ref_pos is a reference for plusing different value.
         * When using the string-by-string, ref_pos increases FAST_PATH_NODE_NUM.
         * When using char-by-char, ref_pos increases 1.
         *
         * @param key Pointer of the lookuping element's string.
         * @param ref_pos Reference of current lookuping position.
         * @param key_size The total size of the lookuping element's string.
         * @param bt Bi-trie that manages 'this' node.
         * @return node* The target child node.
         */
        node * find_trie_node_child(const K_unit* key, size_t& ref_pos,
                                   size_t key_size, const bitrie* bt) const {
            // Find in fast path
            // If find the target node in fpm(fast path manager), we return the
            // fast_path_node
            if (fpm_ != nullptr && (ref_pos + FAST_PATH_NODE_NUM < key_size)) {
                node* fast_path_node =
                    fpm_->lookup_fast_path(key + ref_pos, FAST_PATH_NODE_NUM);
                if (fast_path_node != nullptr) {
                    ref_pos += FAST_PATH_NODE_NUM;
                    return fast_path_node;
                }
            }

            // Find in normal path
            node* target_node = childs_.find(key[ref_pos]);
            ref_pos++;
            return target_node;
        }
    };

    /**
     * @brief Burst-trie's hash node class
     * 
     * Leaf node in burst trie,
     * Take in charge of leading the lookup function to find the lookuping
     * element, the value, in hashtable searching way.
     */
    class hash_node : public node {
        friend class bitrie;

    private:
        slot *key_metas_;
        size_t elem_num_;

        size_t cur_associativity_;

        // normal page_group id
        uint32_t normal_pgid_;
        // special page_group id
        uint32_t special_pgid_;

    public:
        /* Debug helper function */
        void print_slot(int i, int j, page_manager_agent &pm_agent) {
            slot *s = get_slot(i, j);
            cout << i * cur_associativity_ + j << ":" << s->get_special() << ","
                 << s->get_length() << "," << s->get_pos() << ","
                 << s->get_page_id() << ",";
            string str = string(pm_agent.get_content_pointer(s),
                                s->get_length());
            cout << str;
            T v = pm_agent.get_value(s);
            cout << "=" << v << "\n";
        }

        // Print the element in current node in above print_slot() format
        void print_key_metas(page_manager_agent &pm_agent) {
            for (int i = 0; i != BUCKET_NUM; i++) {
                for (int j = 0; j != cur_associativity_; j++) {
                    print_slot(i, j, pm_agent);
                }
                cout << "---\n";
            }
            cout << endl;
        }

    public:
        hash_node(trie_node* p, const string& prefix, page_manager* pm,
                  size_t need_associativity = 1)
            : node(node_type::HASH_NODE, p, prefix.data(), prefix.size()),
              cur_associativity_(need_associativity > ASSOCIATIVITY
                                    ? ASSOCIATIVITY
                                    : need_associativity),
              elem_num_(0),
              normal_pgid_(pm->require_group_id(group_type::NORMAL_GROUP)),
              special_pgid_(pm->require_group_id(group_type::SPECIAL_GROUP)) {
            // Hash node must have the prefix for element string concat        
            node::set_prefix(prefix);

            key_metas_ = new slot[cur_associativity_ * BUCKET_NUM]();
        }

        /**
         * @brief Virtual function for page_manager resize.
         * 
         * The virtual function implementing for page_manager page resize.
         * Move the elements from old page_manager into new page_manager.
         * 
         * @param old_pm Source page manager for resizing.
         * @param new_pm Dest page manager for resizing.
         * @param resize_type Resizing type(special or normal).
         */
        void traverse_for_pgm_resize(page_manager* old_pm, page_manager* new_pm,
                                     group_type resize_type) {
            int old_normal_pgid = normal_pgid_;
            int old_special_pgid = special_pgid_;
            int new_normal_pgid = new_pm->require_group_id(group_type::NORMAL_GROUP);
            int new_special_pgid = new_pm->require_group_id(group_type::SPECIAL_GROUP);

            if (resize_type == group_type::SPECIAL_GROUP) {
                set_special_pgid(new_special_pgid);
            } else if (resize_type == group_type::NORMAL_GROUP) {
                set_normal_pgid(new_normal_pgid);
            }

            page_manager_agent old_pm_agent = old_pm->get_page_manager_agent(old_normal_pgid, old_special_pgid);
            page_manager_agent new_pm_agent = new_pm->get_page_manager_agent(new_normal_pgid, new_special_pgid);

            for (int i = 0; i != BUCKET_NUM; i++) {
                for (int j = 0; j != cur_associativity_; j++) {
                    slot* s = get_slot(i, j);

                    // Ignore the slot that not belong to current resize group_type
                    if (s->is_empty() || get_group_type(s) != resize_type) continue;

                    // Get the content from old page_manager and write it to the
                    // new page_manager
                    s->set_slot(new_pm_agent.insert_element(
                        old_pm_agent.get_content_pointer(s), s->get_length(),
                        old_pm_agent.get_value(s)));
                }
            }
        }

        // For bitrie deconstructor
        virtual void delete_me() { delete this; }

        // Deconstructor
        ~hash_node() { delete[] key_metas_; }

        /*---- Get function ---*/
        inline slot* get_slot(size_t bucketid, size_t slotid) const {
            return key_metas_ + bucketid * cur_associativity_ + slotid;
        }

        /**
         * @brief Get the column store slot object
         *
         * For eliminating the index update in dynamic_expand, we store the
         * column-store-index in v2k instead of row-store-index. So if we have a
         * column-store-index, we use this function to get the target slot.
         *
         * @param column_store_index Index that counted in column-store way.
         * @return slot* Target slot that match the column_store_index.
         */
        inline slot* get_column_store_slot(int column_store_index) const {
            return key_metas_ +
                    (cur_associativity_ * (column_store_index % BUCKET_NUM)) +
                    column_store_index / BUCKET_NUM;
        }

        inline int get_column_store_index(const slot* const s) const {
            return BUCKET_NUM * (get_index(s) % cur_associativity_) +
                    (get_index(s) / cur_associativity_);
        }

        inline int get_index(const slot* const s) const {
            return s - key_metas_;
        }

        inline int get_index(const size_t bucketid, const size_t slotid) const {
            return bucketid * cur_associativity_ + slotid;
        }

        size_t get_normal_pgid() const { return normal_pgid_; }

        size_t get_special_pgid() const { return special_pgid_; }

        /*---- Set function ---*/
        inline void set_normal_pgid(size_t new_normal_pgid) { normal_pgid_ = new_normal_pgid; }

        inline void set_special_pgid(size_t new_special_pgid) { special_pgid_ = new_special_pgid; }

        /*---- Reduce memory foot-print mechanism ---*/
        /* 
         * We use 2 mechanisms for reducing burst's trie memory foot-print:
         *      1. Dynamic expand: Reduce unnecessary memory allocation
         *      2. Cuckoo hash: Increase slot utilization rate
         */
        /**
         * @brief Dynamic expand function.
         * 
         * This function will expand the key_metas to two times of original key_metas.
         * If return -1, it means that current size of key_metas is maximum.
         * 
         * @return int A empty slotid.
         */
        int dynamic_expand() {
            // Already max associativity
            // We cannot expand anymore, return -1
            if (cur_associativity_ == ASSOCIATIVITY) return -1;

            // Get the associativity we need, expand 2 times of cur_associativity
            unsigned int need_associativity = cur_associativity_ << 1;
            if (need_associativity > ASSOCIATIVITY) {
                need_associativity = ASSOCIATIVITY;
            }

            // Allocate a bigger memory for new key_metas
            slot* new_key_metas = new slot[need_associativity * BUCKET_NUM]();

            for (int i = 0; i != BUCKET_NUM; i++) {
                for (int j = 0; j != need_associativity; j++) {
                    slot* cur_new_slot =
                        new_key_metas + i * need_associativity + j;
                    if (j < cur_associativity_) {
                        slot* cur_slot = key_metas_ + i * cur_associativity_ + j;
                        cur_new_slot->set_slot(*cur_slot);
                    } else {
                        cur_new_slot->set_slot(0, 0, 0, 0);
                    }
                }
            }

            // Switch the old key_metas to the new key_metas and release the old
            // key_metas
            delete[] key_metas_;
            key_metas_ = new_key_metas;

            int ret_slotid = cur_associativity_;
            // update current associativity
            cur_associativity_ = need_associativity;

            return ret_slotid;
        }

        /*---- 2. Cuckoo hash function ---*/
        /*---- 2.1 Helper function ---*/
        // Return previous slotid in current bucket. 
        inline int get_previous_slotid_in_same_bucket(int slotid) const {
            return slotid == 0 ? -1 : slotid - 1;
        }

        // Return the first empty slotid or last slotid in current bucket
        inline int get_kick_2_slotid_in_bucket(int bucketid) const {
            for (int i = 0; i != cur_associativity_; i++)
                if (get_slot(bucketid, i)->is_empty()) return i;

            return cur_associativity_ - 1;
        }

        // Return another possible bucketid that the slot *s can be at
        inline size_t get_another_bucketid(page_manager_agent& pm_agent, slot* s,
                                           size_t current_bucketid) const {
            const char* key = pm_agent.get_content_pointer(s);
            size_t bucketid1 = hash(key, s->get_length(), 1) % BUCKET_NUM;
            size_t bucketid2 = hash(key, s->get_length(), 2) % BUCKET_NUM;
            return current_bucketid == bucketid1 ? bucketid2 : bucketid1;
        }

        /*---- 2.2 Cuckoo hash function ---*/
        /**
         * @brief Work as normal cuckoo hashing function.
         *
         * Firstly, store the original key_metas in case that the cuckoo hash
         * function failed. Secondly, create a extra slot for the slot being
         * kicked out. Thirdly, repeately pick a slot to swap the extra slot
         * until we pick a empty slot that make the extra slot to be
         * empty(success) or the repeat time exceed the threshold(fail).
         *
         * @param bucketid Bucketid that needs a empty slot.
         * @param bt Bi-trie that manages 'this' node.
         * @return int A empty slot_id in bucketid.
         */
        int cuckoo_hash(size_t bucketid, bitrie* bt) {
            // Set up the backup for recovery if the cuckoo hash fail
            slot* key_metas_backup = new slot[BUCKET_NUM * cur_associativity_]();
            memcpy(key_metas_backup, key_metas_,
                   BUCKET_NUM * cur_associativity_ * sizeof(slot));

            // Slot index that prepare for returned
            int ret_index = -1;

            /*
             * key_metas:   | x | x | x | x |   extra_slot: | y |
             *              | x | x | x | x |   slot waited to be cuckoo-hashed
             *              | x | x | x | x |
             */
            slot* extra_slot = new slot(0, 0, 0, 0);

            /* cur_process_bucketid, cur_process_slotid indicate the extra_slot's destination */
            int cur_process_bucketid = bucketid;
            int cur_process_slotid = cur_associativity_ - 1;

            page_manager_agent pm_agent =
                bt->pm->get_page_manager_agent(normal_pgid_, special_pgid_);

            map<T, int> searchPoint_wait_2_be_update;
            for (int cuckoo_hash_time = 0; cuckoo_hash_time != BUCKET_NUM * ASSOCIATIVITY * DEFAULT_CUCKOO_HASH_RATIO;
                 cuckoo_hash_time++) {
                /*
                 * The get_previous_slotid_in_same_bucket() will cause the -1 if
                 * there aren't any available slot can be cuckoo hash
                 * If the cur_process_slotid equals to -1, it means that all
                 * elements in current bucket's is anti-moved
                 */
                if (cur_process_slotid == -1) break;

                // Get the slot* we are replacing destination
                int cur_process_index =
                    get_index(cur_process_bucketid, cur_process_slotid);
                slot* cur_process_slot = get_slot(cur_process_bucketid, cur_process_slotid);

                /* Check that whether the cur_process_slot is anti-moved */
                // Get the another bucketid the cur_process_slot can be at
                int cur_kick_to_bucketid = get_another_bucketid(
                    pm_agent, cur_process_slot, cur_process_bucketid);
                // If the cur_process_bucketid == cur_kick_to_bucketid, we
                // process previous slotid
                if (cur_process_bucketid == cur_kick_to_bucketid ||
                    cur_process_index == ret_index) {
                    cur_process_slotid =
                        get_previous_slotid_in_same_bucket(cur_process_slotid);
                    continue;
                }

                /* Checking work is done, executing the cuckoo hashing */
                // Swap the extra slot content with the current process slot
                extra_slot->swap(cur_process_slot);

                // Add this value=index for the searchPoint index updateing
                searchPoint_wait_2_be_update[pm_agent.get_value(cur_process_slot)] =
                    get_column_store_index(cur_process_slot);

                // The first time swap the extra_slot indicate the
                // cur_process_index is ret_index
                if (ret_index == -1) {
                    ret_index = cur_process_index;
                }

                // Cur_process_slot is a empty slot, cuckoo hash is done
                if (extra_slot->is_empty()) {
                    delete[] key_metas_backup;
                    delete extra_slot;

                    bt->apply_the_changed_search_point(
                        searchPoint_wait_2_be_update);

                    // Return slot_id
                    return ret_index % cur_associativity_;
                }

                // Update the current bucketid and slotid which are the
                // replacing destination in next iteration
                cur_process_bucketid = cur_kick_to_bucketid;
                cur_process_slotid = get_kick_2_slotid_in_bucket(cur_kick_to_bucketid);
            }

            // Recover the key_metas
            memcpy(key_metas_, key_metas_backup,
                   BUCKET_NUM * cur_associativity_ * sizeof(slot));

            delete[] key_metas_backup;
            delete extra_slot;

            // The cuckoo hash time exceeds Max_loop return -1 as slotid to
            // indicate cuckoo hash failed
            return -1;
        }

        /*---- Searching function ---*/
        /**
         * @brief Search the target element in specific bucket that match string(key, key_size).
         *
         * @param bucketid Searching bucket.
         * @param key Pointer of searching element's string.
         * @param key_size Size of searching element's string.
         * @param pm_agent Reference of page manager agent that contains current
         * hash node's needed page group.
         * @return found_result Found result of matching string(key, key_size)
         * in current bucket.
         */
        found_result find_in_bucket(size_t bucketid, const K_unit* key,
                                    size_t key_size, page_manager_agent& pm_agent) {
            for (int i = 0; i != cur_associativity_; i++) {
                slot* target_slot = get_slot(bucketid, i);
                if (target_slot->is_empty())
                    return found_result(false, T(), bucketid, i);

                if (key_equal(key, key_size, pm_agent.get_content_pointer(target_slot),
                            target_slot->get_length()))
                    return found_result(true, pm_agent.get_value(target_slot), bucketid, i);
            }
            return found_result(false, T(), bucketid, -1);
        }

        /**
         * @brief Search the target element in hash node that match string(key, key_size).
         * 
         * @param key Pointer of searching element's string.
         * @param key_size Size of searching element's string.
         * @param pm Reference of page manager agent that contains current
         * hash node's needed page group.
         * @return found_result Found result of matching string(key, key_size)
         * in current hash node.
         */
        found_result search_kv_in_hashnode(const K_unit* key, size_t key_size,
                                           page_manager* pm) {
            page_manager_agent pm_agent = 
                    pm->get_page_manager_agent(normal_pgid_, special_pgid_);
            // If found the existed target in bucket_id1 or bucket_id2, just
            // return the iterator for being modified or read
            size_t bucket_id1 = hash(key, key_size, 1) % BUCKET_NUM;
            found_result res1 = find_in_bucket(bucket_id1, key, key_size, pm_agent);
            if (res1.exist()) return res1;

            size_t bucket_id2 = hash(key, key_size, 2) % BUCKET_NUM;
            found_result res2 = find_in_bucket(bucket_id2, key, key_size, pm_agent);
            if (res2.exist()) return res2;

            // If the code reach here it means the target doesn't exist
            // We try to return the result that locating at a un-full bucket
            if (res1.is_bucket_full())
                return res2;
            else
                return res1;
        }

        /**
         * @brief Insert a element into a hash node.
         * 
         * @param key Pointer of inserting element's string.
         * @param key_size Size of inserting element's string.
         * @param bt Bi-trie that manages 'this' node.
         * @param v Value of searching element's string.
         * @param fr Found result that contains the inserting bucketid, slotid, existence information.
         */
        void insert_kv_in_hashnode(const K_unit* key, size_t key_size, bitrie* bt,
                                   T v, found_result fr) {
            size_t bucketid = fr.get_bucketid();
            int slotid = fr.get_slotid();
            page_manager_agent pm_agent = bt->pm->get_page_manager_agent(normal_pgid_, special_pgid_);

            /* 
             * If the slotid == -1, it means that we need extra empty slot in current bucketid
             * Our current strategy:    =(if slotid == -1)=>    1.dynamic expand 
             *                          =(if slotid == -1)=>    2.cuckoo hash 
             *                          =(if slotid == -1)=>    3.burst and re-insert
             * Also we can use cuckoo hash => dynamic expand => 
             * burst and re-insert alternatively for a better slot utilization rate 
             * but in-efficient(cuckoo hash too much)
             */
            bool must_burst = (slotid == -1) &&
                              (slotid = dynamic_expand()) == -1 &&
                              (slotid = cuckoo_hash(bucketid, bt)) == -1;

            if (must_burst) {
                const string& prefix = this->node::get_prefix();    

                // While the resize() in burst() will free the original key's memory, we store
                // the content of original key in temp_string
                string temp_string = string(key, key_size);

                // The replacing_node_of_this(trie_node) is the node replacing
                // this(hash_node) node after burst() in original trie. 
                trie_node* replacing_node_of_this =
                    bt->burst(burst_package(key_metas_, BUCKET_NUM,
                                            cur_associativity_, pm_agent),
                              this->node::get_parent(), prefix);

                // We can continue to insert the original element from
                // replacing_node_of_this after burst() instead of from t_root
                bt->insert_kv_in_bitrie(replacing_node_of_this,
                                        temp_string.data(), key_size, v,
                                        prefix.data(), prefix.size());
                delete this;
                return;
            }

            // Deal with the situation that current insert() is a element
            // modification instead of a element insert
            if (fr.exist()) {
              // The STR-ID will be re-written later at
              // pm_agent.insert_element()

              // The ID-STR in v2k is modifying
              bt->erase_element_in_v2k(fr.get_value());

            } else {
              // Only not modifying the element will the elem_num increase
              elem_num_++;
            }

            assert(slotid != -1 && slotid >= 0 && slotid < cur_associativity_);

            slot* target_slot = get_slot(bucketid, slotid);

            // Ask page_manager that whether there is a place for new element
            /*
             * if page_manager pm don't have enough memory, pm will take charge
             * of the resizeing, and update all hash_nodes' normal_pgid, special_pgid. 
             * And so the pm_agent.insert_element() below can write
             * the content to a updated page group*/
            /*
             * if page_manager pm have enough memory, the pm_agent.insert_element()
             *  will write the content to original page group
             */

            // In case, the key is invalidated in resize()
            if(!pm_agent.try_insert(key_size)) {
                // While the resize() will free the original key's memory, we store
                // the content of original key in temp_string
                string temp_string = string(key, key_size);

                bt->pm->resize(get_group_type(key_size), bt);
                pm_agent = bt->pm->get_page_manager_agent(normal_pgid_, special_pgid_);

                // Page manager agent will take charge of the element writing
                // work and return a slot with position that element been written
                target_slot->set_slot(
                    pm_agent.insert_element(temp_string.data(), key_size, v));

                // Set v2k
                bt->set_v2k(v, this, get_column_store_index(target_slot));

                return;
            }

            // Page manager agent will take charge of the element writing work
            // And return a slot with position that element been written
            target_slot->set_slot(pm_agent.insert_element(key, key_size, v));

            // Set v2k
            bt->set_v2k(v, this, get_column_store_index(target_slot));
            
            return;
        }
    };

    /**
     * @brief Bursting element temporarily stored class.
     * 
     * When a hash_node is about to burst, it will create a burst_package with
     * all its elements. And the burst() function will receive a burst_package
     * and do the burst work.
     *
     * This class provides helper function for burst():
     *      1. Element(slot) traverse: is_empty(), top(), pop()
     *      2. Elements' common prefix calculation
     *      3. Updating function for page_manager to call when page_manager
     *          resize is activated
     *      
     *
     * Burst_package is treated like a stack from out side using is_empty(),
     * pop(), top() to traverse its element. The way traverse burst_package like
     * a stack will eliminate the element rewrites in page_manager resize()
     * while the written elements are removed from elems_.
     */
    class burst_package {
    private:
        page_manager_agent pm_agent_;
        vector<slot> elems_;

    public:
        burst_package(slot* elems, size_t bucket_num,
                      size_t associativity, page_manager_agent pm_agent)
            : pm_agent_(pm_agent) {
            for (int i = 0; i != bucket_num; i++) {
                for (int j = 0; j != associativity; j++) {
                    slot* s = elems + i * associativity + j;
                    if (s->is_empty()) break;
                    elems_.push_back(*s);
                }
            }
        }

        /**
         * @brief Update the element in current burst package from old page
         * manager to new page manager.
         *
         * Invoked by page_manager's notify_burst_package() function when occur
         * a page_manager resize().
         * 
         * @param new_pm New page manager that current burst package's element moving to.
         * @param resize_type The moving type of the elements.
         */
        void update_burst_package(page_manager* new_pm, group_type resize_type) {
            size_t n_group_id = new_pm->require_group_id(group_type::NORMAL_GROUP);
            size_t s_group_id =new_pm->require_group_id(group_type::SPECIAL_GROUP);

            page_manager_agent new_pm_agent = new_pm->get_page_manager_agent(n_group_id, s_group_id);
            for (auto& s : elems_) {
                // ignore the slot that not belong to current resize group
                // type
                if (get_group_type(&s) != resize_type) continue;

                // get the content from old page_manager and write it to the
                // new page_manager
                s.set_slot(new_pm_agent.insert_element(pm_agent_.get_content_pointer(&s),
                                    s.get_length(), pm_agent_.get_value(&s)));
            }
            pm_agent_.set_page_group(resize_type, new_pm_agent.get_page_group(resize_type));
        }

        /*---- Get function ---*/
        inline page_manager_agent get_agent() const { return pm_agent_; }

        /*---- Element traverse function ---*/
        const slot top() const { return elems_.back(); }

        void pop() { elems_.pop_back(); }

        inline bool is_empty() const { return elems_.size() == 0; }

        /*---- Element common prefix calculation function ---*/
        inline unsigned int cal_common_prefix_len(
            const char* s1, unsigned int cur_longest_prefix_len, const char* s2,
            unsigned int new_key_size) const {
            if (cur_longest_prefix_len > new_key_size) {
                cur_longest_prefix_len = new_key_size;
            }
            for (int i = 0; i != cur_longest_prefix_len; i++) {
                    if (s1[i] != s2[i]) {
                    return i;
                }
            }
            return cur_longest_prefix_len;
        }

        // Return the common prefix between elements
        string get_common_prefix() const {
          const char* ret_key_pointer = nullptr;
          unsigned int common_prefix_len = INT_MAX;

          for (int i = 0; i != elems_.size() && common_prefix_len != 0; i++) {
                char* key = pm_agent_.get_content_pointer(&(elems_[i]));
                if (ret_key_pointer == nullptr) ret_key_pointer = key;

                // Update the common_prefix_len
                unsigned int cur_com_prefix_len =
                    cal_common_prefix_len(ret_key_pointer, common_prefix_len, key,
                                        elems_[i].get_length());
                if (common_prefix_len > cur_com_prefix_len)
                    common_prefix_len = cur_com_prefix_len;
          }
          return string(ret_key_pointer, common_prefix_len);
        }      

        /*---- Debug function ---*/
        void print_bp() {
            cout << "-----------------\n";
            for(int i=0; i!= elems_.size(); i++){
                elems_[i].print_slot(pm_agent_);
            }
            cout << endl;
        }
    };

    /**
     * @brief Burst the original hash node into a sub burst trie.
     * 
     * When the hash_node's element cannot dynamic-expand() and cuckoo-hash()
     * anymore, we burst the element in current hash_node into a burst-trie with
     * several small size of hash_node or trie_node linking with hash_nodes
     * children and return the burst-trie's root to let the caller place current
     * burst-trie in original burst-trie.
     *
     * Before:
     * hash_node(size:100)
     *
     *       burst()↓
     * 
     * After:
     *                  root: trie_node(size:0)
     *                   |          |         |
     *  |hash_node(size:25)|trie_node(size:1)|hash_node(size:25)|)
     *                              |
     *                      hash_node(size:49)
     * 
     * @param bp Burst package that contains the elements in original hash node.
     * @param orig_parent Parent of original hash node.
     * @param orig_prefix Prefix of original hash node.
     * @return trie_node* Sub burst trie's root after burst().
     */
    trie_node* burst(burst_package bp, trie_node* orig_parent, const string &orig_prefix) {
        // Add bp to page_manager's notify_list in case occur a page_manager resize()
        pm->register_burst_package(&bp);

        // The return header
        trie_node* ret_trie_root = new trie_node(orig_parent, orig_prefix.data(), orig_prefix.size());

        // Link the current node to trie: replace the orignal root or replace
        // the node in original parent
        if (orig_parent == nullptr)
            set_t_root(ret_trie_root);
        else
            orig_parent->add_child(orig_prefix.back(), ret_trie_root);

        // Get the common_prefix to eliminate the redundant-burst
        string common_prefix = bp.get_common_prefix();

        const char* common_prefix_key = common_prefix.data();
        const unsigned int common_prefix_key_size = common_prefix.size();

        // New prefix = prior prefix + common chain prefix
        string prefix = orig_prefix + common_prefix;

        // Create the common prefix trie chain with several single trie_node
        // The number of node is common_prefix_key_size
        trie_node* parent = ret_trie_root;
        for (int i = 0; i != common_prefix_key_size; i++) {
            trie_node* cur_trie_node =
                new trie_node(parent, prefix.data(),
                                prefix.size() - common_prefix_key_size + i + 1);
            parent->add_child(common_prefix_key[i], cur_trie_node);
            parent = cur_trie_node;
        }

        // Insert the elements truncated after common_prefix_len
        while (!bp.is_empty()) {
                slot s = bp.top();

                char* new_key = bp.get_agent().get_content_pointer(&s) + common_prefix_key_size;
                size_t length_left = s.get_length() - common_prefix_key_size;
                T v = bp.get_agent().get_value(&s);

                insert_kv_in_bitrie(parent, new_key, length_left, v, prefix.data(), prefix.size());
                bp.pop();
        }

        // Remove bp from notify list in page_manager while bp's element are done
        pm->remove_burst_package(&bp);

        return ret_trie_root;
    }


    /**
     * @brief Storage manager class.
     * 
     *
     * Page manager divides its storage into two part: Normal and Special
     * Each part contains several page group. Each page group contains several
     * page. Keys and values are placed in page like:
     * key0value0key1value1key2value2 Each hash_node only store elements in ONE
     * page group, denoted by normal_pgid and special_pgid in hash_node.
     *
     * Each element can be found by the slot's information:
     *      The is_special, length, pg_id, pos variables in slot will lead to a
     * location that store the element
     *
     * |=======================Page manager=======================|
     * |                             |                            |
     * | Normal page group:          | Special page group:        |
     * |                             |                            |
     * | |===Page group 0===|        | |===Page group 0===|       |
     * | |                  |        | |                  |       |
     * | | |=Page 0=|       |        | | |=Page 0=|       |       |
     * | | |        |       |        | | |        |       |       |
     * | | |  keys  | ...   | ...    | | |  keys  | ...   | ...   |
     * | | | values |       |        | | | values |       |       |
     * | | |========|       |        | | |========|       |       |
     * | |                  |        | |                  |       |
     * | |==================|        | |==================|       |
     * |                             |                            |
     * |==========================================================|
     *
     */
    class page_manager {
    public:
        class page_group {
            class page {
            private:
                friend class page_group;

                unsigned int cur_pos;
                char* content;
                
                inline static unsigned int calc_align(unsigned int n, unsigned align) {
                    return ((n + align - 1) & (~(align - 1)));
                }

            public:
                page() : cur_pos(0), content(nullptr) {}

                // Only call this function the page will start to allocate memory in content
                void init_page(size_t size_per_page) {
                    if (content == nullptr) { 
                        content = (char*)malloc(size_per_page); 
                    }
                }

                // Alignment controls whether we place element in different alignment
                void append_impl(const K_unit* key, size_t key_size, T& value,
                                 unsigned int alignment = 1) {
                    // Write the string
                    memcpy(content + cur_pos, key,
                                key_size * sizeof(K_unit));
                    // Write the value
                    memcpy(content + cur_pos + key_size * sizeof(K_unit),
                                &value, sizeof(T));
                    cur_pos += calc_align(key_size * sizeof(K_unit) + sizeof(T),
                                          alignment);
                }

                ~page() {
                    if (content != nullptr) {
                        free(content);
                        content = nullptr;
                    }
                }
            };

            page* pages;
            int cur_page_id;
            bool is_special;

        public:
            page_group() : pages(nullptr), cur_page_id(-1), is_special(false) {}

            // Only call this function the page group will start to allocate memory in pages
            void init_pg(group_type init_type) {
                is_special = init_type == group_type::SPECIAL_GROUP;
                cur_page_id = 0;
                pages = new page[is_special ? DEFAULT_SPECIAL_PAGE_NUMBER : DEFAULT_NORMAL_PAGE_NUMBER]();
                pages[0].init_page(is_special ? DEFAULT_SPECIAL_PAGE_SIZE
                                       : DEFAULT_NORMAL_PAGE_SIZE);
            }

            /*---- Get function ---*/
            inline char* get_content_pointer_in_page(const slot* const s) const {
                return pages[s->get_page_id()].content + s->get_pos();
            }

            inline T get_value_in_page(const slot* const s) const {
                T v;
                memcpy(&v, get_content_pointer_in_page(s) + s->get_length(),
                            sizeof(T));
                return v;
            }

            inline size_t get_cur_page_id() const {
                return cur_page_id;
            }

            inline size_t get_max_page_id() const {
                return is_special ? DEFAULT_SPECIAL_PAGE_NUMBER : DEFAULT_NORMAL_PAGE_NUMBER;
            }

            inline size_t get_max_per_page_size() const {
                return is_special ? DEFAULT_SPECIAL_PAGE_SIZE : DEFAULT_NORMAL_PAGE_SIZE;
            }

            /*---- Insert element function ---*/
            /**
             * @brief Try to insert a element to page manager.
             * 
             * Try to insert element to current page group and return availability.
             * 
             * @param try_insert_key_size Size of a try-inserted element.
             * @return true Page manager have enough space.
             * @return false Page manager is full.
             */
            bool try_insert(size_t try_insert_key_size) const {
                if (cur_page_id + 1 < get_max_page_id()) return true;
                if ((pages[cur_page_id].cur_pos +
                     try_insert_key_size * sizeof(K_unit) + sizeof(T)) <
                    get_max_per_page_size())
                    return true;
                return false;
            }

            /**
             * @brief Insert string(key, key_size) and value to page.
             * 
             * @param key Pointer of the inserted element's string.
             * @param key_size Size of the inserted element's string.
             * @param v Value of the inserted element.
             * @return slot The inserted location.
             */
            slot write_kv_to_page(const K_unit* key, size_t key_size, T v) {
                // Test whether the need_size is more than the left space
                // If yes, init new page, write key and value
                // If no, write key and value
                size_t need_size = key_size * sizeof(K_unit) + sizeof(T);

                if (pages[cur_page_id].cur_pos + need_size >
                    (is_special ? DEFAULT_SPECIAL_PAGE_SIZE
                                : DEFAULT_NORMAL_PAGE_SIZE)) {
                    cur_page_id++;
                    pages[cur_page_id].init_page(
                        is_special ? DEFAULT_SPECIAL_PAGE_SIZE
                                   : DEFAULT_NORMAL_PAGE_SIZE);
                }

                // get page being written
                page& target_page = pages[cur_page_id];

                // record position before updating and status modify
                slot ret_slot =
                    slot(is_special, key_size, target_page.cur_pos, cur_page_id);

                // write content
                target_page.append_impl(key, key_size, v,
                                        is_special ? DEFAULT_SPECIAL_ALIGNMENT
                                                   : DEFAULT_NORMAL_ALIGNMENT);

                return ret_slot;
            }

            ~page_group() {
                delete []pages;
                pages = nullptr;
            }

            /*---- External memory report interface ---*/
            uint64_t get_page_group_memory() {
                return (get_cur_page_id() + 1) * (is_special
                                                        ? DEFAULT_SPECIAL_PAGE_SIZE
                                                        : DEFAULT_NORMAL_PAGE_SIZE);
            }

        };
        
    private:
        page_group* normal_pg;
        page_group* special_pg;

        size_t n_size;
        size_t s_size;

    public:
        page_manager(size_t normal_page_group_number, size_t special_page_group_number)
            : normal_pg(new page_group[normal_page_group_number]),
              special_pg(new page_group[special_page_group_number]),
              n_size(0),
              s_size(0) {
            // Init normal page group according to the normal_page_group_number
            for (int i = 0; i != normal_page_group_number; i++)
                init_a_new_page_group(group_type::NORMAL_GROUP, i);

            // Init special page group according to the special_page_group_number
            for (int i = 0; i != special_page_group_number; i++)
                init_a_new_page_group(group_type::SPECIAL_GROUP, i);
        }

        ~page_manager() {
            delete []normal_pg;
            normal_pg = nullptr;
            delete []special_pg;
            special_pg = nullptr;
        }

        /**
         * @brief For hash_node to require a normal_pgid and special_pgid
         *
         * This function will check the usage of current page group to chose a
         * page group id to return in balance.
         *
         * @param gt The requiring group type(Normal or Special).
         * @return size_t A balanced page group id.
         */
        inline size_t require_group_id(group_type gt) {
            // Processing page_group
            size_t cur_size =
                (gt == group_type::NORMAL_GROUP ? n_size : s_size);
            page_group* cur_pgs =
                (gt == group_type::NORMAL_GROUP ? normal_pg : special_pg);

            size_t least_page_page_group_id = -1;
            size_t least_page = SIZE_MAX;

            // Return a page_group with least page number for load-balance,
            // reduce page_manager resize()
            for (int i = 0; i != cur_size; i++) {
                size_t cur_least_page = cur_pgs[i].get_cur_page_id();
                if (least_page > cur_least_page) {
                    least_page = cur_least_page;
                    least_page_page_group_id = i;
                }
            }
            return least_page_page_group_id;
        }

        // Return a page_manager agent to take charge of the element getting, writing
        inline page_manager_agent get_page_manager_agent(int n_pg,
                                                         int s_pg) {
            return page_manager_agent(n_pg == -1 ? nullptr : normal_pg + n_pg,
                                        s_pg == -1 ? nullptr : special_pg + s_pg);
        }


        /* 
         * Oberserver design pattern
         * page_manager is a Subject that have the function of register, remove,
         * notify(traverse_for_pgm_resize) those burst_package in burst()
         */
        vector<burst_package*> notify_list;

        /**
         * @brief Add a burst package into current page manager's notify list.
         * 
         * @param add_bp_ptr A burst package in a on-going burst() function.
         */
        void register_burst_package(burst_package *add_bp_ptr) {
            notify_list.push_back(add_bp_ptr);
        }

        /**
         * @brief Remove a burst package from current page manager's notify list.
         * 
         * @param rm_bp_ptr A burst package in a finished burst() function.
         */
        void remove_burst_package(const burst_package *const rm_bp_ptr){
            for(auto it = notify_list.begin(); it!=notify_list.end(); it++) {
                if( *it == rm_bp_ptr){ 
                    notify_list.erase(it);
                    return;
                }
            }
        }

        /**
         * @brief Update those burst packages in notify_list.
         * 
         * @param new_pm New page manager the element moved to.
         * @param resize_type Whether move the special or normal elements.
         */
        void notify_burst_package(page_manager *new_pm, group_type resize_type) {
            for (auto bp_ptr : notify_list) {
                bp_ptr->update_burst_package(new_pm, resize_type);
            }
        }

        /**
         * @brief Expand the page manager.
         * 
         * @param resize_type Whether move the special or normal elements.
         * @param bt Bi-trie that manages this page manager
         * @param expand_ratio The resizing ratio.
         */
        void resize(group_type resize_type, bitrie* bt, size_t expand_ratio = 1) {
            page_manager* new_pm;
            if (resize_type == group_type::SPECIAL_GROUP) {
                new_pm = new page_manager(0, s_size << expand_ratio);
            } else {
                new_pm = new page_manager(n_size << expand_ratio, 0);
            }

            // Try insert, if failed, we reallocate the page groups,
            // update the pgid in hashnodes and return
            node* root = bt->t_root;
            root->traverse_for_pgm_resize(this, new_pm, resize_type);

            // Notify the bursting burst_package that your elements have been
            // changed because of the resize
            notify_burst_package(new_pm, resize_type);

            // Old page_manager <=swap=> new page_manager
            swap(new_pm, resize_type);

            delete new_pm;
            return;
        }

        /*---- External memory report interface ---*/
        uint64_t get_page_manager_memory() {
            uint64_t total_page_manager_memory = 0;
            for (size_t i = 0; i != n_size; i++) {
                total_page_manager_memory += normal_pg[i].get_page_group_memory();
            }
            for (size_t i = 0; i != s_size; i++) {
                total_page_manager_memory += special_pg[i].get_page_group_memory();
            }
            return total_page_manager_memory;
        }

    private:
        void init_a_new_page_group(group_type init_type, size_t page_group_index) {
            if (init_type == group_type::SPECIAL_GROUP) {
                s_size++;
                special_pg[page_group_index].init_pg(init_type);
                return;
            } else if (init_type == group_type::NORMAL_GROUP) {
                n_size++;
                normal_pg[page_group_index].init_pg(init_type);
                return;
            } else {
                cout << "initing a undefined group type page group!" << endl;
                assert(false);
                exit(0);
                return;
            }
        }

        /*---- Set function ---*/
        void set_n_size(size_t new_n_size) { n_size = new_n_size; }

        void set_s_size(size_t new_s_size) { s_size = new_s_size; }

        void set_normal_pg(page_group* temp_normal_pg) {
            normal_pg = temp_normal_pg;
        }

        void set_special_pg(page_group* temp_special_pg) {
            special_pg = temp_special_pg;
        }

        /*---- Single swap function ---*/
        void swap(page_manager* pm, group_type gt) {
            /*---- Normal part swap ---*/
            // swap the normal page group
            if (gt == group_type::NORMAL_GROUP) {
                page_group* temp_normal_pg = normal_pg;
                normal_pg = pm->normal_pg;
                pm->set_normal_pg(temp_normal_pg);

                // swap the n_size
                int temp_n_size = n_size;
                set_n_size(pm->n_size);
                pm->set_n_size(temp_n_size);
                return;
            } else {
                /*---- Special part swap ---*/
                // swap the special page group
                page_group* temp_special_pg = special_pg;
                special_pg = pm->special_pg;
                pm->set_special_pg(temp_special_pg);

                // swap the s_size
                int temp_s_size = s_size;
                set_s_size(pm->s_size);
                pm->set_s_size(temp_s_size);
                return;
            }
            cout << "swapping undefined group type!" << endl;
            assert(false);
            exit(0);
            return;
        }

        /*---- Double swap function ---*/
        void swap(page_manager *pm){
            swap(pm, group_type::NORMAL_GROUP);
            swap(pm, group_type::SPECIAL_GROUP);
        }
    };

    /**
     * @brief Helper class for hash_node get its page_groups.
     * 
     */
    class page_manager_agent {
    private:
        typename page_manager::page_group* n_group;
        typename page_manager::page_group* s_group;

        public:
        page_manager_agent(typename page_manager::page_group* ng,
                        typename page_manager::page_group* sg)
            : n_group(ng), s_group(sg) {}

        /*---- Set function ---*/
        inline void set_page_group(const group_type get_type,
                typename page_manager::page_group *const update_page_group) {
            get_type == group_type::SPECIAL_GROUP ? (s_group = update_page_group)
                                                : (n_group = update_page_group);
        }

        /*---- Get function ---*/
        inline typename page_manager::page_group* get_page_group(slot* s) const {
            return get_group_type(s->get_length()) == group_type::SPECIAL_GROUP
                    ? s_group
                    : n_group;
        }

        inline typename page_manager::page_group* get_page_group(const group_type get_type) const {
            return get_type == group_type::SPECIAL_GROUP ? s_group : n_group;
        }

        inline char* get_content_pointer(const slot* const s) const {
            return s->is_special() ? s_group->get_content_pointer_in_page(s)
                                : n_group->get_content_pointer_in_page(s);
        }

        inline T get_value(const slot* const s) const {
            return s->is_special() ? s_group->get_value_in_page(s)
                                : n_group->get_value_in_page(s);
        }

        /**
         * @brief Try to insert element to its right group and return availability.
         * 
         * @param key_size Length of element being tried to insert in page group.
         * @return true If the page group have enough space.
         * @return false Otherwise.
         */
        inline bool try_insert(const size_t key_size) const {
            return get_group_type(key_size) == group_type::SPECIAL_GROUP
                    ? s_group->try_insert(key_size)
                    : n_group->try_insert(key_size);
        }

        /**
         * @brief Insert element to its right group and return the slot(position).
         * 
         * @param key Pointer of element being inserted in page group.
         * @param key_size Length of element being inserted in page group.
         * @param v Value of element being inserted in page group.
         * @return slot Slot that encoding the storing location of inserted element.
         */
        inline slot insert_element(const K_unit* key, const size_t key_size, const T v) {
            return get_group_type(key_size) == group_type::SPECIAL_GROUP
                    ? s_group->write_kv_to_page(key, key_size, v)
                    : n_group->write_kv_to_page(key, key_size, v);
        }
    };

    /**
     * @brief String recovery class
     * 
     * Use a search point, we can recover the complete string from the bi-trie.
     * A search point contains a node and a index, denoting the location that a
     * element been stored.
     */
    class search_point {
       private:
        node* target_node_;
        int index_;

       public:
        search_point() : target_node_(nullptr), index_(-1) {}
        search_point(node* target_node, int index) : target_node_(target_node), index_(index) {}

        void set_index(int index) { index_ = index; }

        string get_string(page_manager* pm) const {
            if (target_node_ == nullptr) return string();

            // Static buffer for content filling
            static char * buffer = (char*)malloc(DEFAULT_SPECIAL_PAGE_SIZE);

            // Fill the prefix string content
            size_t length = target_node_->get_prefix().size();
            memcpy(buffer, target_node_->get_prefix().data(), length);

            // Get the suffix string
            if (index_ != -1) {
                // Get the page_manager_agent
                hash_node* hnode = (hash_node*)target_node_;
                page_manager_agent pm_agent = pm->get_page_manager_agent(
                    hnode->get_normal_pgid(), hnode->get_special_pgid());
                    
                // Get stored location(slot)
                slot* sl = hnode->get_column_store_slot(index_);

                // Fill the suffix string content
                memcpy(buffer + length, pm_agent.get_content_pointer(sl), sl->get_length());
                length += sl->get_length();
            }
            return string(buffer, length);
        }
    };

    /**
     * @brief Found result recording class
     * 
     * Record the information that a element existence, location(bucketid,
     * slotid), value
     * 
     */
    class found_result {
        bool found;
        T v;
        size_t bucketid;
        int slotid;

    public:
        found_result(bool f, T vv, size_t bid, int sid)
            : found(f), v(vv), bucketid(bid), slotid(sid) {}

        bool exist() const { return found; }

        bool is_bucket_full() const { return slotid == -1; }

        T get_value() const { return v; }

        size_t get_bucketid() const { return bucketid; }

        int get_slotid() const { return slotid; }
    };

    /**
     * @brief Insert a element into 'this' bi-trie.
     * 
     * @param start_node The node we start to find the inserting node.
     * @param key Pointer of the inserting element's string.
     * @param key_size Size of the inserting element's string.
     * @param v Value of the inserting element.
     * @param prefix_key Prefix pointer of the inserting element's string before start_node.
     * @param prefix_key_size Prefix size of the inserting element's string before start_node.
     */
    void insert_kv_in_bitrie(node* start_node, const K_unit* key,
                             size_t key_size, T v,
                             const K_unit* prefix_key = nullptr,
                             size_t prefix_key_size = 0) {
        node* current_node = start_node;

        // The pos update is moved to find_trie_node_child(fast-path or
        // non-fast-path way) while the pos increment
        for (size_t ref_pos = 0; ref_pos < key_size;) {
            switch (current_node->get_node_type()) {
                case node_type::TRIE_NODE: {
                    trie_node* orig_tnode = (trie_node*)current_node;
                    // Return the hitted trie_node* or create a new
                    // trie_node with a hash_node son
                    current_node = orig_tnode->find_trie_node_child(key, ref_pos, key_size, this);

                    if(current_node == nullptr){
                        string new_prefix = string(prefix_key, prefix_key_size) + string(key, ref_pos);
                        // Create a corresponding hash_node and add it to current
                        // trie_node's child representation
                        current_node =
                            new hash_node(orig_tnode, new_prefix, pm);
                        orig_tnode->add_child(key[ref_pos - 1], current_node);
                    } 

                } break;
                case node_type::HASH_NODE: {
                    hash_node* hnode = (hash_node*)current_node;
                    found_result res = hnode->search_kv_in_hashnode(key + ref_pos,
                                                            key_size - ref_pos, pm);
                    hnode->insert_kv_in_hashnode(key + ref_pos, key_size - ref_pos, this, v, res);
                    return;
                } break;
                default:{
                    cout << "wrong type!";
                    exit(0);
                    return;
                }
            }
        }

        current_node->insert_value_in_node(string(prefix_key, prefix_key_size) + string(key, key_size), v, this);
        return;
    }

    /**
     * @brief Find a element in 'this' bi-trie.
     * 
     * @param start_node The node we start to find the inserting node.
     * @param key Pointer of the searching element's string.
     * @param key_size Size of the searching element's string.
     * @return found_result Found result that contains the element's bucketid, slotid, existence and value.
     */
    found_result search_kv_in_bitrie(node* start_node, const K_unit* key, size_t key_size) const {
        node* current_node = start_node;

        // The pos update is moved to find_trie_node_child(fast-path or
        // non-fast-path way) while the pos increment
        for (size_t ref_pos = 0; ref_pos < key_size;) {
            switch (current_node->get_node_type()) {
                case node_type::TRIE_NODE: {
                    // return the hitted trie_node* or create a new
                    // trie_node with a hash_node son
                    current_node = ((trie_node*)current_node)->find_trie_node_child(key, ref_pos, key_size, this);

                    if(current_node == nullptr)
                        return found_result(false, T(), -1, -1);
                } break;
                case node_type::HASH_NODE: {
                    hash_node* hnode = (hash_node*)current_node;
                    return hnode->search_kv_in_hashnode(key + ref_pos,
                                                            key_size - ref_pos, pm);
                } break;
                default:
                    cout << "wrong type!";
                    exit(0);
            }
        }

        // Find a key in node's only value
        return current_node->search_kv_in_node();
    }

    /*---- Set function ---*/
    void set_t_root(node* node) { t_root = node; }

    void set_search_point_index(T v, int index) { 
        v2k[v].set_index(index); 
    }

    void set_v2k(T v, node* node, int index) {
        v2k[v] = search_point(node, index);
    }

    void erase_element_in_v2k(T v) {
        v2k.erase(v2k.find(v));
    }

    /**
     * @brief Batch updating the search_points to v2k
     *
     * This function will be called in cuckoo_hash() while the cuckoo hashing
     * will modify the key_metas's layout that modifies the index of elements.
     * So in cuckoo_hash(), we store the modified elements' index in a map and batch
     * update the modified elements' index here.
     *
     * @param search_points Contain the modified elements' value and index.
     */
    void apply_the_changed_search_point(map<T, int>& search_points) {
        for (auto it = search_points.begin(); it != search_points.end(); it++)
            set_search_point_index(it->first, it->second);
    }

    boost::unordered_map<T, search_point> v2k;
    page_manager *pm;
    node* t_root;

public:
    bitrie():pm(new page_manager(1, 1)),
                t_root(new hash_node(nullptr, string(), pm)) { }

    // Deconstructor
    ~bitrie() {
        t_root->delete_me();
        delete pm;
    }

    /*---- External accessing interface ---*/
    /**
     * @brief Access specified element by value(T).
     * 
     * @param v The value of the element to find.
     * @return string The key of the element to find.
     */
    string operator[](T v) { return v2k[v].get_string(pm); }

    /**
     * @brief Access specified element by key(string).
     * 
     * @param key_string The key of the element to find.
     * @return T The value of the element to find.
     */
    T operator[](const string& key_string) const {
        return search_kv_in_bitrie(t_root, key_string.data(), key_string.size())
            .get_value();
    }

    /**
     * @brief Checks if the bi-trie have the element that value equals to v.
     * 
     * @param v Checking target element's value.
     * @return true Target element exist.
     * @return false Target element doesn't exist.
     */
    bool exist(T v) const { return v2k.find(v) != v2k.end(); }

    /**
     * @brief Checks if the bi-trie have the element that key equals to key_string.
     * 
     * @param key_string Checking target element's key.
     * @return true Target element exist.
     * @return false Target element doesn't exist.
     */
    bool exist(const string& key_string) const {
        return search_kv_in_bitrie(t_root, key_string.data(), key_string.size())
            .exist();
    }

    // insert operation
    /**
     * @brief Insert a element into bi-trie.
     * 
     * @param key_string Inserting element's key.
     * @param v Inserting element's value.
     */
    void insert_kv(const string& key_string, T v) {
        insert_kv_in_bitrie(t_root, key_string.data(), key_string.size(), v);
        return;
    }

    /*---- External cleaning interface ---*/
    /**
     * @brief Shrink-to-fit bi-trie's storage.
     * 
     */
    void storage_resize() {
        // zero at last means that we don't need to expand the page_manager
        pm->resize(group_type ::NORMAL_GROUP, this, 0);
        pm->resize(group_type ::SPECIAL_GROUP, this, 0);
    }
};