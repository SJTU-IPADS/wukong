#pragma once

class mymath{ //math will conflict with other lib; so it's named mymath
public:
    inline uint64_t static floor(uint64_t original,uint64_t n){
		if(n==0){
			assert(false);
		}
		if(original%n == 0){
			return original;
		}
		return original - original%n;
	}
    inline uint64_t static hash_mod(uint64_t n,uint64_t m){
        if(m==0){
            assert(false);
        }
        return n%m;
    }
    static uint64_t hash(uint64_t key){
		key = (~key) + (key << 21); // key = (key << 21) - key - 1;
		key = key ^ (key >> 24);
		key = (key + (key << 3)) + (key << 8); // key * 265
		key = key ^ (key >> 14);
		key = (key + (key << 2)) + (key << 4); // key * 21
		key = key ^ (key >> 28);
		key = key + (key << 31);
		return key;
	}


};
