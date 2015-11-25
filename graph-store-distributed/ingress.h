#pragma once 
#include <global_cfg.h>

class ingress{

public:
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
	static int vid2mid(uint64_t vid,int m_num){
		return vid%m_num;
	}

};