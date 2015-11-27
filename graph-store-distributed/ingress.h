#pragma once 
#include <global_cfg.h>

class ingress{

public:
	static int* mid_table;
	static uint64_t size;
	static void create_table(uint64_t size){
		mid_table = new int[size];
		for(uint64_t i=0;i<size;i++){
			mid_table[i]=-1;
		}
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
	static int vid2mid(uint64_t vid,int m_num){
		return vid%m_num;
	}

};