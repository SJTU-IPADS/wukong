#include "old_graph_storage.h"

old_graph_storage::old_graph_storage() {
	pthread_spin_init(&allocation_lock, 0);
	for (int i = 0; i < num_locks; i++) {
		pthread_spin_init(&fine_grain_locks[i], 0);
	}
};

void old_graph_storage::init(RdmaResource* _rdma, uint64_t machine_num, uint64_t machine_id) {
	rdma        = _rdma;
	m_num       = machine_num;
	m_id        = machine_id;
	slot_num    = 1000000 * global_hash_header_million;
	header_num	= (slot_num / cluster_size) / indirect_ratio * (indirect_ratio - 1);
	indirect_num = (slot_num / cluster_size) / indirect_ratio;

	vertex_addr	= (vertex*)(rdma->get_buffer());
	edge_addr	= (edge*)(rdma->get_buffer() + slot_num * sizeof(vertex));

	if (rdma->get_memorystore_size() <= slot_num * sizeof(vertex)) {
		std::cout << "No enough memory to store edge" << std::endl;
		exit(-1);
	}
	max_edge_ptr = (rdma->get_memorystore_size() - slot_num * sizeof(vertex)) / sizeof(edge);
	new_edge_ptr = 0;

	#pragma omp parallel for num_threads(20)
	for (uint64_t i = 0; i < slot_num; i++) {
		vertex_addr[i].key = local_key();
	}
	// if(global_use_loc_cache){
	// 	assert(false);
	// }
}
uint64_t old_graph_storage::insertKey(local_key key) {
	uint64_t vertex_ptr;
	uint64_t bucket_id = key.hash() % header_num;
	uint64_t lock_id = bucket_id % num_locks;
	uint64_t slot_id = 0;
	bool found = false;
	pthread_spin_lock(&fine_grain_locks[lock_id]);
	//last slot is used as next pointer
	while (!found) {
		for (uint64_t i = 0; i < cluster_size - 1; i++) {
			slot_id = bucket_id * cluster_size + i;
			if (vertex_addr[slot_id].key == key) {
				cout << "inserting duplicate key" << endl;
				assert(false);
			}
			if (vertex_addr[slot_id].key == local_key()) {
				vertex_addr[slot_id].key = key;
				found = true;
				break;
			}
		}
		if (found) {
			break;
		} else {
			slot_id = bucket_id * cluster_size + cluster_size - 1;
			if (vertex_addr[slot_id].key != local_key()) {
				bucket_id = vertex_addr[slot_id].key.vid;
				//continue and jump to next bucket
				continue;
			} else {
				pthread_spin_lock(&allocation_lock);
				if (used_indirect_num >= indirect_num) {
					assert(false);
				}
				vertex_addr[slot_id].key.vid = header_num + used_indirect_num;
				used_indirect_num++;
				pthread_spin_unlock(&allocation_lock);
				bucket_id = vertex_addr[slot_id].key.vid;
				slot_id = bucket_id * cluster_size + 0;
				vertex_addr[slot_id].key = key;
				//break the while loop since we successfully insert
				break;
			}
		}
	}
	pthread_spin_unlock(&fine_grain_locks[lock_id]);
	assert(vertex_addr[slot_id].key == key);
	return slot_id;
}
uint64_t old_graph_storage::atomic_alloc_edges(uint64_t num_edge) {
	uint64_t curr_edge_ptr;
	pthread_spin_lock(&allocation_lock);
	curr_edge_ptr = new_edge_ptr;
	new_edge_ptr += num_edge;
	pthread_spin_unlock(&allocation_lock);
	if (new_edge_ptr >= max_edge_ptr) {
		cout << "atomic_alloc_edges out of memory !!!! " << endl;
		exit(-1);
	}
	return curr_edge_ptr;
}

void old_graph_storage::atomic_batch_insert(vector<edge_triple>& vec_spo, vector<edge_triple>& vec_ops) {

	uint64_t curr_edge_ptr = atomic_alloc_edges(2 * (vec_spo.size() + vec_ops.size()));
	uint64_t total_size = 2 * (vec_spo.size() + vec_ops.size());
	uint64_t curr_edge_ptr_old = curr_edge_ptr;
	uint64_t start;
	start = 0;
	while (start < vec_spo.size()) {
		uint64_t end = start + 1;
		while (end < vec_spo.size()
		        && vec_spo[start].s == vec_spo[end].s
		      ) {
			end++;
		}
		local_key key = local_key(vec_spo[start].s, OUT, 1);
		uint64_t vertex_ptr = insertKey(key);
		local_val val = local_val(2 * (end - start), curr_edge_ptr);
		vertex_addr[vertex_ptr].val = val;
		for (uint64_t i = start; i < end; i++) {
			edge_addr[curr_edge_ptr].val = vec_spo[i].p;
			edge_addr[curr_edge_ptr + (end - start) ].val = vec_spo[i].o;
			curr_edge_ptr++;
		}
		curr_edge_ptr += (end - start);
		start = end;
	}

	start = 0;
	while (start < vec_ops.size()) {
		uint64_t end = start + 1;
		while (end < vec_ops.size()
		        && vec_ops[start].o == vec_ops[end].o
		      ) {
			end++;
		}
		local_key key = local_key(vec_ops[start].o, IN, 1);
		uint64_t vertex_ptr = insertKey(key);
		local_val val = local_val(2 * (end - start), curr_edge_ptr);
		vertex_addr[vertex_ptr].val = val;
		for (uint64_t i = start; i < end; i++) {
			edge_addr[curr_edge_ptr].val = vec_ops[i].p;
			edge_addr[curr_edge_ptr + (end - start)].val = vec_ops[i].s;
			curr_edge_ptr++;
		}
		curr_edge_ptr += (end - start);
		start = end;
	}
	assert(curr_edge_ptr_old + total_size == curr_edge_ptr);
}
void old_graph_storage::print_memory_usage() {
	uint64_t used_header_slot = 0;
	for (int x = 0; x < header_num + indirect_num; x++) {
		for (int y = 0; y < cluster_size - 1; y++) {
			uint64_t i = x * cluster_size + y;
			if (vertex_addr[i].key == local_key()) {
				//empty slot, skip it
				continue;
			}
			used_header_slot++;
		}
	}
	cout << "graph_storage direct_header= " << header_num*cluster_size*sizeof(vertex) / 1048576 << " MB " << endl;
	cout << "                  real_data= " << used_header_slot*sizeof(vertex) / 1048576 << " MB " << endl;
	cout << "                   next_ptr= " << header_num*sizeof(vertex) / 1048576 << " MB " << endl;
	cout << "                 empty_slot= " << (header_num * cluster_size - header_num - used_header_slot)
	     *sizeof(vertex) / 1048576 << " MB " << endl;

	uint64_t used_indirect_slot = 0;
	uint64_t used_indirect_bucket = 0;
	for (int x = header_num; x < header_num + indirect_num; x++) {
		bool all_empty = true;
		for (int y = 0; y < cluster_size - 1; y++) {
			uint64_t i = x * cluster_size + y;
			if (vertex_addr[i].key == local_key()) {
				//empty slot, skip it
				continue;
			}
			all_empty = false;
			used_indirect_slot++;
		}
		if (!all_empty) {
			used_indirect_bucket++;
		}
	}
	cout << "graph_storage indirect_header= " << indirect_num*cluster_size*sizeof(vertex) / 1048576 << " MB " << endl;
	cout << "               not_empty_data= " << used_indirect_bucket*cluster_size*sizeof(vertex) / 1048576 << " MB " << endl;
	cout << "                    real_data= " << used_indirect_slot*sizeof(vertex) / 1048576 << " MB " << endl;


	cout << "graph_storage use " << used_indirect_num << " / " << indirect_num 	<< " indirect_num" << endl;
	cout << "graph_storage use " << slot_num*sizeof(vertex) / 1048576 << " MB for vertex data" << endl;

	cout << "graph_storage edge_data= " << new_edge_ptr*sizeof(edge) / 1048576 << "/"
	     << max_edge_ptr*sizeof(edge) / 1048576 << " MB " << endl;
}

vertex old_graph_storage::get_vertex_local(local_key key) {
	uint64_t bucket_id = key.hash() % header_num;
	while (true) {
		for (uint64_t i = 0; i < cluster_size; i++) {
			uint64_t slot_id = bucket_id * cluster_size + i;
			if (i < cluster_size - 1) {
				//data part
				if (vertex_addr[slot_id].key == key) {
					//we found it
					return vertex_addr[slot_id];
				}
			} else {
				if (vertex_addr[slot_id].key != local_key()) {
					//next pointer
					bucket_id = vertex_addr[slot_id].key.vid;
					//break from for loop, will go to next bucket
					break;
				} else {
					return vertex();
				}
			}
		}
	}
}
vertex old_graph_storage::get_vertex_remote(int tid, local_key key) {
	char *local_buffer = rdma->GetMsgAddr(tid);
	uint64_t bucket_id = key.hash() % header_num;
	vertex ret;
	if (rdmacache.lookup(key, ret)) {
		return ret;
	}
	while (true) {
		uint64_t start_addr = sizeof(vertex) * bucket_id * cluster_size;
		uint64_t read_length = sizeof(vertex) * cluster_size;
		rdma->RdmaRead(tid, mymath::hash_mod(key.vid, m_num), (char *)local_buffer, read_length, start_addr);
		vertex* ptr = (vertex*)local_buffer;
		for (uint64_t i = 0; i < cluster_size; i++) {
			if (i < cluster_size - 1) {
				if (ptr[i].key == key) {
					//we found it
					rdmacache.insert(ptr[i]);
					return ptr[i];
				}
			} else {
				if (ptr[i].key != local_key()) {
					//next pointer
					bucket_id = ptr[i].key.vid;
					//break from for loop, will go to next bucket
					break;
				} else {
					return vertex();
				}
			}
		}
	}
}

edge* old_graph_storage::get_edges_global(int tid, uint64_t id, int direction, int predict, int* size) {
	if ( mymath::hash_mod(id, m_num) == m_id) {
		return get_edges_local(tid, id, direction, predict, size);
	}
	local_key key = local_key(id, direction, 1);
	vertex v = get_vertex_remote(tid, key);
	if (v.key == local_key()) {
		*size = 0;
		return NULL;
	}
	char *local_buffer = rdma->GetMsgAddr(tid);
	uint64_t start_addr  = sizeof(vertex) * slot_num + sizeof(edge) * (v.val.ptr);
	uint64_t read_length = sizeof(edge) * v.val.size;
	rdma->RdmaRead(tid, mymath::hash_mod(id, m_num), (char *)local_buffer, read_length, start_addr);
	edge* result_ptr = (edge*)local_buffer;
	//*size=v.val.size;
	int half_size = v.val.size / 2;
	int begin = -1;
	int real_size = 0;
	for (int i = 0; i < half_size; i++) {
		if (result_ptr[i].val == predict && begin == -1) {
			begin = i;
		}
		if (result_ptr[i].val == predict) {
			real_size++;
		}
	}
	result_ptr = result_ptr + half_size + begin;
	*size = real_size;
	return result_ptr;
}
edge* old_graph_storage::get_edges_local(int tid, uint64_t id, int direction, int predict, int* size) {
	assert(mymath::hash_mod(id, m_num) == m_id ||  is_pid(id));
	local_key key = local_key(id, direction, 1);
	vertex v = get_vertex_local(key);
	if (v.key == local_key()) {
		*size = 0;
		return NULL;
	}
	//*size=v.val.size;
	//uint64_t ptr=v.val.ptr;

	edge* result_ptr = (edge*)(&(edge_addr[v.val.ptr]));
	int half_size = v.val.size / 2;
	int begin = -1;
	int real_size = 0;
	for (int i = 0; i < half_size; i++) {
		if (result_ptr[i].val == predict && begin == -1) {
			begin = i;
		}
		if (result_ptr[i].val == predict) {
			real_size++;
		}
	}
	result_ptr = result_ptr + half_size + begin;
	*size = real_size;
	return result_ptr;
}

void old_graph_storage::insert_vector(tbb_vector_table& table, uint64_t index_id, uint64_t value_id) {
	tbb_vector_table::accessor a;
	table.insert(a, index_id);
	a->second.push_back(value_id);
}

void old_graph_storage::init_index_table() {

	uint64_t t1 = timer::get_usec();

	#pragma omp parallel for num_threads(8)
	for (int x = 0; x < header_num + indirect_num; x++) {
		for (int y = 0; y < cluster_size - 1; y++) {
			uint64_t i = x * cluster_size + y;
			if (vertex_addr[i].key == local_key()) {
				//empty slot, skip it
				continue;
			}
			uint64_t vid = vertex_addr[i].key.vid;
			if (vertex_addr[i].key.dir == IN) {
				//TODO
				//only use type_index now
			} else {
				uint64_t size = vertex_addr[i].val.size;
				uint64_t edge_ptr = vertex_addr[i].val.ptr;
				for (int j = 0; j < size / 2; j++) {
					if (edge_addr[edge_ptr + j].val == global_rdftype_id) {
						insert_vector(type_table, edge_addr[edge_ptr + j + size / 2].val, vid);
					}
				}
			}
		}
	}
	uint64_t t2 = timer::get_usec();

	for ( tbb_vector_table::iterator i = type_table.begin(); i != type_table.end(); ++i ) {
		uint64_t curr_edge_ptr = atomic_alloc_edges(i->second.size());
		local_key key = local_key(i->first, IN, 0);
		uint64_t vertex_ptr = insertKey(key);
		local_val val = local_val(i->second.size(), curr_edge_ptr);
		vertex_addr[vertex_ptr].val = val;
		for (uint64_t k = 0; k < i->second.size(); k++) {
			edge_addr[curr_edge_ptr].val = i->second[k];
			curr_edge_ptr++;
			type_index_edge_num++;
		}
	}
	for ( tbb_vector_table::iterator i = src_predict_table.begin(); i != src_predict_table.end(); ++i ) {
		uint64_t curr_edge_ptr = atomic_alloc_edges(i->second.size());
		local_key key = local_key(i->first, IN, 0);
		uint64_t vertex_ptr = insertKey(key);
		local_val val = local_val(i->second.size(), curr_edge_ptr);
		vertex_addr[vertex_ptr].val = val;
		for (uint64_t k = 0; k < i->second.size(); k++) {
			edge_addr[curr_edge_ptr].val = i->second[k];
			curr_edge_ptr++;
			predict_index_edge_num++;
		}
	}
	for ( tbb_vector_table::iterator i = dst_predict_table.begin(); i != dst_predict_table.end(); ++i ) {
		uint64_t curr_edge_ptr = atomic_alloc_edges(i->second.size());
		local_key key = local_key(i->first, OUT, 0);
		uint64_t vertex_ptr = insertKey(key);
		local_val val = local_val(i->second.size(), curr_edge_ptr);
		vertex_addr[vertex_ptr].val = val;
		for (uint64_t k = 0; k < i->second.size(); k++) {
			edge_addr[curr_edge_ptr].val = i->second[k];
			curr_edge_ptr++;
			predict_index_edge_num++;
		}
	}
	tbb_vector_table().swap(src_predict_table);
	tbb_vector_table().swap(dst_predict_table);
	uint64_t t3 = timer::get_usec();
	cout << (t2 - t1) / 1000 << " ms for parallel generate tbb_table " << endl;
	cout << (t3 - t2) / 1000 << " ms for sequence insert tbb_table to graph_storage" << endl;

}

edge* old_graph_storage::get_index_edges_local(int tid, uint64_t index_id, int direction, int* size) {
	local_key key = local_key(index_id, direction, 0);
	vertex v = get_vertex_local(key);
	if (v.key == local_key()) {
		*size = 0;
		return NULL;
	}
	*size = v.val.size;
	edge* result_ptr = (edge*)(&(edge_addr[v.val.ptr]));
	return result_ptr;

//    * size=0;
//    return NULL;
	//return get_edges_local(tid,index_id,direction,0,size);
};
