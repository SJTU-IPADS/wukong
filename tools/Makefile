graph_distributed:
	cd ../graph-store-distributed ; make graph_store ; cd ../tools
micro_micro:
	cd ../micro/;make micro;cd ../tools
micro_verb:
	cd ../micro/;make verb;cd ../tools
index_server:
	g++ -std=c++11 index_server.cpp -o index_server.out
minimal_index_server:
	g++ -std=c++11 minimal_index_server.cpp -o minimal_index_server.out
semantic_index_server:
	g++ -std=c++11 semantic_index_server.cpp -o semantic_index_server
convert_format:
	g++ -std=c++11 convert_format.cpp -o convert_format.out
clean:
	rm *.out