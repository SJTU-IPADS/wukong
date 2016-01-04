graph_store:
	/usr/bin/mpic++ --std=c++11 -I${BOOST_INCLUDE} -fopenmp test_graph_distributed.cpp rdma_lib.cc global_cfg.cpp \
	${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a -o test_graph_distributed.out -lzmq -lrt -libverbs -ltbb 
trinity_simulate_test:
	/usr/bin/mpic++ --std=c++11 -I${BOOST_INCLUDE} -fopenmp trinity_simulate_test.cpp rdma_lib.cc global_cfg.cpp \
	${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a -o trinity_simulate_test.out -lzmq -lrt -libverbs -ltbb 
clean:
	rm *.out; rm *.o