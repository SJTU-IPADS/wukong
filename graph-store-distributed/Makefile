graph_store:
	mpic++ --std=c++11 -I${BOOST_INCLUDE}  test_traverser_keeppath.cpp rdma_lib.cc global_cfg.cpp \
	${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a -o test_traverser_keeppath.out -lzmq -lrt -libverbs
clean:
	rm *.out; rm *.o