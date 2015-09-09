test:
	mpic++ --std=c++11 -I${BOOST_INCLUDE}  test_traverser_keeppath.cpp rdma_lib.cc \
	${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a -o test_traverser_keeppath.out -lzmq -lrt -libverbs
hello:
	mpic++ -I${BOOST_INCLUDE} -L${BOOST_LIB} hello.cpp -lboost_mpi -lboost_serialization  -o hello.out
	
clean:
	rm *.out; rm *.o