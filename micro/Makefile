micro:
	mpic++ --std=c++11 -I${BOOST_INCLUDE} micro_bench.cpp ../graph-store-distributed/rdma_lib.cc \
	${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a -o micro.out -lzmq -lrt -libverbs
verb:
	mpic++ --std=c++11 -I${BOOST_INCLUDE} micro_verb.cpp ../graph-store-distributed/rdma_lib.cc \
	${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a -o verb.out -lzmq -lrt -libverbs
hello:
	mpic++ -I${BOOST_INCLUDE} -L${BOOST_LIB} hello.cpp -lboost_mpi -lboost_serialization  -o hello.out
