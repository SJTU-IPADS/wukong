test:
	mpic++ -std=c++11 -I${BOOST_INCLUDE} -L${BOOST_LIB} test_traverser_keeppath.cpp -lboost_mpi -lboost_serialization -o test_traverser_keeppath.out

hello:
	mpic++ -I${BOOST_INCLUDE} -L${BOOST_LIB} hello.cpp -lboost_mpi -lboost_serialization -o hello.out
	
clean:
	rm *.out; rm *.o