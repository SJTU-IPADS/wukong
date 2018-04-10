About this
======
This is the reporsity of a constrained version of RDMA lib used in DrTM and DrTM+. 
It mainly provides fucntions to establish RDMA connections and wrappers for some basic RDMA operations including (READ,WRITE and ATMOIC_CMP_AND_SWP). 
We plan to merge more advanced functions in this library soon.

Dependencies
------------
* C++ 11 required 
* zeromq 4.0.5 or higher
* Mellanox OFED v3.0-2.0.1 stack or higher

Usage
------------
The simplest way to establish RDMA connections is to use the bootstrapRDMA function providedd by the library.
Every node in the cluster shall call this function during initilization time. 
The input network, together with the tcp port used for RDMA connection and the intended RDMA memory area are passed to this function.
The library assumes a network defined as [id,ip] pair, where id is a 16bit integer and ip is a string such as 10.0.0.100, which can be used for TCP connection.
More detailed information can be found in rdmaio.h, which is very compact and small. 

The memory buffer registered must be continuesly in virtual address space, and we do not support register multi regions.
The reason has 2 folds, first, RDMA has limited pd and mr which is used for memory registeration, thus we cannot register 
arbitrary number of memory regions. Secondly, the region can be handled through application logic, thus no physical seperation is needed. 


