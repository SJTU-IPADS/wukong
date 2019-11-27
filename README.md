# Wukong for Linked Data

Wukong is a fast and distributed graph-structured store that leverages efficient graph exploration to proivde highly concurrent and low-latency query processing over big linked data.


## Feature Highlights

* High-performance and scalable in-memory graph store
* Fast and concurrent SPARQL query processing by graph exloration
* Fast communication by leveraging RDMA feature of InfiniBand network
* A GPU extension of query engine for heterogenous (CPU/GPU) cluster
* A type-centric SPARQL query plan optimizer

For more details see [Wukong Project](http://ipads.se.sjtu.edu.cn/projects/wukong), including new features, roadmap, instructions, etc.


## Getting Started

* [Installation](docs/INSTALL.md)
* [Tutorials](docs/TUTORIALS.md)
* [Manual](docs/COMMANDS.md)
* [GPU extension](docs/gpu/TUTORIALS.md)
* [Q&A](docs/QA.md)


## License

Wukong is released under the [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0.html).

If you use Wukong in your research, please cite our paper:
   
    @inproceedings{osdi2016wukong,
     author = {Shi, Jiaxin and Yao, Youyang and Chen, Rong and Chen, Haibo and Li, Feifei},
     title = {Fast and Concurrent RDF Queries with RDMA-based Distributed Graph Exploration},
     booktitle = {12th USENIX Symposium on Operating Systems Design and Implementation},
     series = {OSDI '16},
     year = {2016},
     month = Nov,
     isbn = {978-1-931971-33-1},
     address = {GA},
     pages = {317--332},
     url = {https://www.usenix.org/conference/osdi16/technical-sessions/presentation/shi},
     publisher = {USENIX Association},
    }


## Academic and Reference Papers

[**OSDI**] [Fast and Concurrent RDF Queries with RDMA-based Distributed Graph Exploration](docs/papers/wukong-osdi16.pdf). Jiaxin Shi, Youyang Yao, Rong Chen, Haibo Chen, and Feifei Li. Proceedings of 12th USENIX Symposium on Operating Systems Design and Implementation, Savannah, GA, US, Nov, 2016. 

[**SOSP**] [Sub-millisecond Stateful Stream Querying over Fast-evolving Linked Data](docs/papers/wukong+s-sosp17.pdf). Yunhao Zhang, Rong Chen, and Haibo Chen. Proceedings of the 26th ACM Symposium on Operating Systems Principles, Shanghai, China, October, 2017. 

[**USENIX ATC**] [Fast and Concurrent RDF Queries using RDMA-assisted GPU Graph Exploration](docs/papers/wukong+g-atc18.pdf). Siyuan Wang, Chang Lou, Rong Chen, and Haibo Chen. Proceedings of 2018 USENIX Annual Technical Conference, Boston, MA, US, July 2018.

[**USENIX ATC**] [Pragh: Locality-preserving Graph Traversal with Split Live Migration](docs/papers/wukong+m-atc19.pdf). Xiating Xie, Xingda Wei, Rong Chen, and Haibo Chen. Proceedings of 2019 USENIX Annual Technical Conference, Renton, WA, US, July 2019.



