# NNDescent.cpp

NNDescent.cpp is a (WIP) C++ implementation of the Nearest Neighbor Descent algorithm. The NND algorithm creates an approximation of an k-Nearest Neighbors graph for a large dataset. The goal of this project is to develop a version of NND that can be highly distributed by handling small portions of the developing graph at a time.
The NND algorithm presented in ["*Efficient k-nearest neighbor graph construction for generic similarity measures*" by Dong et. al.] (https://doi.org/10.1145/1963405.1963487)
[Leland McInnes](https://github.com/lmcinnes), author of [UMAP] (https://github.com/lmcinnes/umap) and [PyNNDescent](https://github.com/lmcinnes/pynndescent) has helped greatly by providing guidance on this project.

# Project Goals

- Header only library
- Have only the C++ Standard Library as a dependancy
- Produce a generic and efficient implementation

Currently, this project uses the C++17 standard, and will likely adopt C++20 to leverage Concepts and Modules.

# Project State

So far, a rough, serial version of NND has been implemented. Since the graph is randomly initialized at the time of writing, this algorithm basically does not converge for large datasets. This project has a lot further to go before being usable, including:

- Incorporation of random projection trees to initalize the graph.
- Pay off technical debt associated with not having dedicated dev time to a structure dedicated to handling direct interactions with data.
- Test building a final graph from smaller graphs made inside each tree.
    - If that works, distribute it.
- Optimize the data structures the algorithm uses.
- Include an optional type erasure scheme that reduces code bloat that may be caused by instantiating multiple templates of the algorithm.


