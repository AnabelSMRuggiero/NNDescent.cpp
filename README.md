# Recursive Nearest Neighbor Descent (rNND)

rNND is a (WIP) C++ implementation of the Nearest Neighbor Descent algorithm. The NND algorithm creates an approximation of an k-Nearest Neighbors graph for a large dataset. The goal of this project is to develop a version of NND that can be highly distributed by handling small portions of the developing graph at a time.
The NND algorithm is presented in ["*Efficient k-nearest neighbor graph construction for generic similarity measures*" by Dong et. al.](https://doi.org/10.1145/1963405.1963487)
[Leland McInnes](https://github.com/lmcinnes), author of [UMAP](https://github.com/lmcinnes/umap) and [PyNNDescent](https://github.com/lmcinnes/pynndescent) has helped greatly by providing guidance on this project.


Embedded meta graph connecting each block to the three blocks with the most neighbors out of block.
![Meta Graph Embedding](Meta-Graph-Embedding.png)


# Project Goals

- Header only library
- Have only the C++ Standard Library as a dependancy
- Produce a generic and efficient implementation

This project uses the C++20 standard.

# The Billion-Scale Approximate Nearest Neighbor Search Challenge 

I have submitted this project to the T2 track of the Billion-Scale Approximate Nearest Neighbor Search Challenge. More information on this challenge may be found [here](https://big-ann-benchmarks.com/). This algorithm will not be implementing vector quantitization, as it is still aiming to serve dimensionality reduction algorithms; I seek to build an index that retains as much spatial information as reasonibly possible.

The abstract for the submission is:

PyNNDescent is a core component of the Python implementation of Uniform Manifold Approximation and Projection (UMAP). UMAP's primary computational bottleneck is the formation of an approximate k nearest neighbors graph. PyNNDescent tackles forming the graph by combining two algorithms: Random Projection Trees (RPTrees) and Nearest Neighbor Descent (NND). RPTrees covers one of NND's weaknesses by initializing the graph. This is achieved by spatially partitioning the input data, and forming subgraphs within each leaf. This initialization is followed by adding new neighbors to each point at random. Then, NND iterates over the vertices in the graph. NND searches for new candidate neighbors by polling the neighbors of neighbors.

Recursive Nearest Neighbor Descent (rNND) is both a reimplementation of and iteration upon PyNNDescent. rNND is currently being developed in C++ as a header only library with the C++ Standard Library as its only core dependency. The algorithmic improvements rNND seeks to implement aim to address two problems with NND: low locality of computations and requiring random access to the entire graph.

To tackle these weaknesses, rNND uses the results of RPTrees to cluster the entire data set into blocks of data for each spatial partition. This allows reconceptualizing NND iterations into operations between blocks of data. However, this leads to the question of how to tackle the additional bookkeeping of the blockwise operations. Due to the way RPTrees partitions space, rNND assumes that the points in each leaf have the majority of their k nearest neighbors in the k' nearest leaves. For truly massive datasets, the k' nearest leaves problem can use this same analysis; the leaves can be clustered, and each leaf can search the k'' nearest clusters. This analysis will be used to efficiently organize and distribute computations between blocks of data.

# Project State

So far, a rough, serial version of NND has been implemented. Since the graph is randomly initialized at the time of writing, this algorithm basically does not converge for large datasets. This project has a lot further to go before being usable, including:

Currently in progress:
- Incorporation of random projection trees to initalize the graph.
- Pay off technical debt associated with not having dedicated dev time to a structure dedicated to handling direct interactions with data.
- Test building a final graph from smaller graphs made inside each tree.

Further down the road:
- Optimization
- Parallelization
    - Distributed computing
- Optimize the data structures the algorithm uses.


# A Note on the Use of std::valarray

So, I finally took a closer look at MSVC's std::valarray implementation. Turns out, apparently a template expression implementation would cause a breaking ABI change for MSVC. What this means for this project is yet to be determined as I do have some technical debt associated with data structures that I have yet to pay off.

Since I'm limiting myself to either the STL or what ever I write for the project's core libraries, I will still be using std::valarray because it is easy to develop with (unless I end up taking a shot at trying to write my own replacement a bit further down the line). Ultimately, it boils down to:

- I recommend using gcc's or LLVM's STL implementations, unless...
- You are compiling with Intel Performance Primatives (Intel's compiler swaps out the STL implementation's std::valarry with their own)
- By the time this project is in a "release" state, it'll be modular enough to use your own implementations/use a library like Eigen for vector math.

I guess this is a great time for me to make the jump to c++20 if I'm going to switch STLs.
