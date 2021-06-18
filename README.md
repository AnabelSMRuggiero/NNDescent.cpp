# NNDescent.cpp

NNDescent.cpp is a (WIP) C++ implementation of the Nearest Neighbor Descent algorithm. The NND algorithm creates an approximation of an k-Nearest Neighbors graph for a large dataset. The goal of this project is to develop a version of NND that can be highly distributed by handling small portions of the developing graph at a time.
The NND algorithm is presented in ["*Efficient k-nearest neighbor graph construction for generic similarity measures*" by Dong et. al.](https://doi.org/10.1145/1963405.1963487)
[Leland McInnes](https://github.com/lmcinnes), author of [UMAP](https://github.com/lmcinnes/umap) and [PyNNDescent](https://github.com/lmcinnes/pynndescent) has helped greatly by providing guidance on this project.

# Project Goals

- Header only library
- Have only the C++ Standard Library as a dependancy
- Produce a generic and efficient implementation

Currently, this project uses the C++17 standard, and will ~~likely adopt C++20 to leverage Concepts and Modules~~ absolutely adopt C++20 just because of std::span. Everything else is icing.

# Project State

So far, a rough, serial version of NND has been implemented. Since the graph is randomly initialized at the time of writing, this algorithm basically does not converge for large datasets. This project has a lot further to go before being usable, including:

- Incorporation of random projection trees to initalize the graph.
- Pay off technical debt associated with not having dedicated dev time to a structure dedicated to handling direct interactions with data.
- Test building a final graph from smaller graphs made inside each tree.
    - If that works, distribute it across multiple processes.
- Optimize the data structures the algorithm uses.
- Include an optional type erasure scheme that reduces code bloat that may be caused by instantiating multiple templates of the algorithm.

# A Note on the Use of std::valarray

So, I finally took a closer look at MSVC's std::valarray implementation. Turns out, apparently a template expression implementation would cause a breaking ABI change for MSVC. What this means for this project is yet to be determined as I do have some technical debt associated with data structures that I have yet to pay off.

Since I'm limiting myself to either the STL or what ever I write for the project's core libraries, I will still be using std::valarray because it is easy to develop with (unless I end up taking a shot at trying to write my own replacement a bit further down the line). Ultimately, it boils down to:

- I recommend using gcc's or LLVM's STL implementations, unless...
- You are compiling with Intel Performance Primatives (Intel's compiler swaps out the STL implementation's std::valarry with their own)
- By the time this project is in a "release" state, it'll be modular enough to use your own implementations/use a library like Eigen for vector math.

I guess this is a great time for me to make the jump to c++20 if I'm going to switch STLs.
