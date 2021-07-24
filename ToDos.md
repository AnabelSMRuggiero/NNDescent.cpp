#General
- Refactor RNG stuff
- Think of more sensical names for the headers
- Get a better idea of the actual data that will be fed into the algorithm.
  - Assume always array of numerics?
    - Front end some sort of generalized PCA?
    - I could just be boring and front end SVD (which doesn't really work for non-euclidian easily, but would be needed for generalized PCA init)

- Template BlockIndicies

- Fix my I/O. My I/O is bad. (I spend so much time in the stream operator).
  - I/O works okay when compiled on linux with Clang (??)
  - Pretty sure the way I do it might be UB

- An actual, good, pair hashing func that's not from a library.

- Rethink template arguements. The declarations are starting to get a bit long.
  - Instead of keeping them in terms of the fundamental types, I can instead declare the templates in terms of the composed types.

- Template out the manually vectorized functions. I should be able to make them generic enough that I don't need too many specializations.
  - AVX512 can use more registers
  - Implement some way to set the maxBatching parameter more elegantly

#RPTrees
- Move forest building outside of constructor.
- Add in tree merge functionality
- Think of a better way to refine tree splitting than if statement into goto.
- Rewrite TreeLeaf

#NNDescent
- This will be forever out from now, but test builing data array inside of the control structure itself to see if it reduces cache misses.
- Try spinning up branchless block bruteforcing.
- Template MetaGraph stuff
  - Partially done
- Rethink verticies. There should be some way to set up making operations on them more efficient.
  - Also set it up so NearestPair calculations can update neighbor lists.
  - Prototyped CacheLineVertex

- Second pass on breaking up BlockSitching and NearestNodeDistances. Mainly a concern at the parallelism stage. Do I need to transform from a contiguous section of memory?

- paramterize/pass as arguement the COM distance functor for metagraph

- Most of my run time is in distance calcs. While optimizing the calculation will be important, I need to squeeze as much as I can out of every call.


- On QueryHotPath: I want copies when I use the queryHint member, but not really when I'm passing in hints.

- UndirectedGraph BlockIndices template specialization.

- I convert the initial directed block graphs into an directed graph using block indecies and an undirected graph without the neighbors. Don't think I use the initial graph after that.
  - I can sink the memory into the undirected graph constructor.

#Parallelization and Distributed Computing
- Do a second pass on the MetaGraph procedures 
- Points of caution for parallelization
  - Splitting schemes
    - Making sure only one thread creates a splitting vector
  - QueryContexts
    - Checking to avoid double calculating.

- I think I'll end up doing a tasking model where a producer wraps the work and resources in a functor and passes the functor to the thread.