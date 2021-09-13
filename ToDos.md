#General

- Think of more sensical names for the headers
- Get a better idea of the actual data that will be fed into the algorithm.
  - Assume always array of numerics?
    - Front end some sort of generalized PCA?
    - I could just be boring and front end SVD (which doesn't really work for non-euclidian easily, but would be needed for generalized PCA init)


- Fix my I/O. My I/O is bad. (I spend so much time in the stream operator).
  - I/O works okay when compiled on linux with Clang (??)

- An actual, good, pair hashing func that's not from a library.

- Template out the manually vectorized functions. I should be able to make them generic enough that I don't need too many specializations.
  - AVX512 can use more registers
  - Implement some way to set the maxBatching parameter more elegantly

- With parallel index building, I have almost 66% more loads from cache. Why?
  - 

#RPTrees
- Think of a better way to refine tree splitting than if statement into goto.
  - Removed in serial case, left for part of the parallel case.

#NNDescent
- This will be forever out from now, but test builing data array inside of the control structure itself to see if it reduces cache misses.
- Try spinning up branchless block bruteforcing.
  - Almost irrelevant at this point.

- Template MetaGraph stuff
  - Partially done

- Rethink verticies. There should be some way to set up making operations on them more efficient.
  - Also set it up so NearestPair calculations can update neighbor lists.
  - Prototyped CacheLineVertex

- Most of my run time is in distance calcs. While optimizing the calculation will be important, I need to squeeze as much as I can out of every call.

- UndirectedGraph BlockIndices template specialization.

- Remove the assumption that the 0th element is block 0.

- Add in search bailout

#Parallelization and Distributed Computing
- Do a second pass on the MetaGraph procedures 
- Points of caution for parallelization
  - Splitting schemes
    - Making sure only one thread creates a splitting vector

