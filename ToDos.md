#General
- Refactor RNG stuff
- Think of more sensical names for the headers
- Get a better idea of the actual data that will be fed into the algorithm.
  - Assume always array of numerics?
    - Front end some sort of generalized PCA?
    - I could just be boring and front end SVD (which doesn't really work for non-euclidian easily, but would be needed for generalized PCA init)

- Try turning space metrics into functors, and have the type as a template parameter
  - Compilers tend to be very good at inlining functor calls.

- I just realized, some distance metrics might wanna return an integral type (binary distances like hamming)
- Template BlockIndicies

- Fix my I/O. My I/O is bad. (I spend so much time in the stream operator).

- An actual, good, pair hashing func that's not from a library.

#RPTrees
- Move forest building outside of constructor.
- Add in tree merge functionality
- Think of a better way to refine tree splitting than if statement into goto.


#NNDescent
- Add in some more robust benchmarking
- This will be forever out from now, but test builing data array inside of the control structure itself to see if it reduces cache misses.
- Try spinning up branchless block bruteforcing.
- Template MetaGraph stuff
  - Partially done
- Rethink verticies. There should be some way to set up making operations on them more efficient.
  - Also set it up so NearestPair calculations can update neighbor lists.
  - Prototyped CacheLineVertex
- Wrap searching function prototypes into a search context
  - In progress
- Should I move results caching outside of the query context? I kinda think having the results cached internally and not using the operator returns is an anti-patern.
- Search leafGraphs don't need distances.
  - Form undirected search graphs

- Second pass on breaking up BlockSitching and NearestNodeDistances. Mainly a concern at the parallelism stage. Do I need to transform from a contiguous section of memory?

- paramterize/pass as arguement the COM distance functor for metagraph

- Rewrite TreeLeaf
- Most of my run time is in distance calcs. While optimizing the calculation will be important, I need to squeeze as much as I can out of every call.
  - I'm only storing results from calculations on one side for every query. Figure out how to save calculations for both sides.

- Maybe replace the initial bruteforce joins with the current blockwise join prototype. Use the results of NearestNode searches to seed this.
  - Could affect recall? Might mean I don't do searches that would likely come up dead anyways, but it might mean I miss out searches that are relevant.
  - But if it is faster, it could mean I could do more initial joins to seed a more diverse pool of searches.

- On QueryHotPath: I want copies when I use the queryHint member, but not really when I'm passing in hints.

- UndirectedGraph BlockIndices template specialization.

#Parallelization and Distributed Computing
- Do a second pass on the MetaGraph procedures 
- Points of caution for parallelization
  - Splitting schemes
    - Making sure only one thread creates a splitting vector
  - QueryContexts
    - Checking to avoid double calculating.

- I think I'll end up doing a tasking model where a producer wraps the work and resources in a functor and passes the functor to the thread.