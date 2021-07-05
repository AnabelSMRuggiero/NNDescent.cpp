#General
- Refactor RNG stuff
- Refine implementation of space metrics and dot product. 
  - std::inner_product exists
- Think of more sensical names for the headers
- Get a better idea of the actual data that will be fed into the algorithm.
  - Assume always array of numerics?
    - Front end some sort of generalized PCA?
    - I could just be boring and front end SVD (which doesn't really work for non-euclidian easily, but would be needed for generalized PCA init)

- I just realized, some distance metrics might wanna return an integral type (binary distances like hamming)

#RPTrees
- Add way to track best splits
  - With the direction I'm planning on going this probably won't be needed.
- ~~Move forest building outside of constructor.~~
- Add in tree merge functionality
- Think of a better way to refine tree splitting than if statement into goto.
- Check to see if omitting the rejected split copy violates the invariance of sum of indicies


#NNDescent
- Add in some more robust benchmarking
  - Add in recall checking
- This will be forever out from now, but test builing data array inside of the control structure itself to see if it reduces cache misses.
- Try spinning up branchless block bruteforcing.
- Update algorithm/data structure prototypes to reflect data blocking
- Template MetaGraph stuff
- Rethink verticies. There should be some way to set up making operations on them more efficient.
  - Also set it up so NearestPair calculations can update neighbor lists.
  - Prototyped CacheLineVertex
- Wrap searching function prototypes into a search context

#Parallelization and Distributed Computing
- Do a second pass on the MetaGraph proceedures (assuming the results look good; seems promising so far)
- Points of caution for parallelization
  - Splitting schemes
    - Making sure only one thread creates a splitting vector
  - QueryContexts
    - Checking to avoid double calculating.