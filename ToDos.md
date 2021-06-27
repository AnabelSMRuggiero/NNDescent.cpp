#General
- Refactor RNG stuff
- Refine implementation of space metrics and dot product. 
  - std::inner_product exists
  - also valarray.apply exists. That probably exploits the expression templates valarray is implemented with
    - doesn't work with static_cast because the applied func needs to return the exact same type
- Think of more sensical names for the headers


#RPTrees
- Add way to track best splits
- ~~Move forest building outside of constructor.~~
- Add in tree merge functionality

- Check to see if omitting the rejected split copy violates the invariance of sum of indicies


#NNDescent
- Add in some more robust benchmarking
  - Add in recall checking
- This will be forever out from now, but test builing data array inside of the control structure itself to see if it reduces cache misses.

#Parallelization and Distributed Computing
- Do a second pass on the MetaGraph proceedures (assuming the results look good; seems promising so far)