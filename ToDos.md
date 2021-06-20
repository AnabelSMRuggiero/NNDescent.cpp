#General
- Refactor RNG stuff
- Refine implementation of space metrics and dot product. 
  - std::inner_product exists
  - also valarray.apply exists. That probably exploits the expression templates valarray is implemented with
- Think of more sensical names for the headers


#RPTrees
- Add way to track best splits
- ~~Move forest building outside of constructor.~~
- Add in tree merge functionality
- Do a second pass on design of tree structure.
  - Specifically focus on how I want to handle copying.

#NNDescent
- Add in some more robust benchmarking
  - Add in recall checking
- This will be forever out from now, but test builing data array inside of the control structure itself to see if it reduces cache misses.