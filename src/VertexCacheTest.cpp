#include "NND/GraphStructures/CachableVertex.hpp"
using namespace nnd;
int main(){
    auto cache = MakeVertexCache<float>(10);
    CachableVertex<float> cachable = cache.Take();
    {
        CachableVertex<float> another = cache.Take();
        another->resize(14);
    }
    {
        CachableVertex<float> again = cache.Take();
        (*again).push_back({0, 0.0});
    }

    
    return 0;
}