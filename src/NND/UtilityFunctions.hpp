#ifndef NND_UTILITYFUNCTIONS_H
#define NND_UTILITYFUNCTIONS_H
#include <utility>

namespace nnd{

template<TriviallyCopyable IndexType, typename FloatType>
bool const NeighborDistanceComparison(const std::pair<IndexType, FloatType>& neighborA, const std::pair<IndexType, FloatType>& neighborB){
    return neighborA.second < neighborB.second;
};

template<TriviallyCopyable IndexType, typename FloatType>
bool const NeighborIdentityCheck(const std::pair<IndexType, FloatType>& neighborA, const std::pair<IndexType, FloatType>& neighborB){
    return neighborA.first == neighborB.first;
};

//Todo: template this so it can handle arbitrary float types
template<TriviallyCopyable IndexType, typename FloatType>
struct NeighborSearchFunctor{
    
    size_t searchValue;

    NeighborSearchFunctor() : searchValue(0){};

    bool operator()(std::pair<IndexType, FloatType> currentValue){
        return currentValue.first == searchValue;
    }

};



}

#endif //NND_UTILITYFUNCTIONS_H