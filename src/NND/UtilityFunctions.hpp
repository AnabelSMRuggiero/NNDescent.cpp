/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_UTILITYFUNCTIONS_H
#define NND_UTILITYFUNCTIONS_H
#include <utility>

#include "../Utilities/DataDeserialization.hpp"

namespace nnd{

template<TriviallyCopyable IndexType, typename FloatType>
bool constexpr NeighborDistanceComparison(const std::pair<IndexType, FloatType>& neighborA, const std::pair<IndexType, FloatType>& neighborB){
    return neighborA.second < neighborB.second;
};

template<TriviallyCopyable IndexType, typename FloatType>
bool constexpr NeighborIdentityCheck(const std::pair<IndexType, FloatType>& neighborA, const std::pair<IndexType, FloatType>& neighborB){
    return neighborA.first == neighborB.first;
};


template<TriviallyCopyable IndexType, typename DistType>
struct NeighborSearchFunctor{
    
    IndexType searchValue;

    NeighborSearchFunctor(IndexType searchValue) : searchValue(searchValue){};

    bool operator()(std::pair<IndexType, DistType> currentValue){
        return currentValue.first == searchValue;
    }

};

template<TriviallyCopyable IndexType, typename DistType>
struct NeighborUnderDist{
    
    IndexType threshold;

    NeighborUnderDist(DistType threshold) : threshold(threshold){};

    bool operator()(std::pair<IndexType, DistType> currentValue){
        return currentValue.first < threshold;
    }

};

template<typename DistType>
struct NeighborOverDist{
    
    DistType threshold;

    NeighborOverDist(DistType threshold) : threshold(threshold){};

    template<TriviallyCopyable IndexType>
    bool operator()(const std::pair<IndexType, DistType> currentValue){
        return currentValue.second >= threshold;
    }

};




}

#endif //NND_UTILITYFUNCTIONS_H