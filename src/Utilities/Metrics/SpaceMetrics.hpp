/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_SPACEMETRICS_HPP
#define NND_SPACEMETRICS_HPP

#include <valarray>
#include <cmath>
#include <execution>
#include <numeric>
//#include <functional>

#include <immintrin.h>

#include "Euclidean.hpp"

namespace nnd{


template<size_t p, typename DataEntry, typename RetValue = double>
RetValue PNorm(const DataEntry& point){
    using Extent = typename DataEntry::value_type;
    RetValue acc = std::transform_reduce(point.begin(),
                          point.end(),
                          RetValue(0),
                          std::plus<RetValue>(),
                          [](const Extent& extent)->RetValue{ return std::pow(extent, p);});

    return std::pow(acc, 1.0/p);
}

template<typename DataEntry>
void Normalize(DataEntry& entry){
    using Extent = typename DataEntry::value_type;
    Extent norm = PNorm<2, DataEntry, Extent>(entry);

    for(auto& component: entry) component /= norm;
}


template<typename VectorA, typename VectorB, typename RetType=float>
RetType Dot(const VectorA& pointA, const VectorB& pointB){
    using ExtentA = typename VectorA::value_type;
    using ExtentB = typename VectorB::value_type;
    auto transformFunc = [](ExtentA operandA, ExtentB operandB){
        return static_cast<RetType>(operandA) * static_cast<RetType>(operandB);
    };

    RetType accum = std::transform_reduce(std::execution::unseq,
                                    pointA.begin(),
                                    pointA.end(),
                                    pointB.begin(),
                                    RetType(0),
                                    std::plus<RetType>(),
                                    transformFunc);

    return accum;
};


}

#endif 