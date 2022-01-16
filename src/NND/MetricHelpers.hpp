/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_METRICHELPERS_HPP
#define NND_METRICHELPERS_HPP

#include <type_traits>

#include "../ann/Type.hpp"
#include "../ann/Metrics/Euclidean.hpp"
#include "MemoryInternals.hpp"

namespace nnd{
//template<template<size_t> typename NormFunctor>
void BatchDispatcher(const ann::vector_span<const float> pointFrom,
                    std::span<const ann::vector_span<const float>> pointsTo,
                    std::span<float> resultLocation) noexcept {

    auto dispatch = [&]<size_t numPoints>(std::integral_constant<size_t, numPoints>){
        return BatchEuclideanNorm(pointFrom,
                                  std::span<const ann::vector_span<const float>, numPoints>{pointsTo},
                                  std::span<float, numPoints>{resultLocation});
    };

    switch(pointsTo.size()){
        /*
        case 15: 
            return dispatch(std::integral_constant<size_t, 15>{});
        case 14: 
            return dispatch(std::integral_constant<size_t, 14>{});
        case 13: 
            return dispatch(std::integral_constant<size_t, 13>{});
        case 12: 
            return dispatch(std::integral_constant<size_t, 12>{});
        case 11: 
            return dispatch(std::integral_constant<size_t, 11>{});
        case 10: 
            return dispatch(std::integral_constant<size_t, 10>{});
        case 9: 
            return dispatch(std::integral_constant<size_t, 9>{});
        case 8: 
            return dispatch(std::integral_constant<size_t, 8>{});
        */
        case 7: 
            return dispatch(std::integral_constant<size_t, 7>{});
        case 6: 
            return dispatch(std::integral_constant<size_t, 6>{});
        case 5: 
            return dispatch(std::integral_constant<size_t, 5>{});
        case 4: 
            return dispatch(std::integral_constant<size_t, 4>{});
        case 3: 
            return dispatch(std::integral_constant<size_t, 3>{});
        case 2: 
            return dispatch(std::integral_constant<size_t, 2>{});
        case 1: 
            
            resultLocation[0] = EuclideanNorm(AlignedSpan<const float, 32>{pointsTo[0]}, AlignedSpan<const float, 32>{pointFrom});
            return;
        default:
            assert(false);
            return;
    }
}

std::pmr::vector<float> EuclideanBatcher(const ann::vector_span<const float> pointFrom, std::span<const ann::vector_span<const float>> pointsTo) noexcept {
    
    constexpr size_t maxBatch = 7;
    std::pmr::vector<float> retVector(pointsTo.size(), internal::GetThreadResource());
    
    size_t index = 0;

    for( ; (index+maxBatch)< pointsTo.size(); index += maxBatch){
        std::span<const ann::vector_span<const float>, maxBatch> partialBatch{pointsTo.begin()+index, maxBatch};
        std::span<float, maxBatch> batchOutput{retVector.begin()+index, maxBatch};
        BatchEuclideanNorm(pointFrom, partialBatch, batchOutput);
    }
    
    if(index<pointsTo.size()){
        size_t remainder = pointsTo.size() - index;
        std::span<const ann::vector_span<const float>> partialBatch{pointsTo.begin()+index, remainder};
        std::span<float> batchOutput{retVector.begin()+index, remainder};
        BatchDispatcher(pointFrom, partialBatch, batchOutput);
    }

    return retVector;

}



struct EuclideanMetricPair{
    using DistType = float;
    float operator()(const AlignedSpan<const float> lhsVector, const AlignedSpan<const float> rhsVector) const{
        return EuclideanNorm<AlignedSpan<const float>, AlignedSpan<const float>, float>(lhsVector, rhsVector);
    };
    
    std::pmr::vector<float> operator()(ann::vector_span<const float> lhsVector, std::span<const ann::vector_span<const float>> rhsVectors) const{
        return EuclideanBatcher(lhsVector, rhsVectors);
    };

    
};

struct EuclideanComDistance{
    using DistType = float;
    float operator()(const AlignedSpan<const float> dataVector, const AlignedSpan<const float> comVector) const{
        return EuclideanNorm<AlignedSpan<const float>, AlignedSpan<const float>, float>(comVector, dataVector);
    };
    
    std::pmr::vector<float> operator()(ann::vector_span<const float> comVector, std::span<const ann::vector_span<const float>> rhsVectors) const{
        return EuclideanBatcher(comVector, rhsVectors);
    };
};

struct EuclideanMetricSet{
    using DataToData_t = EuclideanMetricPair;
    using DataToCom_t = EuclideanMetricPair;
    using ComToCom_t = EuclideanMetricPair;

    [[no_unique_address]] EuclideanMetricPair dataToData{};
    [[no_unique_address]] EuclideanMetricPair dataToCom{};
    [[no_unique_address]] EuclideanMetricPair comToCom{};
    //data to data
    //data to COM
    //COM to COM

};
}

#endif