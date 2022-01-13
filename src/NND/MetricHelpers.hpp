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
#include "../ann/Type.hpp"
#include "../ann/Metrics/Euclidean.hpp"
#include "MemoryInternals.hpp"

namespace nnd{
void EuclideanDispatch(const AlignedSpan<const float> pointFrom,
                    std::span<const AlignedSpan<const float>> pointsTo,
                    std::span<float> resultLocation) noexcept {

    switch(pointsTo.size()){
        [[unlikely]] case 7: 
            return BatchEuclideanNorm<7>(pointFrom,
                                        std::span<const AlignedSpan<const float>, 7>{pointsTo},
                                        std::span<float, 7>{resultLocation});
        case 6: 
            return BatchEuclideanNorm<6>(pointFrom,
                                        std::span<const AlignedSpan<const float>, 6>{pointsTo},
                                        std::span<float, 6>{resultLocation});
        case 5: 
            return BatchEuclideanNorm<5>(pointFrom,
                                        std::span<const AlignedSpan<const float>, 5>{pointsTo},
                                        std::span<float, 5>{resultLocation});
        case 4: 
            return BatchEuclideanNorm<4>(pointFrom,
                                        std::span<const AlignedSpan<const float>, 4>{pointsTo},
                                        std::span<float, 4>{resultLocation});
        case 3: 
            return BatchEuclideanNorm<3>(pointFrom,
                                        std::span<const AlignedSpan<const float>, 3>{pointsTo},
                                        std::span<float, 3>{resultLocation});
        case 2: 
            return BatchEuclideanNorm<2>(pointFrom,
                                        std::span<const AlignedSpan<const float>, 2>{pointsTo},
                                        std::span<float, 2>{resultLocation});
        case 1: 
            
            resultLocation[0] = EuclideanNorm(pointsTo[0], pointFrom);
            return;
        default:
            assert(false);
            return;
    }
}

std::pmr::vector<float> EuclideanBatcher(const AlignedSpan<const float> pointFrom, std::span<const AlignedSpan<const float>> pointsTo) noexcept {
    
    std::pmr::vector<float> retVector(pointsTo.size(), internal::GetThreadResource());
    
    size_t index = 0;

    for( ; (index+6)< pointsTo.size(); index += 7){
        std::span<const AlignedSpan<const float>, 7> partialBatch{pointsTo.begin()+index, 7};
        std::span<float, 7> batchOutput{retVector.begin()+index, 7};
        BatchEuclideanNorm<7>(pointFrom, partialBatch, batchOutput);
    }
    
    if(index<pointsTo.size()){
        size_t remainder = pointsTo.size() - index;
        std::span<const AlignedSpan<const float>> partialBatch{pointsTo.begin()+index, remainder};
        std::span<float> batchOutput{retVector.begin()+index, remainder};
        EuclideanDispatch(pointFrom, partialBatch, batchOutput);
    }

    return retVector;

}



struct EuclideanMetricPair{
    using DistType = float;
    float operator()(const AlignedSpan<const float> lhsVector, const AlignedSpan<const float> rhsVector) const{
        return EuclideanNorm<AlignedSpan<const float>, AlignedSpan<const float>, float>(lhsVector, rhsVector);
    };
    
    std::pmr::vector<float> operator()(AlignedSpan<const float> lhsVector, std::span<const AlignedSpan<const float>> rhsVectors) const{
        return EuclideanBatcher(lhsVector, rhsVectors);
    };

    
};

struct EuclideanComDistance{
    using DistType = float;
    float operator()(const AlignedSpan<const float> dataVector, const AlignedSpan<const float> comVector) const{
        return EuclideanNorm<AlignedSpan<const float>, AlignedSpan<const float>, float>(comVector, dataVector);
    };
    
    std::pmr::vector<float> operator()(AlignedSpan<const float> comVector, std::span<const AlignedSpan<const float>> rhsVectors) const{
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