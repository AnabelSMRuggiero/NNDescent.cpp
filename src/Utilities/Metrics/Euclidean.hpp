/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_EUCLIDEAN_HPP
#define NND_EUCLIDEAN_HPP

#include <immintrin.h>
#include <type_traits>
#include <bit>
#include <array>
#include <span>
#include <cassert>

#include "SpaceMetrics.hpp"

#include "../Type.hpp"

namespace nnd{

template<typename VectorA, typename VectorB, typename RetType=double>
RetType EuclideanNorm(const VectorA& pointA, const VectorB& pointB){
    using ExtentA = typename VectorA::value_type;
    using ExtentB = typename VectorB::value_type;
    auto transformFunc = [](ExtentA operandA, ExtentB operandB){
        RetType diff = static_cast<RetType>(operandA) - static_cast<RetType>(operandB);
        return diff*diff;
    };
    RetType accum = std::transform_reduce(std::execution::unseq,
                                    std::begin(pointA),
                                    std::end(pointA),
                                    std::begin(pointB),
                                    RetType(0),
                                    std::plus<RetType>(),
                                    transformFunc);
    return std::sqrt(accum);
};


template<size_t numPointsTo>
void BatchEuclideanNorm(const AlignedSpan<const float> pointB,
                        std::span<const AlignedSpan<const float>, numPointsTo> pointsTo,
                        std::span<float, numPointsTo> resultLocation) noexcept {
    static_assert(numPointsTo<=7 && numPointsTo>=2);
    

    std::array<__m256, numPointsTo> accumulators;
    for (__m256& accumulator: accumulators) accumulator = _mm256_setzero_ps();

    std::array<__m256, numPointsTo> toComponents;

    __m256 fromComponent1, fromComponent2;

    size_t index = 0;

    

    [[likely]] if(pointB.size()>=8){
        //Pre load first set of elements
        fromComponent1 = _mm256_load_ps(&(pointB[0]));

        for (size_t i = 0; i < numPointsTo; i+=1) toComponents[i] = NTLoadFloat(&(pointsTo[i][0]));
        //for (size_t i = 0; i < numPointsTo; i += 1) toComponents[i] = _mm256_load_ps(&(pointsTo[i][0]));
            
        //Core computation loop
        for(;index+15<pointB.size(); index+=8){
            fromComponent2 = _mm256_load_ps(&(pointB[index+8]));

            for(size_t j = 0; j<numPointsTo; j+=1) toComponents[j] = _mm256_sub_ps(toComponents[j], fromComponent1);
            for(size_t j = 0; j<numPointsTo; j+=1) accumulators[j] = _mm256_fmadd_ps(toComponents[j], toComponents[j], accumulators[j]);
                
            //Load for next iteration
            //for(size_t j = 0; j<numPointsTo; j+=1) toComponents[j] = _mm256_load_ps(&(pointsTo[j][index+8]));
            for(size_t j = 0; j<numPointsTo; j+=1) toComponents[j] = NTLoadFloat(&(pointsTo[j][index+8]));
            fromComponent1 = fromComponent2;
        }
        
        
        //Already have fromComponent1 loaded for the last iter
        for(size_t j = 0; j<numPointsTo; j+=1) toComponents[j] = _mm256_sub_ps(toComponents[j], fromComponent1);
        for(size_t j = 0; j<numPointsTo; j+=1) accumulators[j] = _mm256_fmadd_ps(toComponents[j], toComponents[j], accumulators[j]);

        index +=8;
        fromComponent2 = _mm256_setzero_ps();
        //reduce the results
        for(size_t j = 0; j<2; j+=1){
            for (auto& accumulator: accumulators){
                accumulator = _mm256_hadd_ps(accumulator, fromComponent2);
            }
        }

        for (size_t j = 0; j<numPointsTo; j+=1){
            
            //This constexpr branch works with MSVC and Clang, haven't tried GCC, but I suspect it should.
            if constexpr (std::is_union_v<__m256>){
                resultLocation[j] = accumulators[j].m256_f32[0] + accumulators[j].m256_f32[4];
            } else{
                resultLocation[j] = accumulators[j][0] + accumulators[j][4];
            }
            //__GNUC__
        }

    }
    //Take care of the excess. I should be able to remove this when I get alignment right
    for ( ; index<pointB.size(); index += 1){
        for (size_t j = 0; j<numPointsTo; j+=1){
            float diff = pointsTo[j][index] - pointB[index];
            resultLocation[j] += diff*diff;
        }
    }
    //Last I checked, this emits an avx sqrt on clang
    for (auto& res: resultLocation) res = std::sqrt(res);


}

/*
template<size_t numPointsTo, typename Alloc = std::allocator<AlignedSpan<const float>>>
std::vector<float> BatchEuclideanNorm(const std::vector<AlignedSpan<const float>, Alloc>& pointsTo, const AlignedSpan<const float>& pointB){
    static_assert(numPointsTo<=7 && numPointsTo>=2);
    //size_t prefetchPeriod = 16;
    //assert(pointsTo.size() == 7);
    std::array<__m256, numPointsTo> accumulators;
    for (__m256& accumulator: accumulators){
        accumulator = _mm256_setzero_ps();
    }

    std::array<__m256, numPointsTo> toComponents;

    __m256 fromComponent1, fromComponent2;

    size_t index = 0;

    std::vector<float> result(numPointsTo);
    

    [[likely]] if(pointB.size()>=8){
        //Pre load first set of elements
        fromComponent1 = _mm256_load_ps(&(pointB[0]));
        //fromComponent1 = NTLoadFloat(&(pointB[0]));
        for (size_t i = 0; i < numPointsTo; i+=1){
            //toComponents[i] = _mm256_load_ps(&(pointsTo[i][0]));
            toComponents[i] = NTLoadFloat(&(pointsTo[i][0]));
        }
        

        
        //Core computation loop
        for(;index+15<pointB.size(); index+=8){
            fromComponent2 = _mm256_load_ps(&(pointB[index+8]));
            for(size_t j = 0; j<numPointsTo; j+=1) toComponents[j] = _mm256_sub_ps(toComponents[j], fromComponent1);
                
            for(size_t j = 0; j<numPointsTo; j+=1) accumulators[j] = _mm256_fmadd_ps(toComponents[j], toComponents[j], accumulators[j]);
                
            //Load for next iteration
            //for(size_t j = 0; j<numPointsTo; j+=1) toComponents[j] = _mm256_load_ps(&(pointsTo[j][index+8]));
            for(size_t j = 0; j<numPointsTo; j+=1) toComponents[j] = NTLoadFloat(&(pointsTo[j][index+8]));
            fromComponent1 = fromComponent2;
        }
        
        
        //Already have fromComponent1 loaded for the last iter
        for(size_t j = 0; j<numPointsTo; j+=1) toComponents[j] = _mm256_sub_ps(toComponents[j], fromComponent1);
        for(size_t j = 0; j<numPointsTo; j+=1) accumulators[j] = _mm256_fmadd_ps(toComponents[j], toComponents[j], accumulators[j]);

        index +=8;
        fromComponent2 = _mm256_setzero_ps();
        //reduce the results
        for(size_t j = 0; j<2; j+=1){
            for (auto& accumulator: accumulators){
                accumulator = _mm256_hadd_ps(accumulator, fromComponent2);
            }
        }

        for (size_t j = 0; j<numPointsTo; j+=1){
            
            //This constexpr branch works with MSVC and Clang, haven't tried GCC, but I suspect it should.
            if constexpr (std::is_union_v<__m256>){
                result[j] = accumulators[j].m256_f32[0] + accumulators[j].m256_f32[4];
            } else{
                result[j] = accumulators[j][0] + accumulators[j][4];
            }
            //__GNUC__
        }

    }
    //Take care of the excess. I should be able to remove this when I get alignment right
    for ( ; index<pointB.size(); index += 1){
        for (size_t j = 0; j<numPointsTo; j+=1){
            float diff = pointsTo[j][index] - pointB[index];
            result[j] += diff*diff;
        }
    }
    //I know there's an avx instruction for this, but just wanna wrap up for now.
    for (auto& res: result){
        res = std::sqrt(res);
    }

    return result;

}
*/



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

std::vector<float> EuclideanBatcher(const AlignedSpan<const float> pointFrom, std::span<const AlignedSpan<const float>> pointsTo) noexcept {
    
    std::vector<float> retVector(pointsTo.size());
    
    
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
    //RetType resultBatch;
    //resultBatch.reserve(pointsTo.size());

    

}

/*
template<typename Alloc = std::allocator<AlignedSpan<const float>>>
std::vector<float> EuclideanBatcher(const AlignedSpan<const float>& pointFrom, std::span<const AlignedSpan<const float>> pointsTo){
    
    std::vector<float> retVector(pointsTo.size());
    switch(pointsTo.size()){
        case 7: 
            return BatchEuclideanNorm<7>(pointsTo, pointFrom);
        case 6: 
            return BatchEuclideanNorm<6>(pointsTo, pointFrom);
        case 5: 
            return BatchEuclideanNorm<5>(pointsTo, pointFrom);
        case 4: 
            return BatchEuclideanNorm<4>(pointsTo, pointFrom);
        case 3: 
            return BatchEuclideanNorm<3>(pointsTo, pointFrom);
        case 2: 
            return BatchEuclideanNorm<2>(pointsTo, pointFrom);
        case 1: 
            
            retVector.push_back(EuclideanNorm(pointsTo[0], pointFrom));
            return retVector;
        default:
            break;
    }
    
    size_t index = 0;


    for( ; (index+6)< pointsTo.size();){
        std::vector<AlignedSpan<const float>> partialBatch;
        partialBatch.reserve(7);
        for (size_t i = 0; i<7; i += 1, index+=1){
            partialBatch.push_back(pointsTo[index]);
        }
        std::vector<float> partialResult = EuclideanBatcher(pointFrom, partialBatch);
        for(const auto res: partialResult) retVector.push_back(res);
    }
    
    if(index<pointsTo.size()){
        std::vector<AlignedSpan<const float>> partialBatch;
        partialBatch.reserve(pointsTo.size()-index);
        for (; index<pointsTo.size(); index += 1){
            partialBatch.push_back(pointsTo[index]);
        }
        std::vector<float> partialResult = EuclideanBatcher(pointFrom, partialBatch);
        for(const auto res: partialResult) retVector.push_back(res);
    }

    return retVector;
    //RetType resultBatch;
    //resultBatch.reserve(pointsTo.size());

    

}
*/

struct EuclideanMetricPair{
    using DistType = float;
    float operator()(const AlignedSpan<const float> lhsVector, const AlignedSpan<const float> rhsVector) const{
        return EuclideanNorm<AlignedSpan<const float>, AlignedSpan<const float>, float>(lhsVector, rhsVector);
    };
    
    std::vector<float> operator()(AlignedSpan<const float> lhsVector, std::span<const AlignedSpan<const float>> rhsVectors) const{
        return EuclideanBatcher(lhsVector, rhsVectors);
    };

    
};

struct EuclideanComDistance{
    using DistType = float;
    float operator()(const AlignedSpan<const float> dataVector, const AlignedSpan<const float> comVector) const{
        return EuclideanNorm<AlignedSpan<const float>, AlignedSpan<const float>, float>(comVector, dataVector);
    };
    
    std::vector<float> operator()(AlignedSpan<const float> comVector, std::span<const AlignedSpan<const float>> rhsVectors) const{
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