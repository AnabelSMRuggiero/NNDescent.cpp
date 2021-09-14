/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_ANGULAR_HPP
#define NND_ANGULAR_HPP

#include <immintrin.h>
#include <type_traits>

#include "../Type.hpp"

namespace nnd{

//Assumes data is already normalized
template<typename VectorA, typename VectorB, typename RetType=double>
RetType AngularMetric(const VectorA& pointA, const VectorB& pointB){
    using ExtentA = typename VectorA::value_type;
    using ExtentB = typename VectorB::value_type;

    auto transformFunc = [](ExtentA operandA, ExtentB operandB){
        return static_cast<RetType>(operandA) * static_cast<RetType>(operandB); 
    };
    RetType accum = std::transform_reduce(std::execution::unseq,
                                    std::begin(pointA),
                                    std::end(pointA),
                                    std::begin(pointB),
                                    RetType(0),
                                    std::plus<RetType>(),
                                    transformFunc);
    return static_cast<RetType>(1.0) - accum;
};


template<size_t numPointsTo>
std::vector<float> BatchAngularMetric(const std::vector<AlignedSpan<const float>>& pointsTo, const AlignedSpan<const float>& pointB){
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
        for (size_t i = 0; i < numPointsTo; i+=1){
            toComponents[i] = _mm256_load_ps(&(pointsTo[i][0]));
        }
        

        
        //Core computation loop
        for(;index+15<pointB.size(); index+=8){
            fromComponent2 = _mm256_load_ps(&(pointB[index+8]));
                
            for(size_t j = 0; j<numPointsTo; j+=1) accumulators[j] = _mm256_fmadd_ps(toComponents[j], fromComponent1, accumulators[j]);
                
            //Load for next iteration
            for(size_t j = 0; j<numPointsTo; j+=1) toComponents[j] = _mm256_load_ps(&(pointsTo[j][index+8]));

            fromComponent1 = fromComponent2;
        }
        
        
        //Already have fromComponent1 loaded for the last iter
        for(size_t j = 0; j<numPointsTo; j+=1) accumulators[j] = _mm256_fmadd_ps(toComponents[j], fromComponent1, accumulators[j]);

        index +=8;
        fromComponent2 = _mm256_setzero_ps();
        //reduce the results
        for(size_t j = 0; j<2; j+=1){
            for (auto& accumulator: accumulators){
                accumulator = _mm256_hadd_ps(accumulator, fromComponent2);
            }
        }

        for (size_t j = 0; j<numPointsTo; j+=1){
            /*
            #ifdef __clang__
            result[j] = accumulators[j][0] + accumulators[j][4];
            #endif
            #ifdef _MSC_VER
            result[j] = accumulators[j].m256_f32[0] + accumulators[j].m256_f32[4];
            #endif
            */
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
            result[j] += pointsTo[j][index] * pointB[index];
        }
    }
    //I know there's an avx instruction for this, but just wanna wrap up for now.
    for (auto& res: result){
        res = 1.0f - res;
    }

    return result;

}

std::vector<float> AngularBatcher(const AlignedSpan<const float>& pointFrom, const std::vector<AlignedSpan<const float>>& pointsTo){
    
    std::vector<float> retVector;
    switch(pointsTo.size()){
        case 7: 
            return BatchAngularMetric<7>(pointsTo, pointFrom);
        case 6: 
            return BatchAngularMetric<6>(pointsTo, pointFrom);
        case 5: 
            return BatchAngularMetric<5>(pointsTo, pointFrom);
        case 4: 
            return BatchAngularMetric<4>(pointsTo, pointFrom);
        case 3: 
            return BatchAngularMetric<3>(pointsTo, pointFrom);
        case 2: 
            return BatchAngularMetric<2>(pointsTo, pointFrom);
        case 1: 
            
            retVector.push_back(AngularMetric(pointsTo[0], pointFrom));
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
        std::vector<float> partialResult = AngularBatcher(pointFrom, partialBatch);
        for(const auto res: partialResult) retVector.push_back(res);
    }
    
    if(index<pointsTo.size()){
        std::vector<AlignedSpan<const float>> partialBatch;
        partialBatch.reserve(pointsTo.size()-index);
        for (; index<pointsTo.size(); index += 1){
            partialBatch.push_back(pointsTo[index]);
        }
        std::vector<float> partialResult = AngularBatcher(pointFrom, partialBatch);
        for(const auto res: partialResult) retVector.push_back(res);
    }

    return retVector;
    //RetType resultBatch;
    //resultBatch.reserve(pointsTo.size());

    

}

struct AngularMetricPair{
    using DistType = float;
    float operator()(const AlignedSpan<const float> lhsVector, const AlignedSpan<const float> rhsVector) const{
        return AngularMetric<AlignedSpan<const float>, AlignedSpan<const float>, float>(lhsVector, rhsVector);
    };
    
    std::vector<float> operator()(AlignedSpan<const float> lhsVector, const std::vector<AlignedSpan<const float>>& rhsVectors) const{
        return AngularBatcher(lhsVector, rhsVectors);
    };
};

}

#endif