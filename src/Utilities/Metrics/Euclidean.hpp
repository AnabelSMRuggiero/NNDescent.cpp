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


template<size_t numPointsTo, size_t prefetchPeriod, typename Alloc = std::allocator<AlignedSpan<const float>>>
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
        for (size_t i = 0; i < numPointsTo; i+=1){
            toComponents[i] = _mm256_load_ps(&(pointsTo[i][0]));
        }
        while(index+(8*prefetchPeriod - 1)<pointB.size()){
            //Prefetch future data
            _mm_prefetch(&(pointB[index+(8*prefetchPeriod)]), _MM_HINT_T0);
            for (size_t j = 0; j < numPointsTo; j+=1){
                _mm_prefetch(&(pointsTo[index+(8*prefetchPeriod)]), _MM_HINT_T0);
            }

            for (size_t j = 0; j<prefetchPeriod; j+=1){
            //Core computation loop

                fromComponent2 = _mm256_load_ps(&(pointB[index+8]));
                for(size_t j = 0; j<numPointsTo; j+=1){
                    toComponents[j] = _mm256_sub_ps(toComponents[j], fromComponent1);
                    accumulators[j] = _mm256_fmadd_ps(toComponents[j], toComponents[j], accumulators[j]);
                    //Load for next iteration
                    //toComponents[j] = _mm256_load_ps(&(pointsTo[j][index+8]));
                }
                for(size_t j = 0; j<numPointsTo; j+=1) toComponents[j] = _mm256_load_ps(&(pointsTo[j][index+8]));
                fromComponent1 = fromComponent2;
                
                index+=8;
            }
        }
        for(;index+15<pointB.size(); index+=8){
            fromComponent2 = _mm256_load_ps(&(pointB[index+8]));
            for(size_t j = 0; j<7; j+=1){
                toComponents[j] = _mm256_sub_ps(toComponents[j], fromComponent1);
                accumulators[j] = _mm256_fmadd_ps(toComponents[j], toComponents[j], accumulators[j]);
                //Load for next iteration
                //toComponents[j] = _mm256_load_ps(&(pointsTo[j][index+8]));
            }
            for(size_t j = 0; j<numPointsTo; j+=1) toComponents[j] = _mm256_load_ps(&(pointsTo[j][index+8]));
            fromComponent1 = fromComponent2;
        }
        //Already have fromComponent1 loaded for the last iter
        for(size_t j = 0; j<numPointsTo; j+=1){
            toComponents[j] = _mm256_sub_ps(toComponents[j], fromComponent1);
            accumulators[j] = _mm256_fmadd_ps(toComponents[j], toComponents[j], accumulators[j]);
            //Load for next iteration
            //toComponents[j] = _mm256_loadu_ps(&(pointsTo[j][index+8]));
        }

        fromComponent2 = _mm256_setzero_ps();
        //reduce the results
        for(size_t j = 0; j<2; j+=1){
            for (auto& accumulator: accumulators){
                accumulator = _mm256_hadd_ps(accumulator, fromComponent2);
            }
        }

        for (size_t j = 0; j<numPointsTo; j+=1){
            result[j] = accumulators[j][0] + accumulators[j][4];
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
    for (auto& res: result) res = std::sqrt(res);
        

    return result;

}



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

/*
namespace internal{
    static const size_t euclideanMaxBatch = 7;
}
*/
//const Container<DataIndexType>& LHSIndecies, DataIndexType RHSIndex, const DataEntry& queryData
//lhsData, queryData
//template<typename VectorSetA, typename VectorB, typename RetType = std::vector<double>>
template<typename Alloc = std::allocator<AlignedSpan<const float>>>
std::vector<float> EuclideanBatcher(const AlignedSpan<const float>& pointFrom, const std::vector<AlignedSpan<const float>, Alloc>& pointsTo){
    
    std::vector<float> retVector;
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

struct EuclideanMetricPair{
    using DistType = float;
    float operator()(const AlignedSpan<const float> lhsVector, const AlignedSpan<const float> rhsVector) const{
        return EuclideanNorm<AlignedSpan<const float>, AlignedSpan<const float>, float>(lhsVector, rhsVector);
    };
    
    template<typename Alloc = std::allocator<AlignedSpan<const float>>>
    std::vector<float> operator()(AlignedSpan<const float> lhsVector, const std::vector<AlignedSpan<const float>, Alloc>& rhsVectors) const{
        return EuclideanBatcher(lhsVector, rhsVectors);
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