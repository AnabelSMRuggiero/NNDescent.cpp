/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#include <vector>
#include <span>
#include <array>
#include <cmath>

#include <immintrin.h>


std::vector<float> BatchEuclideanNorm(const std::vector<std::span<const float>>& pointsTo, const std::span<const float>& pointB){
    
    //assert(pointsTo.size() == 7);
    std::array<__m256, 7> accumulators;
    for (__m256& accumulator: accumulators){
        accumulator = _mm256_setzero_ps();
    }

    std::array<__m256, 7> toComponents;

    __m256 fromComponent1, fromComponent2;

    size_t index = 0;

    std::vector<float> result(7);
    
    //size_t prefetchPeriod = 4;

    [[likely]] if(pointB.size()>=8){
        //I can't guarantee memory allignment yet, so unalligned loads for now
        fromComponent1 = _mm256_loadu_ps(&(pointB[0]));
        for (size_t i = 0; i < 7; i+=1){
            toComponents[i] = _mm256_loadu_ps(&(pointsTo[i][0]));
        }
        for(;index+15<pointB.size(); index+=8){
            fromComponent2 = _mm256_loadu_ps(&(pointB[index+8]));
            for(size_t j = 0; j<7; j+=1){
                toComponents[j] = _mm256_sub_ps(toComponents[j], fromComponent1);
                accumulators[j] = _mm256_fmadd_ps(toComponents[j], toComponents[j], accumulators[j]);
                //Load for next iteration
                toComponents[j] = _mm256_loadu_ps(&(pointsTo[j][index+8]));
            }
            fromComponent1 = fromComponent2;
        }
        //Already have fromComponent1 loaded for the last iter
        for(size_t j = 0; j<7; j+=1){
            toComponents[j] = _mm256_sub_ps(toComponents[j], fromComponent1);
            accumulators[j] = _mm256_fmadd_ps(toComponents[j], toComponents[j], accumulators[j]);
            //Load for next iteration
            //toComponents[j] = _mm256_loadu_ps(&(pointsTo[j][index+8]));
        }
        index +=8;
        fromComponent2 = _mm256_setzero_ps();
        //reduce the results
        for(size_t j = 0; j<2; j+=1){
            for (auto& accumulator: accumulators){
                accumulator = _mm256_hadd_ps(accumulator, fromComponent2);
            }
        }

        for (size_t j = 0; j<7; j+=1){
            result[j] = accumulators[j][0] + accumulators[j][4];
        }

    }
    //Take care of the excess. I should be able to remove this when I get alignment right
    for ( ; index<pointB.size(); index += 1){
        for (size_t j = 0; j<7; j+=1){
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