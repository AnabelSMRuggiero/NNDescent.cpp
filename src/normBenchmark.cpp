#include <numeric>
#include <span>
#include <algorithm>
#include <execution>
#include <bit>
#include <random>
#include <chrono>
#include <cassert>
#include <array>

#include <immintrin.h>

#include "NND/SpaceMetrics.hpp"
#include "Utilities/Data.hpp"
#include "Utilities/DataDeserialization.hpp"
/*

template<typename DataTypeA, typename DataTypeB, typename RetType=double>
RetType EuclideanNorm(const std::valarray<DataTypeA>& pointA, const std::valarray<DataTypeB>& pointB){
    auto transformFunc = [](DataTypeA operandA, DataTypeB operandB){
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


*/
/*
template<typename DataTypeA, typename DataTypeB, typename RetType=double>
std::vector<RetType> BatchEuclideanNorm(const std::vector<std::span<const DataTypeA>>& pointsTo, const std::valarray<DataTypeB>& pointB){
    std::vector<RetType> results(pointsTo.size());

    auto transformFunc = [&](DataTypeA operandA, size_t index){
        std::vector<RetType> transformedElements(pointsTo.size());
        auto elementWiseTransform = [&](std::span<const DataTypeA> targetArray){
            RetType diff = targetArray[index] - operandA;
            return diff*diff;
        };
        std::transform(std::execution::unseq, pointsTo.begin(), pointsTo.end(), transformedElements.begin(), elementWiseTransform);
        //RetType diff = static_cast<RetType>(operandA) - static_cast<RetType>(operandB);
        return transformedElements;
    };

    auto accumulator = [] (std::vector<RetType> operandA, std::vector<RetType> operandB){
        for (size_t i = 0; i<operandA.size(); i+=1){
          operandA[i] += operandB[i];
        }
        return operandA;
    };
    std::vector<RetType> accum;
    for (size_t i = 0; i<pointB.size(); i+=1){
        std::vector<RetType> tmp = transformFunc(pointB[i], i);
        accum = accumulator(accum, tmp);
    }
    
    //std::vector<RetType> accum = std::transform_reduce(
    //                                std::begin(pointB),
    //                                std::end(pointB),
    //                                std::views::iota(0).begin(),
    //                                results,
    //                                accumulator,
    //                                transformFunc);
    
    return accum;

}
*/


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





/* 
    Some functions for benchmarking. From Chandler Carruth's talk at Cppcon2015: https://www.youtube.com/watch?v=nXaxk27zwlk
    Don't use these in actual production code!
*/

static void escape(void* p){
    asm volatile("" : : "g"(p) : "memory");
}

static void clobber(){
    asm volatile("" : : : "memory");
}

// End benchmarking functions

int main(){

    static const std::endian dataEndianness = std::endian::big;

    std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");
    nnd::DataSet<std::valarray<float>> mnistFashionTrain(trainDataFilePath, 28*28, 60'000, &nnd::ExtractNumericArray<float,dataEndianness>);

    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), mnistFashionTrain.numberOfSamples - 1);

    
    auto eNorm = &nnd::EuclideanNorm<float,float,float>;
    auto beNorm = &BatchEuclideanNorm;
    clobber();
    std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();
    /*
    std::vector<size_t> targetPoints(7);
    std::vector<std::span<const float>> targetSpans;
    for (auto& point: targetPoints){
        point = rngDist(rngEngine);
        targetSpans.emplace_back(mnistFashionTrain[point]);
    }

    size_t startPoint = rngDist(rngEngine);

    std::vector<float> serialRes;

    for (const auto point: targetPoints){
        serialRes.push_back(nnd::EuclideanNorm<float,float,float>(mnistFashionTrain[startPoint], mnistFashionTrain[point]));
        
    }
    std::vector<float> batchRes = BatchEuclideanNorm(targetSpans, std::span<const float>(mnistFashionTrain[startPoint]));
    */
    
    for(size_t i = 0; i<1'000'000; i+=1){
        std::vector<size_t> targetPoints(7);
        std::vector<std::span<const float>> targetSpans;
        for (auto& point: targetPoints){
            point = rngDist(rngEngine);
            targetSpans.emplace_back(mnistFashionTrain[point]);
        }

        size_t startPoint = rngDist(rngEngine);

        for (const auto point: targetPoints){
            float dist = nnd::EuclideanNorm<float,float,float>(mnistFashionTrain[startPoint], mnistFashionTrain[point]);
            escape(&dist);
        }
    }
    std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 1 dist at a time" << std::endl;
    
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<1'000'000; i+=1){
        std::vector<size_t> targetPoints(7);
        std::vector<std::span<const float>> targetSpans;
        for (auto& point: targetPoints){
            point = rngDist(rngEngine);
            targetSpans.emplace_back(mnistFashionTrain[point]);
        }

        size_t startPoint = rngDist(rngEngine);

        std::vector<float> dists = BatchEuclideanNorm(targetSpans, mnistFashionTrain[startPoint]);
        escape(dists.data());
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 7 dist at a time" << std::endl;
    
    return 0;
}