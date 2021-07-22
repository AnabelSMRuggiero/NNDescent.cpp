#include <numeric>
#include <span>
#include <algorithm>
#include <execution>
#include <bit>
#include <random>
#include <chrono>
#include <cassert>
#include <array>
#include <memory>
#include <type_traits>

#include <immintrin.h>

#include "Utilities/Type.hpp"

#include "NND/SpaceMetrics.hpp"
#include "Utilities/Data.hpp"
#include "Utilities/DataDeserialization.hpp"


using namespace nnd;



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



template<size_t prefetchPeriod = 8>
std::vector<float> BatchEuclideanNorm(const std::vector<AlignedSpan<const float>>& pointsTo, const AlignedSpan<const float>& pointB){
    
    //size_t prefetchPeriod = 16;
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
        //Pre load first set of elements
        fromComponent1 = _mm256_load_ps(&(pointB[0]));
        for (size_t i = 0; i < 7; i+=1){
            toComponents[i] = _mm256_load_ps(&(pointsTo[i][0]));
        }
        while(index+15<pointB.size()){
            //Prefetch future data
            _mm_prefetch(&(pointB[index+(8*prefetchPeriod)]), _MM_HINT_T0);
            for (size_t j = 0; j < 7; j+=1){
                _mm_prefetch(&(pointsTo[index+(8*prefetchPeriod)]), _MM_HINT_T0);
            }

            for (size_t j = 0; j<prefetchPeriod; j+=1){
            //Core computation loop
                for(;index+15<pointB.size(); index+=8){
                    fromComponent2 = _mm256_load_ps(&(pointB[index+8]));
                    for(size_t j = 0; j<7; j+=1){
                        toComponents[j] = _mm256_sub_ps(toComponents[j], fromComponent1);
                        accumulators[j] = _mm256_fmadd_ps(toComponents[j], toComponents[j], accumulators[j]);
                        //Load for next iteration
                        toComponents[j] = _mm256_load_ps(&(pointsTo[j][index+8]));
                    }
                    fromComponent1 = fromComponent2;
                }
            }
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

template<>
std::vector<float> BatchEuclideanNorm<0>(const std::vector<AlignedSpan<const float>>& pointsTo, const AlignedSpan<const float>& pointB){
    
    //size_t prefetchPeriod = 16;
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
        //Pre load first set of elements
        fromComponent1 = _mm256_load_ps(&(pointB[0]));
        for (size_t i = 0; i < 7; i+=1){
            toComponents[i] = _mm256_load_ps(&(pointsTo[i][0]));
        }
        

        
        //Core computation loop
        for(;index+15<pointB.size(); index+=8){
            fromComponent2 = _mm256_load_ps(&(pointB[index+8]));
            for(size_t j = 0; j<7; j+=1){
                toComponents[j] = _mm256_sub_ps(toComponents[j], fromComponent1);
                accumulators[j] = _mm256_fmadd_ps(toComponents[j], toComponents[j], accumulators[j]);
                //Load for next iteration
                toComponents[j] = _mm256_load_ps(&(pointsTo[j][index+8]));
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




// End benchmarking functions

int main(){
    //std::unique_ptr<float[]>test(new (std::align_val_t(32)) float[8]);
    static const std::endian dataEndianness = std::endian::big;
    AlignedArray<float> (*test)(std::ifstream&, size_t) =  &nnd::ExtractNumericArray<AlignedArray<float>, dataEndianness>;
    std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");
    nnd::DataSet<AlignedArray<float>> mnistFashionTrain(trainDataFilePath, 28*28, 60'000, &nnd::ExtractNumericArray<AlignedArray<float>, dataEndianness>);

    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), mnistFashionTrain.numberOfSamples - 1);

    std::chrono::time_point<std::chrono::steady_clock> runStart, runEnd;
    

    
    /*
    std::vector<size_t> targetPoints(7);
    std::vector<AlignedSpan<const float>> targetSpans;
    for (auto& point: targetPoints){
        point = rngDist(rngEngine);
        targetSpans.emplace_back(mnistFashionTrain[point]);
    }

    size_t startPoint = rngDist(rngEngine);
    
    std::vector<float> serialRes;

    for (const auto point: targetPoints){
        serialRes.push_back(nnd::EuclideanNorm<AlignedArray,float,float,float>(mnistFashionTrain[startPoint], mnistFashionTrain[point]));
        
    }
    std::vector<float> batchRes = BatchEuclideanNorm(targetSpans, AlignedSpan<const float>(mnistFashionTrain[startPoint]));

    for (auto dist: serialRes){
        std::cout << dist << ' ';
    }
    std::cout << std::endl;

    for (auto dist: batchRes){
        std::cout << dist << ' ';
    }
    std::cout << std::endl;
    */
    /*
    runStart = std::chrono::steady_clock::now();
    for (size_t i = 0; i<10'000'000; i += 1){
        std::vector<size_t> targetPoints(7);
        std::vector<AlignedSpan<const float>> targetSpans;
        for (auto& point: targetPoints){
            point = rngDist(rngEngine);
            targetSpans.emplace_back(mnistFashionTrain[point]);
        }
        escape(targetSpans.data());

        size_t startPoint = rngDist(rngEngine);
        escape(&startPoint);
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for index picking" << std::endl;
    */
    /*
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<10'000'000; i+=1){
        std::vector<size_t> targetPoints(7);
        std::vector<AlignedSpan<const float>> targetSpans;
        for (auto& point: targetPoints){
            point = rngDist(rngEngine);
            targetSpans.emplace_back(mnistFashionTrain[point]);
        }
        escape(targetSpans.data());

        size_t startPoint = rngDist(rngEngine);
        escape(&startPoint);

        for (const auto point: targetPoints){
            float dist = nnd::EuclideanNorm<AlignedArray,float,float,float>(mnistFashionTrain[startPoint], mnistFashionTrain[point]);
            escape(&dist);
        }
    }
    
    
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 1 dist at a time" << std::endl;
    */
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<100'000'000; i+=1){
        std::vector<size_t> targetPoints(7);
        //std::vector<std::span<const float>> targetSpans;
        
        std::vector<AlignedSpan<const float>> targetSpans;
        for (auto& point: targetPoints){
            point = rngDist(rngEngine);
            targetSpans.push_back(AlignedSpan<const float>(mnistFashionTrain[point]));
        }
        escape(targetSpans.data());

        size_t startPoint = rngDist(rngEngine);
        escape(&startPoint);

        std::vector<float> dists = BatchEuclideanNorm<0>(targetSpans, AlignedSpan<const float>(mnistFashionTrain[startPoint]));
        escape(dists.data());
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 7 dist at a time (no prefetch)" << std::endl;
    
    rngEngine.seed(0);
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<100'000'000; i+=1){
        std::vector<size_t> targetPoints(7);
        //std::vector<std::span<const float>> targetSpans;
        
        std::vector<AlignedSpan<const float>> targetSpans;
        for (auto& point: targetPoints){
            point = rngDist(rngEngine);
            targetSpans.push_back(AlignedSpan<const float>(mnistFashionTrain[point]));
        }
        escape(targetSpans.data());

        size_t startPoint = rngDist(rngEngine);
        escape(&startPoint);

        std::vector<float> dists = BatchEuclideanNorm<2>(targetSpans, AlignedSpan<const float>(mnistFashionTrain[startPoint]));
        escape(dists.data());
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 7 dist at a time (prefetch=2)" << std::endl;

    rngEngine.seed(0);
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<100'000'000; i+=1){
        std::vector<size_t> targetPoints(7);
        //std::vector<std::span<const float>> targetSpans;
        
        std::vector<AlignedSpan<const float>> targetSpans;
        for (auto& point: targetPoints){
            point = rngDist(rngEngine);
            targetSpans.push_back(AlignedSpan<const float>(mnistFashionTrain[point]));
        }
        escape(targetSpans.data());

        size_t startPoint = rngDist(rngEngine);
        escape(&startPoint);

        std::vector<float> dists = BatchEuclideanNorm<4>(targetSpans, AlignedSpan<const float>(mnistFashionTrain[startPoint]));
        escape(dists.data());
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 7 dist at a time (prefetch=4)" << std::endl;
    

    rngEngine.seed(0);
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<100'000'000; i+=1){
        std::vector<size_t> targetPoints(7);
        //std::vector<std::span<const float>> targetSpans;
        
        std::vector<AlignedSpan<const float>> targetSpans;
        for (auto& point: targetPoints){
            point = rngDist(rngEngine);
            targetSpans.push_back(AlignedSpan<const float>(mnistFashionTrain[point]));
        }
        escape(targetSpans.data());

        size_t startPoint = rngDist(rngEngine);
        escape(&startPoint);

        std::vector<float> dists = BatchEuclideanNorm<8>(targetSpans, AlignedSpan<const float>(mnistFashionTrain[startPoint]));
        escape(dists.data());
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 7 dist at a time (prefetch=8)" << std::endl;
    

    rngEngine.seed(0);
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<100'000'000; i+=1){
        std::vector<size_t> targetPoints(7);
        //std::vector<std::span<const float>> targetSpans;
        
        std::vector<AlignedSpan<const float>> targetSpans;
        for (auto& point: targetPoints){
            point = rngDist(rngEngine);
            targetSpans.push_back(AlignedSpan<const float>(mnistFashionTrain[point]));
        }
        escape(targetSpans.data());

        size_t startPoint = rngDist(rngEngine);
        escape(&startPoint);

        std::vector<float> dists = BatchEuclideanNorm<16>(targetSpans, AlignedSpan<const float>(mnistFashionTrain[startPoint]));
        escape(dists.data());
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 7 dist at a time (prefetch=16)" << std::endl;
    
    rngEngine.seed(0);
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<100'000'000; i+=1){
        std::vector<size_t> targetPoints(7);
        //std::vector<std::span<const float>> targetSpans;
        
        std::vector<AlignedSpan<const float>> targetSpans;
        for (auto& point: targetPoints){
            point = rngDist(rngEngine);
            targetSpans.push_back(AlignedSpan<const float>(mnistFashionTrain[point]));
        }
        escape(targetSpans.data());

        size_t startPoint = rngDist(rngEngine);
        escape(&startPoint);

        std::vector<float> dists = BatchEuclideanNorm<32>(targetSpans, AlignedSpan<const float>(mnistFashionTrain[startPoint]));
        escape(dists.data());
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 7 dist at a time (prefetch=32)" << std::endl;
  
    return 0;
}