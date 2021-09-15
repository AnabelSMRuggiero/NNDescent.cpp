/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

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
#include <cstring>

#include <immintrin.h>

#include "Utilities/Type.hpp"

#include "Utilities/Metrics/SpaceMetrics.hpp"
#include "Utilities/Data.hpp"
#include "Utilities/DataDeserialization.hpp"

#include "NND/GraphStructures.hpp"
#include "NND/UtilityFunctions.hpp"

using namespace nnd;


template<typename IndexType, typename FloatType>
struct SortedVertex{

    using iterator = typename std::vector<std::pair<IndexType, FloatType>>::iterator;
    using const_iterator = typename std::vector<std::pair<IndexType, FloatType>>::const_iterator;
    std::vector<std::pair<IndexType, FloatType>> neighbors;
    //std::vector<size_t> reverseNeighbor;

    SortedVertex(): neighbors(0){};

    SortedVertex(size_t numNeighbors): neighbors(0) {
        this->neighbors.reserve(numNeighbors + 1);
    };

    //GraphVertex(GraphVertex&& rval): neighbors(std::forward<std::vector<std::pair<IndexType, FloatType>>>(rval.neighbors)){};
    //Incorporate size checking in here?
    bool PushNeighbor(std::pair<IndexType, FloatType> newNeighbor){
        if (newNeighbor.second > neighbors.back().second) return false;
        //neighbors.push_back(newNeighbor);
        //[&](std::pair<IndexType, FloatType>& elementToCompare){return NeighborDistanceComparison<IndexType, FloatType>()}
        //auto insertionPoint = std::upper_bound(neighbors.begin(), neighbors.end()-1, neighbors.back(), NeighborDistanceComparison<IndexType, FloatType>);
        size_t index = neighbors.size();
        for ( ; index>0; index -= 1){
            if (NeighborDistanceComparison<IndexType, FloatType>(neighbors[index-1], newNeighbor)) break;
        }
        
        neighbors.push_back(newNeighbor);
        std::memmove(&neighbors[index+1], &neighbors[index], sizeof(std::pair<IndexType, FloatType>)*(neighbors.size()-1 - index));
        neighbors[index] = newNeighbor;
        //size_t index = std::transform_reduce(std::execution::unseq, neighbors.begin(), neighbors.end()-1, size_t(0), std::plus<size_t>(),
        //    [&](std::pair<IndexType,FloatType>& entry){
        //        return NeighborDistanceComparison<IndexType, FloatType>(entry, newNeighbor);
        //    });
        //std::rotate(std::execution::unseq, neighbors.begin()+index, neighbors.end()-1, neighbors.end());
        //std::rotate(std::execution::unseq, insertionPoint, neighbors.end()-1, neighbors.end());
        neighbors.pop_back();
        return true;
    };

    std::pair<IndexType, FloatType> PushNeighbor(std::pair<IndexType, FloatType> newNeighbor, ReturnRemoved){
        if (newNeighbor.second > neighbors[0].second) return newNeighbor;
        //neighbors.push_back(newNeighbor);

        size_t index = neighbors.size();
        for ( ; index>0; index -= 1){
            if (NeighborDistanceComparison<IndexType, FloatType>(neighbors[index-1], newNeighbor)) break;
        }

        neighbors.push_back(newNeighbor);
        std::memmove(&neighbors[index+1], &neighbors[index], sizeof(std::pair<IndexType, FloatType>)*(neighbors.size()-1 - index));
        neighbors[index] = newNeighbor;

        std::pair<IndexType, FloatType> retValue = neighbors.back();
        neighbors.pop_back();

        return retValue;
    };

    void JoinPrep(){
        std::sort(neighbors.begin(), neighbors.end(), NeighborDistanceComparison<IndexType, FloatType>);
    }

    void UnPrep(){
        //std::sort_heap(neighbors.begin(), neighbors.end(), NeighborDistanceComparison<IndexType, FloatType>);
        return; //noop
    }
    
    FloatType PushThreshold() const noexcept{
        return neighbors.back().second;
    }

    //Object Composition stuff below here

    constexpr void pop_back(){
        neighbors.pop_back();
    }

    constexpr void push_back(const std::pair<IndexType, FloatType>& value){
        neighbors.push_back(value);
    }

    //template<typename PairReferenceType>
    constexpr void push_back(std::pair<IndexType, FloatType>&& value){
        neighbors.push_back(std::forward<std::pair<IndexType, FloatType>>(value));
    }

    constexpr std::pair<IndexType, FloatType>& operator[](size_t i){
        return neighbors[i];
    }

    constexpr const std::pair<IndexType, FloatType>& operator[](size_t i) const{
        return neighbors[i];
    }

    constexpr std::pair<IndexType, FloatType>& operator[](BlockIndecies i){
        // I'm assuming the block number is correct
        return neighbors[i.dataIndex];
    }

    constexpr const std::pair<IndexType, FloatType>& operator[](BlockIndecies i) const{
        // I'm assuming the block number is correct
        return neighbors[i.dataIndex];
    }

    size_t size() const noexcept{
        return neighbors.size();
    }
    
    constexpr iterator begin() noexcept{
        return neighbors.begin();
    }

    constexpr const_iterator begin() const noexcept{
        return neighbors.begin();
    }

    constexpr const_iterator cbegin() const noexcept{
        return neighbors.cbegin();
    }

    constexpr iterator end() noexcept{
        return neighbors.end();
    }

    constexpr const_iterator end() const noexcept{
        return neighbors.end();
    }

    constexpr const_iterator cend() const noexcept{
        return neighbors.cend();
    }

    constexpr void resize(size_t count){
        neighbors.resize(count);
    }

    constexpr iterator erase(const_iterator first, const_iterator last) {
        return neighbors.erase(first, last);
    }

    //private:
    
};



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
    //std::unique_ptr<float[]>test(new (std::align_val_t(32)) float[8]);
    static const std::endian dataEndianness = std::endian::big;
    

    std::mt19937_64 rngEngine(0);
    std::uniform_real_distribution<float> rngDist(0.1, 0.9);

    SortedVertex<BlockIndecies, float> sorted(10);

    SortedVertex<BlockIndecies, float> sortedCopy(10);

    GraphVertex<BlockIndecies, float> heaped(10);

    GraphVertex<BlockIndecies, float> heapedCopy(10);
    
    for (size_t i = 0; i<10; i+=1){
        sortedCopy.push_back({{0,0}, 0.9});
        heapedCopy.push_back({{0,0}, 0.9});
    }
        
    sorted = sortedCopy;
    heaped = heapedCopy;

    std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();
    for (size_t i = 0; i < 10'000; i+=1){
        escape(sorted.neighbors.data());
        for (size_t j = 0; j < 100'000; j+=1){
            sorted.PushNeighbor({{0,0}, rngDist(rngEngine)});
            escape(sorted.neighbors.data());
        }
        sorted = sortedCopy;
    }
    std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for sorted pushes" << std::endl;

    rngEngine.seed(0);

    std::chrono::time_point<std::chrono::steady_clock> heapedStart = std::chrono::steady_clock::now();
    for (size_t i = 0; i < 10'000; i+=1){
        escape(heaped.neighbors.data());
        for (size_t j = 0; j < 100'000; j+=1){
            heaped.PushNeighbor({{0,0}, rngDist(rngEngine)});
            escape(heaped.neighbors.data());
        }
        heaped = heapedCopy;
    }
    std::chrono::time_point<std::chrono::steady_clock> heapedEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(heapedEnd - heapedStart).count() << "s total for heaped pushes" << std::endl;

    
    /*
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<1'000'000; i+=1){
        

        size_t startPoint = rngDist(rngEngine);
        escape(&startPoint);

        std::vector<float> dists = BatchEuclideanNorm<7>(targetSpans, AlignedSpan<const float>(mnistFashionTrain[startPoint]));
        escape(dists.data());
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 8 dist at a time (no prefetch)" << std::endl;
    */
    
    /*
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<1'000'000; i+=1){
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

        std::vector<float> dists = BatchEuclideanNorm<7>(targetSpans, AlignedSpan<const float>(mnistFashionTrain[startPoint]));
        escape(dists.data());
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 7 dist at a time (no prefetch)" << std::endl;
    
    rngEngine.seed(0);
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<1'000'000; i+=1){
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

        std::vector<float> dists = BatchEuclideanNorm<7,2>(targetSpans, AlignedSpan<const float>(mnistFashionTrain[startPoint]));
        escape(dists.data());
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 7 dist at a time (prefetch=2)" << std::endl;

    rngEngine.seed(0);
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<1'000'000; i+=1){
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

        std::vector<float> dists = BatchEuclideanNorm<7,4>(targetSpans, AlignedSpan<const float>(mnistFashionTrain[startPoint]));
        escape(dists.data());
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 7 dist at a time (prefetch=4)" << std::endl;
    

    rngEngine.seed(0);
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<1'000'000; i+=1){
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

        std::vector<float> dists = BatchEuclideanNorm<7,8>(targetSpans, AlignedSpan<const float>(mnistFashionTrain[startPoint]));
        escape(dists.data());
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 7 dist at a time (prefetch=8)" << std::endl;
    

    rngEngine.seed(0);
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<1'000'000; i+=1){
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

        std::vector<float> dists = BatchEuclideanNorm<7,16>(targetSpans, AlignedSpan<const float>(mnistFashionTrain[startPoint]));
        escape(dists.data());
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 7 dist at a time (prefetch=16)" << std::endl;
    
    rngEngine.seed(0);
    runStart = std::chrono::steady_clock::now();
    for(size_t i = 0; i<1'000'000; i+=1){
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

        std::vector<float> dists = BatchEuclideanNorm<7,32>(targetSpans, AlignedSpan<const float>(mnistFashionTrain[startPoint]));
        escape(dists.data());
    }
    runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for 7 dist at a time (prefetch=32)" << std::endl;
    */
    return 0;
}