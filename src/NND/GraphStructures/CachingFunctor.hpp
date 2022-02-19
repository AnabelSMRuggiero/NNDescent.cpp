/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_CACHINGFUNCTOR_HPP
#define NND_CACHINGFUNCTOR_HPP

#include <algorithm>
#include <memory_resource>
#include <span>
#include <vector>

#include "../FunctorErasure.hpp"
#include "../Type.hpp"
#include "NND/UtilityFunctions.hpp"
#include "ann/Type.hpp"

#include "Graph.hpp"
#include "GraphVertex.hpp"

namespace nnd {

template<typename DistanceType>
struct cached_result {
    size_t queryIndex;
    std::span<size_t> targetIndecies;
    std::span<DistanceType> distances;
};

template<typename DistanceType>
struct reverse_graph {
    using iterator = typename Graph<DataIndex_t, DistanceType>::iterator;
    using const_iterator = typename Graph<DataIndex_t, DistanceType>::const_iterator;
    Graph<DataIndex_t, DistanceType> reverseGraph;
    std::vector<NodeTracker> nodesJoined;
    size_t cachedGraphSize{ 0 };

    constexpr iterator begin() noexcept { return reverseGraph.begin(); }

    constexpr const_iterator begin() const noexcept { return reverseGraph.begin(); }

    constexpr const_iterator cbegin() const noexcept { return reverseGraph.cbegin(); }

    constexpr iterator end() noexcept { return reverseGraph.begin() + cachedGraphSize + 1; }

    constexpr const_iterator end() const noexcept { return reverseGraph.begin() + cachedGraphSize + 1; }

    constexpr const_iterator cend() const noexcept { return reverseGraph.begin() + cachedGraphSize + 1; }

    size_t size() const noexcept { return cachedGraphSize + 1; }
};



template<typename DistanceType>
struct cache_state {

    cache_state(size_t maxBlockSize, size_t numNeighbors)
        : maxBlockSize{ maxBlockSize }, numNeighbors{ numNeighbors },
          results{ .reverseGraph = { maxBlockSize, numNeighbors }, .nodesJoined = [maxBlockSize]() {
                                  std::vector<NodeTracker> initVec;
                                  initVec.resize(maxBlockSize);
                                  std::ranges::for_each(initVec, [&](auto& tracker) { tracker.resize(maxBlockSize); });
                                  return initVec;
                              }() } {}
    
    size_t maxBlockSize;
    size_t numNeighbors;
    std::vector<cached_result<DistanceType>> accumulatedResults;
    reverse_graph<DistanceType> results;
    std::pmr::monotonic_buffer_resource cacheMemory{ 16'800 };

    cache_state& reset_results(){

        for (auto& vertex : results.reverseGraph){
            vertex.resize(0);
        }
        for (auto& tracker : results.nodesJoined){
            tracker.clear();
            tracker.resize(maxBlockSize);
        }

        results.cachedGraphSize = 0;
        return *this;
    }

    void reduce_results() {
        for (const auto& result : accumulatedResults) {
            const size_t queryIndex = result.queryIndex;

            for (size_t i = 0; i < result.targetIndecies.size(); i += 1) {
                const size_t target = result.targetIndecies[i];
                const DistanceType distance = result.distances[i];

                results.cachedGraphSize = std::max(target, results.cachedGraphSize);

                results.nodesJoined[target][queryIndex] = true;

                // results.reverseGraph[target].push_back({static_cast<DataIndex_t>(queryIndex), distance});

                int diff = numNeighbors - results.reverseGraph[target].size();
                switch (diff) {
                    case 0:
                        results.reverseGraph[target].PushNeighbor({ static_cast<DataIndex_t>(queryIndex), distance });
                        break;
                    case 1:
                        results.reverseGraph[target].push_back({ static_cast<DataIndex_t>(queryIndex), distance });
                        results.reverseGraph[target].JoinPrep();
                        break;
                    default:
                        results.reverseGraph[target].push_back({ static_cast<DataIndex_t>(queryIndex), distance });
                }
            }
        }
        
        accumulatedResults.clear();
        cacheMemory.release();
    }
};



template<typename DistType>
struct caching_functor {

  private:
    cache_state<DistType>& resultsCache;

  public:
    erased_metric<DistType> metricFunctor;
    // Graph<DataIndex_t, DistType> reverseGraph;
    // std::vector<NodeTracker> nodesJoined;

    caching_functor(cache_state<DistType>& resultsCache, erased_metric<DistType> metricFunctor)
        : resultsCache{ resultsCache.reset_results() }, metricFunctor(std::move(metricFunctor)){};

    caching_functor() = default;

    caching_functor(const caching_functor&) = default;

    caching_functor& operator=(const caching_functor&) = default;

    DistType operator()(const size_t queryIndex, const size_t targetIndex) {

        DistType distance = this->metricFunctor(queryIndex, targetIndex);

        std::pmr::polymorphic_allocator<> alloc{ &resultsCache.cacheMemory };

        auto addNode = [&]() -> cached_result<DistType> {
            size_t* indexStorage = alloc.allocate_object<size_t>();
            DistType* distanceStorage = alloc.allocate_object<DistType>();

            *indexStorage = targetIndex;
            *distanceStorage = distance;

            return { .queryIndex = queryIndex, .targetIndecies = { indexStorage, 1 }, .distances = { distanceStorage, 1 } };
        };

        resultsCache.accumulatedResults.emplace_back(addNode());


        return distance;
    };

    std::ranges::contiguous_range auto operator()(const size_t queryIndex, std::span<const size_t> targetIndecies) {

        std::ranges::contiguous_range auto distances = this->metricFunctor(queryIndex, targetIndecies);

        std::pmr::polymorphic_allocator<> alloc{ &resultsCache.cacheMemory };

        auto addNode = [&]() -> cached_result<DistType> {
            size_t numIndecies = targetIndecies.size();
            size_t* indexStorage = alloc.allocate_object<size_t>(numIndecies);
            DistType* distanceStorage = alloc.allocate_object<DistType>(numIndecies);

            std::ranges::copy(targetIndecies, indexStorage);
            std::ranges::copy(distances, distanceStorage);

            return { .queryIndex = queryIndex,
                     .targetIndecies = { indexStorage, numIndecies },
                     .distances = { distanceStorage, numIndecies } };
        };

        resultsCache.accumulatedResults.emplace_back(addNode());

        return distances;
    };


};

} // namespace nnd

#endif