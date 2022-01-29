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

#include <vector>
#include <span>
#include <memory_resource>
#include <algorithm>

#include "NND/UtilityFunctions.hpp"
#include "ann/Type.hpp"
#include "../FunctorErasure.hpp"
#include "../Type.hpp"

#include "GraphVertex.hpp"
#include "Graph.hpp"

namespace nnd{

template<typename DistType>
struct CachingFunctor{

    private:

    struct CachedResult{
        size_t queryIndex;
        std::span<size_t> targetIndecies;
        std::span<DistType> distances;
    };

    std::vector<CachedResult> accumulatedResults;
    std::pmr::monotonic_buffer_resource cacheMemory{16'800};

    struct ReverseGraph{
        using iterator = typename Graph<DataIndex_t, DistType>::iterator;
        using const_iterator = typename Graph<DataIndex_t, DistType>::const_iterator;
        Graph<DataIndex_t, DistType> reverseGraph;
        std::vector<NodeTracker> nodesJoined;
        size_t cachedGraphSize{0};

        constexpr iterator begin() noexcept{
            return reverseGraph.begin();
        }

        constexpr const_iterator begin() const noexcept{
            return reverseGraph.begin();
        }

        constexpr const_iterator cbegin() const noexcept{
            return reverseGraph.cbegin();
        }

        constexpr iterator end() noexcept{
            return reverseGraph.begin() + cachedGraphSize + 1;
        }

        constexpr const_iterator end() const noexcept{
            return reverseGraph.begin() + cachedGraphSize + 1;
        }

        constexpr const_iterator cend() const noexcept{
            return reverseGraph.begin() + cachedGraphSize + 1;
        }

        size_t size() const noexcept{
            return cachedGraphSize + 1;
        }
    };

    ReverseGraph results;

    public:

    DispatchFunctor<DistType> metricFunctor;
    //Graph<DataIndex_t, DistType> reverseGraph;
    //std::vector<NodeTracker> nodesJoined;

    size_t numNeighbors;
    size_t maxBlockSize;

    CachingFunctor(DispatchFunctor<DistType>& metricFunctor, size_t maxBlockSize, size_t numNeighbors):
        results{
            .reverseGraph = {maxBlockSize, numNeighbors},
            .nodesJoined = [maxBlockSize](){
                std::vector<NodeTracker> initVec;
                initVec.resize(maxBlockSize);
                std::ranges::for_each(initVec, [&](auto& tracker){ tracker.resize(maxBlockSize); });
                return initVec;
            }()
        },
        metricFunctor(metricFunctor), 
        numNeighbors(numNeighbors),
        maxBlockSize(maxBlockSize) {};

    CachingFunctor() = default;

    CachingFunctor(const CachingFunctor&) = default;

    CachingFunctor& operator= (const CachingFunctor&) = default;

    DistType operator()(const size_t queryIndex, const size_t targetIndex){

        DistType distance = this->metricFunctor(queryIndex, targetIndex);

        std::pmr::polymorphic_allocator<> alloc{&cacheMemory};

        auto addNode = [&] ()->CachedResult {
            size_t* indexStorage = alloc.allocate_object<size_t>();
            DistType* distanceStorage = alloc.allocate_object<DistType>();

            *indexStorage = targetIndex;
            *distanceStorage = distance;

            return {
                .queryIndex = queryIndex, 
                .targetIndecies = {indexStorage, 1},
                .distances = {distanceStorage, 1}
            };
        };

        accumulatedResults.emplace_back(addNode());

        /*
        cachedGraphSize = std::max(targetIndex, cachedGraphSize);

        int diff = numNeighbors - reverseGraph[targetIndex].size();

        switch(diff){
            case 0:
                reverseGraph[targetIndex].PushNeighbor({static_cast<DataIndex_t>(queryIndex), distance});
                break;
            case 1:
                reverseGraph[targetIndex].push_back({static_cast<DataIndex_t>(queryIndex), distance});
                reverseGraph[targetIndex].JoinPrep();
                break;
            default:
                reverseGraph[targetIndex].push_back({static_cast<DataIndex_t>(queryIndex), distance});
                break;
        }

        nodesJoined[targetIndex][queryIndex] = true;
        */
        return distance;
    };

    std::ranges::contiguous_range auto operator()(const size_t queryIndex, std::span<const size_t> targetIndecies){

        
        std::ranges::contiguous_range auto distances = this->metricFunctor(queryIndex, targetIndecies);
        
        std::pmr::polymorphic_allocator<> alloc{&cacheMemory};

        auto addNode = [&] ()->CachedResult {
            size_t numIndecies = targetIndecies.size();
            size_t* indexStorage = alloc.allocate_object<size_t>(numIndecies);
            DistType* distanceStorage = alloc.allocate_object<DistType>(numIndecies);

            std::ranges::copy(targetIndecies, indexStorage);
            std::ranges::copy(distances, distanceStorage);

            return {
                .queryIndex = queryIndex, 
                .targetIndecies = {indexStorage, numIndecies},
                .distances = {distanceStorage, numIndecies}
            };
        };

        accumulatedResults.emplace_back(addNode());

        //CachedResult& result = accumulatedResults.back();


        
        /*
        for(const auto& index: targetIndecies) cachedGraphSize = std::max(index, cachedGraphSize);
        for(const auto& index: targetIndecies) nodesJoined[index][queryIndex] = true;
        */
        /* Loop through and add results to reverse graph */
        /*
        for (size_t i = 0; i<targetIndecies.size(); i+=1){
            int diff = numNeighbors - reverseGraph[targetIndecies[i]].size();
            switch(diff){
                case 0:
                    reverseGraph[targetIndecies[i]].PushNeighbor({static_cast<DataIndex_t>(queryIndex), distances[i]});
                    break;
                case 1:
                    reverseGraph[targetIndecies[i]].push_back({static_cast<DataIndex_t>(queryIndex), distances[i]});
                    reverseGraph[targetIndecies[i]].JoinPrep();
                    break;
                default:
                    reverseGraph[targetIndecies[i]].push_back({static_cast<DataIndex_t>(queryIndex), distances[i]});
            }
        }
        */
        return distances;
    };

    void ReduceResults(){
        for (const auto& result : accumulatedResults){
            const size_t queryIndex = result.queryIndex;

            for (size_t i = 0; i<result.targetIndecies.size(); i+=1){
                const size_t target = result.targetIndecies[i];
                const DistType distance = result.distances[i];

                results.cachedGraphSize = std::max(target, results.cachedGraphSize);
                
                results.nodesJoined[target][queryIndex] = true;


                //results.reverseGraph[target].push_back({static_cast<DataIndex_t>(queryIndex), distance});
                
                int diff = numNeighbors - results.reverseGraph[target].size();
                switch(diff){
                    case 0:
                        results.reverseGraph[target].PushNeighbor({static_cast<DataIndex_t>(queryIndex), distance});
                        break;
                    case 1:
                        results.reverseGraph[target].push_back({static_cast<DataIndex_t>(queryIndex), distance});
                        results.reverseGraph[target].JoinPrep();
                        break;
                    default:
                        results.reverseGraph[target].push_back({static_cast<DataIndex_t>(queryIndex), distance});
                }
                
            }
        }
        /*
        for(auto& cachedVec : results.reverseGraph){
            if(cachedVec.size() >= numNeighbors){
                std::partial_sort(cachedVec.begin(), cachedVec.begin() + numNeighbors, cachedVec.end(), edge_ops::lessThan);
                cachedVec.resize(numNeighbors);
            }
        }
        */
        accumulatedResults.clear();
        cacheMemory.release();
    }

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum){
        results.cachedGraphSize = 0;
        for (auto& vertex: results.reverseGraph){
            vertex.resize(0);
        }
        for (auto& tracker: results.nodesJoined){
            tracker.clear();
            tracker.resize(maxBlockSize);
        }
        //for (auto& minDist: minDists){
        //    minDist = std::numeric_limits<DistType>::max();
        //}
        this->metricFunctor.SetBlocks(lhsBlockNum, rhsBlockNum);
    }

    using iterator = typename std::vector<GraphVertex<size_t, DistType>>::iterator;
    using const_iterator = typename std::vector<GraphVertex<size_t, DistType>>::const_iterator;

    size_t size() const noexcept{
        return results.cachedGraphSize + 1;
    }

    ReverseGraph& AccessCache(){

        ReduceResults();
        return results;
    }

    /*
    constexpr iterator begin() noexcept{
        return reverseGraph.begin();
    }

    constexpr const_iterator begin() const noexcept{
        return reverseGraph.begin();
    }

    constexpr const_iterator cbegin() const noexcept{
        return reverseGraph.cbegin();
    }

    constexpr iterator end() noexcept{
        return reverseGraph.begin() + cachedGraphSize + 1;
    }

    constexpr const_iterator end() const noexcept{
        return reverseGraph.begin() + cachedGraphSize + 1;
    }

    constexpr const_iterator cend() const noexcept{
        return reverseGraph.begin() + cachedGraphSize + 1;
    }
    */

};

}

#endif