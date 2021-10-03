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


#include "../../Utilities/Type.hpp"
#include "../../Utilities/Metrics/FunctorErasure.hpp"
#include "../Type.hpp"

#include "GraphVertex.hpp"
#include "Graph.hpp"

namespace nnd{

template<typename DistType>
struct CachingFunctor{

    DispatchFunctor<DistType> metricFunctor;
    //DistanceCache<DistType> cache;
    Graph<DataIndex_t, DistType> reverseGraph;
    std::vector<NodeTracker> nodesJoined;
    //std::vector<DistType> minDists;
    size_t numNeighbors;
    size_t maxBlockSize;

    CachingFunctor(DispatchFunctor<DistType>& metricFunctor, size_t maxBlockSize, size_t numNeighbors):
        metricFunctor(metricFunctor), 
        reverseGraph(maxBlockSize, numNeighbors),
        nodesJoined(maxBlockSize, NodeTracker(maxBlockSize)),
        numNeighbors(numNeighbors),
        maxBlockSize(maxBlockSize) {};

    CachingFunctor() = default;

    CachingFunctor(const CachingFunctor&) = default;

    CachingFunctor& operator= (const CachingFunctor&) = default;

    DistType operator()(const size_t queryIndex, const size_t targetIndex){
        DistType distance = this->metricFunctor(queryIndex, targetIndex);
        cachedGraphSize = std::max(targetIndex, cachedGraphSize);
        //minDists[targetIndex] = std::min(minDists[targetIndex], distance);
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
        /*
        if(reverseGraph[targetIndex].size() == numNeighbors){
            reverseGraph[targetIndex].PushNeighbor({queryIndex, distance});
        } else if(reverseGraph[targetIndex].size() == numNeighbors-1){
            reverseGraph[targetIndex].push_back({queryIndex, distance});
            reverseGraph[targetIndex].JoinPrep();
        } else{
            reverseGraph[targetIndex].push_back({queryIndex, distance});
        }
        */
        nodesJoined[targetIndex][queryIndex] = true;
        
        return distance;
    };

    std::vector<DistType> operator()(const size_t queryIndex, const std::vector<size_t>& targetIndecies){
        for(const auto& index: targetIndecies) cachedGraphSize = std::max(index, cachedGraphSize);
        for(const auto& index: targetIndecies) nodesJoined[index][queryIndex] = true;
        std::vector<DistType> distances = this->metricFunctor(queryIndex, targetIndecies);

        
        
        for (size_t i = 0; i<targetIndecies.size(); i+=1){
            //cachedGraphSize = std::max(targetIndecies[i], cachedGraphSize);
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
            //nodesJoined[targetIndecies[i]][queryIndex] = true;
        }
        
        /*
        const size_t* start = targetIndecies.data();
        const size_t* end = start + targetIndecies.size();

        DistType* distancePtr = distances.data();
        for ( ; start<end; start++){    
            cachedGraphSize = std::max(*start, cachedGraphSize);
            int diff = numNeighbors - reverseGraph[targetIndecies[i]].size();
            switch(diff){
                case 0:
                    reverseGraph[targetIndecies[i]].PushNeighbor({queryIndex, distances[i]});
                    break;
                case 1:
                    reverseGraph[targetIndecies[i]].push_back({queryIndex, distances[i]});
                    reverseGraph[targetIndecies[i]].JoinPrep();
                    break;
                default:
                    reverseGraph[targetIndecies[i]].push_back({queryIndex, distances[i]});
            }
            nodesJoined[targetIndecies[i]][queryIndex] = true;
        }
        */
        return distances;
    };

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum){
        cachedGraphSize = 0;
        for (auto& vertex: reverseGraph){
            vertex.resize(0);
        }
        for (auto& tracker: nodesJoined){
            tracker = NodeTracker(maxBlockSize);
        }
        //for (auto& minDist: minDists){
        //    minDist = std::numeric_limits<DistType>::max();
        //}
        this->metricFunctor.SetBlocks(lhsBlockNum, rhsBlockNum);
    }

    using iterator = typename std::vector<GraphVertex<size_t, DistType>>::iterator;
    using const_iterator = typename std::vector<GraphVertex<size_t, DistType>>::const_iterator;

    size_t size() const noexcept{
        return cachedGraphSize + 1;
    }
    
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

    private:
    size_t cachedGraphSize;
};

}

#endif