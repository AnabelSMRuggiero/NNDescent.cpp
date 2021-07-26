/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/


#ifndef NND_SUBGRAPHQUERY_HPP
#define NND_SUBGRAPHQUERY_HPP

#include <vector>
#include <unordered_map>
#include <limits>
#include <cassert>
#include <cstdint>
#include <utility>
#include <optional>

#include "../Utilities/Data.hpp"
#include "../Utilities/DataDeserialization.hpp"
#include "../Utilities/Metrics/FunctorErasure.hpp"
#include "GraphStructures.hpp"

namespace nnd{

template<TriviallyCopyable DataIndexType, typename DataEntry, typename DistType>
struct SubProblemData{

    const Graph<DataIndexType, DistType>& subGraph;
    const DataBlock<DataEntry>& dataBlock;

    SubProblemData(const Graph<DataIndexType, DistType>& subGraph, const DataBlock<DataEntry>& dataBlock): subGraph(subGraph), dataBlock(dataBlock) {};
};







template<TriviallyCopyable IndexType, typename DataEntry, std::floating_point FloatType>
std::tuple<size_t, size_t, FloatType> BruteNearestNodes(SubProblemData<IndexType, DataEntry, FloatType> subGraphA,
                  SubProblemData<IndexType, DataEntry, FloatType> subGraphB,
                  SpaceMetric<DataEntry, DataEntry, FloatType>  distanceFunctor){

    std::pair<size_t, size_t> bestPair;
    FloatType bestDistance(std::numeric_limits<FloatType>::max());


    for(size_t i = 0; i<subGraphA.dataBlock.size(); i+=1){
        for (size_t j = 0; j<subGraphB.dataBlock.size(); j+=1){
            FloatType distance = distanceFunctor(subGraphA.dataBlock[i], subGraphB.dataBlock[j]);
            if (distance < bestDistance){
                bestDistance = distance;
                bestPair = std::pair<size_t, size_t>(i,j);
            }
        }
    }
    
    return {bestPair.first, bestPair.second, bestDistance};
}

template<typename DataEntry, typename DataView, std::floating_point COMExtentType, typename DistType>
GraphVertex<size_t, DistType> QueryCOMNeighbors(const AlignedArray<COMExtentType>& centerOfMass,
                                                     const SubProblemData<size_t, DataEntry, DistType> subProb, 
                                                     const int numCandidates,
                                                     SpaceMetric<AlignedSpan<const COMExtentType>, DataView, COMExtentType> distanceFunctor){

    GraphVertex<size_t, DistType> COMneighbors(numCandidates);
    
    //Just gonna dummy it and select the first few nodes. Since the splitting process is randomized, this is a totally random selection, right? /s
    NodeTracker nodesVisited(subProb.dataBlock.size());
    for (size_t i = 0; i < numCandidates; i+=1){
        COMneighbors.neighbors.push_back(std::pair<size_t, DistType>(i,
                                          distanceFunctor(centerOfMass, subProb.dataBlock.blockData[i])));
        nodesVisited[i] = true;
    }
    std::make_heap(COMneighbors.neighbors.begin(), COMneighbors.neighbors.end(), NeighborDistanceComparison<size_t, COMExtentType>);

    bool breakVar = false;
    GraphVertex<size_t, COMExtentType> newState(COMneighbors);
    while (!breakVar){
        breakVar = true;   
        for (const auto& curCandidate: COMneighbors){
            for (const auto& joinTarget: subProb.subGraph[curCandidate.first]){
                if(nodesVisited[joinTarget.first]) continue;
                nodesVisited[joinTarget.first] = true;
                COMExtentType distance = distanceFunctor(centerOfMass, subProb.dataBlock[joinTarget.first]);
                if (distance < newState[0].second){
                    newState.PushNeighbor({joinTarget.first, distance});
                    breakVar = false;
                }
                
            }
        }

        COMneighbors = newState;
    }

    return COMneighbors;
}

template<typename DistType>
struct QueryPoint{
    const GraphVertex<size_t, DistType>& queryHint;
    const size_t dataIndex;
    QueryPoint(const GraphVertex<size_t, DistType>& hint, const size_t index): queryHint(hint), dataIndex(index){}
};

template<typename DataIndexType, typename DataEntry, typename DataView, typename DistType, typename COMExtentType>
GraphVertex<DataIndexType, DistType> QueryHintFromCOM(const AlignedArray<COMExtentType>& centerOfMass,
                                                       const SubProblemData<DataIndexType, DataEntry, DistType> subProb,
                                                       const std::uint32_t numCandidates,
                                                       SpaceMetric<AlignedSpan<const COMExtentType>, DataView, DistType> comDistanceFunctor){
    GraphVertex<DataIndexType, COMExtentType> comNeighbors = QueryCOMNeighbors<DataIndexType, DataEntry, DataView, COMExtentType, DistType>(centerOfMass, subProb, numCandidates, comDistanceFunctor);
    GraphVertex<DataIndexType, DistType> retHint;
    for (auto& hint: comNeighbors){
        //This should be an emplace_back
        retHint.push_back({hint.first, std::numeric_limits<DistType>::max()});
    }
    return retHint;
}

template<typename DistType>
struct DefaultQueryFunctor{


    DispatchFunctor<DistType>& distanceFunctor;

    DefaultQueryFunctor(DispatchFunctor<DistType>& distanceFunctor): distanceFunctor(distanceFunctor){};
    
    DistType operator()(size_t LHSIndex, size_t RHSIndex) const{
        return this->distanceFunctor(LHSIndex, RHSIndex);
    }

    std::vector<DistType> operator()(const std::vector<size_t>& LHSIndecies, size_t RHSIndex) const{
        return this->distanceFunctor(LHSIndecies, RHSIndex);
    }

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum) {
        this->distanceFunctor->SetBlocks(lhsBlockNum, rhsBlockNum);
    }

    auto CachingFunctor(DistanceCache<DistType>& cache) const {
        cache.reserve(dataBlock.size()*dataBlock.size());
        auto cachingFunctor = [&](std::vector<size_t> LHSIndecies, size_t RHSIndex) -> std::vector<DistType>{
            std::vector<DistType> distances = (*this)(LHSIndecies, RHSIndex);
            for (size_t i = 0; i<LHSIndecies.size(); i+=1){
                cache[std::pair{LHSIndecies[i], RHSIndex}] = distances[i];
            }
            return distances;
        };

        return cachingFunctor;
    }

    auto CachedFunctor(DistanceCache<DistType>& cache) const {
        auto cachedFunctor = [&](std::vector<size_t> LHSIndecies, size_t RHSIndex) -> std::vector<DistType>{
            std::vector<size_t> distToCompute;
            std::vector<DistType> precomputedDists;
            for(size_t i = 0; i<LHSIndecies.size(); i += 1){
                auto result = cache.find(std::pair{RHSIndex, LHSIndecies[i]});
                if(result == cache.end()) distToCompute.push_back(LHSIndecies[i]);
                else precomputedDists.push_back(result->second);
            }
            if (distToCompute.size() == 0) return precomputedDists;
            std::vector<DistType> newDists = (*this)(distToCompute, RHSIndex);
            std::vector<DistType> results;
            size_t newIndex = 0;
            for(size_t i=0; i<LHSIndecies.size(); i+=1){
                if (LHSIndecies[i] != distToCompute[newIndex]) results.push_back(precomputedDists[i-newIndex]);
                else{
                    results.push_back(newDists[newIndex]);
                    newIndex += 1;
                }
            }
            return results;
        };

        return cachedFunctor;
    }
};

namespace internal{
    static const size_t maxBatch = 7;
}

template<typename DistType>
struct QueryContext{
    const UndirectedGraph<size_t> subGraph;
    const GraphVertex<size_t, DistType> queryHint;
    const int querySearchDepth;
    const DefaultQueryFunctor<DistType> defaultQueryFunctor;
    const size_t blockNumber;
    //std::unordered_map<BlockNumberType, Graph<IndexType, DistType>> neighborCandidates;
    //SpaceMetric<DataView, DataView, DistType> distanceFunctor;

    QueryContext(const Graph<size_t, DistType>& subGraph,
                 const GraphVertex<size_t, DistType> queryHint,
                 DefaultQueryFunctor<DistType>& defaultQueryFunctor,
                 const int querySearchDepth,
                 const size_t blockNumber):
                    subGraph(subGraph),
                    queryHint(std::move(queryHint)),
                    querySearchDepth(querySearchDepth),
                    defaultQueryFunctor(defaultQueryFunctor),
                    blockNumber(blockNumber){
            
            //defaultQueryFunctor = DefaultQueryFunctor<IndexType, DataEntry, DistType>(distanceFunctor, dataBlock);
    };

    //Nearest Node Distance
    //make checking this in parallel safe
    std::tuple<size_t, size_t, DistType> operator*(const QueryContext& rhs) const{
           return NearestNodes(rhs);
    }
    

    template<typename QueryType>
    GraphVertex<IndexType, DistType> operator||(QueryPoint<DistType>& queryPoint) const {
        return QueryHotPath(queryPoint.queryHint, queryPoint.dataIndex, defaultQueryFunctor);
    }

    //I want copies when I use the queryHint member, but not really when I'm passing in hints. ???
    //Figure this out later
    template<typename QueryFunctor = decltype(defaultQueryFunctor)>
    GraphVertex<IndexType, DistType> QueryHotPath(GraphVertex<IndexType, DistType>& initVertex,
                                                  const IndexType queryIndex,
                                                  const QueryFunctor& queryFunctor,
                                                  std::optional<NodeTracker> previousVisits = std::nullopt) const {

        NodeTracker nodesVisited = previousVisits.value_or( NodeTracker(dataBlock.size()) );

        int sizeDif = initVertex.size() - queryHint.size();
        //if sizeDif is negative, fill to numCandidates
        if(sizeDif<0){
            //Gotta avoid dupes
            NodeTracker nodesInHint(dataBlock.size());
            for (const auto& hint: initVertex){
                nodesInHint[hint.first] = true;
            }
            int indexOffset(0);
            for (int i = 0; i < -sizeDif; i += 1){
                while (nodesInHint[queryHint[i+indexOffset].first]){
                    indexOffset += 1;
                }
                initVertex.push_back(queryHint[i+indexOffset]);
            }
        }
        std::vector<IndexType> initComputations;
        for (auto& queryStart: initVertex){
            initComputations.push_back(queryStart.first);
            nodesVisited[queryStart.first] = true;
        }
        std::vector<DistType> initDistances = queryFunctor(initComputations, queryIndex, queryData);
        for (size_t i = 0; i<initVertex.size(); i+=1){
            initVertex[i].second = initDistances[i];
        }
        /*
        for (auto& queryStart: initVertex){
            queryStart.second = queryFunctor(queryStart.first, queryIndex, queryData);
            nodesVisited[queryStart.first] = true;
        }
        */
        std::make_heap(initVertex.begin(), initVertex.end(), NeighborDistanceComparison<IndexType, DistType>);
        //if sizeDif is positive, reduce to numCandidates
        for (int i = 0; i < sizeDif; i+=1){
            std::pop_heap(initVertex.begin(), initVertex.end(), NeighborDistanceComparison<IndexType, DistType>);
            initVertex.pop_back();
        }
        GraphVertex<IndexType, DistType> compareTargets;
        compareTargets.resize(querySearchDepth);

        std::vector<IndexType> joinQueue;
        const size_t maxBatch = internal::maxBatch;
        joinQueue.reserve(maxBatch);

        NodeTracker nodesCompared(dataBlock.size());

        bool breakVar = true;
        while (breakVar){
            std::partial_sort_copy(initVertex.begin(), initVertex.end(), compareTargets.begin(), compareTargets.end(), NeighborDistanceComparison<IndexType, DistType>);
            breakVar = false;
            size_t numCompared = 0;
            for (const auto& neighbor: compareTargets){
                if (nodesCompared[neighbor.first]){
                    numCompared += 1;
                    continue;
                }
                const std::vector<IndexType>& currentNeighbor = subGraph[neighbor.first];
                for (const auto& joinTarget: currentNeighbor){
                    if (nodesVisited[joinTarget] == true) continue;
                    nodesVisited[joinTarget] = true;
                    joinQueue.push_back(joinTarget);
                    if (joinQueue.size() == maxBatch) goto computeBatch; //double break
                    /*
                    DistType distance = queryFunctor(joinTarget, queryIndex, queryData);
                    if (distance < initVertex[0].second){
                        initVertex.PushNeighbor({joinTarget, distance});
                        breakVar = false;
                    }
                    */
                }
                nodesCompared[neighbor.first] = true;
                numCompared += 1;
            }
            computeBatch:
            std::vector<DistType> distances = queryFunctor(joinQueue, queryIndex, queryData);
            for (size_t i = 0; i<distances.size(); i+=1){
                breakVar = initVertex.PushNeighbor({joinQueue[i], distances[i]}) || breakVar;
            }
            joinQueue.resize(0);
            breakVar = breakVar || numCompared != querySearchDepth;
        }
        return initVertex;
    }

    std::tuple<size_t, size_t, DistType> NearestNodes(const QueryContext& rhs) const{

        this->defaultQueryFunctor.SetBlocks(this->blockNumber, rhs.blockNumber);

        std::pair<size_t, size_t> bestPair;
        DistType bestDistance(std::numeric_limits<DistType>::max());
        //NodeTracker nodesVisitedA(subGraphA.dataBlock.size());
        //NodeTracker nodesVisitedB(subGraphB.dataBlock.size());

        for(const auto& starterA: this->queryHint.neighbors){
            //nodesVisitedA[starterA.first] = true;
            for (const auto& starterB: rhs.queryHint.neighbors){
                //nodesVisitedB[starterB.first] = true;
                DistType distance = distanceFunctor(starterA.first, starterB.first);
                if (distance < bestDistance){
                    bestDistance = distance;
                    bestPair = std::pair<size_t, size_t>(starterA.first, starterB.first);
                }
            }
        }

        bool breakVar = false;
        while (!breakVar){
            breakVar = true;
            std::pair<size_t, size_t> tmpPair = bestPair;
            for (const auto& neighborA: this->subGraph[bestPair.first]){
                //if (!nodesVisitedA[neighborA.first]){
                DistType distance = distanceFunctor(neighborA, tmpPair.second);
                if (distance < bestDistance){
                    bestDistance = distance;
                    tmpPair.first = neighborA;
                    breakVar = false;
                }
                    //nodesVisitedA[neighborA.first] = true;
                //}  
                
                for (const auto& neighborOfNeighborA: this->subGraph[neighborA]){
                    //if (nodesVisitedA[neighborOfNeighborA.first]) continue;
                    //nodesVisitedA[neighborOfNeighborA.first] = true;
                    DistType distance = this->distanceFunctor(neighborOfNeighborA, tmpPair.second);
                    if (distance < bestDistance){
                        bestDistance = distance;
                        tmpPair.first = neighborOfNeighborA;
                        breakVar = false;
                    }
                }
            }
            for (const auto& neighborB: rhs.subGraph[bestPair.second]){
                //if (!nodesVisitedB[neighborB.first]){
                    DistType distance = this->distanceFunctor(tmpPair.first, neighborB);
                if (distance < bestDistance){
                    bestDistance = distance;
                    tmpPair.second = neighborB;
                    breakVar = false;
                }
                //  nodesVisitedB[neighborB.first] = true;
                //}
                for (const auto& neighborOfNeighborB: rhs.subGraph[neighborB]){
                    //nodesVisitedB[neighborOfNeighborB.first] = true;
                    DistType distance = this->distanceFunctor(tmpPair.first, neighborOfNeighborB);
                    if (distance < bestDistance){
                        bestDistance = distance;
                        tmpPair.second = neighborOfNeighborB;
                        breakVar = false;
                    }
                }
            }
            bestPair = tmpPair;
        }
        
        return {bestPair.first, bestPair.second, bestDistance};
    }

};

}

#endif 