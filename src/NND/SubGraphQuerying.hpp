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
#include <type_traits>

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
std::tuple<size_t, size_t, FloatType> BruteNearestNodes(const Graph<IndexType, FloatType>& subGraphA,
                  const DataBlock<DataEntry>& dataBlockA,
                  const Graph<IndexType, FloatType>& subGraphB,
                  const DataBlock<DataEntry>& dataBlockB,
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

template<std::floating_point COMExtent, typename DistType>
GraphVertex<size_t, DistType> QueryCOMNeighbors(const size_t pointIndex,
                                                     const Graph<size_t, DistType>& subProb, 
                                                     const size_t numCandidates,
                                                     SinglePointFunctor<COMExtent>& distanceFunctor){

    GraphVertex<size_t, COMExtent> COMneighbors(numCandidates);
    
    //Just gonna dummy it and select the first few nodes. Since the splitting process is randomized, this is a totally random selection, right? /s
    NodeTracker nodesVisited(subProb.size());
    for (size_t i = 0; i < numCandidates; i+=1){
        COMneighbors.push_back(std::pair<size_t, COMExtent>(i,
                                distanceFunctor(pointIndex, i)));
        nodesVisited[i] = true;
    }
    COMneighbors.JoinPrep();
    //std::make_heap(COMneighbors.begin(), COMneighbors.neighbors.end(), NeighborDistanceComparison<size_t, COMExtent>);

    bool breakVar = false;
    GraphVertex<size_t, COMExtent> newState(COMneighbors);
    while (!breakVar){
        breakVar = true;   
        for (const auto& curCandidate: COMneighbors){
            for (const auto& joinTarget: subProb[curCandidate.first]){
                if(nodesVisited[joinTarget.first]) continue;
                nodesVisited[joinTarget.first] = true;
                COMExtent distance = distanceFunctor(pointIndex, joinTarget.first);
                if (distance < newState.PushThreshold()){
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

template<typename DistType, typename COMExtent>
GraphVertex<size_t, DistType> QueryHintFromCOM(const size_t metaPointIndex,
                                                const Graph<size_t, DistType> subProb,
                                                const size_t numCandidates,
                                                SinglePointFunctor<COMExtent>& distanceFunctor){
    GraphVertex<size_t, COMExtent> comNeighbors = QueryCOMNeighbors<COMExtent, DistType>(metaPointIndex, subProb, numCandidates, distanceFunctor);
    if constexpr (std::is_same<DistType, COMExtent>()){
        for (auto& neighbor: comNeighbors){
            neighbor.second = std::numeric_limits<DistType>::max();
        }
        return comNeighbors;
    } else{
        GraphVertex<size_t, DistType> retHint;
        for (auto& hint: comNeighbors){
            //This should be an emplace_back
            retHint.push_back({hint.first, std::numeric_limits<DistType>::max()});
        }
        return retHint;
    }
}

template<typename DistType, typename DistanceFunctor>
struct DefaultQueryFunctor{


    DistanceFunctor distanceFunctor;

    DefaultQueryFunctor(DistanceFunctor& distanceFunctor): distanceFunctor(distanceFunctor){};
    
    DistType operator()(size_t LHSIndex, size_t RHSIndex) const{
        return this->distanceFunctor(LHSIndex, RHSIndex);
    }

    std::vector<DistType> operator()(size_t lhsIndex, const std::vector<size_t>& rhsIndecies) const{
        return this->distanceFunctor(lhsIndex, rhsIndecies);
    }

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum) {
        this->distanceFunctor.SetBlocks(lhsBlockNum, rhsBlockNum);
    }

    auto CachingFunctor(DistanceCache<DistType>& cache) const {
        //cache.reserve(dataBlock.size()*dataBlock.size());
        auto cachingFunctor = [&](const size_t lhsIndex, const std::vector<size_t>& rhsIndecies) -> std::vector<DistType>{
            std::vector<DistType> distances = (*this)(lhsIndex, rhsIndecies);
            for (size_t i = 0; i<rhsIndecies.size(); i+=1){
                cache[std::pair{lhsIndex, rhsIndecies[i]}] = distances[i];
            }
            return distances;
        };

        return cachingFunctor;
    }

    auto CachedFunctor(DistanceCache<DistType>& cache) const {
        auto cachedFunctor = [&](const size_t lhsIndex, std::vector<size_t>& rhsIndecies) -> std::vector<DistType>{
            std::vector<size_t> distToCompute;
            std::vector<DistType> precomputedDists;
            for(size_t i = 0; i<rhsIndecies.size(); i += 1){
                auto result = cache.find(std::pair{rhsIndecies[i], lhsIndex});
                if(result == cache.end()) distToCompute.push_back(rhsIndecies[i]);
                else precomputedDists.push_back(result->second);
            }
            if (distToCompute.size() == 0) return precomputedDists;
            std::vector<DistType> newDists = (*this)(lhsIndex, distToCompute);
            std::vector<DistType> results;
            size_t newIndex = 0;
            for(size_t i=0; i<rhsIndecies.size(); i+=1){
                if (newIndex == distToCompute.size() || rhsIndecies[i] != distToCompute[newIndex]) results.push_back(precomputedDists[i-newIndex]);
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
    size_t querySize;
    size_t querySearchDepth;
    //DefaultQueryFunctor<DistType, DistanceFunctor> defaultQueryFunctor;
    const size_t blockNumber{size_t(-1)};
    const size_t blockSize{size_t(-1)};
    //std::unordered_map<BlockNumberType, Graph<IndexType, DistType>> neighborCandidates;
    //SpaceMetric<DataView, DataView, DistType> distanceFunctor;

    QueryContext() = default;

    QueryContext(const Graph<size_t, DistType>& subGraph,
                 const GraphVertex<size_t, DistType>&& queryHint,
                 //DefaultQueryFunctor<DistType, DistanceFunctor> defaultQueryFunctor,
                 const int querySearchDepth,
                 const size_t blockNumber,
                 const size_t blockSize):
                    subGraph(subGraph),
                    queryHint(std::move(queryHint)),
                    querySearchDepth(querySearchDepth),
                    //defaultQueryFunctor(defaultQueryFunctor),
                    blockNumber(blockNumber),
                    blockSize(blockSize){
            querySize = queryHint.size();
            //defaultQueryFunctor = DefaultQueryFunctor<IndexType, DataEntry, DistType>(distanceFunctor, dataBlock);
    };
    
    /*
    template<typename QueryType>
    GraphVertex<size_t, DistType> operator||(QueryPoint<DistType>& queryPoint) const {
        return Query(queryPoint.queryHint, queryPoint.dataIndex, defaultQueryFunctor);
    }
    */
   
    
    //Figure this out later
    template<typename QueryFunctor>
    GraphVertex<size_t, DistType> Query(GraphVertex<size_t, DistType>& initVertex,
                                                  const size_t queryIndex, //Can realistically be any parameter passed through to the Functor
                                                  QueryFunctor& queryFunctor,
                                                  std::optional<NodeTracker> previousVisits = std::nullopt) const {

        NodeTracker nodesVisited;
        if (previousVisits){
            nodesVisited = previousVisits.value();
            if (initVertex.size()<querySize){
                ReverseQueryInit(initVertex,
                                 queryIndex, 
                                 queryFunctor,
                                 nodesVisited);
            }
        } else {
            nodesVisited = NodeTracker(blockSize);
            ForwardQueryInit(initVertex,
                             queryIndex, 
                             queryFunctor,
                             nodesVisited);
            
        }
        GraphVertex<size_t, DistType> compareTargets;
        compareTargets.resize(querySearchDepth);

        std::vector<size_t> joinQueue;
        const size_t maxBatch = internal::maxBatch;
        joinQueue.reserve(maxBatch);

        NodeTracker nodesCompared(blockSize);

        bool breakVar = true;
        while (breakVar){
            std::partial_sort_copy(initVertex.begin(), initVertex.end(), compareTargets.begin(), compareTargets.end(), NeighborDistanceComparison<size_t, DistType>);
            breakVar = false;
            size_t numCompared = 0;
            for (const auto& neighbor: compareTargets){
                if (nodesCompared[neighbor.first]){
                    numCompared += 1;
                    continue;
                }
                const std::vector<size_t>& currentNeighbor = subGraph[neighbor.first];
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
            std::vector<DistType> distances = queryFunctor(queryIndex, joinQueue);
            for (size_t i = 0; i<distances.size(); i+=1){
                breakVar = initVertex.PushNeighbor({joinQueue[i], distances[i]}) || breakVar;
            }
            joinQueue.resize(0);
            breakVar = breakVar || numCompared != querySearchDepth;
        }
        return initVertex;
    }

    template<typename QueryFunctor>
    void ForwardQueryInit(GraphVertex<size_t, DistType>& initVertex,
                            const size_t queryIndex, //Can realistically be any parameter passed through to the Functor
                            QueryFunctor& queryFunctor,
                            NodeTracker& nodesJoined) const{
        int sizeDif = initVertex.size() - querySize;
        //if sizeDif is negative, fill to numCandidates
        if(sizeDif<0){
            //initVertex.reserve(queryHint.size());
            for (const auto& hint: initVertex){
                nodesJoined[hint.first] = true;
            }
            int indexOffset(0);
            for (int i = 0; initVertex.size() < querySize; i += 1){
                while (i+indexOffset < queryHint.size() && nodesJoined[queryHint[i+indexOffset].first]){
                    indexOffset += 1;
                }
                [[likely]] if(i+indexOffset<queryHint.size()){
                    
                    initVertex.push_back(queryHint[i+indexOffset]);
                    nodesJoined[queryHint[i+indexOffset].first] = true;
                } else{
                    while(nodesJoined[indexOffset]){
                        indexOffset += 1;
                    }
                    initVertex.push_back({indexOffset, std::numeric_limits<DistType>::max()});
                    nodesJoined[indexOffset] = true;
                    
                }
            }
        }
        std::vector<size_t> initComputations;
        for (auto& queryStart: initVertex){
            initComputations.push_back(queryStart.first);
            nodesJoined[queryStart.first] = true;
        }
        std::vector<DistType> initDistances = queryFunctor(queryIndex, initComputations);
        for (size_t i = 0; i<initVertex.size(); i+=1){
            initVertex[i].second = initDistances[i];
        }

        std::make_heap(initVertex.begin(), initVertex.end(), NeighborDistanceComparison<size_t, DistType>);
        //if sizeDif is positive, reduce to numCandidates
        //for (int i = 0; i < sizeDif; i+=1){
        //    std::pop_heap(initVertex.begin(), initVertex.end(), NeighborDistanceComparison<size_t, DistType>);
        //    initVertex.pop_back();
        //}
    }

    template<typename QueryFunctor>
    void ReverseQueryInit(GraphVertex<size_t, DistType>& initVertex,
                            const size_t queryIndex, //Can realistically be any parameter passed through to the Functor
                            QueryFunctor& queryFunctor,
                            NodeTracker& previousVisits) const{

        int sizeDif = querySize - initVertex.size();
        
        
        int indexOffset(0);
        std::vector<size_t> initComputations;
        for (int i = 0; i < sizeDif; i += 1){
            while (previousVisits[queryHint[i+indexOffset].first]){
                indexOffset += 1;
            }
            initComputations.push_back(queryHint[i+indexOffset].first);
            previousVisits[queryHint[i+indexOffset].first] = true;
        }

        std::vector<DistType> initDistances = queryFunctor(queryIndex, initComputations);
        for (size_t i = 0; i<initDistances.size(); i+=1){
            initVertex.push_back({initComputations[i], initDistances[i]});
        }
        initVertex.JoinPrep();
    }

    template<typename QueryFunctor>
    std::tuple<size_t, size_t, DistType> NearestNodes(const QueryContext& rhs, QueryFunctor& distanceFunctor) const{

        distanceFunctor.SetBlocks(this->blockNumber, rhs.blockNumber);

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
                    DistType distance = distanceFunctor(neighborOfNeighborA, tmpPair.second);
                    if (distance < bestDistance){
                        bestDistance = distance;
                        tmpPair.first = neighborOfNeighborA;
                        breakVar = false;
                    }
                }
            }
            for (const auto& neighborB: rhs.subGraph[bestPair.second]){
                //if (!nodesVisitedB[neighborB.first]){
                    DistType distance = distanceFunctor(tmpPair.first, neighborB);
                if (distance < bestDistance){
                    bestDistance = distance;
                    tmpPair.second = neighborB;
                    breakVar = false;
                }
                //  nodesVisitedB[neighborB.first] = true;
                //}
                for (const auto& neighborOfNeighborB: rhs.subGraph[neighborB]){
                    //nodesVisitedB[neighborOfNeighborB.first] = true;
                    DistType distance = distanceFunctor(tmpPair.first, neighborOfNeighborB);
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