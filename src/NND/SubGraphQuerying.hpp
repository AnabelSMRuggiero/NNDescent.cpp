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
#include <ranges>
#include <functional>
#include <concepts>

#include "../Utilities/Data.hpp"
#include "../Utilities/DataSerialization.hpp"
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

template<std::floating_point COMExtent, typename DistType, typename IndexType, typename Functor>
GraphVertex<IndexType, COMExtent> QueryCOMNeighbors(const size_t pointIndex,
                                                     const Graph<IndexType, DistType>& subProb, 
                                                     const size_t numCandidates,
                                                     Functor& distanceFunctor){

    GraphVertex<IndexType, COMExtent> COMneighbors(numCandidates);
    //auto problemView = subProb.GetOffsetView(indexOffset);
    //Just gonna dummy it and select the first few nodes. Since the splitting process is randomized, this is a totally random selection, right? /s
    NodeTracker nodesVisited(subProb.size());
    for (size_t i = 0; i < numCandidates; i+=1){
        COMneighbors.push_back(std::pair<IndexType, COMExtent>(i,
                                distanceFunctor(pointIndex, i)));
        nodesVisited[i] = true;
    }
    COMneighbors.JoinPrep();
    //std::make_heap(COMneighbors.begin(), COMneighbors.neighbors.end(), NeighborDistanceComparison<size_t, COMExtent>);

    bool breakVar = false;
    GraphVertex<IndexType, COMExtent> newState(COMneighbors);
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

/*
template<typename DistType>
struct QueryPoint{
    const GraphVertex<size_t, DistType>& queryHint;
    const size_t dataIndex;
    QueryPoint(const GraphVertex<size_t, DistType>& hint, const size_t index): queryHint(hint), dataIndex(index){}
};
*/

template<typename DistType, typename COMExtent, typename IndexType>
GraphVertex<IndexType, DistType> QueryHintFromCOM(const size_t metaPointIndex,
                                                const Graph<IndexType, DistType>& subProb,
                                                const size_t numCandidates,
                                                SinglePointFunctor<COMExtent>& distanceFunctor){
    GraphVertex<IndexType, COMExtent> comNeighbors = QueryCOMNeighbors<COMExtent>(metaPointIndex, subProb, numCandidates, distanceFunctor);
    if constexpr (std::is_same<DistType, COMExtent>()){
        for (auto& neighbor: comNeighbors){
            neighbor.second = std::numeric_limits<DistType>::max();
        }
        return comNeighbors;
    } else{
        GraphVertex<IndexType, DistType> retHint;
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
    static constexpr size_t maxBatch = 7;
}

template<std::unsigned_integral IndexType, std::totally_ordered DistType>
struct QueryContext{

    const UndirectedGraph<IndexType> subGraph;
    const GraphVertex<IndexType, DistType> queryHint;
    size_t querySize;
    size_t querySearchDepth;
    const GraphFragment_t graphFragment{GraphFragment_t(-1)};
    const BlockNumber_t blockNumber{BlockNumber_t(-1)};
    size_t blockSize{size_t(-1)};

    QueryContext() = default;

    QueryContext(const Graph<IndexType, DistType>& subGraph,
                 GraphVertex<IndexType, DistType>&& queryHint,
                 const int querySearchDepth,
                 const size_t graphFragment,
                 const size_t blockNumber,
                 const size_t blockSize):
                 
                    subGraph(BuildUndirectedGraph(subGraph)),
                    queryHint(std::move(queryHint)),
                    querySearchDepth(querySearchDepth),
                    graphFragment(graphFragment),
                    blockNumber(blockNumber),
                    blockSize(blockSize){
            querySize = this->queryHint.size();
            //defaultQueryFunctor = DefaultQueryFunctor<IndexType, DataEntry, DistType>(distanceFunctor, dataBlock);
    };


    QueryContext(std::ifstream& inFile): subGraph(Extract<UndirectedGraph<IndexType>>(inFile)),
                                         //subGraph(inFile),
                                         queryHint(Extract<GraphVertex<IndexType, DistType>>(inFile)),
                                         querySize(Extract<size_t>(inFile)),
                                         querySearchDepth(Extract<size_t>(inFile)),
                                         graphFragment(Extract<GraphFragment_t>(inFile)),
                                         blockNumber(Extract<BlockNumber_t>(inFile)),
                                         blockSize(Extract<size_t>(inFile)) {}
    
    QueryContext(QueryContext&&) = default;


    template<typename QueryFunctor>
    GraphVertex<IndexType, DistType> Query(const std::vector<IndexType>& initHints,
                                           const size_t queryIndex, //Can realistically be any parameter passed through to the Functor
                                           QueryFunctor& queryFunctor) const{

        
        auto [vertex, nodeTracker] = ForwardQueryInit(initHints,
                                                      queryIndex, 
                                                      queryFunctor);
        

        return std::move(QueryLoop(vertex, queryIndex, queryFunctor, nodeTracker));

    }
    
    //Figure this out later
    template<typename QueryFunctor>
    GraphVertex<IndexType, DistType>& Query(GraphVertex<IndexType, DistType>& initVertex,
                                           const size_t queryIndex, //Can realistically be any parameter passed through to the Functor
                                           QueryFunctor& queryFunctor,
                                           NodeTracker& nodesVisited) const {

        if (initVertex.size()<querySize){
            ReverseQueryInit(initVertex,
                                queryIndex, 
                                queryFunctor,
                                nodesVisited);
        }

        return QueryLoop(initVertex, queryIndex, queryFunctor, nodesVisited);
    }
        

        
        //constexpr size_t bufferSize = sizeof(size_t)*(internal::maxBatch+2)  + sizeof(std::pmr::vector<size_t>);
        //char stackBuffer[bufferSize];
        //std::pmr::monotonic_buffer_resource stackResource(stackBuffer, bufferSize);
        
            
        
        //std::pmr::vector<size_t>& joinQueue= *(new (stackResource.allocate(sizeof(std::pmr::vector<size_t>))) std::pmr::vector<size_t>(&stackResource));
    template<typename QueryFunctor>
    GraphVertex<IndexType, DistType>& QueryLoop(GraphVertex<IndexType, DistType>& initVertex,
                                           const size_t queryIndex, //Can realistically be any parameter passed through to the Functor
                                           QueryFunctor& queryFunctor,
                                           NodeTracker& nodesVisited) const{
        std::vector<size_t> joinQueue;
        constexpr size_t maxBatch = internal::maxBatch; //Compile time constant
        joinQueue.reserve(maxBatch);

        NodeTracker nodesCompared(blockSize);
        auto notVisited = [&](const auto index){ return !nodesVisited[index]; };

        auto notCompared = [&](const auto index)->bool{ 
            return (nodesCompared[index]) ?
                false :
                !(nodesCompared[index] = std::none_of(subGraph[index].begin(), subGraph[index].end(), notVisited));
        };
        auto toNeighborView = [&](const auto index){ return subGraph[index]; }; //returns a view into a data block

        
        
        bool breakVar = true;
        while (breakVar){
            

            auto toNeighbor = [&](const auto edge) mutable {

                return edge.first; 
            };

            for(const auto joinTarget : initVertex 
                                        | std::views::transform(toNeighbor)
                                        | std::views::take(querySearchDepth)    
                                        | std::views::filter(notCompared) 
                                        | std::views::transform(toNeighborView)            
                                        | std::views::join
                                        | std::views::filter(notVisited)
                                        | std::views::take(maxBatch)){
                joinQueue.push_back(joinTarget);
                nodesVisited[joinTarget] = true;
            }
            
            
            std::ranges::contiguous_range auto distances = queryFunctor(queryIndex, joinQueue);
            breakVar = std::transform_reduce(joinQueue.begin(), joinQueue.end(), distances.begin(),
                                             joinQueue.size() == maxBatch,
                                             std::logical_or<bool>{},
                                             [&](const size_t index, const DistType distance){ 
                                                 return initVertex.PushNeighbor({static_cast<IndexType>(index), distance});    
            });

            joinQueue.clear();
        }
        return initVertex;
    }

    template<typename QueryFunctor>
    std::pair<GraphVertex<IndexType, DistType>, NodeTracker> ForwardQueryInit(const std::vector<IndexType>& startHint,
                                                                const size_t queryIndex, //Can realistically be any parameter passed through to the Functor
                                                                QueryFunctor& queryFunctor) const{
        
        
        NodeTracker nodesJoined(blockSize);
        for (const auto& hint : startHint) nodesJoined[hint] = true;

        std::vector<size_t> initDestinations = [&](){
            if constexpr(std::is_same_v<IndexType, size_t>) return startHint;
            else{
                std::vector<size_t> initDistances(startHint.size());
                std::ranges::transform(startHint, initDistances.begin(), std::identity{});
                return initDistances;
            }
        }();

        auto notJoined = [&](const auto& index){ return !nodesJoined[index]; };
        
        auto padIndecies = [&](const auto& range){
            for (const auto& index: range | std::views::filter(notJoined)
                                              | std::views::take(querySize - initDestinations.size())){
                initDestinations.push_back(index);
                nodesJoined[index] = true;
            }
        };

        if (initDestinations.size() < querySize){
            padIndecies(queryHint | std::views::transform([&](const auto& pair){return pair.first;}));
            [[unlikely]] if (initDestinations.size()<querySize) 
                padIndecies(std::views::iota(size_t{0}, blockSize));
        }

        std::ranges::contiguous_range auto initDistances = queryFunctor(queryIndex, initDestinations);
        

        GraphVertex<IndexType, DistType> retVertex(initDistances.size()); //This constructor merely reserves
        retVertex.resize(initDistances.size());

        std::ranges::transform(initDestinations, initDistances, retVertex.begin(),
            [&](const auto& index, const auto& distance){
                return typename GraphVertex<IndexType, DistType>::EdgeType{index, distance};
        });
        retVertex.JoinPrep();

        return {retVertex, nodesJoined};
        
    }

    template<typename QueryFunctor>
    void ReverseQueryInit(GraphVertex<IndexType, DistType>& initVertex,
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

        std::ranges::contiguous_range auto initDistances = queryFunctor(queryIndex, initComputations);
        for (size_t i = 0; i<initDistances.size(); i+=1){
            initVertex.push_back({static_cast<IndexType>(initComputations[i]), initDistances[i]});
        }
        initVertex.JoinPrep();
    }

    template<typename QueryFunctor>
    std::tuple<IndexType, IndexType, DistType> NearestNodes(const QueryContext& rhs, QueryFunctor& distanceFunctor) const{

        //distanceFunctor.SetBlocks(this->blockNumber, rhs.blockNumber);

        std::pair<IndexType, IndexType> bestPair;
        DistType bestDistance(std::numeric_limits<DistType>::max());
        //NodeTracker nodesVisitedA(subGraphA.dataBlock.size());
        //NodeTracker nodesVisitedB(subGraphB.dataBlock.size());

        for(const auto& starterA: this->queryHint){
            //nodesVisitedA[starterA.first] = true;
            for (const auto& starterB: rhs.queryHint){
                //nodesVisitedB[starterB.first] = true;
                DistType distance = distanceFunctor(starterA.first, starterB.first);
                if (distance < bestDistance){
                    bestDistance = distance;
                    bestPair = std::pair<IndexType, IndexType>(starterA.first, starterB.first);
                }
            }
        }

        bool breakVar = false;
        while (!breakVar){
            breakVar = true;
            std::pair<IndexType, IndexType> tmpPair = bestPair;
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
    
    void serialize(std::ofstream& outFile) const {
        


        auto outputter = BindSerializer(outFile);

        

        outputter(this->subGraph.graphBlock);

        outputter(this->queryHint);

        outputter(this->querySize);
        outputter(this->querySearchDepth);
        
        outputter(this->graphFragment);
        outputter(this->blockNumber);
        outputter(this->blockSize);

    }

};




}

#endif 