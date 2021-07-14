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

#include "../Utilities/Data.hpp"
#include "../Utilities/DataDeserialization.hpp"
#include "GraphStructures.hpp"

namespace nnd{

template<TriviallyCopyable DataIndexType, typename DataEntry, typename DistType>
struct SubProblemData{

    const Graph<DataIndexType, DistType>& subGraph;
    const DataBlock<DataEntry>& dataBlock;

    SubProblemData(const Graph<DataIndexType, DistType>& subGraph, const DataBlock<DataEntry>& dataBlock): subGraph(subGraph), dataBlock(dataBlock) {};
};




//Maybe a block specific one that reads i.blockNumber from a BlockIndecies
struct NodeTracker{

    using reference = std::vector<bool>::reference;
    using const_reference = std::vector<bool>::const_reference;
    using size_type = std::vector<bool>::size_type;

    std::vector<bool> flags;

    NodeTracker(size_t graphSize): flags(graphSize, false){};

    constexpr reference operator[](size_type i){
        return flags[i];
    }

    constexpr const_reference operator[](size_type i) const {
        return flags[i];
    }

    constexpr reference operator[](BlockIndecies i){
        //Assuming block index lines up here;
        return flags[i.dataIndex];
    }

    constexpr const_reference operator[](BlockIndecies i) const{
        //Assuming block index lines up here;
        return flags[i.dataIndex];
    }

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

template<TriviallyCopyable DataIndexType, typename DataEntry, std::floating_point COMExtentType, typename DistType>
GraphVertex<DataIndexType, DistType> QueryCOMNeighbors(const std::valarray<DistType>& centerOfMass,
                                                     const SubProblemData<DataIndexType, DataEntry, DistType> subProb, 
                                                     const int numCandidates,
                                                     SpaceMetric<std::valarray<COMExtentType>, DataEntry, COMExtentType> distanceFunctor){

    GraphVertex<DataIndexType, DistType> COMneighbors(numCandidates);
    
    //Just gonna dummy it and select the first few nodes. Since the splitting process is randomized, this is a totally random selection, right? /s
    NodeTracker nodesVisited(subProb.dataBlock.size());
    for (size_t i = 0; i < numCandidates; i+=1){
        COMneighbors.neighbors.push_back(std::pair<DataIndexType, DistType>(i,
                                          distanceFunctor(centerOfMass, subProb.dataBlock.blockData[i])));
        nodesVisited[i] = true;
    }
    std::make_heap(COMneighbors.neighbors.begin(), COMneighbors.neighbors.end(), NeighborDistanceComparison<DataIndexType, COMExtentType>);

    bool breakVar = false;
    GraphVertex<DataIndexType, COMExtentType> newState(COMneighbors);
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

template<TriviallyCopyable IndexType, typename QueryType, typename DistType>
struct QueryPoint{
    const GraphVertex<IndexType, DistType>& queryHint;
    const QueryType& queryData;
    const IndexType dataIndex;
    QueryPoint(const GraphVertex<IndexType, DistType>& hint, const QueryType& data, const IndexType index): queryHint(hint), queryData(data), dataIndex(index){}
};

template<typename DataIndexType, typename DataEntry, typename DistType, typename COMExtentType>
GraphVertex<DataIndexType, DistType> QueryHintFromCOM(const std::valarray<COMExtentType>& centerOfMass,
                                                       const SubProblemData<DataIndexType, DataEntry, DistType> subProb,
                                                       const std::uint32_t numCandidates,
                                                       SpaceMetric<std::valarray<COMExtentType>, DataEntry, DistType> comDistanceFunctor){
    GraphVertex<DataIndexType, COMExtentType> comNeighbors = QueryCOMNeighbors<DataIndexType, DataEntry, COMExtentType, DistType>(centerOfMass, subProb, numCandidates, comDistanceFunctor);
    GraphVertex<DataIndexType, DistType> retHint;
    for (auto& hint: comNeighbors){
        //This should be an emplace_back
        retHint.push_back({hint.first, std::numeric_limits<DistType>::max()});
    }
    return retHint;
}

template<typename DataIndexType, typename DataEntry, typename DistType>
struct DefaultQueryFunctor{
    const DataBlock<DataEntry>& dataBlock;
    SpaceMetric<DataEntry, DataEntry, DistType> distanceFunctor;

    //DefaultQueryFunctor(SpaceMetric<DataEntry, DataEntry, DistType> distanceFunctor, const DataBlock<DataEntry>& dataBlock):
    //    distanceFunctor(distanceFunctor), dataBlock(dataBlock){
    //};
    
    DistType operator()(DataIndexType LHSIndex, DataIndexType RHSIndex, const DataEntry& queryData) const{
        return this->distanceFunctor(dataBlock[LHSIndex], queryData);
    }
};

template<TriviallyCopyable BlockNumberType, TriviallyCopyable IndexType, typename DataEntry, typename DistType>
struct QueryContext{
    const Graph<IndexType, DistType>& subGraph;
    const DataBlock<DataEntry>& dataBlock;
    const GraphVertex<IndexType, DistType> queryHint;
    const int numCandidates;
    const DefaultQueryFunctor<IndexType, DataEntry, DistType> defaultQueryFunctor;
    //std::unordered_map<BlockNumberType, Graph<IndexType, DistType>> neighborCandidates;
    SpaceMetric<DataEntry, DataEntry, DistType> distanceFunctor;

    QueryContext(const Graph<IndexType, DistType>& subGraph,
                 const DataBlock<DataEntry>& dataBlock,
                 const GraphVertex<IndexType, DistType> queryHint,
                 SpaceMetric<DataEntry, DataEntry, DistType> distanceFunctor,
                 const int numCandidates):
                    subGraph(subGraph),
                    dataBlock(dataBlock),
                    queryHint(std::move(queryHint)),
                    numCandidates(numCandidates),
                    distanceFunctor(distanceFunctor),
                    defaultQueryFunctor(dataBlock, distanceFunctor){
            
            //defaultQueryFunctor = DefaultQueryFunctor<IndexType, DataEntry, DistType>(distanceFunctor, dataBlock);
    };

    /*
    QueryContext(const Graph<IndexType, DistType>& subGraph,
                 const DataBlock<DataEntry>& dataBlock,
                 const std::valarray<COMExtent>& centerOfMass,
                 const int numCandidates,
                 SpaceMetric<DataEntry, DataEntry, DistType> distanceFunctor,
                 SpaceMetric<std::valarray<COMExtent>, DataEntry, DistType> comDistanceFunctor):
                    subGraph(subGraph), dataBlock(dataBlock), numCandidates(numCandidates), neighborCandidates(), distanceFunctor(distanceFunctor){
        const SubProblemData thisSub{subGraph, dataBlock};
        queryHint = QueryCOMNeighbors<IndexType, DataEntry, DistType>(centerOfMass, thisSub, numCandidates, comDistanceFunctor);
        for (auto& hint: queryHint){
            hint.second = std::numeric_limits<DistType>::max();
        }
    };
    */

    //Nearest Node Distance
    //make checking this in parallel safe
    std::tuple<IndexType, IndexType, DistType> operator*(const QueryContext& rhs) const{
           return NearestNodes(rhs);
    }

    std::pair<Graph<IndexType, DistType>, Graph<IndexType, DistType>> operator||(const QueryContext& rhs) const{
        //Get my updates by querying my data against RHS    Get RHS updates by querying RHS data against mine
        return {rhs.QuerySubGraph(*this),                   this->QuerySubGraph(rhs)};
    }
    //

    template<typename QueryType>
    GraphVertex<IndexType, DistType> operator||(const QueryPoint<IndexType, QueryType, DistType>& queryPoint) const {
        return QueryHotPath(queryPoint.queryHint, queryPoint.queryData, queryPoint.dataIndex, defaultQueryFunctor);
    }

    //I want copies when I use the queryHint member, but not really when I'm passing in hints. ???
    //Figure this out later
    template<typename QueryType, typename QueryFunctor = decltype(defaultQueryFunctor)>
    GraphVertex<IndexType, DistType> QueryHotPath(GraphVertex<IndexType, DistType> initVertex,
                                                  const QueryType& queryData,
                                                  const IndexType queryIndex,
                                                  const QueryFunctor& queryFunctor) const {
        NodeTracker nodesVisited(dataBlock.size());
        int sizeDif = initVertex.size() - numCandidates;
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
        for (auto& queryStart: initVertex){
            queryStart.second = queryFunctor(queryStart.first, queryIndex, queryData);
            nodesVisited[queryStart.first] = true;
        }
        std::make_heap(initVertex.begin(), initVertex.end(), NeighborDistanceComparison<IndexType, DistType>);
        //if sizeDif is positive, reduce to numCandidates
        for (int i = 0; i < sizeDif; i+=1){
            std::pop_heap(initVertex.begin(), initVertex.end(), NeighborDistanceComparison<IndexType, DistType>);
            initVertex.pop_back();
        }

        GraphVertex<IndexType, DistType> newState(initVertex);
        bool breakVar = false;
        while (!breakVar){
            breakVar = true;
            for (const auto& neighbor: initVertex){
                const GraphVertex<IndexType, DistType>& currentNeighbor = subGraph[neighbor.first];
                for (const auto& joinTarget: currentNeighbor){
                    if (nodesVisited[joinTarget.first] == true) continue;
                    nodesVisited[joinTarget.first] = true;
                    DistType distance = queryFunctor(joinTarget.first, queryIndex, queryData);
                    if (distance < newState[0].second){
                        newState.PushNeighbor({joinTarget.first, distance});
                        breakVar = false;
                    }
                }
            }
            initVertex = newState;
        }
        return initVertex;
    }

    Graph<IndexType, DistType> QuerySubGraph(const QueryContext& rhs) const{

        Graph<IndexType, DistType> retGraph;

        for (size_t i = 0; i<rhs.dataBlock.size(); i += 1){
            DataEntry queryData = rhs.dataBlock[i];
            retGraph.push_back(QueryHotPath(queryHint, queryData, i, defaultQueryFunctor));
        }
        return retGraph;
    }

    std::tuple<IndexType, IndexType, DistType> NearestNodes(const QueryContext& rhs) const{

        assert(this->distanceFunctor == rhs.distanceFunctor);

        std::pair<size_t, size_t> bestPair;
        DistType bestDistance(std::numeric_limits<DistType>::max());
        //NodeTracker nodesVisitedA(subGraphA.dataBlock.size());
        //NodeTracker nodesVisitedB(subGraphB.dataBlock.size());

        for(const auto& starterA: this->queryHint.neighbors){
            //nodesVisitedA[starterA.first] = true;
            for (const auto& starterB: rhs.queryHint.neighbors){
                //nodesVisitedB[starterB.first] = true;
                DistType distance = distanceFunctor(this->dataBlock[starterA.first], rhs.dataBlock[starterB.first]);
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
                DistType distance = distanceFunctor(this->dataBlock[neighborA.first], rhs.dataBlock[tmpPair.second]);
                if (distance < bestDistance){
                    bestDistance = distance;
                    tmpPair.first = neighborA.first;
                    breakVar = false;
                }
                    //nodesVisitedA[neighborA.first] = true;
                //}  
                
                for (const auto& neighborOfNeighborA: this->subGraph[neighborA.first]){
                    //if (nodesVisitedA[neighborOfNeighborA.first]) continue;
                    //nodesVisitedA[neighborOfNeighborA.first] = true;
                    DistType distance = this->distanceFunctor(this->dataBlock[neighborOfNeighborA.first], rhs.dataBlock[tmpPair.second]);
                    if (distance < bestDistance){
                        bestDistance = distance;
                        tmpPair.first = neighborOfNeighborA.first;
                        breakVar = false;
                    }
                }
            }
            for (const auto& neighborB: rhs.subGraph[bestPair.second]){
                //if (!nodesVisitedB[neighborB.first]){
                    DistType distance = this->distanceFunctor(this->dataBlock[tmpPair.first], rhs.dataBlock[neighborB.first]);
                if (distance < bestDistance){
                    bestDistance = distance;
                    tmpPair.second = neighborB.first;
                    breakVar = false;
                }
                //  nodesVisitedB[neighborB.first] = true;
                //}
                for (const auto& neighborOfNeighborB: rhs.subGraph[neighborB.first]){
                    //nodesVisitedB[neighborOfNeighborB.first] = true;
                    DistType distance = this->distanceFunctor(this->dataBlock[tmpPair.first], rhs.dataBlock[neighborOfNeighborB.first]);
                    if (distance < bestDistance){
                        bestDistance = distance;
                        tmpPair.second = neighborOfNeighborB.first;
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