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

#include "../Utilities/Data.hpp"
#include "../Utilities/DataDeserialization.hpp"
#include "GraphStructures.hpp"

namespace nnd{

template<TriviallyCopyable IndexType, typename DataEntry, std::floating_point FloatType>
struct SubProblemData{
    const Graph<IndexType, FloatType>& subGraph;
    const DataBlock<DataEntry>& dataBlock;
};





struct NodeTracker{

    using reference = std::vector<bool>::reference;
    using size_type = std::vector<bool>::size_type;

    std::vector<bool> flags;

    NodeTracker(size_t graphSize): flags(graphSize, false){};

    reference operator[](size_type i){
        return flags[i];
    }

    reference operator[](BlockIndecies i){
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

template<TriviallyCopyable IndexType, typename DataEntry, std::floating_point FloatType>
GraphVertex<IndexType, FloatType> QueryCOMNeighbors(const std::valarray<FloatType>& centerOfMass,
                                                     const SubProblemData<IndexType, DataEntry, FloatType> subProb, 
                                                     const int numCandidates,
                                                     SpaceMetric<std::valarray<FloatType>, DataEntry, FloatType> distanceFunctor){

    GraphVertex<IndexType, FloatType> COMneighbors(numCandidates);
    
    //Just gonna dummy it and select the first few nodes. Since the splitting process is randomized, this is a totally random selection, right? /s
    NodeTracker nodesVisited(subProb.dataBlock.size());
    for (size_t i = 0; i < numCandidates; i+=1){
        COMneighbors.neighbors.push_back(std::pair<IndexType, FloatType>(i,
                                          distanceFunctor(centerOfMass, subProb.dataBlock.blockData[i])));
        nodesVisited[i] = true;
    }
    std::make_heap(COMneighbors.neighbors.begin(), COMneighbors.neighbors.end(), NeighborDistanceComparison<IndexType, FloatType>);

    bool breakVar = false;
    GraphVertex<IndexType, FloatType> newState(COMneighbors);
    while (!breakVar){
        breakVar = true;   
        for (const auto& curCandidate: COMneighbors){
            for (const auto& joinTarget: subProb.subGraph[curCandidate.first]){
                if(nodesVisited[joinTarget.first]) continue;
                nodesVisited[joinTarget.first] = true;
                FloatType distance = distanceFunctor(centerOfMass, subProb.dataBlock[joinTarget.first]);
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
    GraphVertex<IndexType, DistType>& queryHint;
    const QueryType& queryData;
};


template<TriviallyCopyable IndexType, typename DataEntry, typename DistType>
struct QueryContext{
    const Graph<IndexType, DistType>& subGraph;
    const DataBlock<DataEntry>& dataBlock;
    GraphVertex<IndexType, DistType> queryHint;
    const int numCandidates;
    std::unordered_map<size_t, Graph<IndexType, DistType>> neighborCandidates;
    std::unordered_map<size_t, DistType> distances;
    SpaceMetric<DataEntry, DataEntry, DistType> distanceFunctor;

    QueryContext(const Graph<IndexType, DistType>& subGraph,
                 const DataBlock<DataEntry>& dataBlock,
                 GraphVertex<IndexType, DataEntry> queryHint,
                 SpaceMetric<DataEntry, DataEntry, DistType> distanceFunctor,
                 const int numCandidates): subGraph(subGraph), dataBlock(dataBlock), queryHint(std::move(queryHint)), numCandidates(numCandidates), neighborCandidates(), distanceFunctor(distanceFunctor){};

    QueryContext(const Graph<IndexType, DistType>& subGraph,
                 const DataBlock<DataEntry>& dataBlock,
                 const std::valarray<DistType>& centerOfMass,
                 const int numCandidates,
                 SpaceMetric<DataEntry, DataEntry, DistType> distanceFunctor,
                 SpaceMetric<std::valarray<DistType>, DataEntry, DistType> comDistanceFunctor):
                    subGraph(subGraph), dataBlock(dataBlock), numCandidates(numCandidates), neighborCandidates(), distanceFunctor(distanceFunctor){
        const SubProblemData thisSub{subGraph, dataBlock};
        queryHint = QueryCOMNeighbors<IndexType, DataEntry, DistType>(centerOfMass, thisSub, numCandidates, comDistanceFunctor);
        for (auto& hint: queryHint){
            hint.second = std::numeric_limits<DistType>::max();
        }
    };

    //Nearest Node Distance
    //make checking this in parallel safe
    DistType operator*(QueryContext& rhs){
        auto result = this->distances.find(rhs.dataBlock.blockNumber);
        if(result == distances.end()){
            DistType distance = NearestNodes(rhs);
            this->distances[rhs.dataBlock.blockNumber] = distance;
            rhs.distances[this->dataBlock.blockNumber] = distance;
        }
        return this->distances[rhs.dataBlock.blockNumber];

    }

    void operator||(QueryContext& rhs){
        auto result = this->neighborCandidates.find(rhs.dataBlock.blockNumber);
        if (result != neighborCandidates.end()) return;
        //Query RHS data against mine
        rhs.neighborCandidates[this->dataBlock.blockNumber] = this->QuerySubGraph(rhs);
        //Query My data against RHS
        this->neighborCandidates[rhs.dataBlock.blockNumber] = rhs.QuerySubGraph(*this);
    }
    //

    template<typename QueryType>
    GraphVertex<IndexType, DistType> operator||(const QueryPoint<IndexType, QueryType, DistType>& queryPoint) const {
        return QueryHotPath(queryPoint.queryHint, queryPoint.queryData);
    }

    //I want copies when I use the queryHint member, but not really when I'm passing in hints. ???
    //Figure this out later
    template<typename QueryType>
    GraphVertex<IndexType, DistType> QueryHotPath(GraphVertex<IndexType, DistType> initVertex,
                                                   const QueryType& queryData) const {
        NodeTracker nodesVisited(dataBlock.size());
        
        for (auto& queryStart: initVertex){
            queryStart.second = this->distanceFunctor(queryData, dataBlock[queryStart.first]);
        }
        std::make_heap(initVertex.begin(), initVertex.end(), NeighborDistanceComparison<IndexType, DistType>);

        GraphVertex<IndexType, DistType> newState(initVertex);
        bool breakVar = false;
        while (!breakVar){
            breakVar = true;
            for (const auto& neighbor: initVertex){
                const GraphVertex<IndexType, DistType>& currentNeighbor = subGraph[neighbor.first];
                for (const auto& joinTarget: currentNeighbor){
                    if (nodesVisited[joinTarget.first] == true) continue;
                    nodesVisited[joinTarget.first] = true;
                    DistType distance = this->distanceFunctor(queryData, dataBlock[joinTarget.first]);
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

    private:

    Graph<IndexType, DistType> QuerySubGraph(const QueryContext& rhs){

        Graph<IndexType, DistType> retGraph;

        for (size_t i = 0; i<rhs.dataBlock.size(); i += 1){
            DataEntry queryData = rhs.dataBlock[i];
            retGraph.push_back(QueryHotPath(queryHint, queryData));
        }
        return retGraph;
    }

    DistType NearestNodes(const QueryContext& rhs){

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
        
        return bestDistance;
    }

};

}

#endif 