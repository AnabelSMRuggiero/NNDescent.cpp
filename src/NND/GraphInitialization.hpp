/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_GRAPHINITIALIZATION_HPP
#define NND_GRAPHINITIALIZATION_HPP

#include <utility>
#include <vector>
#include <algorithm>

#include "../RPTrees/Forest.hpp"

#include "./Type.hpp"

#include "./GraphStructures.hpp"
#include "./MetaGraph.hpp"
#include "./BlockwiseAlgorithm.hpp"


#include "../Utilities/Data.hpp"
#include "../Utilities/Metrics/FunctorErasure.hpp"

namespace nnd{

template<typename DataEntry>
std::pair<IndexMaps<size_t>, std::vector<DataBlock<DataEntry>>> PartitionData(const RandomProjectionForest& treeToMap, const DataSet<DataEntry>& sourceData){
    //There might be a more elegant way to do this with templates. I tried.
    auto boundContructor = [](const DataSet<DataEntry>& dataSource, std::span<const size_t> dataPoints, size_t blockNumber){ 
        return DataBlock<DataEntry>(dataSource, dataPoints, blockNumber);
    };
    
    DataMapper<DataEntry, DataBlock<DataEntry>, decltype(boundContructor)> dataMapper(sourceData, boundContructor);
    CrawlTerminalLeaves(treeToMap, dataMapper);
    
    std::vector<DataBlock<DataEntry>> retBlocks = std::move(dataMapper.dataBlocks);
    IndexMaps<size_t> retMaps = {
        std::move(dataMapper.splitToBlockNum),
        std::move(dataMapper.blockIndexToSource),
        std::move(dataMapper.sourceToBlockIndex),
        std::move(dataMapper.sourceToSplitIndex)
    };

    return {retMaps, std::move(retBlocks)};
}

template<typename DistType>
std::vector<Graph<size_t, DistType>> InitializeBlockGraphs(const size_t numBlocks, const std::vector<size_t>& blockSizes, const size_t numNeighbors, DispatchFunctor<DistType> distanceFunctor){
    
    std::vector<Graph<size_t, DistType>> blockGraphs(0);
    blockGraphs.reserve(blockSizes.size());
    for (size_t i =0; i<numBlocks; i+=1){
        distanceFunctor.SetBlocks(i,i);
        blockGraphs.push_back(BruteForceBlock<DistType>(numNeighbors, blockSizes[i], distanceFunctor));
    }

    return blockGraphs;
}

template<typename BlockNumberType, typename DataIndexType, typename DistType>
Graph<BlockIndecies, DistType> ToBlockIndecies(const Graph<DataIndexType, DistType>& blockGraph, const BlockNumberType blockNum){
    Graph<BlockIndecies, DistType> newGraph(blockGraph.size(), blockGraph[0].size());
    for (size_t j = 0; const auto& vertex: blockGraph){
        newGraph[j].resize(blockGraph[j].size());
        for(size_t k = 0; const auto& neighbor: vertex){
            newGraph[j][k] = {{blockNum, neighbor.first}, neighbor.second};
            k++;
        }   
        j++;
    }
    return newGraph;
}

template <typename DataEntry, typename DataView, typename DistType, typename COMExtent>
Graph<size_t, DistType> GenerateQueryHints(const std::vector<Graph<size_t, DistType>>& blockGraphs,
                                                  const std::vector<DataBlock<DataEntry>>& dataBlocks,
                                                  const MetaGraph<COMExtent>& metaGraph,
                                                  const size_t numNeighbors,
                                                  SpaceMetric<AlignedSpan<const COMExtent>, DataView, COMExtent> distanceFunctor){
    
    Graph<size_t, DistType> retGraph;
    for(size_t i = 0; i<metaGraph.points.size(); i+=1){
        retGraph.push_back(QueryHintFromCOM<DataEntry, DataView, DistType, COMExtent>(metaGraph.points[i].centerOfMass, 
                                                                                           {blockGraphs[i], dataBlocks[i]}, 
                                                                                           numNeighbors, 
                                                                                           distanceFunctor));
    }

    return retGraph;

}


template<typename DistType, typename COMExtent, typename DistanceFunctor>
std::vector<BlockUpdateContext<DistType, DistanceFunctor>> InitializeBlockContexts(const std::vector<Graph<size_t, DistType>>& blockGraphs,
                                                                  const MetaGraph<COMExtent>& metaGraph,
                                                                  Graph<size_t, DistType>& queryHints,
                                                                  const int queryDepth,
                                                                  DistanceFunctor& distanceFunctor){
                                                                        
    std::vector<BlockUpdateContext<DistType, DistanceFunctor>> blockUpdateContexts;
    blockUpdateContexts.reserve(blockGraphs.size());
    //template<typename size_t, typename DataEntry, typename DistType, typename COMExtentType>
    for (size_t i = 0; i<blockGraphs.size(); i+=1){

        QueryContext<DistType, DistanceFunctor> queryContext(blockGraphs[i], std::move(queryHints[i]), DefaultQueryFunctor<DistType, DistanceFunctor>(distanceFunctor), queryDepth, i, blockGraphs[i].size());
            /*
            const Graph<size_t, DistType>& subGraph,
                 const GraphVertex<size_t, DistType> queryHint,
                 DefaultQueryFunctor<DistType>& defaultQueryFunctor,
                 const int querySearchDepth,
                 const size_t blockNumber,
                 const size_t blockSize
            */

        blockUpdateContexts.emplace_back(blockGraphs[i],
                                         std::move(queryContext),
                                         metaGraph.verticies.size());

        blockUpdateContexts.back().currentGraph = ToBlockIndecies(blockGraphs[i], i);
        blockUpdateContexts.back().blockJoinTracker[i] = true;
    }

    return blockUpdateContexts;
}

template<typename DistType>
using InitialJoinHints = std::unordered_map<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>;

template<typename DistType, typename DistanceFunctor>
std::pair<Graph<size_t, DistType>, InitialJoinHints<DistType>> NearestNodeDistances(const std::vector<BlockUpdateContext<DistType, DistanceFunctor>>& blockUpdateContexts,
                                                        const MetaGraph<DistType>& metaGraph,
                                                        const size_t maxNearestNodeNeighbors){

    std::unordered_set<ComparisonKey<size_t>> nearestNodeDistQueue;

    for (size_t i = 0; const auto& vertex: metaGraph.verticies){
        for (const auto& neighbor: vertex){
            nearestNodeDistQueue.insert({i, neighbor.first});
        }
        i++;
    }
    
    std::vector<ComparisonKey<size_t>> distancesToCompute;
    distancesToCompute.reserve(nearestNodeDistQueue.size());
    for (const auto& pair: nearestNodeDistQueue){
        distancesToCompute.push_back(pair);
    }
    
    std::vector<std::tuple<size_t, size_t, DistType>> nnDistanceResults(nearestNodeDistQueue.size());
    auto nnDistanceFunctor = [&](const ComparisonKey<size_t> blockNumbers) -> std::tuple<size_t, size_t, DistType>{
        return blockUpdateContexts[blockNumbers.first].queryContext.NearestNodes(blockUpdateContexts[blockNumbers.second].queryContext,
                                                                                 blockUpdateContexts[blockNumbers.first].queryContext.defaultQueryFunctor);
    };

    std::transform(std::execution::unseq, distancesToCompute.begin(), distancesToCompute.end(), nnDistanceResults.begin(), nnDistanceFunctor);

    std::unordered_map<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>> blockJoinHints;

    for (size_t i = 0; i<distancesToCompute.size(); i += 1){
        blockJoinHints[distancesToCompute[i]] = nnDistanceResults[i];
    }

    Graph<size_t, DistType> nearestNodeDistances(metaGraph.verticies.size(), maxNearestNodeNeighbors);
    for(size_t i = 0; const auto& result: nnDistanceResults){
        
        nearestNodeDistances[distancesToCompute[i].first].push_back({distancesToCompute[i].second,
                                                                     std::get<2>(result)});
        nearestNodeDistances[distancesToCompute[i].second].push_back({distancesToCompute[i].first,
                                                                      std::get<2>(result)});
        //nearestNeighbors.push_back({pair.first, std::get<2>(pair.second)});
        i++;
    }

    auto sortFunctor = [=] (GraphVertex<size_t, DistType>& vertex){
        std::sort(std::execution::unseq, vertex.begin(), vertex.end(), NeighborDistanceComparison<size_t, DistType>);
        vertex.resize(maxNearestNodeNeighbors);
    };
    std::for_each(std::execution::unseq, nearestNodeDistances.begin(), nearestNodeDistances.end(), sortFunctor);

    return {std::move(nearestNodeDistances), std::move(blockJoinHints)};
}


template<typename DistType, typename DistanceFunctor>
void StitchBlocks(const Graph<size_t, DistType>& nearestNodeDistances,
                  const InitialJoinHints<DistType>& stitchHints,
                  std::vector<BlockUpdateContext<DistType, DistanceFunctor>>& blockUpdateContexts,
                  CachingFunctor<DistType>& cachingFunctor){

    std::unordered_set<ComparisonKey<size_t>> initBlockJoinQueue;
    for(size_t i = 0; const auto& vertex: nearestNodeDistances){
        for(size_t j = 0; j<nearestNodeDistances[0].size(); j+=1){
            initBlockJoinQueue.insert({i, vertex[j].first});
        }
        i++;
    }

    std::vector<ComparisonKey<size_t>> initBlockJoins;
    initBlockJoins.reserve(initBlockJoinQueue.size());
    for (const auto& pair: initBlockJoinQueue){
        blockUpdateContexts[pair.first].blockJoinTracker[pair.second] = true;
        blockUpdateContexts[pair.second].blockJoinTracker[pair.first] = true;
        initBlockJoins.push_back(pair);
    }
    

    std::vector<std::pair<JoinResults<size_t, DistType>, JoinResults<size_t, DistType>>> initUpdates(initBlockJoins.size());
    

    auto initBlockJoin = [&](const ComparisonKey<size_t> blockNumbers) -> std::pair<JoinResults<size_t, DistType>, JoinResults<size_t, DistType>>{
        
        auto [blockNums, stitchHint] = *(stitchHints.find(blockNumbers));
        if (blockNums.first != blockNumbers.first) stitchHint = {std::get<1>(stitchHint), std::get<0>(stitchHint), std::get<2>(stitchHint)};
        auto blockLHS = blockUpdateContexts[blockNumbers.first];
        auto blockRHS = blockUpdateContexts[blockNumbers.second];
        JoinHints<size_t> LHShint;
        LHShint[std::get<0>(stitchHint)] = {std::get<1>(stitchHint)};
        JoinHints<size_t> RHShint;
        RHShint[std::get<1>(stitchHint)] = {std::get<0>(stitchHint)};
        /*
        DistanceCache<DistType> distanceCache;
        distanceCache.reserve(50*50); //This is a touch janky, but place holder while I get code working again

        
        blockLHS.queryContext.defaultQueryFunctor.SetBlocks(blockNumbers.first, blockNumbers.second);
        auto cachingDistanceFunctor = blockLHS.queryContext.defaultQueryFunctor.CachingFunctor(distanceCache);
        auto cachedDistanceFunctor = blockRHS.queryContext.defaultQueryFunctor.CachedFunctor(distanceCache);
        */
        cachingFunctor.SetBlocks(blockNumbers.first, blockNumbers.second);

        std::pair<JoinResults<size_t, DistType>, JoinResults<size_t, DistType>> retPair;
        retPair.first = BlockwiseJoin(LHShint,
                                      blockLHS.currentGraph,
                                      blockLHS.joinPropagation,
                                      blockRHS.queryContext,
                                      cachingFunctor);

        cachingFunctor.metricFunctor.SetBlocks(blockNumbers.second, blockNumbers.first);
        ReverseBlockJoin(RHShint,
                         blockRHS.currentGraph,
                         blockRHS.joinPropagation,
                         blockLHS.queryContext,
                         cachingFunctor,
                         cachingFunctor.metricFunctor);

        for(size_t i = 0; const auto& vertex: cachingFunctor.reverseGraph){
            if(vertex.size()>0){
                retPair.second.push_back({i, vertex});
            }
            i++;
        }

        

        return retPair;
    };

    std::transform(std::execution::seq, initBlockJoins.begin(), initBlockJoins.end(), initUpdates.begin(), initBlockJoin);
    int initGraphUpdates(0);
    for (size_t i = 0; i<initUpdates.size(); i += 1){
        ComparisonKey<size_t> blocks = initBlockJoins[i];
        std::pair<JoinResults<size_t, DistType>, JoinResults<size_t, DistType>>& updates = initUpdates[i];
        for (auto& result: updates.first){
            initGraphUpdates += ConsumeVertex(blockUpdateContexts[blocks.first].currentGraph[result.first], result.second, blocks.second);
        }
        for (auto& result: updates.second){
            initGraphUpdates += ConsumeVertex(blockUpdateContexts[blocks.second].currentGraph[result.first], result.second, blocks.first);
        }

    }

    
    

    
    //Initial filling of comparison targets.
    std::vector<ComparisonMap<size_t, size_t>> queueMaps;
    queueMaps.reserve(blockUpdateContexts.size());
    for (size_t i = 0; i<blockUpdateContexts.size(); i+=1){
        queueMaps.push_back(InitializeComparisonQueues<size_t, size_t, float>(blockUpdateContexts[i].currentGraph, i));
    }
    
    //std::vector<JoinMap<size_t, size_t>> joinHints;

    for(size_t i = 0; i<queueMaps.size(); i+=1){
        ComparisonMap<size_t, size_t>& comparisonMap = queueMaps[i];
        
        blockUpdateContexts[i].joinsToDo = InitializeJoinMap<DistType>(blockUpdateContexts, comparisonMap, blockUpdateContexts[i].blockJoinTracker);
    }
}

}

#endif