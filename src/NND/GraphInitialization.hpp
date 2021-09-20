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
#include <future>

#include "../RPTrees/Forest.hpp"

#include "./Type.hpp"

#include "./GraphStructures.hpp"
#include "./MetaGraph.hpp"
#include "./BlockwiseAlgorithm.hpp"


#include "../Utilities/Data.hpp"
#include "../Utilities/Metrics/FunctorErasure.hpp"

namespace nnd{

template<typename DataEntry>
std::pair<IndexMaps<size_t>, std::vector<DataBlock<typename DataEntry::value_type>>> PartitionData(const RandomProjectionForest& treeToMap, const DataSet<DataEntry>& sourceData, const size_t startIndex=0){
    //There might be a more elegant way to do this with templates. I tried.

    using DataType = typename DataEntry::value_type;

    auto boundContructor = [entryLength = sourceData.sampleLength](const DataSet<DataEntry>& dataSource, std::span<const size_t> dataPoints, size_t blockNumber){ 
        return DataBlock<DataType>(dataSource, dataPoints, entryLength, blockNumber);
    };
    
    DataMapper<DataEntry, DataBlock<DataType>, decltype(boundContructor)> dataMapper(sourceData, boundContructor, startIndex);
    CrawlTerminalLeaves(treeToMap, dataMapper);
    
    std::vector<DataBlock<DataType>> retBlocks = std::move(dataMapper.dataBlocks);
    IndexMaps<size_t> retMaps = {
        std::move(dataMapper.splitToBlockNum),
        std::move(dataMapper.blockIndexToSource),
        std::move(dataMapper.sourceToBlockIndex),
        std::move(dataMapper.sourceToSplitIndex)
    };


    
    return {retMaps, std::move(retBlocks)};
}

template<typename DataEntry, typename Pool>
std::pair<IndexMaps<size_t>, std::vector<DataBlock<typename DataEntry::value_type>>> PartitionData(const RandomProjectionForest& treeToMap, const DataSet<DataEntry>& sourceData, Pool& threadPool,  const size_t startIndex=0){
    //There might be a more elegant way to do this with templates. I tried.

    /*
    auto boundContructor = [](const DataSet<DataEntry>& dataSource, std::span<const size_t> dataPoints, size_t blockNumber){ 
        return DataBlock<DataEntry>(dataSource, dataPoints, blockNumber);
    };
    */
    
    using DataType = typename DataEntry::value_type;

    threadPool.StartThreads();

    DataMapper<DataEntry, void, void> dataMapper(sourceData, startIndex);
    
    auto mapTask = [&]()mutable{
        CrawlTerminalLeaves(treeToMap, dataMapper);    
    };
    
    threadPool.DelegateTask(mapTask);

    std::vector<std::unique_ptr<std::promise<DataBlock<DataType>>>> blockPromise;
    
    auto constructorGenerator = [&, i = startIndex](const size_t splitIdx, std::span<const size_t> indecies)mutable{
        blockPromise.push_back(std::make_unique<std::promise<DataBlock<DataType>>>());

        auto blockBuilder = [&, &promise = *(blockPromise.back()), indecies, i](){
            promise.set_value(DataBlock<DataType>(sourceData, indecies, sourceData.sampleLength, i));
        };

        threadPool.DelegateTask(blockBuilder);

        i++;
    };

    CrawlTerminalLeaves(treeToMap, constructorGenerator);
    
    std::vector<DataBlock<DataType>> retBlocks;
    for(auto& promisePtr: blockPromise){
        std::future<DataBlock<DataType>> blockFuture = promisePtr->get_future();
        retBlocks.push_back(blockFuture.get());
    }

    threadPool.StopThreads();

    //std::vector<DataBlock<DataEntry>> retBlocks = std::move(dataMapper.dataBlocks);
    IndexMaps<size_t> retMaps = {
        std::move(dataMapper.splitToBlockNum),
        std::move(dataMapper.blockIndexToSource),
        std::move(dataMapper.sourceToBlockIndex),
        std::move(dataMapper.sourceToSplitIndex)
    };


    
    return {retMaps, std::move(retBlocks)};
}

template<typename DistType>
std::vector<Graph<size_t, DistType>> InitializeBlockGraphs(const size_t numBlocks,
                                                           const std::vector<size_t>& blockSizes,
                                                           const size_t numNeighbors,
                                                           DispatchFunctor<DistType> distanceFunctor){
    
    std::vector<Graph<size_t, DistType>> blockGraphs(0);
    blockGraphs.reserve(blockSizes.size());
    for (size_t i =0; i<numBlocks; i+=1){
        distanceFunctor.SetBlocks(i,i);
        blockGraphs.push_back(BruteForceBlock<DistType>(numNeighbors, blockSizes[i], distanceFunctor));
    }

    return blockGraphs;
}



template <typename DistType, typename COMExtent>
Graph<size_t, DistType> GenerateQueryHints(const OffsetSpan<Graph<size_t, DistType>> blockGraphs,
                                                  const MetaGraph<COMExtent>& metaGraph,
                                                  const size_t numNeighbors,
                                                  SinglePointFunctor<COMExtent> distanceFunctor){
    
    Graph<size_t, DistType> retGraph;
    size_t blockOffset = 0;
    for(size_t i = blockOffset; i<metaGraph.points.size() + blockOffset; i+=1){
        distanceFunctor.SetBlock(i);
        retGraph.push_back(QueryHintFromCOM<DistType, COMExtent>(i, 
                                                                 blockGraphs[i], 
                                                                 numNeighbors, 
                                                                 distanceFunctor));
    }

    return retGraph;

}


template<typename DistType, typename COMExtent>
std::unique_ptr<BlockUpdateContext<DistType>[]> InitializeBlockContexts(std::vector<Graph<size_t, DistType>>& blockGraphs,
                                                                  const MetaGraph<COMExtent>& metaGraph,
                                                                  Graph<size_t, DistType>& queryHints,
                                                                  const int queryDepth,
                                                                  const size_t blockNumStart){
                                                                        
    std::unique_ptr<BlockUpdateContext<DistType>[]> blockUpdateContexts = std::make_unique<BlockUpdateContext<DistType>[]>(blockGraphs.size());
    //blockUpdateContexts.reserve(blockGraphs.size());
    //template<typename size_t, typename DataEntry, typename DistType, typename COMExtentType>
    for (size_t i = 0; i<blockGraphs.size(); i+=1){

        QueryContext<DistType> queryContext(blockGraphs[i],
                                            std::move(queryHints[i]),
                                            queryDepth,
                                            i + blockNumStart,
                                            blockGraphs[i].size());

        
        BlockUpdateContext<DistType>* blockLocation = &blockUpdateContexts[i];

        blockLocation->~BlockUpdateContext<DistType>();
        new(blockLocation) BlockUpdateContext<DistType>(std::move(blockGraphs[i]),
                                                        std::move(queryContext),
                                                        metaGraph.verticies.size());

        //blockUpdateContexts.back().currentGraph = ToBlockIndecies(blockGraphs[i], i);
        blockUpdateContexts[i].blockJoinTracker[i] = true;
    }

    return blockUpdateContexts;
}

template<typename DistType>
using InitialJoinHints = std::unordered_map<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>;

template<typename DistType>
std::pair<Graph<size_t, DistType>, InitialJoinHints<DistType>> NearestNodeDistances(OffsetSpan<const BlockUpdateContext<DistType>> blockUpdateContexts,
                                                        const MetaGraph<DistType>& metaGraph,
                                                        const size_t maxNearestNodeNeighbors,
                                                        DispatchFunctor<DistType>& distanceFunctor){

    std::unordered_set<ComparisonKey<size_t>> nearestNodeDistQueue;
    //const size_t startIndex = blockUpdateContexts[0].queryContext.blockNumber;

    size_t blockIdxOffset = blockUpdateContexts.Offset();

    for (size_t i = blockIdxOffset; const auto& vertex: metaGraph.verticies){
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
                                                                                 distanceFunctor);
    };

    std::transform(std::execution::unseq, distancesToCompute.begin(), distancesToCompute.end(), nnDistanceResults.begin(), nnDistanceFunctor);

    std::unordered_map<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>> blockJoinHints;

    for (size_t i = 0; i<distancesToCompute.size(); i += 1){
        blockJoinHints[distancesToCompute[i]] = nnDistanceResults[i];
    }

    Graph<size_t, DistType> nearestNodeDistances(metaGraph.verticies.size(), maxNearestNodeNeighbors);
    OffsetSpan distOffsetView = nearestNodeDistances.GetOffsetView(blockIdxOffset);
    for(size_t i = 0; const auto& result: nnDistanceResults){
        
        distOffsetView[distancesToCompute[i].first].push_back({distancesToCompute[i].second,
                                                                     std::get<2>(result)});
        distOffsetView[distancesToCompute[i].second].push_back({distancesToCompute[i].first,
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


template<typename DistType>
void StitchBlocks(const Graph<size_t, DistType>& nearestNodeDistances,
                  const InitialJoinHints<DistType>& stitchHints,
                  OffsetSpan<BlockUpdateContext<DistType>> blockUpdateContexts,
                  CachingFunctor<DistType>& cachingFunctor){

    size_t blockIdxOffset = blockUpdateContexts.Offset();

    //OffsetSpan distView = nearestNodeDistances.GetOffsetView(blockIdxOffset);

    std::unordered_set<ComparisonKey<size_t>> initBlockJoinQueue;
    for(size_t i = blockIdxOffset; const auto& vertex: nearestNodeDistances){
        for(size_t j = 0; j<vertex.size(); j+=1){
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
        auto& blockLHS = blockUpdateContexts[blockNumbers.first];
        auto& blockRHS = blockUpdateContexts[blockNumbers.second];
        JoinHints<size_t> LHShint;
        LHShint[std::get<0>(stitchHint)] = {std::get<1>(stitchHint)};
        JoinHints<size_t> RHShint;
        RHShint[std::get<1>(stitchHint)] = {std::get<0>(stitchHint)};

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
        
        for(size_t i = 0; auto& vertex: cachingFunctor.reverseGraph){
            EraseRemove(vertex, blockRHS.currentGraph[i].PushThreshold());
            /*
            NeighborOverDist<DistType> comparison(blockRHS.currentGraph[i].PushThreshold());
            vertex.erase(std::remove_if(vertex.begin(),
                                        vertex.end(),
                                        comparison),
                        vertex.end());
            */
        }
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
        queueMaps.push_back(InitializeComparisonQueues<size_t, size_t, DistType>(blockUpdateContexts[i].currentGraph, i));
    }
    
    //std::vector<JoinMap<size_t, size_t>> joinHints;

    for(size_t i = 0; i<queueMaps.size(); i+=1){
        ComparisonMap<size_t, size_t>& comparisonMap = queueMaps[i];
        
        blockUpdateContexts[i].joinsToDo = InitializeJoinMap<DistType>(blockUpdateContexts, comparisonMap, blockUpdateContexts[i].blockJoinTracker);
    }
}

template<typename DataType, typename DistType, typename COMExtent>
std::unique_ptr<BlockUpdateContext<DistType>[]> BuildGraph(const std::vector<DataBlock<DataType>>& dataBlocks,
                                                                          const MetaGraph<COMExtent>& metaGraph,
                                                                          DispatchFunctor<DistType>& dispatch,
                                                                          std::vector<size_t>&& sizes,
                                                                          const HyperParameterValues& hyperParams,
                                                                          std::execution::sequenced_policy){


    std::vector<Graph<size_t, float>> blockGraphs = InitializeBlockGraphs<float>(sizes.size(), sizes, hyperParams.indexParams.blockGraphNeighbors, dispatch);

    OffsetSpan<Graph<size_t, float>> blockGraphView = {{blockGraphs.begin(), blockGraphs.size()}, dataBlocks[0].blockNumber};

    DataComDistance<DataType, COMExtent, EuclideanMetricPair> comFunctor(metaGraph, dataBlocks);

    Graph<size_t, float> queryHints = GenerateQueryHints<float, float>(blockGraphView, metaGraph, hyperParams.indexParams.blockGraphNeighbors, comFunctor);


    std::unique_ptr<BlockUpdateContext<DistType>[]> blockUpdateContexts = InitializeBlockContexts<DistType, COMExtent>(blockGraphs, 
                                                                                         metaGraph,
                                                                                         queryHints,
                                                                                         hyperParams.indexParams.queryDepth,
                                                                                         metaGraph.GetBlockOffset());
    
    OffsetSpan<BlockUpdateContext<DistType>> blockSpan(blockUpdateContexts.get(), blockGraphs.size(), dataBlocks[0].blockNumber);
    OffsetSpan<const BlockUpdateContext<DistType>> constBlockSpan(blockUpdateContexts.get(), blockGraphs.size(), dataBlocks[0].blockNumber);
    CachingFunctor<float> cacher(dispatch, hyperParams.splitParams.maxTreeSize, hyperParams.indexParams.blockGraphNeighbors);

    auto [nearestNodeDistances, stitchHints] = NearestNodeDistances(constBlockSpan, metaGraph, hyperParams.indexParams.nearestNodeNeighbors, dispatch);
    StitchBlocks(nearestNodeDistances, stitchHints, blockSpan, cacher);
    
    
    //int iteration(1);
    int graphUpdates(1);
    while(graphUpdates>0){
        graphUpdates = 0;
        for(size_t i = 0; i<blockSpan.size(); i+=1){
            for (auto& joinList: blockUpdateContexts[i].joinsToDo){
                graphUpdates += UpdateBlocks(blockUpdateContexts[i], blockUpdateContexts[joinList.first], cacher);
                blockUpdateContexts[joinList.first].joinsToDo.erase(i);
            }
        }
        for (auto& context: blockSpan){
            context.SetNextJoins();
        }

    }

    return blockUpdateContexts;
}


}

#endif