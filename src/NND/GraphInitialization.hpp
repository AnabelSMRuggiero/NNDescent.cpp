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

#include <cstddef>
#include <utility>
#include <vector>
#include <algorithm>
#include <future>
#include <memory>

#include "RPTrees/Forest.hpp"

#include "Type.hpp"

#include "GraphStructures.hpp"
#include "MetaGraph.hpp"
#include "BlockwiseAlgorithm.hpp"

#include "ann/Data.hpp"
#include "FunctorErasure.hpp"
#include "MetricHelpers.hpp"
#include "NND/GraphStructures/CachingFunctor.hpp"


namespace nnd{

template<typename DataType>
std::pair<IndexMaps<size_t>, std::vector<DataBlock<DataType>>> PartitionData(const RandomProjectionForest& treeToMap, const DataSet<DataType>& sourceData, const size_t startIndex=0){


    //using DataType = typename DataEntry::value_type;

    auto boundContructor = [entryLength = sourceData.SampleLength()](const DataSet<DataType>& dataSource, std::span<const size_t> dataPoints, size_t blockNumber){ 
        return DataBlock<DataType>(dataSource, dataPoints, entryLength, blockNumber);
    };
    
    DataMapper<DataType, DataBlock<DataType>, decltype(boundContructor)> dataMapper(sourceData, boundContructor, startIndex);
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

template<typename DataType, typename Pool>
std::pair<IndexMaps<size_t>, std::vector<DataBlock<DataType>>> PartitionData(const RandomProjectionForest& treeToMap, const DataSet<DataType>& sourceData, Pool& threadPool,  const size_t startIndex=0){
    
    //using DataType = typename DataEntry::value_type;

    threadPool.StartThreads();

    DataMapper<DataType, void, void> dataMapper(sourceData, startIndex);
    
    auto mapTask = [&]()mutable{
        CrawlTerminalLeaves(treeToMap, dataMapper);    
    };
    
    threadPool.DelegateTask(mapTask);

    std::vector<std::unique_ptr<std::promise<DataBlock<DataType>>>> blockPromise;
    
    auto constructorGenerator = [&, i = startIndex](const size_t splitIdx, std::span<const size_t> indecies)mutable{
        blockPromise.push_back(std::make_unique<std::promise<DataBlock<DataType>>>());

        auto blockBuilder = [&, &promise = *(blockPromise.back()), indecies, i](){
            promise.set_value(DataBlock<DataType>(sourceData, indecies, sourceData.SampleLength(), i));
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
std::vector<Graph<DataIndex_t, DistType>> InitializeBlockGraphs(const size_t numBlocks,
                                                           std::span<const std::size_t> blockSizes,
                                                           const size_t numNeighbors,
                                                           erased_binary_binder<DistType> distanceFunctor){
    
    std::vector<Graph<DataIndex_t, DistType>> blockGraphs(0);
    blockGraphs.reserve(blockSizes.size());
    for (size_t i =0; i<numBlocks; i+=1){
        blockGraphs.push_back(BruteForceBlock<DistType>(numNeighbors, blockSizes[i], distanceFunctor(i,i)));
    }

    return blockGraphs;
}

template <typename DistType, typename COMExtent, typename IndexType>
Graph<IndexType, DistType> old_GenerateQueryHints(std::span<Graph<IndexType, DistType>> blockGraphs,
                                              size_t numNeighbors,
                                              erased_unary_binder<COMExtent> distanceFunctor){
    
    Graph<IndexType, DistType> retGraph;
    for(size_t i = 0; i<blockGraphs.size(); i+=1){
        retGraph.push_back(QueryHintFromCOM<DistType, COMExtent>(i, 
                                                                 blockGraphs[i], 
                                                                 numNeighbors, 
                                                                 distanceFunctor(i)));
    }

    return retGraph;

}

template <typename DistType, typename IndexType>
Graph<IndexType, DistType> GenerateQueryHints(std::span<Graph<IndexType, DistType>> blockGraphs,
                                              size_t numNeighbors){
    
    Graph<IndexType, DistType> retGraph;
    for(size_t i = 0; i<blockGraphs.size(); i+=1){
        //RandomQueryHint(std::size_t block_size, std::size_t num_candidates)
        retGraph.push_back(RandomQueryHint<IndexType, DistType>(blockGraphs[i].size(), 
                                                     numNeighbors));
    }

    return retGraph;

}


template<typename DistType>
ann::dynamic_array<BlockUpdateContext<DistType>> InitializeBlockContexts(std::vector<Graph<DataIndex_t, DistType>>& blockGraphs,
                                                                  Graph<DataIndex_t, DistType>& queryHints,
                                                                  int queryDepth,
                                                                  GraphFragment_t fragment_number=0){
                                                                        
    ann::dynamic_array<BlockUpdateContext<DistType>> blockUpdateContexts{blockGraphs.size()};
    //blockUpdateContexts.reserve(blockGraphs.size());
    //template<typename size_t, typename DataEntry, typename DistType, typename COMExtentType>
    for (size_t i = 0; i<blockGraphs.size(); i+=1){

        QueryContext<DataIndex_t, DistType> queryContext(blockGraphs[i],
                                            std::move(queryHints[i]),
                                            queryDepth,
                                            fragment_number,
                                            i,
                                            blockGraphs[i].size());

        
        BlockUpdateContext<DistType>* blockLocation = &blockUpdateContexts[i];

        blockLocation->~BlockUpdateContext<DistType>();
        new(blockLocation) BlockUpdateContext<DistType>(std::move(blockGraphs[i]),
                                                        std::move(queryContext),
                                                        blockGraphs.size());

        //blockUpdateContexts.back().currentGraph = ToBlockIndecies(blockGraphs[i], i);
        blockUpdateContexts[i].blockJoinTracker[i] = true;
    }

    return blockUpdateContexts;
}

template<typename DistType>
using InitialJoinHints = std::unordered_map<comparison_key<BlockNumber_t>, std::tuple<DataIndex_t, DataIndex_t, DistType>>;

template<typename DistType>
std::pair<Graph<BlockNumber_t, DistType>, InitialJoinHints<DistType>> NearestNodeDistances(std::span<const BlockUpdateContext<DistType>> blockUpdateContexts,
                                                        const MetaGraph<DistType>& metaGraph,
                                                        const size_t maxNearestNodeNeighbors,
                                                        const erased_binary_binder<DistType>& distanceFunctor){

    std::unordered_set<comparison_key<BlockNumber_t>> nearestNodeDistQueue;
    //const size_t startIndex = blockUpdateContexts[0].queryContext.blockNumber;


    for (size_t i = 0; const auto& vertex: metaGraph.verticies){
        for (const auto& neighbor: vertex){
            nearestNodeDistQueue.insert({static_cast<BlockNumber_t>(i), neighbor.first});
        }
        i++;
    }
    
    std::vector<comparison_key<BlockNumber_t>> distancesToCompute;
    distancesToCompute.reserve(nearestNodeDistQueue.size());
    for (const auto& pair: nearestNodeDistQueue){
        distancesToCompute.push_back(pair);
    }
    
    std::vector<std::tuple<DataIndex_t, DataIndex_t, DistType>> nnDistanceResults(nearestNodeDistQueue.size());
    auto nnDistanceFunctor = [&](const comparison_key<BlockNumber_t> blockNumbers) -> std::tuple<DataIndex_t, DataIndex_t, DistType>{
        return blockUpdateContexts[blockNumbers.first].queryContext.NearestNodes(blockUpdateContexts[blockNumbers.second].queryContext,
                                                                                 distanceFunctor(blockNumbers.first, blockNumbers.second));
    };

    std::transform(std::execution::unseq, distancesToCompute.begin(), distancesToCompute.end(), nnDistanceResults.begin(), nnDistanceFunctor);

    std::unordered_map<comparison_key<BlockNumber_t>, std::tuple<DataIndex_t, DataIndex_t, DistType>> blockJoinHints;

    for (size_t i = 0; i<distancesToCompute.size(); i += 1){
        blockJoinHints[distancesToCompute[i]] = nnDistanceResults[i];
    }

    Graph<BlockNumber_t, DistType> nearestNodeDistances(metaGraph.verticies.size(), maxNearestNodeNeighbors);
    
    for(size_t i = 0; const auto& result: nnDistanceResults){
        
        nearestNodeDistances[distancesToCompute[i].first].push_back({distancesToCompute[i].second,
                                                                     std::get<2>(result)});
        nearestNodeDistances[distancesToCompute[i].second].push_back({distancesToCompute[i].first,
                                                                      std::get<2>(result)});
        //nearestNeighbors.push_back({pair.first, std::get<2>(pair.second)});
        i++;
    }

    auto sortFunctor = [=] (GraphVertex<BlockNumber_t, DistType>& vertex){
        std::sort(std::execution::unseq, vertex.begin(), vertex.end(), edge_ops::lessThan);
        vertex.resize(maxNearestNodeNeighbors);
    };
    std::for_each(std::execution::unseq, nearestNodeDistances.begin(), nearestNodeDistances.end(), sortFunctor);

    return {std::move(nearestNodeDistances), std::move(blockJoinHints)};
}


template<typename DistType>
void StitchBlocks(const Graph<BlockNumber_t, DistType>& nearestNodeDistances,
                  const InitialJoinHints<DistType>& stitchHints,
                  std::span<BlockUpdateContext<DistType>> blockUpdateContexts,
                  const size_t graphFragment,
                  const erased_binary_binder<DistType>& bound_blocks,
                  cache_state<DistType>& result_cache){



    //OffsetSpan distView = nearestNodeDistances.GetOffsetView(blockIdxOffset);

    std::unordered_set<comparison_key<BlockNumber_t>> initBlockJoinQueue;
    for(size_t i = 0; const auto& vertex: nearestNodeDistances){
        for(size_t j = 0; j<vertex.size(); j+=1){
            initBlockJoinQueue.insert({static_cast<BlockNumber_t>(i), vertex[j].first});
        }
        i++;
    }

    std::vector<comparison_key<BlockNumber_t>> initBlockJoins;
    initBlockJoins.reserve(initBlockJoinQueue.size());
    for (const auto& pair: initBlockJoinQueue){
        blockUpdateContexts[pair.first].blockJoinTracker[pair.second] = true;
        blockUpdateContexts[pair.second].blockJoinTracker[pair.first] = true;
        initBlockJoins.push_back(pair);
    }
    

    std::vector<std::pair<JoinResults<DistType>, JoinResults<DistType>>> initUpdates(initBlockJoins.size());
    

    auto initBlockJoin = [&](const comparison_key<BlockNumber_t> blockNumbers) -> std::pair<JoinResults<DistType>, JoinResults<DistType>>{
        
        auto [blockNums, stitchHint] = *(stitchHints.find(blockNumbers));
        if (blockNums.first != blockNumbers.first) stitchHint = {std::get<1>(stitchHint), std::get<0>(stitchHint), std::get<2>(stitchHint)};
        auto& blockLHS = blockUpdateContexts[blockNumbers.first];
        auto& blockRHS = blockUpdateContexts[blockNumbers.second];
        JoinHints LHShint;
        LHShint[std::get<0>(stitchHint)] = {std::get<1>(stitchHint)};
        JoinHints RHShint;
        RHShint[std::get<1>(stitchHint)] = {std::get<0>(stitchHint)};

        ;

        std::pair<JoinResults<DistType>, JoinResults<DistType>> retPair;
        retPair.first = BlockwiseJoin(LHShint,
                                      blockLHS.currentGraph,
                                      blockLHS.joinPropagation,
                                      blockRHS.queryContext,
                                      caching_functor<DistType>{result_cache, bound_blocks(blockNumbers.first, blockNumbers.second)});
        
        ReverseBlockJoin(RHShint,
                         blockRHS.currentGraph,
                         blockRHS.joinPropagation,
                         blockLHS.queryContext,
                         result_cache,
                         bound_blocks(blockNumbers.second, blockNumbers.first));
        
        for(size_t i = 0; auto& vertex: result_cache.results){
            EraseRemove(vertex, blockRHS.currentGraph[i].PushThreshold());
        }
        
        for(size_t i = 0; const auto& vertex: result_cache.results){
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
        comparison_key<BlockNumber_t> blocks = initBlockJoins[i];
        std::pair<JoinResults<DistType>, JoinResults<DistType>>& updates = initUpdates[i];
        for (auto& result: updates.first){
            initGraphUpdates += ConsumeVertex(blockUpdateContexts[blocks.first].currentGraph[result.first], result.second, graphFragment, blocks.second);
        }
        for (auto& result: updates.second){
            initGraphUpdates += ConsumeVertex(blockUpdateContexts[blocks.second].currentGraph[result.first], result.second, graphFragment, blocks.first);
        }

    }

    
    

    
    //Initial filling of comparison targets.
    std::vector<ComparisonMap> queueMaps;
    queueMaps.reserve(blockUpdateContexts.size());
    for (size_t i = 0; i<blockUpdateContexts.size(); i+=1){
        queueMaps.push_back(InitializeComparisonQueues<DistType>(blockUpdateContexts[i].currentGraph, i));
    }
    
    //std::vector<JoinMap<size_t, size_t>> joinHints;

    for(size_t i = 0; i<blockUpdateContexts.size(); i+=1){
        ComparisonMap& comparisonMap = queueMaps[i];
        
        blockUpdateContexts[i].joinsToDo = InitializeJoinMap<DistType>(blockUpdateContexts, comparisonMap, blockUpdateContexts[i].blockJoinTracker);
    }
}

template<typename DistType, typename COMExtent>
ann::dynamic_array<BlockUpdateContext<DistType>> BuildGraph(const MetaGraph<COMExtent>& metaGraph,
                                                           erased_binary_binder<DistType> dispatch,
                                                           const hyper_parameters& hyperParams,
                                                           erased_unary_binder<COMExtent> comFunctor){


    std::vector<Graph<DataIndex_t, float>> blockGraphs = InitializeBlockGraphs<float>(metaGraph.size(), metaGraph.weights, hyperParams.indexParams.blockGraphNeighbors, dispatch);

    std::span<Graph<DataIndex_t, float>> blockGraphView = {blockGraphs.begin(), blockGraphs.size()};

    Graph<DataIndex_t, float> queryHints = old_GenerateQueryHints<float, float>(blockGraphView, hyperParams.indexParams.blockGraphNeighbors, comFunctor);


    ann::dynamic_array<BlockUpdateContext<DistType>> blockUpdateContexts = InitializeBlockContexts<DistType>(blockGraphs,
                                                                                         queryHints,
                                                                                         hyperParams.indexParams.queryDepth);
    
    std::span<BlockUpdateContext<DistType>> blockSpan{blockUpdateContexts};
    std::span<const BlockUpdateContext<DistType>> constBlockSpan(blockUpdateContexts.data(), blockUpdateContexts.size());
    cache_state<float> cache(hyperParams.splitParams.maxTreeSize, hyperParams.indexParams.blockGraphNeighbors);

    auto [nearestNodeDistances, stitchHints] = NearestNodeDistances(constBlockSpan, metaGraph, hyperParams.indexParams.nearestNodeNeighbors, dispatch);
    StitchBlocks(nearestNodeDistances, stitchHints, blockSpan, metaGraph.FragmentNumber(), dispatch, cache);
    
    
    //int iteration(1);
    int graphUpdates(1);
    while(graphUpdates>0){
        graphUpdates = 0;
        for(size_t i = 0; i<blockSpan.size(); i+=1){
            for (auto& joinList: blockSpan[i].joinsToDo){
                graphUpdates += UpdateBlocks(blockSpan[i], blockSpan[joinList.first], dispatch, cache);
                blockSpan[joinList.first].joinsToDo.erase(i);
            }
        }
        for (auto& context: blockSpan){
            context.SetNextJoins();
        }

    }

    return blockUpdateContexts;
}

template<typename DistType>
ann::dynamic_array<BlockUpdateContext<DistType>> BuildGraphRedux(const ann::dynamic_array<candidate_set>& candidates,
                                                           erased_binary_binder<DistType> dispatch,
                                                           const hyper_parameters& hyperParams){

    ann::dynamic_array<std::size_t> sizes{ candidates | std::views::transform([](const auto& array){return array.size();})};

    std::vector<Graph<DataIndex_t, float>> blockGraphs = InitializeBlockGraphs<float>(candidates.size(), sizes, hyperParams.indexParams.blockGraphNeighbors, dispatch);

    std::span<Graph<DataIndex_t, float>> blockGraphView = {blockGraphs.begin(), blockGraphs.size()};

    Graph<DataIndex_t, float> queryHints = GenerateQueryHints(blockGraphView, hyperParams.indexParams.blockGraphNeighbors);


    ann::dynamic_array<BlockUpdateContext<DistType>> blockUpdateContexts = InitializeBlockContexts<DistType>(blockGraphs,
                                                                                         queryHints,
                                                                                         hyperParams.indexParams.queryDepth);
    for(std::size_t i = 0; i<candidates.size(); ++i){
        for (std::size_t j = 0; j<candidates[i].size(); ++j){
            for(const auto candidate : candidates[i][j]){
                add_candidate(blockUpdateContexts[i].joinsToDo, j, candidate);
            }
        }
    }

    std::span<BlockUpdateContext<DistType>> blockSpan{blockUpdateContexts};
    //std::span<const BlockUpdateContext<DistType>> constBlockSpan(blockUpdateContexts.data(), blockUpdateContexts.size());
    cache_state<float> cache(hyperParams.splitParams.maxTreeSize, hyperParams.indexParams.blockGraphNeighbors);

     
    
    //int iteration(1);
    int graphUpdates(1);
    while(graphUpdates>0){
        graphUpdates = 0;
        for(size_t i = 0; i<blockSpan.size(); i+=1){
            for (auto& joinList: blockSpan[i].joinsToDo){
                graphUpdates += UpdateBlocks(blockSpan[i], blockSpan[joinList.first], dispatch, cache);
                blockSpan[joinList.first].joinsToDo.erase(i);
            }
        }
        for (auto& context: blockSpan){
            context.SetNextJoins();
        }
        std::cout << graphUpdates <<std::endl;
    }

    return blockUpdateContexts;
}


}

#endif