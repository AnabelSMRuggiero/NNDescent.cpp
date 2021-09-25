/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#include <bit>
#include <memory>
#include <memory_resource>
#include <execution>
#include <functional>
#include <array>
#include <iostream>
#include <ranges>
#include <type_traits>
#include <algorithm>

#include "Utilities/Type.hpp"
#include "Utilities/Data.hpp"
#include "Utilities/Metrics/Euclidean.hpp"

#include "Parallelization/ThreadPool.hpp"

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"

#include "NND/Type.hpp"
#include "NND/Parallel-Algorithm/FreeFunctions.hpp"

using namespace nnd;

template<typename COMExtent>
struct FragmentGraph{
    std::vector<size_t> weights;
    DataBlock<COMExtent> points;
    Graph<GraphFragment_t, COMExtent> verticies;
    QueryContext<GraphFragment_t, COMExtent> queryContext;

    size_t size() const{
        return points.numEntries;
    }

};

template<typename COMExtent>
size_t WeightedEuclideanCOM(const MetaGraph<COMExtent>& metaGraph, AlignedSpan<COMExtent> writeLocation){
    const DataBlock<COMExtent>& dataBlock = metaGraph.points;
    const std::vector<size_t>& weights = metaGraph.weights;
    size_t totalWeight = 0;
    for (size_t i = 0; i<dataBlock.size(); i += 1){
        totalWeight += weights[i];
        for(size_t j = 0; j<dataBlock[i].size(); j += 1){
            writeLocation[j] += dataBlock[i][j] * weights[i];
        }
    }

    for(auto& extent: writeLocation) extent /= totalWeight;

    return totalWeight;
}

template<typename COMExtent, typename MetricSet, typename COMFunctor>
FragmentGraph<COMExtent> BuildFragmentGraph(std::span<const MetaGraph<COMExtent>>& metaGraphs, const size_t neighbors, const size_t queryDepth, MetricSet metricSet, COMFunctor COMCalculator){
    std::vector<size_t> weights(0);
    weights.reserve(metaGraphs.size());
    DataBlock<COMExtent> points(metaGraphs.size(), metaGraphs[0].points.entryLength, 0);
    Graph<GraphFragment_t, COMExtent> verticies(metaGraphs.size(), neighbors);
    //SinglePointFunctor<COMExtent> functor(DataComDistance<DataEntry, COMExtent, MetricPair>(*this, dataBlocks, metricFunctor));
    weights.reserve(metaGraphs.size());
    for (size_t i = 0; const auto& graph: metaGraphs){
        //(graph.size());
        weights.push_back(COMCalculator(graph, points[i]));
        i++;
    }

    AlignedArray<COMExtent> centerOfMass(points.entryLength);
    COMCalculator(points, {centerOfMass.GetAlignedPtr(0), centerOfMass.size()});

    BruteForceGraph<COMExtent>(verticies, neighbors, points, metricSet.dataToData);
    for(auto& vertex: verticies){
        std::sort(vertex.begin(), vertex.end(), NeighborDistanceComparison<size_t, COMExtent>);
    }

    auto neighborFunctor = [&](size_t, size_t pointIndex){
        return metricSet.comToCom({centerOfMass.GetAlignedPtr(0), centerOfMass.size()}, points[pointIndex]);
    };
    GraphVertex<GraphFragment_t, COMExtent> queryHint = QueryCOMNeighbors<COMExtent, COMExtent, GraphFragment_t>(0, verticies, neighbors, neighborFunctor);

    QueryContext<GraphFragment_t,COMExtent> queryContext(verticies,
                                         std::move(queryHint),
                                         queryDepth,
                                         0,
                                         GraphFragment_t{std::numeric_limits<GraphFragment_t>::max()},
                                         points.size());

    return FragmentGraph<COMExtent>{weights, std::move(points), verticies, std::move(queryContext)};
}

template<typename COMExtent>
using FragmentStitchHint = std::tuple<BlockNumber_t, BlockNumber_t, COMExtent>;
/*
    using COMExtent = float;
    auto nnFunctor = [&](const size_t lhsIndex, const size_t rhsIndex)->auto{
        return metricSet.comToCom(metaGraphA.points[lhsIndex], metaGraphB.points[rhsIndex]);
    };
    std::tuple<BlockNumber_t, BlockNumber_t, COMExtent> nnDist = metaGraphA.queryContext.NearestNodes(metaGraphB.queryContext, nnFunctor);

    for (auto& block: graphA){
        block.blockJoinTracker = NodeTracker(graphB.size());
    }
    for (auto& block: graphB){
        block.blockJoinTracker = NodeTracker(graphA.size());
    }

    std::vector<std::pair<BlockNumber_t, BlockNumber_t>> joinQueue{{std::get<0>(nnDist), std::get<1>(nnDist)}};
    
    for (size_t i = 0; i<metaGraphA.verticies[0].size(); i++){
        joinQueue.push_back({metaGraphA.verticies[joinQueue.front().first][i].first,
                             metaGraphB.verticies[joinQueue.front().second][i].first});
    }
*/

std::vector<FragmentStitchHint<float>> GetFragmentStitchHints(const MetaGraph<float>& lhsFragment, const MetaGraph<float>& rhsFragment, const ErasedMetricPair<float>& functor){
    using COMExtent = float;

    std::vector<FragmentStitchHint<COMExtent>> cachedCalcs;
    cachedCalcs.reserve(30);
    auto hintCacher = [&](const size_t lhsIndex, const size_t rhsIndex)->COMExtent{
        COMExtent retVal = functor(lhsIndex, rhsIndex);
        cachedCalcs.push_back({lhsIndex, rhsIndex, retVal});
        return retVal;
    };

    lhsFragment.queryContext.NearestNodes(rhsFragment.queryContext, functor);

    std::sort(cachedCalcs.begin(), cachedCalcs.end(), [](const FragmentStitchHint<COMExtent>& a, const FragmentStitchHint<COMExtent>& b){
        return std::get<2>(a) < std::get<2>(b);
    });

    std::unordered_set<BlockNumber_t> lhsBlocks;
    std::unordered_set<BlockNumber_t> rhsBlocks;

    std::vector<FragmentStitchHint<COMExtent>> retHints;
    for (const auto& hint: cachedCalcs){
        if (!lhsBlocks.contains(std::get<0>(hint)) && !rhsBlocks.contains(std::get<1>(hint))){
            retHints.push_back(hint);
            lhsBlocks.insert(std::get<0>(hint));
            rhsBlocks.insert(std::get<1>(hint));
        }
    }
    return retHints;
}

template<typename DistType>
size_t HintlessBlockJoin(BlockUpdateContext<DistType>& lhsBlock, BlockUpdateContext<DistType>& rhsBlock, CachingFunctor<DistType>& cacher){
    cacher.SetBlocks(lhsBlock.queryContext.blockNumber, rhsBlock.queryContext.blockNumber);
    std::tuple<DataIndex_t, DataIndex_t, DistType> stitchHint = lhsBlock.queryContext.NearestNodes(rhsBlock.queryContext, cacher.metricFunctor);
    lhsBlock.joinsToDo[rhsBlock.queryContext.blockNumber].insert({std::get<0>(stitchHint), {std::get<1>(stitchHint)}});
    rhsBlock.joinsToDo[lhsBlock.queryContext.blockNumber].insert({std::get<1>(stitchHint), {std::get<0>(stitchHint)}});
    return UpdateBlocks<true>(lhsBlock, rhsBlock, cacher);
}


template<typename DistType, typename COMExtent>
size_t GraphwiseJoin(//MetaGraph<COMExtent>& metaGraphA,
                   std::span<BlockUpdateContext<DistType>> graphA,
                   //MetaGraph<COMExtent>& metaGraphB,
                   std::span<BlockUpdateContext<DistType>> graphB,
                   CachingFunctor<DistType>& cacher,
                   std::vector<FragmentStitchHint<COMExtent>> stitchHints){
    

    std::unordered_set<BlockNumber_t> lhsBlocks;
    std::unordered_set<BlockNumber_t> rhsBlocks;

    size_t totalUpdates{0};

    for (auto& [lhsIndex, rhsIndex, ignore]: stitchHints){
        size_t joinUpdates = HintlessBlockJoin(graphA[lhsIndex], graphB[rhsIndex], cacher);
        if (joinUpdates > 0){
            lhsBlocks.insert(lhsIndex);
            rhsBlocks.insert(rhsIndex);
            totalUpdates += joinUpdates;
        }
    }


    for (auto& lhsIndex: lhsBlocks){
        graphA[lhsIndex].SetNextJoins();
        for(auto& [rhsIndex, ignore]: graphA[lhsIndex].joinsToDo){
            totalUpdates += UpdateBlocks<true>(graphA[lhsIndex], graphB[rhsIndex], cacher);
            graphB[rhsIndex].joinsToDo.erase(lhsIndex);
        }
        //graphA[lhsIndex].SetNextJoins();
    }

    for (auto& rhsIndex: rhsBlocks){
        graphB[rhsIndex].SetNextJoins();
        for(auto& [lhsIndex, ignore]: graphB[rhsIndex].joinsToDo){
            totalUpdates += UpdateBlocks<true>(graphA[lhsIndex], graphB[rhsIndex], cacher);
            graphA[lhsIndex].joinsToDo.erase(rhsIndex);
        }
    }

    size_t graphUpdates = 1;
    while(graphUpdates>0){
        graphUpdates = 0;
        for(size_t i = 0; i<graphA.size(); i+=1){
            for(auto& [rhsIndex, ignore]: graphA[i].joinsToDo){
                graphUpdates += UpdateBlocks<true>(graphA[i], graphB[rhsIndex], cacher);
                graphB[rhsIndex].joinsToDo.erase(i);
            }
        }

        for(size_t i = 0; i<graphB.size(); i+=1){
            for(auto& [lhsIndex, ignore]: graphB[i].joinsToDo){
                graphUpdates += UpdateBlocks<true>(graphA[lhsIndex], graphB[i], cacher);
                graphA[lhsIndex].joinsToDo.erase(i);
            }
        }
        totalUpdates += graphUpdates;
        for (auto& context: graphA){
            context.SetNextJoins();
        }
        for (auto& context: graphB){
            context.SetNextJoins();
        }
    }

    return totalUpdates;
}

using BlockJoinSet = std::unordered_set<std::pair<GraphFragment_t, BlockNumber_t>, IntegralPairHasher<GraphFragment_t, BlockNumber_t>>;

template<typename DistType>
void QueueOOFJoins(const Graph<BlockIndecies, DistType>& currentGraphState,
                   const NodeTracker& fragmentsJoined,
                   const size_t targetFragment,
                   const size_t targetBlock,
                   const Graph<BlockIndecies, DistType>& targetGraphState,
                   BlockJoinSet& joinTargets){
    //std::unordered_set<std::pair<FragmentNumber_t, BlockNumber_t>> joinTargets;
    std::vector<BlockIndecies> neighborsToCheck;
    for (size_t i = 0; const auto& vertex: currentGraphState){
        
        for (const auto neighbor: vertex){
            if (neighbor.first.blockNumber == targetBlock && neighbor.first.graphFragment == targetFragment) neighborsToCheck.push_back(neighbor.first);
        }
        for (const auto& neighbor: neighborsToCheck){
            for (const auto& targetVertexNeighbor: targetGraphState[neighbor.dataIndex]){
                GraphFragment_t neighborFragment = targetVertexNeighbor.first.graphFragment;
                BlockNumber_t neighborBlock = targetVertexNeighbor.first.blockNumber;
                if (fragmentsJoined[neighborFragment]) continue;
                joinTargets.insert({neighborFragment, neighborBlock});
            } 
        
        }
        neighborsToCheck.clear();
        i++;
    }
}

std::pair<std::vector<BlockJoinSet>, std::vector<BlockJoinSet>> InitOOFSearches(std::span<const BlockUpdateContext<float>> graphLHS,
                                                                                        const MetaGraph<float>& metaGraphLHS,
                                                                                        std::span<const NodeTracker> lhsTrackers,
                                                                                        std::span<const BlockUpdateContext<float>> graphRHS,
                                                                                        const MetaGraph<float>& metaGraphRHS,
                                                                                        std::span<const NodeTracker> rhsTrackers){
    
    std::vector<BlockJoinSet> lhsSets;
    lhsSets.reserve(graphLHS.size());

    for (size_t i = 0; const auto& graph: graphLHS){
        BlockJoinSet blockJoinTargets;
        for (size_t j = 0; j<lhsTrackers[i].size(); j+=1){
            if (!lhsTrackers[i][j]) continue;
            QueueOOFJoins(graph.currentGraph, metaGraphLHS.fragmentsJoined, metaGraphRHS.FragmentNumber(), j, graphRHS[j].currentGraph, blockJoinTargets);
        }    
        lhsSets.push_back(std::move(blockJoinTargets));
    }

    std::vector<BlockJoinSet> rhsSets;
    rhsSets.reserve(graphRHS.size());

    for (size_t i=0; const auto& graph: graphRHS){
        BlockJoinSet blockJoinTargets;
        for (size_t j = 0; j<rhsTrackers[i].size(); j+=1){
            if (!rhsTrackers[i][j]) continue;
            QueueOOFJoins(graph.currentGraph, metaGraphRHS.fragmentsJoined, metaGraphLHS.FragmentNumber(), j, graphLHS[j].currentGraph, blockJoinTargets);
        }   
        rhsSets.push_back(std::move(blockJoinTargets)); 
        i++;
    }

    return {lhsSets, rhsSets};
}



std::pair<std::vector<BlockJoinSet>, std::vector<BlockJoinSet>> FindOOFSearches(std::span<const BlockUpdateContext<float>> graphLHS,
                                                                                        const MetaGraph<float>& metaGraphLHS,
                                                                                        std::span<const BlockUpdateContext<float>> graphRHS,
                                                                                        const MetaGraph<float>& metaGraphRHS){
    
    std::vector<BlockJoinSet> lhsSets;
    lhsSets.reserve(graphLHS.size());

    for (const auto& graph: graphLHS){
        BlockJoinSet blockJoinTargets;
        for (size_t j = 0; j<graph.blockJoinTracker.size(); j+=1){
            if (!graph.blockJoinTracker[j]) continue;
            QueueOOFJoins(graph.currentGraph, metaGraphLHS.fragmentsJoined, metaGraphRHS.FragmentNumber(), j, graphRHS[j].currentGraph, blockJoinTargets);
        }    
        lhsSets.push_back(std::move(blockJoinTargets));
    }

    std::vector<BlockJoinSet> rhsSets;
    rhsSets.reserve(graphRHS.size());

    for (const auto& graph: graphRHS){
        BlockJoinSet blockJoinTargets;
        for (size_t j = 0; j<graph.blockJoinTracker.size(); j+=1){
            if (!graph.blockJoinTracker[j]) continue;
            QueueOOFJoins(graph.currentGraph, metaGraphRHS.fragmentsJoined, metaGraphLHS.FragmentNumber(), j, graphLHS[j].currentGraph, blockJoinTargets);
        }   
        rhsSets.push_back(std::move(blockJoinTargets)); 

    }

    return {lhsSets, rhsSets};
}


void StichFragments(std::vector<MetaGraph<float>>& metaGraphs, std::span<std::unique_ptr<BlockUpdateContext<float>[]>> fragments){
    FragmentGraph<float> topGraph = BuildFragmentGraph(std::span<const MetaGraph<float>>(metaGraphs), 10, 4, EuclideanMetricSet{}, WeightedEuclideanCOM);
/*
    {
        //reset blockJoinTrackers
        GetFragmentStitchHints
        GraphwiseJoin
        //collect old trackers
    }
    //InitOOFSearches
    */
}






int main(int argc, char *argv[]){

    //SplittingHeurisitcs firstSplitParams= {125'000, 75'000, 175'000, 0.4f};

    SplittingHeurisitcs firstSplitParams= {60'000, 36'000, 84'000, 0.4f};

    SplittingHeurisitcs splitParams= {205, 123, 287, 0.0f};

    IndexParameters indexParams{12, 40, 35, 6};

    SearchParameters searchParams{10, 6, 5};

    HyperParameterValues parameters{splitParams, indexParams, searchParams};
    static const std::endian dataEndianness = std::endian::native;
    
    std::string trainDataFilePath("./TestData/SIFT-Train.bin");
    DataSet<float> trainData(trainDataFilePath, 128, 1'000'000);

    /*
    std::string testDataFilePath("./TestData/SIFT-Test.bin");
    std::string testNeighborsFilePath("./TestData/SIFT-Neighbors.bin");
    DataSet<float> testData(testDataFilePath, 128, 10'000);
    DataSet<uint32_t> testNeighbors(testNeighborsFilePath, 100, 10'000);
    */
    auto [forest, splittingVectors] = BuildRPForest<EuclidianScheme<float, AlignedArray<float>>>(std::execution::seq, trainData, firstSplitParams);

    //
    std::vector<DynamicArray<size_t>> subSections;
    auto indexArrayMaker = [&subSections](const size_t, std::span<const size_t> indecies){
        DynamicArray<size_t> subSection(indecies.size());
        std::copy(indecies.begin(), indecies.end(), subSection.get());
        subSections.push_back(std::move(subSection));
    };

    CrawlTerminalLeaves(forest, indexArrayMaker);

    std::vector<RandomProjectionForest> forests;
    std::vector<std::remove_reference_t<decltype(splittingVectors)>> splitters;
    for (auto& section: subSections){
        auto [subforest, subSplitters] = BuildRPForest<EuclidianScheme<float, AlignedArray<float>>>(std::execution::seq, trainData, std::move(section), splitParams);

        forests.push_back(std::move(subforest));
        splitters.push_back(std::move(subSplitters));

    }
    using BlockSet = std::vector<DataBlock<float>>;
    std::vector<BlockSet> blockSets;
    
    for (auto& subForest: forests){

        auto blockContructor = [&blockSets, &trainData, blockNum = 0ul](size_t, std::span<const size_t> dataPoints)mutable->auto{ 
        blockSets.back().emplace_back(trainData, dataPoints, trainData.SampleLength(), blockNum++);
        };
        blockSets.emplace_back();
        CrawlTerminalLeaves(subForest, blockContructor);
        //blockContructor.blockNum = 0;
    }

    std::vector<MetaGraph<float>> metaGraphs;
    metaGraphs.reserve(blockSets.size());

    std::unique_ptr<std::unique_ptr<BlockUpdateContext<float>[]>[]> graphs = std::make_unique<std::unique_ptr<BlockUpdateContext<float>[]>[]>(blockSets.size());
    size_t i = 0;
    for(auto& blockSet: blockSets){
        
        //MetaGraph<float> metaGraph = BuildMetaGraphFragment<float>(blockSet, parameters.indexParams, i, EuclideanMetricSet(), EuclideanCOM<float, float>);
        metaGraphs.push_back(BuildMetaGraphFragment<float>(blockSet, parameters.indexParams, i, EuclideanMetricSet(), EuclideanCOM<float, float>));
        DataComDistance<float, float, EuclideanMetricPair> comFunctor(metaGraphs[i], blockSet);
        
        std::unique_ptr<BlockUpdateContext<float>[]> blockContextArr;
        std::span<BlockUpdateContext<float>> blockUpdateContexts;

        MetricFunctor<float, EuclideanMetricPair> euclideanFunctor(blockSet);
        DispatchFunctor<float> testDispatch(euclideanFunctor);
        
        ThreadPool<ThreadFunctors<float, float>> pool(12, euclideanFunctor, comFunctor, splitParams.maxTreeSize, indexParams.blockGraphNeighbors);
        pool.StartThreads();
        blockContextArr = BuildGraph(metaGraphs[i], parameters, pool);
        //blockUpdateContexts = {blockContextArr.get(), blockSet.size()};
        pool.StopThreads();

        graphs[i] = (std::move(blockContextArr));

        i++;
    }
    //CrawlTerminalLeaves()
    

    return 0;
}