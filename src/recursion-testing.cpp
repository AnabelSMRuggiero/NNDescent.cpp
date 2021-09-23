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

#include "Utilities/Type.hpp"
#include "Utilities/Data.hpp"
#include "Utilities/Metrics/Euclidean.hpp"

#include "Parallelization/ThreadPool.hpp"

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"

#include "NND/Type.hpp"
#include "NND/Parallel-Algorithm/FreeFunctions.hpp"

using namespace nnd;

void GraphwiseJoin(MetaGraph<float> metaGraphA, std::span<BlockUpdateContext<float>> graphA, MetaGraph<float> metaGraphB, std::span<BlockUpdateContext<float>> graphB, CachingFunctor<float>& cacher, EuclideanMetricSet metricSet){
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

    for (auto& [lhsIndex, rhsIndex]: joinQueue){
        UpdateBlocks(graphA[lhsIndex], graphB[rhsIndex], cacher);
    }
}


template<typename Sinkee>
void Sink(Sinkee&& objToSink){
    Sinkee consume(std::move(objToSink));
};

int main(int argc, char *argv[]){

    SplittingHeurisitcs firstSplitParams= {125'000, 75'000, 175'000, 0.4f};

    SplittingHeurisitcs splitParams= {205, 123, 287, 0.0f};

    IndexParameters indexParams{12, 40, 35, 6};

    SearchParameters searchParams{10, 6, 5};

    HyperParameterValues parameters{splitParams, indexParams, searchParams};
    static const std::endian dataEndianness = std::endian::native;
    
    std::string trainDataFilePath("./TestData/SIFT-Train.bin");
    DataSet<AlignedArray<float>> trainData(trainDataFilePath, 128, 1'000'000, &ExtractNumericArray<AlignedArray<float>,dataEndianness>);


    std::string testDataFilePath("./TestData/SIFT-Test.bin");
    std::string testNeighborsFilePath("./TestData/SIFT-Neighbors.bin");
    DataSet<AlignedArray<float>> testData(testDataFilePath, 128, 10'000, &ExtractNumericArray<AlignedArray<float>,dataEndianness>);
    DataSet<AlignedArray<uint32_t>> testNeighbors(testNeighborsFilePath, 100, 10'000, &ExtractNumericArray<AlignedArray<uint32_t>,dataEndianness>);

    auto [forest, splittingVectors] = BuildRPForest<EuclidianScheme<AlignedArray<float>, AlignedArray<float>>>(std::execution::seq, trainData, firstSplitParams);

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
        auto [subforest, subSplitters] = BuildRPForest<EuclidianScheme<AlignedArray<float>, AlignedArray<float>>>(std::execution::seq, trainData, std::move(section), splitParams);

        forests.push_back(std::move(subforest));
        splitters.push_back(std::move(subSplitters));

    }
    using BlockSet = std::vector<DataBlock<float>>;
    std::vector<BlockSet> blocksSets;
    
    for (auto& subForest: forests){

        auto blockContructor = [&blocksSets, &trainData, blockNum = 0ul](size_t, std::span<const size_t> dataPoints)mutable->auto{ 
        blocksSets.back().emplace_back(trainData, dataPoints, trainData.sampleLength, blockNum++);
        };
        blocksSets.emplace_back();
        CrawlTerminalLeaves(subForest, blockContructor);
        //blockContructor.blockNum = 0;
    }

    std::unique_ptr<std::unique_ptr<BlockUpdateContext<float>[]>[]> graphs = std::make_unique<std::unique_ptr<BlockUpdateContext<float>[]>[]>(blocksSets.size());
    size_t i = 0;
    for(auto& blockSet: blocksSets){
        
        MetaGraph<float> metaGraph = BuildMetaGraphFragment<float>(blockSet, parameters.indexParams, i, EuclideanMetricSet(), EuclideanCOM<float, float>);
        DataComDistance<float, float, EuclideanMetricPair> comFunctor(metaGraph, blockSet);
        
        std::unique_ptr<BlockUpdateContext<float>[]> blockContextArr;
        std::span<BlockUpdateContext<float>> blockUpdateContexts;

        MetricFunctor<float, EuclideanMetricPair> euclideanFunctor(blockSet);
        DispatchFunctor<float> testDispatch(euclideanFunctor);
        
        ThreadPool<ThreadFunctors<float, float>> pool(12, euclideanFunctor, comFunctor, splitParams.maxTreeSize, indexParams.blockGraphNeighbors);
        pool.StartThreads();
        blockContextArr = BuildGraph(metaGraph, parameters, pool);
        //blockUpdateContexts = {blockContextArr.get(), blockSet.size()};
        pool.StopThreads();

        graphs[i] = (std::move(blockContextArr));

        i++;
    }
    //CrawlTerminalLeaves()


    return 0;
}