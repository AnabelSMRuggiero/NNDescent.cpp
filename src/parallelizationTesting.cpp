/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#include <functional>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <bit>
#include <future>
#include <utility>
#include <memory>
#include <vector>
#include <list>
#include <cassert>
#include <type_traits>
#include <iostream>
#include <optional>
#include <span>
#include <tuple>


#include "Parallelization/AtomicUniquePtr.hpp"
#include "Parallelization/ThreadPool.hpp"
#include "Parallelization/TaskQueuer.hpp"

#include "Utilities/Type.hpp"
#include "Utilities/DataSerialization.hpp"
#include "Utilities/DataDeserialization.hpp"
#include "Utilities/Data.hpp"
#include "Utilities/Metrics/SpaceMetrics.hpp"
#include "Utilities/Metrics/FunctorErasure.hpp"

#include "NND/GraphStructures.hpp"
#include "NND/BlockwiseAlgorithm.hpp"
#include "NND/GraphInitialization.hpp"

#include "NND/Parallel-Algorithm/FreeFunctions.hpp"
#include "NND/Parallel-Algorithm/NearestNodesTask.hpp"
#include "NND/Parallel-Algorithm/InitJoinTask.hpp"
#include "NND/Parallel-Algorithm/GraphUpdateTask.hpp"
#include "NND/Parallel-Algorithm/GraphComparisonTask.hpp"

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"


using namespace nnd;


template<typename TaskResult>
using TaskStates = std::unique_ptr<std::pair<std::promise<TaskResult>, std::future<TaskResult>>[]>;




template<typename DistType>
struct InvertedComparisons{


    std::unique_ptr<std::vector<BlockIndecies>[]> comparisonsQueued;

    const size_t mySize;

    //std::span<const BlockUpdateContext<DistType>> blocks;

    std::vector<BlockIndecies>& operator[](const size_t index){
        return comparisonsQueued[index];
    }

    const std::vector<BlockIndecies>& operator[](const size_t index) const {
        return comparisonsQueued[index];
    }



};
/*
template<typename DistType>
std::unique_ptr<InvertedComparisons<DistType>[]> MakeInvertedComparisonQueues(std::span<const BlockUpdateContext<DistType>> blocks){

    std::unique_ptr retPtr = std::make_unique<InvertedComparisons<DistType>[]>(blocks.size());
    
    for (size_t i = 0; i<blocks.size(); i+=1){
        retptr[i] = {std::make_unique<std::vector<BlockIndecies>[]>(blocks[i].joinPropagation.size()), blocks[i].joinPropagation.size()};
    }

    return retPtr;
}

template<typename DistType>
void InvertComparisonMap(std::span<InvertedComparisons<DistType>> invertedQueues, const size_t sourceBlock, ComparisonMap<size_t, size_t>&& mapToInvert){

    for(const auto& [targetBlock, comparisons]: mapToInvert){
        for(const auto& [sourceIndex, targetIndex]: comparisons){
            invertedQueues[targetBlock][targetIndex].push_back({sourceBlock, sourceIndex})
        }
    }
};
*/

template<typename DistType>
struct DelegatedComparisons{

    std::shared_ptr<std::vector<BlockIndecies>[]> comparisonsQueued;

    std::span<const BlockUpdateContext<DistType>> blocks;

    std::vector<BlockIndecies>& operator[](const size_t index){
        return comparisonsQueued[index];
    }

    const std::vector<BlockIndecies>& operator[](const size_t index) const {
        return comparisonsQueued[index];
    }

};





template<typename DistType>
DelegatedComparisons<DistType> DelegateQueues(InvertedComparisons<DistType>& comparisonsToMove, std::span<const BlockUpdateContext<DistType>> blocks){
    DelegatedComparisons<DistType> retComparisons = {std::move(comparisonsToMove.comparisonsQueued), blocks};
    comparisonsToMove.comparisonsQueued = std::make_unique<std::vector<BlockIndecies>[]>(comparisonsToMove.mySize);
    return retComparisons;
}


using JoinsToDo = std::pair<size_t, std::unordered_map<size_t, std::vector<size_t>>>;


int main(){

    static const std::endian dataEndianness = std::endian::big;

    std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");
    DataSet<AlignedArray<float>> mnistFashionTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<AlignedArray<float>,dataEndianness>);


    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), mnistFashionTrain.numberOfSamples - 1);
    StlRngFunctor<std::mt19937_64, std::uniform_int_distribution, size_t> rngFunctor(std::move(rngEngine), std::move(rngDist));

    EuclidianTrain<AlignedArray<float>, AlignedArray<float>> splittingScheme(mnistFashionTrain);
    TrainingSplittingScheme splitterFunc(splittingScheme);
    
    IndexParamters indexParams{5, 10, 3, 2};

    size_t numBlockGraphNeighbors = 5;
    size_t numCOMNeighbors = 10;
    size_t maxNearestNodes = 3;
    size_t queryDepth = 2;

    SearchParameters searchParams{10, 10, 10};
    size_t numberSearchNeighbors = 10;
    size_t searchQueryDepth = 10;
    size_t maxNewSearches = 10;

    SplittingHeurisitcs splitParams= {16, 140, 60, 180};

    HyperParameterValues parameters{splitParams, indexParams, searchParams};

    RandomProjectionForest rpTreesTrain(size_t(mnistFashionTrain.numberOfSamples), rngFunctor, splitterFunc, splitParams);



    auto [indexMappings, dataBlocks] = PartitionData<AlignedArray<float>>(rpTreesTrain, mnistFashionTrain);

    
    MetaGraph<float> metaGraph(dataBlocks, parameters.indexParams.COMNeighbors, EuclideanMetricPair());

    MetricFunctor<AlignedArray<float>, EuclideanMetricPair> euclideanFunctor(dataBlocks);
    DataComDistance<AlignedArray<float>, float, EuclideanMetricPair> comFunctor(metaGraph, dataBlocks);

    //Begin body of Parallel index build
    ThreadPool<ThreadFunctors<float, float>> pool(12, euclideanFunctor, comFunctor, splitParams.maxTreeSize, 10);

    std::vector<size_t> sizes;
    sizes.reserve(dataBlocks.size());
    for(const auto& block: dataBlocks){
        sizes.push_back(block.size());
    }
    
    

    pool.StartThreads();

    auto nnFuture = NearestNodesToDo(metaGraph, pool);

    BlocksAndState<float> blocks = InitializeBlockContexts(dataBlocks.size(), sizes, parameters, pool);

    auto nnToDo = nnFuture.get();

    std::span<BlockUpdateContext<float>> blockSpan(blocks.blocks.get(), dataBlocks.size());
    std::span<std::atomic<bool>> blockState(blocks.isReady.get(), dataBlocks.size());

    using FirstTask = TaskQueuer<NearestNodesGenerator<float, float>, NearestNodesConsumer<float>>;

    auto nnBuilder = GenerateTaskBuilder<FirstTask>(std::tuple{blockSpan, blockState},
                                         std::tuple{std::move(nnToDo.second), dataBlocks.size(), parameters.indexParams.nearestNodeNeighbors});


    using BlockUpdates = std::vector<std::pair<size_t, JoinResults<size_t, float>>>;

    std::unique_ptr<BlockUpdates[]> updateStorage = std::make_unique<BlockUpdates[]>(blockSpan.size());
    std::span<BlockUpdates> updateSpan{updateStorage.get(), blockSpan.size()};
    using SecondTask = TaskQueuer<InitJoinQueuer<float, float>, InitJoinConsumer<float>>;
    auto initJoinBuilder = GenerateTaskBuilder<SecondTask>(std::tuple{blockSpan, blockState},
                                                           std::tuple{updateSpan});
    


    using ThirdTask = TaskQueuer<GraphUpateQueuer<float, float>, GraphUpdateConsumer<float>>;

    std::vector<bool> blocksUpdated(blockSpan.size());
    auto updateBuilder = GenerateTaskBuilder<ThirdTask>(std::tuple{blockSpan, blockState, updateSpan},
                                                        std::tuple{blockSpan.size(), std::reference_wrapper(blocksUpdated)});



    std::unique_ptr<std::atomic<bool>[]> blocksFinalized = std::make_unique<std::atomic<bool>[]>(blockSpan.size());

    using FourthTask = TaskQueuer<GraphComparisonQueuer<float, float>, void>;

    auto comparisonBuilder = GenerateTaskBuilder<FourthTask>(std::tuple{blockSpan, std::reference_wrapper(blocksUpdated), std::span{blocksFinalized.get(), blockSpan.size()}});

    std::tuple<FirstTask, SecondTask, ThirdTask, FourthTask> tasks{nnBuilder, initJoinBuilder, updateBuilder, comparisonBuilder};

    std::tuple taskArgs = std::tie(nnToDo.first, std::get<0>(tasks).GetTaskArgs(), std::get<1>(tasks).GetTaskArgs(), std::get<2>(tasks).GetTaskArgs());
    
    std::integer_sequence taskIndecies = std::make_index_sequence<std::tuple_size<decltype(tasks)>{}>{};

    auto allConsumed = []<typename... Tasks, size_t... idx>(std::tuple<Tasks...>& tasks, std::index_sequence<idx...>)->bool{
        return (std::get<idx>(tasks).DoneConsuming() && ... );
    };
    
    
    
    
    

    auto taskLoopControl = [&]<typename... Tasks, size_t... idx>(std::tuple<Tasks...>& tasks, std::index_sequence<idx...>){
        (TaskLoopBody<idx>(tasks, taskArgs, pool), ...);
    };

    
    while(!allConsumed(tasks, taskIndecies)){
        taskLoopControl(tasks, taskIndecies);
    }

    pool.Latch();

    ParallelBlockJoins(blockSpan, std::move(blocksFinalized), pool);
    
    //End parallel build

    pool.StopThreads();

    

    return 0;
}