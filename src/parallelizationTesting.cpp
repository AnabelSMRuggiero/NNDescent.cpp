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


#include "Parallelization/AtomicUniquePtr.hpp"
#include "Parallelization/ThreadPool.hpp"

#include "Utilities/Type.hpp"
#include "Utilities/DataSerialization.hpp"
#include "Utilities/DataDeserialization.hpp"
#include "Utilities/Data.hpp"
#include "Utilities/Metrics/SpaceMetrics.hpp"
#include "Utilities/Metrics/FunctorErasure.hpp"

#include "NND/GraphStructures.hpp"
#include "NND/GraphInitialization.hpp"

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"


using namespace nnd;


template<typename DistType, typename COMExtent>
struct ThreadFunctors{
    DispatchFunctor<DistType> dispatchFunctor;
    CachingFunctor<DistType> cache;
    SinglePointFunctor<COMExtent> comDistFunctor;

    ThreadFunctors() = default;

    ThreadFunctors(const ThreadFunctors&) = default;

    ThreadFunctors& operator=(const ThreadFunctors&) = default;

    template<typename DistanceFunctor, typename COMFunctor>
    ThreadFunctors(DistanceFunctor distanceFunctor, COMFunctor comFunctor, size_t maxBlockSize, size_t numNeighbors):
        dispatchFunctor(distanceFunctor),
        cache(dispatchFunctor, maxBlockSize, numNeighbors),
        comDistFunctor(comFunctor) {};

};





template<typename TaskResult>
using TaskStates = std::unique_ptr<std::pair<std::promise<TaskResult>, std::future<TaskResult>>[]>;



template<typename DistType, typename COMExtent>
AtomicPtrArr<BlockUpdateContext<DistType>> InitializeBlockContexts(const size_t numBlocks,
                                                          const std::vector<size_t>& blockSizes,
                                                          const HyperParameterValues& params,
                                                          ThreadPool<ThreadFunctors<DistType, COMExtent>>& threadPool){
    
    AtomicPtrArr<BlockUpdateContext<DistType>> retArr = std::make_unique<AtomicUniquePtr<BlockUpdateContext<DistType>>[]>(blockSizes.size());


    //lambda that generates a function that executes the relevant work
    auto taskGenerator = [&](size_t blockNum, size_t blockSize)-> auto{

        auto task = [=, &atomicPtr = retArr[blockNum]](ThreadFunctors<DistType, COMExtent>& functors) mutable {
            functors.dispatchFunctor.SetBlocks(blockNum, blockNum);
            functors.comDistFunctor.SetBlock(blockNum);
            Graph<size_t, DistType> blockGraph = BruteForceBlock(params.indexParams.blockGraphNeighbors, blockSize, functors.dispatchFunctor);
            
            GraphVertex<size_t, DistType> queryHint = QueryHintFromCOM(blockNum, blockGraph, params.indexParams.blockGraphNeighbors, functors.comDistFunctor);
            QueryContext<DistType> queryContext(blockGraph, std::move(queryHint), params.indexParams.queryDepth, blockNum, numBlocks);
            std::unique_ptr<BlockUpdateContext<DistType>> blockContextPtr = std::make_unique<BlockUpdateContext<DistType>>(blockGraph, std::move(queryContext), numBlocks);
            atomicPtr = std::move(blockContextPtr);
        };

        return std::function(std::move(task));
    };

    
    //generate tasks, return ptrs
    for (size_t i = 0; i<numBlocks; i+=1){
        threadPool.DelegateTask(taskGenerator(i, blockSizes[i]));
    }

    return retArr;
}

template<typename DistType, typename COMExtent>
std::future<std::pair<std::vector<std::optional<ComparisonKey<size_t>>>, std::unique_ptr<size_t[]>>> NearestNodesToDo(const MetaGraph<COMExtent>& metaGraph,
                                                                                                         ThreadPool<ThreadFunctors<DistType, COMExtent>>& threadPool){
    std::promise<std::pair<std::vector<std::optional<ComparisonKey<size_t>>>, std::unique_ptr<size_t[]>>> promise;
    std::future<std::pair<std::vector<std::optional<ComparisonKey<size_t>>>, std::unique_ptr<size_t[]>>> retFuture = promise.get_future();
    auto task = [&, resPromise = std::move(promise)](ThreadFunctors<DistType,COMExtent>&){
        std::unordered_set<ComparisonKey<size_t>> nearestNodeDistQueue;
        std::unique_ptr<size_t[]> distancesPerBlock = std::make_unique<size_t[]>(metaGraph.size());

        for (size_t i = 0; const auto& vertex: metaGraph.verticies){
            for (const auto& neighbor: vertex){
                if(nearestNodeDistQueue.insert({i, neighbor.first}).second){
                    distancesPerBlock[i] += 1;
                    distancesPerBlock[neighbor.first] += 1;
                }
            }
            i++;
        }
        
        std::vector<std::optional<ComparisonKey<size_t>>> distancesToCompute;
        distancesToCompute.reserve(nearestNodeDistQueue.size());
        for (const auto& pair: nearestNodeDistQueue){
            distancesToCompute.push_back(std::make_optional(pair));
        }

        promise.set_value({distancesToCompute, std::move(distancesPerBlock)});
    };

    return std::move(retFuture);
}

template<typename DistType, typename COMExtent>
struct NearestNodesGenerator{

    using BlockPtrPair = std::pair<std::unique_ptr<BlockUpdateContext<DistType>>,std::unique_ptr<BlockUpdateContext<DistType>>>;
    
    
    /*
    std::span<AtomicUniquePtr<BlockUpdateContext<DistType>>> blocks;
    std::vector<std::optional<ComparisonKey<size_t>>>& distancesToCompute;
    size_t nullCounter;
    */
    bool operator()(ThreadPool<ThreadFunctors<DistType, COMExtent>>& pool, AsyncQueue<std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>>& resultsQueue){
        auto nnDistanceTaskGenerator = [&](BlockPtrPair blockPtrs)->auto{

            auto task = [&, ptrs = std::move(blockPtrs)](ThreadFunctors<DistType, COMExtent>& functors) mutable-> std::tuple<size_t, size_t, DistType>{
                const QueryContext<DistType>& lhsQueryContext = ptrs.first.get()->queryContext;
                const QueryContext<DistType>& rhsQueryContext = ptrs.second.get()->queryContext;
                std::tuple<size_t, size_t, DistType> nnDistResult = lhsQueryContext.NearestNodes(rhsQueryContext,
                                                                                                functors.dispatchFunctor);
                /*
                nnDistanceResults[resultIndex].second = {{blockNumbers.first, std::get<0>(nnDistResult)},
                                                {blockNumbers.second, std::get<1>(nnDistResult)},
                                                std::get<2>(nnDistResult)};
                */
               resultsQueue.Put(nnDistResult);
               blocks[lhsQueryContext.blockNumber] = std::move(ptrs.first);
               blocks[rhsQueryContext.blockNumber] = std::move(ptrs.second);
            };

            return task;
        };

        for(auto& blockNums: distancesToCompute){
            if(!blockNums) continue;
            std::unique_ptr<BlockUpdateContext<DistType>> lhsPtr = blocks[*blockNums.first].GetUnique();
            if(!lhsPtr) continue;
            std::unique_ptr<BlockUpdateContext<DistType>> rhsPtr = blocks[*blockNums.second].GetUnique();
            if(!rhsPtr){
                blocks[*blockNums.first] = std::move(lhsPtr);
                continue;
            }

            //I have two valid unique_ptrs I can use
            pool.DelegateTask(nnDistanceTaskGenerator({std::move(lhsPtr), std::move(rhsPtr)}));
            blockNums = std::nullopt;
            ++nullCounter;
        }

        if(nullCounter >= distancesToCompute.size()/2) EraseNulls();

        return distancesToCompute.size()==false;
    }

    private:

    void EraseNulls(){
        std::erase(std::remove_if(distancesToCompute.begin(), distancesToCompute.end(), std::logical_not()),
                   distancesToCompute.end());

        nullCounter = 0;  
    }


    std::span<AtomicUniquePtr<BlockUpdateContext<DistType>>> blocks;
    std::vector<std::optional<ComparisonKey<size_t>>>& distancesToCompute;
    size_t nullCounter = 0;
};

template<typename DistType, typename COMExtent>
struct NearestNodesConsumer{

    NearestNodesConsumer() = default;

    NearestNodesConsumer(std::unique_ptr<size_t[]> distanceCounts, const size_t numBlocks, const size_t numNeighbors): distancesPerBlock(std::move(distanceCounts)),
                            nnNumNeighbors(numNeighbors),
                            nnGraph(numBlocks, numNeighbors),
                            blocksDone(0) {};

    bool operator()(std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>> result){
        joinHints[result.first] = result.second;

        nnGraph[result.first.first].push_back({result.first.second, std::get<2>(result.second)});
        if (distancesPerBlock[result.first.first] == nnGraph[result.first.first].size()){
            GraphVertex<size_t, DistType>& vertex = nnGraph[result.first.first];
            std::partial_sort(vertex.begin(), vertex.begin()+nnNumNeighbors, vertex.end(), NeighborDistanceComparison<size_t, DistType>);
            vertex.resize(nnNumNeighbors);

            for(const auto& neighbor: vertex){
                if(initJoinsQueued.insert({result.first.first, neighbor.first}).second) initJoinsToDo.push_back({result.first.first, neighbor.first});
            }

            ++blocksDone;
        }

        nnGraph[result.first.second].push_back({result.first.first, std::get<2>(result.second)});
        if (distancesPerBlock[result.first.second] == nnGraph[result.first.second].size()){
            GraphVertex<size_t, DistType>& vertex = nnGraph[result.first.second];
            std::partial_sort(vertex.begin(), vertex.begin()+nnNumNeighbors, vertex.end(), NeighborDistanceComparison<size_t, DistType>);
            vertex.resize(nnNumNeighbors);


            for(const auto& neighbor: vertex){
                if(initJoinsQueued.insert({result.first.second, neighbor.first}).second) initJoinsToDo.push_back({result.first.second, neighbor.first});
            }

            ++blocksDone;
        }

        return blocksDone == nnGraph.size();
    }

    bool CanFeedGenerator(){
        return initJoinsToDo.size() > 0;
    }


    private:
    
    std::unique_ptr<size_t[]> distancesPerBlock;
    const size_t nnNumNeighbors;
    Graph<size_t, DistType> nnGraph;
    
    size_t blocksDone;

    std::unordered_map<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>> joinHints;
    std::unordered_set<ComparisonKey<size_t>> initJoinsQueued;
    std::vector<std::optional<ComparisonKey<size_t>>> initJoinsToDo;


};



template<typename Generator, typename Consumer, typename TaskResult>
struct TaskQueuer{

    Generator generator;
    Consumer consumer;

    template<typename... GenArgs, typename... ConsArgs>
    TaskQueuer(std::tuple<GenArgs...>&& generatorArgs, std::tuple<ConsArgs...>&& consumerArgs): 
        generator(std::make_from_tuple<Generator>(std::forward(generatorArgs))),
        consumer(std::make_from_tuple<Consumer>(std::forward(consumerArgs))){};

    TaskQueuer(const TaskQueuer&) = delete;
    TaskQueuer(TaskQueuer&&) = delete;

    bool ConsumeResults(){
        std::list<TaskResult> newResults = incomingResults.TakeAll();
        for (const auto& entry: newResults){
            doneConsuming = consumer(entry);
        }
        return doneConsuming;
    }

    template<typename Pool>
    bool QueueTasks(Pool& pool){
        doneGenerating = generator(pool, incomingResults);
        return doneGenerating;
    }

    bool DoneGenerating(){
        return doneGenerating;
    }

    bool DoneConsuming(){
        return doneConsuming;
    }

    bool IsDone(){
        return DoneConsuming() && DoneGenerating();
    }


    private:
    bool doneGenerating;
    bool doneConsuming;
    AsyncQueue<TaskResult> incomingResults;

};
/*
template<typename DistType, typename COMExtent>
void QueueNearestNodes(std::vector<std::optional<ComparisonKey<size_t>>>& distancesToCompute, const AtomicUniquePtr<BlockUpdateContext<DistType>>* blockUpdateContexts){
    std::unique_ptr<std::optional<std::tuple<BlockIndecies, BlockIndecies, DistType>>[]> nnDistanceResults(distancesToCompute.size());
    auto nnDistanceTaskGenerator = [&](const ComparisonKey<size_t> blockNumbers, const size_t resultIndex)->auto{
        auto task = [&, resultIndex, blockNumbers, results=nnDistanceResults.get()](ThreadFunctors<DistType, COMExtent>& functors) mutable-> std::tuple<size_t, size_t, DistType>{
            const QueryContext<DistType>& lhsQueryContext = blockUpdateContexts[blockNumbers.first].get()->queryContext;
            const QueryContext<DistType>& rhsQueryContext = blockUpdateContexts[blockNumbers.second].get()->queryContext;
            std::tuple<size_t, size_t, DistType> nnDistResult = lhsQueryContext.NearestNodes(rhsQueryContext,
                                                                                             functors.dispatchFunctor);
            nnDistanceResults[resultIndex].second = {{blockNumbers.first, std::get<0>(nnDistResult)},
                                              {blockNumbers.second, std::get<1>(nnDistResult)},
                                              std::get<2>(nnDistResult)};
        };

        return task;
    };



    return std::move(nnDistanceResults);
}
*/

/*
    std::vector<Graph<size_t, DistType>> blockGraphs(0);
    blockGraphs.reserve(blockSizes.size());
    for (size_t i =0; i<numBlocks; i+=1){
        distanceFunctor.SetBlocks(i,i);
        blockGraphs.push_back(BruteForceBlock<DistType>(numNeighbors, blockSizes[i], distanceFunctor));
    }

    return blockGraphs;
*/


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

    ThreadPool<ThreadFunctors<float, float>> pool(12, euclideanFunctor, comFunctor, splitParams.maxTreeSize, 10);

    std::vector<size_t> sizes;
    sizes.reserve(dataBlocks.size());
    for(const auto& block: dataBlocks){
        sizes.push_back(block.size());
    }
    


    pool.StartThreads();

    auto nnFuture = NearestNodesToDo(metaGraph, pool);

    AtomicPtrArr<BlockUpdateContext<float>> blocks = InitializeBlockContexts(dataBlocks.size(), sizes, parameters, pool);



    //auto testRes = futures[sizes.size()-1].second.get();
    

    pool.StopThreads();

    for (size_t i = 0; i<dataBlocks.size(); i+=1){
        //if (!(futures[i])) 
        assert(blocks[i]);
        auto test = blocks[i].GetUnique();
        std::cout << test->currentGraph.size() << std::endl;
    }
    assert(blocks[0] == false);
    return 0;
}