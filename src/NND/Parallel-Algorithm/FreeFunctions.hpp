/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_PARAFREEFUNCTIONS_HPP
#define NND_PARAFREEFUNCTIONS_HPP

#include <future>
#include <vector>

#include "Utilities/Metrics/FunctorErasure.hpp"

#include "../Type.hpp"
#include "../MetaGraph.hpp"
#include "../BlockwiseAlgorithm.hpp"

#include "Parallelization/ThreadPool.hpp"


namespace nnd {

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

template<typename DistType>
struct BlocksAndState{
    std::unique_ptr<BlockUpdateContext<DistType>[]> blocks;
    std::unique_ptr<std::atomic<bool>[]> isReady;
};

template<typename DistType, typename COMExtent>
BlocksAndState<DistType> InitializeBlockContexts(const size_t numBlocks,
                                                          const std::vector<size_t>& blockSizes,
                                                          const HyperParameterValues& params,
                                                          ThreadPool<ThreadFunctors<DistType, COMExtent>>& threadPool){
    
    std::unique_ptr<BlockUpdateContext<DistType>[]> blockArr = std::make_unique<BlockUpdateContext<DistType>[]>(blockSizes.size());
    std::unique_ptr<std::atomic<bool>[]> blockStates = std::make_unique<std::atomic<bool>[]>(blockSizes.size());

    //lambda that generates a function that executes the relevant work
    auto taskGenerator = [&](size_t blockNum, size_t blockSize)-> auto{

        auto task = [=, blockLocation = &(blockArr[blockNum]), blockState = &(blockStates[blockNum])](ThreadFunctors<DistType, COMExtent>& functors) mutable {
            functors.dispatchFunctor.SetBlocks(blockNum, blockNum);
            functors.comDistFunctor.SetBlock(blockNum);
            Graph<size_t, DistType> blockGraph = BruteForceBlock(params.indexParams.blockGraphNeighbors, blockSize, functors.dispatchFunctor);
            
            GraphVertex<size_t, DistType> queryHint = QueryHintFromCOM(blockNum, blockGraph, params.indexParams.blockGraphNeighbors, functors.comDistFunctor);
            QueryContext<DistType> queryContext(blockGraph, std::move(queryHint), params.indexParams.queryDepth, blockNum, blockSizes[blockNum]);
            blockLocation->~BlockUpdateContext<DistType>();
            new(blockLocation) BlockUpdateContext<DistType>(std::move(blockGraph), std::move(queryContext), numBlocks);
            
            *blockState = true;
            //blockLocation = std::make_from_tuple<BlockUpdateContext<DistType>>(std::forward_as_tuple(std::move(blockGraph), std::move(queryContext), numBlocks));
        };

        return task;
    };

    
    //generate tasks, return ptrs
    for (size_t i = 0; i<numBlocks; i+=1){
        threadPool.DelegateTask(taskGenerator(i, blockSizes[i]));
    }

    return {std::move(blockArr), std::move(blockStates)};
}

using AsyncNNResults = std::pair<std::vector<std::optional<ComparisonKey<size_t>>>, std::unique_ptr<size_t[]>>;

template<typename DistType, typename COMExtent>
std::future<AsyncNNResults> NearestNodesToDo(const MetaGraph<COMExtent>& metaGraph,
                                                                                                         ThreadPool<ThreadFunctors<DistType, COMExtent>>& threadPool){
    
    std::shared_ptr<std::promise<AsyncNNResults>> promise = std::make_shared<std::promise<AsyncNNResults>>();
    
    std::future<AsyncNNResults> retFuture = promise->get_future();

    auto task = [&, resPromise = std::move(promise)](ThreadFunctors<DistType,COMExtent>&){
        std::unordered_set<ComparisonKey<size_t>> nearestNodeDistQueue;
        std::unique_ptr<size_t[]> distancesPerBlock = std::make_unique<size_t[]>(metaGraph.points.size());

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

        resPromise->set_value({distancesToCompute, std::move(distancesPerBlock)});
    };

    threadPool.DelegateTask(std::move(task));

    return retFuture;
}


template<typename DistType, typename COMExtent>
void ParallelBlockJoins(std::span<BlockUpdateContext<DistType>> blocks, std::unique_ptr<std::atomic<bool>[]> blockStates, ThreadPool<ThreadFunctors<DistType, COMExtent>>& pool){
    
    std::atomic<size_t> doneBlocks = 0;
    auto updateGenerator = [&](const size_t lhsNum, const size_t rhsNum)->auto{
        auto updateTask = [&, lhsPtr=&(blocks[lhsNum]), rhsPtr=&(blocks[rhsNum])](ThreadFunctors<DistType, COMExtent>& functors){
            UpdateBlocks(*lhsPtr, *rhsPtr, functors.cache);
            lhsPtr->joinsToDo.erase(rhsPtr->queryContext.blockNumber);
            size_t doneInc{0};
            if(lhsPtr->joinsToDo.size() == 0){
                if(lhsPtr->newJoins.size() == 0) doneInc++;
                else lhsPtr->SetNextJoins();
            }
            if(rhsPtr->joinsToDo.erase(lhsPtr->queryContext.blockNumber) == 1){
                if(rhsPtr->joinsToDo.size() == 0){
                    if(rhsPtr->newJoins.size() == 0) doneInc++;
                    else rhsPtr->SetNextJoins();
                }   
            } 
            blockStates[lhsPtr->queryContext.blockNumber] = true;
            blockStates[rhsPtr->queryContext.blockNumber] = true;
            if(doneInc>0) doneBlocks += doneInc;
        };
        return updateTask;
    };
    
    //Case where we start with 0 joins to do?
    while(doneBlocks<blocks.size()){
        //doneBlocks = 0;
        for(size_t i = 0; i<blocks.size(); i+=1){
            bool expectTrue = true;
            bool queued = false;
            if(!blockStates[i].compare_exchange_strong(expectTrue, false)) continue;
            //expectTrue = true;
            if(blocks[i].joinsToDo.size() == 0){
                if(blocks[i].newJoins.size() == 0){
                    blockStates[i] = true;
                    continue;
                }
                blocks[i].SetNextJoins();
            }
            for (auto& joinList: blocks[i].joinsToDo){
                expectTrue = true;
                if(!blockStates[joinList.first].compare_exchange_strong(expectTrue, false)){
                    
                    continue;
                }
                queued = true;
                pool.DelegateTask(updateGenerator(i, joinList.first));
                break;
                
            }
            if(!queued)blockStates[i] = true;
        }
    }
}

template<typename DistType, typename COMExtent>
using NNDPool = ThreadPool<ThreadFunctors<DistType, COMExtent>>;

template<typename DistType, typename COMExtent>
void BuildGraph(NNDPool<DistType, COMExtent>& pool, std::vector<size_t>&& blockSizes, const MetaGraph<COMExtent>& metaGraph, HyperParameterValues& parameters){
    //ThreadPool<ThreadFunctors<float, float>> pool(12, euclideanFunctor, comFunctor, splitParams.maxTreeSize, 10);
    /*
    std::vector<size_t> sizes;
    sizes.reserve(dataBlocks.size());
    for(const auto& block: dataBlocks){
        sizes.push_back(block.size());
    }
    */
    

    pool.StartThreads();

    auto nnFuture = NearestNodesToDo(metaGraph, pool);

    BlocksAndState<float> blocks = InitializeBlockContexts(dataBlocks.size(), blockSizes, parameters, pool);

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
}

}

#endif