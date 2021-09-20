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

#include "ParallelizationObjects.hpp"
#include "NearestNodesTask.hpp"
#include "InitJoinTask.hpp"
#include "GraphUpdateTask.hpp"
#include "GraphComparisonTask.hpp"


namespace nnd {


template<typename DistType, typename COMExtent>
BlocksAndState<DistType> InitializeBlockContexts(const size_t numBlocks,
                                                          const std::vector<size_t>& blockSizes,
                                                          const HyperParameterValues& params,
                                                          ThreadPool<ThreadFunctors<DistType, COMExtent>>& threadPool,
                                                          const size_t startBlock){
    
    std::unique_ptr<BlockUpdateContext<DistType>[]> blockArr = std::make_unique<BlockUpdateContext<DistType>[]>(blockSizes.size());
    std::unique_ptr<std::atomic<bool>[]> blockStates = std::make_unique<std::atomic<bool>[]>(blockSizes.size());
    OffsetSpan<BlockUpdateContext<DistType>> blockView(blockArr.get(), blockSizes.size(), startBlock);
    OffsetSpan<std::atomic<bool>> stateView(blockStates.get(), blockSizes.size(), startBlock);
    //lambda that generates a function that executes the relevant work
    auto taskGenerator = [&](size_t blockNum, size_t blockSize)-> auto{

        auto task = [=, blockLocation = &(blockView[blockNum]), blockState = &(stateView[blockNum])](ThreadFunctors<DistType, COMExtent>& functors) mutable {
            functors.dispatchFunctor.SetBlocks(blockNum, blockNum);
            functors.comDistFunctor.SetBlock(blockNum);
            Graph<size_t, DistType> blockGraph = BruteForceBlock(params.indexParams.blockGraphNeighbors, blockSize, functors.dispatchFunctor);
            
            GraphVertex<size_t, DistType> queryHint = QueryHintFromCOM(blockNum, blockGraph, params.indexParams.blockGraphNeighbors, functors.comDistFunctor);
            QueryContext<DistType> queryContext(blockGraph, std::move(queryHint), params.indexParams.queryDepth, blockNum, blockSizes[blockNum - startBlock]);
            blockLocation->~BlockUpdateContext<DistType>();
            new(blockLocation) BlockUpdateContext<DistType>(std::move(blockGraph), std::move(queryContext), numBlocks, startBlock);
            
            *blockState = true;
            //blockLocation = std::make_from_tuple<BlockUpdateContext<DistType>>(std::forward_as_tuple(std::move(blockGraph), std::move(queryContext), numBlocks));
        };

        return task;
    };

    
    //generate tasks, return ptrs
    for (size_t i = 0; i<numBlocks; i+=1){
        threadPool.DelegateTask(taskGenerator(i+startBlock, blockSizes[i]));
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

        //size_t blockOffset = metaGraph.GetBlockOffset();

        OffsetSpan<size_t> distancesView(distancesPerBlock.get(), metaGraph.size(), metaGraph.GetBlockOffset());

        for (size_t i = distancesView.Offset(); const auto& vertex: metaGraph.verticies){
            for (const auto& neighbor: vertex){
                if(nearestNodeDistQueue.insert({i, neighbor.first}).second){
                    distancesView[i] += 1;
                    distancesView[neighbor.first] += 1;
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
void ParallelBlockJoins(OffsetSpan<BlockUpdateContext<DistType>> blocks, OffsetSpan<std::atomic<bool>> blockStates, ThreadPool<ThreadFunctors<DistType, COMExtent>>& pool){
    
    std::atomic<size_t> doneBlocks = 0;
    auto updateGenerator = [&](const size_t lhsNum, const size_t rhsNum)->auto{
        auto updateTask = [&, lhsPtr=&(blocks[lhsNum]), rhsPtr=&(blocks[rhsNum])](ThreadFunctors<DistType, COMExtent>& functors){
            UpdateBlocks(*lhsPtr, *rhsPtr, functors.cache);
            lhsPtr->joinsToDo.erase(rhsPtr->queryContext.blockNumber);
            size_t doneInc{0};
            if(lhsPtr->joinsToDo.size() == 0){
                lhsPtr->SetNextJoins();
                if(lhsPtr->joinsToDo.size() == 0) doneInc++;
                //if(lhsPtr->newJoins.size() == 0) doneInc++;
                //else lhsPtr->SetNextJoins();
            }
            if(rhsPtr->joinsToDo.erase(lhsPtr->queryContext.blockNumber) == 1){
                if(rhsPtr->joinsToDo.size() == 0){
                    rhsPtr->SetNextJoins();
                    if(rhsPtr->joinsToDo.size() == 0) doneInc++;
                    //if(rhsPtr->newJoins.size() == 0) doneInc++;
                    //else rhsPtr->SetNextJoins();
                }   
            }
            if constexpr (debugNND) VerifySubGraphState(lhsPtr->currentGraph, lhsPtr->queryContext.blockNumber);
            if constexpr (debugNND) VerifySubGraphState(rhsPtr->currentGraph, rhsPtr->queryContext.blockNumber);
            blockStates[lhsPtr->queryContext.blockNumber] = true;
            blockStates[rhsPtr->queryContext.blockNumber] = true;
            if(doneInc>0) doneBlocks += doneInc;
        };
        return updateTask;
    };
    //For the case where we start with 0 joins to do
    for (size_t i = blocks.Offset(); i<blocks.size() + blocks.Offset(); i +=1){
        if(blocks[i].joinsToDo.size()==0) doneBlocks++;
    }
    
    mainLoop:
    while(doneBlocks<blocks.size()){
        //doneBlocks = 0;
        for(size_t i = blocks.Offset(); i<blocks.size() + blocks.Offset(); i+=1){
            bool expectTrue = true;
            bool queued = false;
            if(!blockStates[i].compare_exchange_strong(expectTrue, false)) continue;
            //expectTrue = true;
            if(blocks[i].joinsToDo.size() == 0){
                if(blocks[i].newJoins.size() == 0){
                    blockStates[i] = true;
                    //if (firstLoop) doneBlocks++;
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
                //if (firstLoop && blocks[joinList.first].joinsToDo.size() == 0 && blocks[joinList.first].newJoins.size() == 0) doneBlocks++;
                pool.DelegateTask(updateGenerator(i, joinList.first));
                break;
                
            }
            if(!queued)blockStates[i] = true;
        }
        
    }
    pool.Latch();

    for (size_t i = blocks.Offset(); i<blocks.size()+blocks.Offset(); i +=1){
        if(blocks[i].joinsToDo.size()!=0 || blocks[i].newJoins.size() !=0){
            doneBlocks--;
            if(blocks[i].joinsToDo.size()==0) blocks[i].SetNextJoins();
        }
    }
    if (doneBlocks < blocks.size()) goto mainLoop;
}

template<typename DistType, typename COMExtent>
using NNDPool = ThreadPool<ThreadFunctors<DistType, COMExtent>>;

template<typename DistType, typename COMExtent>
using GraphInitTasks = std::tuple<NNTask<DistType, COMExtent>, InitJoinTask<DistType, COMExtent>, UpdateTask<DistType, COMExtent>, ComparisonTask<DistType, COMExtent>>;

template<typename DistType, typename COMExtent>
std::unique_ptr<BlockUpdateContext<DistType>[]> BuildGraph(std::vector<size_t>&& blockSizes, const MetaGraph<COMExtent>& metaGraph, HyperParameterValues& parameters, NNDPool<DistType, COMExtent>& pool){
    //ThreadPool<ThreadFunctors<float, float>> pool(12, euclideanFunctor, comFunctor, splitParams.maxTreeSize, 10);
    /*
    std::vector<size_t> sizes;
    sizes.reserve(dataBlocks.size());
    for(const auto& block: dataBlocks){
        sizes.push_back(block.size());
    }
    */
    

    //pool.StartThreads();

    auto nnFuture = NearestNodesToDo(metaGraph, pool);

    BlocksAndState<DistType> blocks = InitializeBlockContexts(blockSizes.size(), blockSizes, parameters, pool, metaGraph.GetBlockOffset());

    auto nnToDo = nnFuture.get();

    OffsetSpan<BlockUpdateContext<DistType>> blockSpan(blocks.blocks.get(), blockSizes.size(), metaGraph.GetBlockOffset());
    OffsetSpan<std::atomic<bool>> blockState(blocks.isReady.get(), blockSizes.size(), metaGraph.GetBlockOffset());

    

    


    using BlockUpdates = std::vector<std::pair<size_t, JoinResults<size_t, DistType>>>;

    std::unique_ptr<BlockUpdates[]> updateStorage = std::make_unique<BlockUpdates[]>(blockSpan.size());
    OffsetSpan<BlockUpdates> updateSpan{updateStorage.get(), blockSpan.size(), blockSpan.Offset()};

    std::vector<bool> blocksUpdated(blockSpan.size());
    


    std::unique_ptr<std::atomic<bool>[]> blocksFinalized = std::make_unique<std::atomic<bool>[]>(blockSpan.size());

    OffsetSpan<std::atomic<bool>> finalizedView(blocksFinalized.get(), blockSpan.size(), blockSpan.Offset());

    auto nnBuilder = GenerateTaskBuilder<NNTask<DistType, COMExtent>>(std::tuple{blockSpan, blockState},
                                         std::tuple{std::move(nnToDo.second), blockSizes.size(), parameters.indexParams.nearestNodeNeighbors, blockSpan.Offset()});
    auto initJoinBuilder = GenerateTaskBuilder<InitJoinTask<DistType, COMExtent>>(std::tuple{blockSpan, blockState},
                                                           std::tuple{updateSpan});
    auto updateBuilder = GenerateTaskBuilder<UpdateTask<DistType, COMExtent>>(std::tuple{blockSpan, blockState, updateSpan},
                                                        std::tuple{blockSpan.size(), std::reference_wrapper(blocksUpdated), blockSpan.Offset()});
    auto comparisonBuilder = GenerateTaskBuilder<ComparisonTask<DistType, COMExtent>>(std::tuple{blockSpan, std::reference_wrapper(blocksUpdated), finalizedView});

    GraphInitTasks<DistType, COMExtent> tasks(std::move(nnBuilder), std::move(initJoinBuilder), std::move(updateBuilder), std::move(comparisonBuilder));

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

    

    ParallelBlockJoins(blockSpan, finalizedView, pool);

    //pool.Latch();
    
    return std::move(blocks.blocks);
}

}

#endif