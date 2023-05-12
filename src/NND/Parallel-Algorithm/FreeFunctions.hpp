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

#include <cstddef>
#include <functional>
#include <future>
#include <ranges>
#include <vector>

#include "../FunctorErasure.hpp"

#include "../BlockwiseAlgorithm.hpp"
#include "../MetaGraph.hpp"
#include "../Type.hpp"

#include "ann/AlignedMemory/DynamicArray.hpp"
#include "Parallelization/ThreadPool.hpp"

#include "GraphComparisonTask.hpp"
#include "GraphUpdateTask.hpp"
#include "InitJoinTask.hpp"
#include "NearestNodesTask.hpp"
#include "ParallelizationObjects.hpp"

namespace nnd {

template<typename DistType, typename Pool>
BlocksAndState<DistType> InitializeBlockContexts(
    const size_t numBlocks, const size_t graphFragment, std::span<const std::size_t> blockSizes, const hyper_parameters& params,
    Pool& threadPool) {

    ann::dynamic_array<BlockUpdateContext<DistType>> blockArr{blockSizes.size()};
    std::unique_ptr<std::atomic<bool>[]> blockStates = std::make_unique<std::atomic<bool>[]>(blockSizes.size());
    std::span<BlockUpdateContext<DistType>> blockView{blockArr};
    std::span<std::atomic<bool>> stateView(blockStates.get(), blockSizes.size());

    auto taskGenerator = [&](size_t blockNum, size_t blockSize) -> auto {

        auto task = [=, blockLocation = &(blockView[blockNum]), blockState = &(stateView[blockNum])](
                        typename Pool::state& functors) mutable {
            Graph<DataIndex_t, DistType> blockGraph =
                BruteForceBlock(params.index_params.block_graph_neighbors, blockSize, functors.dispatchFunctor(blockNum, blockNum));

            //GraphVertex<DataIndex_t, DistType> queryHint =
            //    QueryHintFromCOM(blockNum, blockGraph, params.indexParams.blockGraphNeighbors, functors.comDistFunctor(blockNum));
            GraphVertex<DataIndex_t, DistType> queryHint = RandomQueryHint<DataIndex_t, DistType>(blockSize, params.index_params.block_graph_neighbors);
            QueryContext<DataIndex_t, DistType> queryContext(
                blockGraph, std::move(queryHint), graphFragment, blockNum, blockSizes[blockNum]);
            blockLocation->~BlockUpdateContext<DistType>();
            new (blockLocation) BlockUpdateContext<DistType>(std::move(blockGraph), std::move(queryContext), numBlocks);

            *blockState = true;
            blockState->notify_all();
            // blockLocation = std::make_from_tuple<BlockUpdateContext<DistType>>(std::forward_as_tuple(std::move(blockGraph),
            // std::move(queryContext), numBlocks));
        };

        return task;
    };

    // generate tasks, return ptrs
    for (size_t i = 0; i < numBlocks; i += 1) {
        threadPool.DelegateTask(taskGenerator(i, blockSizes[i]));
    }

    return { std::move(blockArr), std::move(blockStates) };
}

using AsyncNNResults = std::pair<std::vector<std::optional<comparison_key<BlockNumber_t>>>, std::unique_ptr<size_t[]>>;

template<typename COMExtent, typename Pool>
std::future<AsyncNNResults>
NearestNodesToDo(const MetaGraph<COMExtent>& metaGraph, Pool& threadPool) {

    std::shared_ptr<std::promise<AsyncNNResults>> promise = std::make_shared<std::promise<AsyncNNResults>>();

    std::future<AsyncNNResults> retFuture = promise->get_future();

    auto task = [&, resPromise = std::move(promise)](typename Pool::state&) {
        std::unordered_set<comparison_key<BlockNumber_t>> nearestNodeDistQueue;
        std::unique_ptr<size_t[]> distancesPerBlock = std::make_unique<size_t[]>(metaGraph.points.size());

        // size_t blockOffset = metaGraph.GetBlockOffset();

        std::span<size_t> distancesView(distancesPerBlock.get(), metaGraph.size());

        for (size_t i = 0; const auto& vertex : metaGraph.verticies) {
            for (const auto& neighbor : vertex) {
                if (nearestNodeDistQueue.insert({ static_cast<BlockNumber_t>(i), neighbor.first }).second) {
                    distancesView[i] += 1;
                    distancesView[neighbor.first] += 1;
                }
            }
            i++;
        }

        std::vector<std::optional<comparison_key<BlockNumber_t>>> distancesToCompute;
        distancesToCompute.reserve(nearestNodeDistQueue.size());
        for (const auto& pair : nearestNodeDistQueue) {
            distancesToCompute.push_back(std::make_optional(pair));
        }

        resPromise->set_value({ distancesToCompute, std::move(distancesPerBlock) });
    };

    threadPool.DelegateTask(std::move(task));

    return retFuture;
}

template<typename DistType, typename Pool>
void ParallelBlockJoins( const index_parameters& index_params,
    std::span<BlockUpdateContext<DistType>> blocks, std::span<std::atomic<bool>> blockStates,
    Pool& pool) {

    std::atomic<size_t> doneBlocks = 0;
    auto updateGenerator = [&](const size_t lhsNum, const size_t rhsNum) -> auto {
        auto updateTask = [&, lhsPtr = &(blocks[lhsNum]), rhsPtr = &(blocks[rhsNum])](typename Pool::state& functors) {
            UpdateBlocks(index_params,*lhsPtr, *rhsPtr, functors.dispatchFunctor, functors.cache);
            lhsPtr->joinsToDo.erase(rhsPtr->queryContext.blockNumber);
            size_t doneInc{ 0 };
            if (lhsPtr->joinsToDo.size() == 0) {
                lhsPtr->SetNextJoins();
                if (lhsPtr->joinsToDo.size() == 0) doneInc++;
                // if(lhsPtr->newJoins.size() == 0) doneInc++;
                // else lhsPtr->SetNextJoins();
            }
            if (rhsPtr->joinsToDo.erase(lhsPtr->queryContext.blockNumber) == 1) {
                if (rhsPtr->joinsToDo.size() == 0) {
                    rhsPtr->SetNextJoins();
                    if (rhsPtr->joinsToDo.size() == 0) doneInc++;
                    // if(rhsPtr->newJoins.size() == 0) doneInc++;
                    // else rhsPtr->SetNextJoins();
                }
            }
            //if constexpr (debugNND) VerifySubGraphState(lhsPtr->currentGraph, lhsPtr->queryContext.blockNumber);
            //if constexpr (debugNND) VerifySubGraphState(rhsPtr->currentGraph, rhsPtr->queryContext.blockNumber);
            blockStates[lhsPtr->queryContext.blockNumber] = true;
            blockStates[rhsPtr->queryContext.blockNumber] = true;
            if (doneInc > 0) doneBlocks += doneInc;
        };
        return updateTask;
    };
    // For the case where we start with 0 joins to do
    for (size_t i = 0; i < blocks.size(); i += 1) {
        if (blocks[i].joinsToDo.size() == 0) doneBlocks++;
    }

mainLoop:
    while (doneBlocks < blocks.size()) {
        // doneBlocks = 0;
        for (size_t i = 0; i < blocks.size(); i += 1) {
            bool expectTrue = true;
            bool queued = false;
            if (!blockStates[i].compare_exchange_strong(expectTrue, false)) continue;
            // expectTrue = true;
            if (blocks[i].joinsToDo.size() == 0) {
                if (blocks[i].newJoins.size() == 0) {
                    blockStates[i] = true;
                    // if (firstLoop) doneBlocks++;
                    continue;
                }
                blocks[i].SetNextJoins();
            }
            for (auto& joinList : blocks[i].joinsToDo) {
                expectTrue = true;
                if (!blockStates[joinList.first].compare_exchange_strong(expectTrue, false)) {

                    continue;
                }
                queued = true;
                // if (firstLoop && blocks[joinList.first].joinsToDo.size() == 0 && blocks[joinList.first].newJoins.size() == 0)
                // doneBlocks++;
                pool.DelegateTask(updateGenerator(i, joinList.first));
                break;
            }
            if (!queued) blockStates[i] = true;
        }
    }
    pool.Latch();

    for (size_t i = 0; i < blocks.size(); i += 1) {
        if (blocks[i].joinsToDo.size() != 0 || blocks[i].newJoins.size() != 0) {
            doneBlocks--;
            if (blocks[i].joinsToDo.size() == 0) blocks[i].SetNextJoins();
        }
    }
    if (doneBlocks < blocks.size()) goto mainLoop;
}


template<typename DistType>
using NNDPool = ThreadPool<thread_functors<DistType>>;

template<typename DistType, typename COMExtent>
using old_NNDPool = ThreadPool<old_thread_functors<DistType, COMExtent>>;

template<typename DistType, typename COMExtent>
using GraphInitTasks = std::tuple<
    NNTask<DistType, COMExtent>, InitJoinTask<DistType, COMExtent>, UpdateTask<DistType, COMExtent>, ComparisonTask<DistType, COMExtent>>;

template<typename DistType, typename COMExtent>
ann::dynamic_array<BlockUpdateContext<DistType>> BuildGraph( // std::vector<size_t>&& blockSizes,
    const MetaGraph<COMExtent>& metaGraph, const hyper_parameters& parameters, old_NNDPool<DistType, COMExtent>& pool) {
    // ThreadPool<thread_functors<float, float>> pool(12, euclideanFunctor, comFunctor, splitParams.maxTreeSize, 10);
    /*
    std::vector<size_t> sizes;
    sizes.reserve(dataBlocks.size());
    for(const auto& block: dataBlocks){
        sizes.push_back(block.size());
    }
    */

    // pool.StartThreads();

    auto nnFuture = NearestNodesToDo(metaGraph, pool);

    BlocksAndState<DistType> blocks =
        InitializeBlockContexts(metaGraph.size(), metaGraph.FragmentNumber(), metaGraph.weights, parameters, pool);

    auto nnToDo = nnFuture.get();

    std::span<BlockUpdateContext<DistType>> blockSpan{blocks.blocks};
    std::span<std::atomic<bool>> blockState(blocks.isReady.get(), metaGraph.size());

    using BlockUpdates = std::vector<std::pair<BlockNumber_t, JoinResults<DistType>>>;

    std::unique_ptr<BlockUpdates[]> updateStorage = std::make_unique<BlockUpdates[]>(blockSpan.size());
    std::span<BlockUpdates> updateSpan{ updateStorage.get(), blockSpan.size() };

    std::vector<bool> blocksUpdated(blockSpan.size());

    std::unique_ptr<std::atomic<bool>[]> blocksFinalized = std::make_unique<std::atomic<bool>[]>(blockSpan.size());

    std::span<std::atomic<bool>> finalizedView(blocksFinalized.get(), blockSpan.size());

    auto nnBuilder = GenerateTaskBuilder<NNTask<DistType, COMExtent>>(
        std::tuple{ blockSpan, blockState },
        std::tuple{ std::move(nnToDo.second), metaGraph.size(), parameters.index_params.nearest_node_neighbors });
    auto initJoinBuilder =
        GenerateTaskBuilder<InitJoinTask<DistType, COMExtent>>(std::tuple{ blockSpan, blockState }, std::tuple{ updateSpan });
    auto updateBuilder = GenerateTaskBuilder<UpdateTask<DistType, COMExtent>>(
        std::tuple{ blockSpan, blockState, updateSpan }, std::tuple{ blockSpan.size(), std::reference_wrapper(blocksUpdated) });
    auto comparisonBuilder = GenerateTaskBuilder<ComparisonTask<DistType, COMExtent>>(
        std::tuple{ blockSpan, std::reference_wrapper(blocksUpdated), finalizedView });

    GraphInitTasks<DistType, COMExtent> tasks(
        std::move(nnBuilder), std::move(initJoinBuilder), std::move(updateBuilder), std::move(comparisonBuilder));

    std::tuple taskArgs =
        std::tie(nnToDo.first, std::get<0>(tasks).GetTaskArgs(), std::get<1>(tasks).GetTaskArgs(), std::get<2>(tasks).GetTaskArgs());

    std::integer_sequence taskIndecies = std::make_index_sequence<std::tuple_size<decltype(tasks)>{}>{};

    auto allConsumed = []<typename... Tasks, size_t... idx>(std::tuple<Tasks...> & tasks, std::index_sequence<idx...>)->bool {
        return (std::get<idx>(tasks).DoneConsuming() && ...);
    };

    auto taskLoopControl = [&]<typename... Tasks, size_t... idx>(std::tuple<Tasks...> & tasks, std::index_sequence<idx...>) {
        (TaskLoopBody<idx>(tasks, taskArgs, pool), ...);
    };

    while (!allConsumed(tasks, taskIndecies)) {
        taskLoopControl(tasks, taskIndecies);
    }

    pool.Latch();

    ParallelBlockJoins(blockSpan, finalizedView, pool);

    // pool.Latch();

    return std::move(blocks.blocks);
}

template<typename DistType>
ann::dynamic_array<BlockUpdateContext<DistType>> BuildGraphRedux( // std::vector<size_t>&& blockSizes,
    const ann::dynamic_array<candidate_set>& candidates, const hyper_parameters& parameters, NNDPool<DistType>& pool) {
    // ThreadPool<thread_functors<float, float>> pool(12, euclideanFunctor, comFunctor, splitParams.maxTreeSize, 10);
    /*
    std::vector<size_t> sizes;
    sizes.reserve(dataBlocks.size());
    for(const auto& block: dataBlocks){
        sizes.push_back(block.size());
    }
    */

    // pool.StartThreads();

    ann::dynamic_array<std::size_t> sizes{ candidates | std::views::transform([](const auto& array){return array.size();})};

    auto [blocks, is_ready] =
        InitializeBlockContexts<DistType>(candidates.size(), 0, sizes, parameters, pool);

    auto add_candidates = [&, block_span = std::span{blocks}, block_ready = is_ready.get()] (std::size_t idx, const auto&){
        if (block_ready[idx] != true){
            block_ready[idx].wait(false);
        }
        block_ready[idx] = false;
        for (std::size_t j = 0; j<candidates[idx].size(); ++j){
            for(const auto candidate : candidates[idx][j]){
                add_candidate(block_span[idx].joinsToDo, j, candidate);
            }
        }
        block_ready[idx] = true;
        block_ready[idx].notify_one();
    };

    for(std::size_t i = 0; i<blocks.size(); ++i){
        pool.DelegateTask(std::bind_front(add_candidates, i));
    }

    pool.Latch();

    ParallelBlockJoins(parameters.index_params, std::span{blocks}, std::span{is_ready.get(), blocks.size()}, pool);

    // pool.Latch();

    return std::move(blocks);
}

} // namespace nnd

#endif