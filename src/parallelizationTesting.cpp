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


#include "Parallelization/AtomicUniquePtr.hpp"
#include "Parallelization/ThreadPool.hpp"

#include "Utilities/Type.hpp"
#include "Utilities/DataSerialization.hpp"
#include "Utilities/DataDeserialization.hpp"
#include "Utilities/Data.hpp"
#include "Utilities/Metrics/SpaceMetrics.hpp"
#include "Utilities/Metrics/FunctorErasure.hpp"

#include "NND/GraphStructures.hpp"
#include "NND/BlockwiseAlgorithm.hpp"
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


template<typename Element>
void EraseNulls(std::vector<std::optional<Element>>& optVector){
    optVector.erase(std::remove_if(optVector.begin(), optVector.end(), std::logical_not()),
                             optVector.end());  
}

template<typename TaskResult>
using TaskStates = std::unique_ptr<std::pair<std::promise<TaskResult>, std::future<TaskResult>>[]>;

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
            QueryContext<DistType> queryContext(blockGraph, std::move(queryHint), params.indexParams.queryDepth, blockNum, numBlocks);
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
struct NearestNodesGenerator{

    using TaskResult = std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>;

    using BlockPtrPair = std::pair<BlockUpdateContext<DistType>*, BlockUpdateContext<DistType>*>;
    
    NearestNodesGenerator(std::span<BlockUpdateContext<DistType>> blockSpan,
                          std::span<std::atomic<bool>> blockStates,
                          std::vector<std::optional<ComparisonKey<size_t>>>& nnToDo):
        blocks(blockSpan),
        readyBlocks(blockStates),
        distancesToCompute(nnToDo),
        nullCounter(0) {};
    
    /*
    std::span<AtomicUniquePtr<BlockUpdateContext<DistType>>> blocks;
    std::vector<std::optional<ComparisonKey<size_t>>>& distancesToCompute;
    size_t nullCounter;
    */
    bool operator()(ThreadPool<ThreadFunctors<DistType, COMExtent>>& pool, AsyncQueue<std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>>& resultsQueue){
        auto nnDistanceTaskGenerator = [&](BlockPtrPair blockPtrs)->auto{

            auto task = [&, ptrs = blockPtrs, readyBlocks = this->readyBlocks](ThreadFunctors<DistType, COMExtent>& functors) mutable->void{
                const QueryContext<DistType>& lhsQueryContext = ptrs.first->queryContext;
                const QueryContext<DistType>& rhsQueryContext = ptrs.second->queryContext;
                std::tuple<size_t, size_t, DistType> nnDistResult = lhsQueryContext.NearestNodes(rhsQueryContext,
                                                                                                functors.dispatchFunctor);
                /*
                nnDistanceResults[resultIndex].second = {{blockNumbers.first, std::get<0>(nnDistResult)},
                                                {blockNumbers.second, std::get<1>(nnDistResult)},
                                                std::get<2>(nnDistResult)};
                */
               resultsQueue.Put({{lhsQueryContext.blockNumber, rhsQueryContext.blockNumber}, nnDistResult});

            };

            return task;
        };

        for(auto& blockNums: distancesToCompute){
            /*
            if(!blockNums) continue;
            std::unique_ptr<BlockUpdateContext<DistType>> lhsPtr = blocks[blockNums->first].GetUnique();
            if(!lhsPtr) continue;
            std::unique_ptr<BlockUpdateContext<DistType>> rhsPtr = blocks[blockNums->second].GetUnique();
            if(!rhsPtr){
                blocks[blockNums->first] = std::move(lhsPtr);
                continue;
            }
            */
            if(!blockNums) continue;
            if(!readyBlocks[blockNums->first]) continue;
            if(!readyBlocks[blockNums->second]) continue;

            /*
            bool expectTrue = true;
            if(!readyBlocks[blockNums->first].compare_exchange_strong(expectTrue, false)) continue;
            
            if(!readyBlocks[blockNums->second].compare_exchange_strong(expectTrue, false)){
                readyBlocks[blockNums->first] = true;
                continue;
            }
            */
            BlockUpdateContext<DistType>* rhsPtr = &(blocks[blockNums->second]);
            BlockUpdateContext<DistType>* lhsPtr = &(blocks[blockNums->first]);
            //I have two valid unique_ptrs I can use
            pool.DelegateTask(nnDistanceTaskGenerator({lhsPtr, rhsPtr}));
            blockNums = std::nullopt;
            ++nullCounter;
        }

        if(nullCounter >= distancesToCompute.size()/2){
            EraseNulls(distancesToCompute);
            nullCounter = 0;
        }
        return distancesToCompute.size()==0;
    }

    private:



    //std::span<AtomicUniquePtr<BlockUpdateContext<DistType>>> blocks;
    std::span<BlockUpdateContext<DistType>> blocks;
    std::span<std::atomic<bool>> readyBlocks;
    std::vector<std::optional<ComparisonKey<size_t>>>& distancesToCompute;
    size_t nullCounter;
};

template<typename DistType>
struct NearestNodesConsumer{

    using TaskResult =std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>;

    NearestNodesConsumer() = default;

    NearestNodesConsumer(std::unique_ptr<size_t[]>&& distanceCounts, const size_t numBlocks, const size_t numNeighbors): 
                            distancesPerBlock(std::move(distanceCounts)),
                            nnNumNeighbors(numNeighbors),
                            nnGraph(numBlocks, numNeighbors),
                            blocksDone(0),
                            joinsPerBlock(numBlocks) {};

    bool operator()(std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>> result){
        joinHints[result.first] = result.second;

        nnGraph[result.first.first].push_back({result.first.second, std::get<2>(result.second)});
        if (distancesPerBlock[result.first.first] == nnGraph[result.first.first].size()){
            GraphVertex<size_t, DistType>& vertex = nnGraph[result.first.first];
            std::partial_sort(vertex.begin(), vertex.begin()+nnNumNeighbors, vertex.end(), NeighborDistanceComparison<size_t, DistType>);
            vertex.resize(nnNumNeighbors);

            for(const auto& neighbor: vertex){
                if(initJoinsQueued.insert({result.first.first, neighbor.first}).second){
                    initJoinsToDo.push_back(std::make_optional<StitchHint>(ComparisonKey<size_t>{result.first.first, neighbor.first},
                                            joinHints[{result.first.first, neighbor.first}]));
                    joinsPerBlock[result.first.first] += 1;
                    joinsPerBlock[neighbor.first] += 1;
                }
            }

            ++blocksDone;
        }

        nnGraph[result.first.second].push_back({result.first.first, std::get<2>(result.second)});
        if (distancesPerBlock[result.first.second] == nnGraph[result.first.second].size()){
            GraphVertex<size_t, DistType>& vertex = nnGraph[result.first.second];
            std::partial_sort(vertex.begin(), vertex.begin()+nnNumNeighbors, vertex.end(), NeighborDistanceComparison<size_t, DistType>);
            vertex.resize(nnNumNeighbors);


            for(const auto& neighbor: vertex){
                if(initJoinsQueued.insert({result.first.second, neighbor.first}).second){
                    initJoinsToDo.push_back(std::make_optional<StitchHint>(ComparisonKey<size_t>{result.first.second, neighbor.first},
                                            joinHints[{result.first.second, neighbor.first}]));
                    joinsPerBlock[result.first.second] += 1;
                    joinsPerBlock[neighbor.first] += 1;
                }
            }


            ++blocksDone;
        }

        return blocksDone == nnGraph.size();
    }

    bool CanFeedGenerator(){
        return initJoinsToDo.size() > 0;
    }

    //void OnCompletion

    std::unique_ptr<size_t[]> PassJoinCounts(){
        return std::move(joinsPerBlock);
    }


    private:
    
    std::unique_ptr<size_t[]> distancesPerBlock;
    const size_t nnNumNeighbors;
    Graph<size_t, DistType> nnGraph;
    
    size_t blocksDone;

    std::unordered_map<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>> joinHints;
    std::unordered_set<ComparisonKey<size_t>> initJoinsQueued;
    using StitchHint = std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>;
    std::vector<std::optional<StitchHint>> initJoinsToDo;

    std::unique_ptr<size_t[]> joinsPerBlock;


};

//template<typename Generator, typename Consumer>
auto GenerateTaskBuilder = []<typename... GenArgs, typename... ConsArgs>(std::tuple<GenArgs...>&& generatorArgs, std::tuple<ConsArgs...>&& consumerArgs)->auto{
    auto builder = [generatorArgs, consumerArgs]<typename Generator, typename Consumer>()->TaskQueuer<Generator, Consumer>{
        return TaskQueuer<Generator, Consumer>(generatorArgs, consumerArgs);
    };
    return builder;
};

/*
template<typename Generator, typename Consumer>
struct TaskConstructor{



    template<typename... GenArgs, typename... ConsArgs>
    TaskQueuer(std::tuple<GenArgs...>&& generatorArgs, std::tuple<ConsArgs...>&& consumerArgs): {}

};
*/

template<typename Generator, typename Consumer>
struct TaskQueuer{

    static_assert(std::same_as<typename Generator::TaskResult, typename Consumer::TaskResult>);
    using TaskResult = typename Generator::TaskResult;

    template<typename NextGenerator>
    static constexpr bool hasValidOnCompletion = requires(Consumer cons, NextGenerator& nextGen){
        cons.OnCompletion(nextGen);
    };

    Generator generator;
    Consumer consumer;

    

    template<typename... GenArgs, typename... ConsArgs>
    TaskQueuer(std::tuple<GenArgs...>&& generatorArgs, std::tuple<ConsArgs...>&& consumerArgs): 
        generator(std::make_from_tuple<Generator>(std::forward<std::tuple<GenArgs...>>(generatorArgs))),
        consumer(std::make_from_tuple<Consumer>(std::forward<std::tuple<ConsArgs...>>(consumerArgs))){};

    TaskQueuer(const TaskQueuer&) = delete;
    TaskQueuer(TaskQueuer&&) = delete;

    bool ConsumeResults(){
        std::list<TaskResult> newResults = incomingResults.TryTakeAll();
        for (auto& entry: newResults){
            doneConsuming = consumer(std::move(entry));
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

template<typename Generator>
struct TaskQueuer<Generator, void>{

    using TaskResult = typename Generator::TaskResult;
    static_assert(std::is_void_v<TaskResult>);

    Generator generator;

    template<typename... GenArgs, typename... ConsArgs>
    TaskQueuer(std::tuple<GenArgs...>&& generatorArgs, std::tuple<ConsArgs...>&& consumerArgs): 
        generator(std::make_from_tuple<Generator>(std::forward<std::tuple<GenArgs...>>(generatorArgs))){};

    TaskQueuer(const TaskQueuer&) = delete;
    TaskQueuer(TaskQueuer&&) = delete;


    template<typename Pool>
    bool QueueTasks(Pool& pool){
        doneGenerating = generator(pool, incomingResults);
        return doneGenerating;
    }

    bool DoneGenerating(){
        return doneGenerating;
    }

    bool DoneConsuming(){
        return true;
    }

    bool IsDone(){
        return DoneGenerating();
    }


    private:
    bool doneGenerating;
    bool doneConsuming;
    AsyncQueue<TaskResult> incomingResults;

};

template<typename DistType, typename COMExtent>
struct InitJoinQueuer{

    

    using BlockPtrPair = std::pair<BlockUpdateContext<DistType>*, BlockUpdateContext<DistType>*>;
    using StitchHint = std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>;
    using InitJoinResult = std::pair<JoinResults<size_t, DistType>, JoinResults<size_t, DistType>>;

    using TaskArg = StitchHint;
    using TaskResult = std::pair<ComparisonKey<size_t>, InitJoinResult>;

    bool operator()(ThreadPool<ThreadFunctors<DistType, COMExtent>>& pool, AsyncQueue<std::pair<ComparisonKey<size_t>, InitJoinResult>>& resultsQueue){
        auto joinGenerator = [&](const BlockPtrPair blockPtrs, const std::tuple<size_t, size_t, DistType> stitchHint) -> auto{
        
            auto initJoin = [&, blockPtrs, stitchHint](ThreadFunctors<DistType, COMExtent>& threadFunctors)->void{
                //auto [blockNums, stitchHint] = *(stitchHints.find(blockNumbers));
                //if (blockNums.first != blockNumbers.first) stitchHint = {std::get<1>(stitchHint), std::get<0>(stitchHint), std::get<2>(stitchHint)};
                auto& blockLHS = *(blockPtrs.first);
                auto& blockRHS = *(blockPtrs.second);
                JoinHints<size_t> LHShint;
                LHShint[std::get<0>(stitchHint)] = {std::get<1>(stitchHint)};
                JoinHints<size_t> RHShint;
                RHShint[std::get<1>(stitchHint)] = {std::get<0>(stitchHint)};
                
                
                threadFunctors.cache.SetBlocks(blockLHS.queryContext.blockNumber, blockRHS.queryContext.blockNumber);

                std::pair<JoinResults<size_t, DistType>, JoinResults<size_t, DistType>> retPair;
                retPair.first = BlockwiseJoin(LHShint,
                                            blockLHS.currentGraph,
                                            blockLHS.joinPropagation,
                                            blockRHS.queryContext,
                                            threadFunctors.cache);

                threadFunctors.dispatchFunctor.SetBlocks(blockRHS.queryContext.blockNumber, blockLHS.queryContext.blockNumber);
                ReverseBlockJoin(RHShint,
                                blockRHS.currentGraph,
                                blockRHS.joinPropagation,
                                blockLHS.queryContext,
                                threadFunctors.cache,
                                threadFunctors.dispatchFunctor);

                for(size_t i = 0; const auto& vertex: threadFunctors.cache.reverseGraph){
                    if(vertex.size()>0){
                        retPair.second.push_back({i, vertex});
                    }
                    i++;
                }

                resultsQueue.Put({ComparisonKey<size_t>{blockLHS.queryContext.blockNumber, blockRHS.queryContext.blockNumber}, std::move(retPair)});
            };

            return initJoin;
        };

        for(auto& stitchHint: initJoinsToDo){

            if(!stitchHint) continue;
            if(!readyBlocks[blockNums->first]) continue;
            if(!readyBlocks[blockNums->second]) continue;
            BlockUpdateContext<DistType>* rhsPtr = &(blocks[stitchHint->first.second]);
            BlockUpdateContext<DistType>* lhsPtr = &(blocks[stitchHint->first.first]);
            //I have two valid unique_ptrs I can use
            pool.DelegateTask(joinGenerator({lhsPtr, rhsPtr}, stitchHint.second));
            stitchHint = std::nullopt;
            ++nullCounter;
        }

        if(nullCounter >= initJoinsToDo.size()/2){
            EraseNulls(initJoinsToDo);
            nullCounter = 0;
        }

        return initJoinsToDo.size()==0;
    }


    private:

    std::span<BlockUpdateContext<DistType>> blocks;
    std::span<std::atomic<bool>> readyBlocks;
    //std::span<size_t> joinsPerBlock;

    size_t nullCounter;
    //using StitchHint = std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>;
    std::vector<std::optional<StitchHint>>& initJoinsToDo;
};

template<typename DistType>
struct InitJoinConsumer{

    using BlockUpdates = std::vector<std::pair<size_t, JoinResults<size_t, DistType>>>;

    InitJoinConsumer() = default;

    InitJoinConsumer(const size_t numBlocks): 
                            graphUpdates(std::make_unique<BlockUpdates[]>(numBlocks)),
                            numBlocks(numBlocks),
                            joinsPerBlock(),
                            blocksDone(0){};

    using InitJoinResult = std::pair<JoinResults<size_t, DistType>, JoinResults<size_t, DistType>>;
    using NextTaskArg = size_t;

    bool operator()(std::pair<ComparisonKey<size_t>, InitJoinResult> result){
        graphUpdates[result.first.first].push_back(std::move(result.second.first));
        graphUpdates[result.first.second].push_back(std::move(result.second.second));

        if(joinsPerBlock){
            if(graphUpdates[result.first.first].size() == joinsPerBlock[result.first.first]){
                resultsToReduce.push_back(std::make_optional<size_t>(result.first.first));
                blocksDone++;
            }
            if(graphUpdates[result.first.second].size() == joinsPerBlock[result.first.second]){
                resultsToReduce.push_back(std::make_optional<size_t>(result.first.second));
                blocksDone++;
            }
        }
        return blocksDone == numBlocks;
    }

    bool CanFeedGenerator(){
        return resultsToReduce.size() > 0;
    }

    void UpdateExpectedJoins(std::unique_ptr<size_t[]>&& expectedJoins){
        this->joinsPerBlock = std::move(expectedJoins);

        for(size_t i = 0; i<numBlocks; i+=1){
            if(graphUpdates[i].size() == joinsPerBlock[i]){
                resultsToReduce.push_back(std::make_optional<size_t>(i));
                blocksDone++;
            }
        }
    }

    private:
    
    std::unique_ptr<BlockUpdates[]> graphUpdates;
    const size_t numBlocks;

    std::unique_ptr<size_t[]> joinsPerBlock;
    size_t blocksDone;

    std::vector<std::optional<size_t>>& resultsToReduce;

};



template<typename DistType, typename COMExtent>
struct GraphUpateQueuer{

    using BlockUpdates = std::vector<std::pair<size_t, JoinResults<size_t, DistType>>>;

    using TaskResult = std::pair<size_t, ComparisonMap<size_t, size_t>>;
    using TaskArgs = size_t;

    bool operator()(ThreadPool<ThreadFunctors<DistType, COMExtent>>& pool, AsyncQueue<TaskResult>& resultsQueue){
        auto updateGenerator = [&](const size_t blockToUpdate){

            auto updateTask = [&, blockPtr = &(blocks[blockToUpdate]), updates = std::move(graphUpdates[blockToUpdate]), readyBlocks] mutable{
                for (auto& joinUpdates: updates){
                    for(auto& vertex: joinUpdates.second){
                        ConsumeVertex(blockPtr->currentGraph[vertex.first], vertex.second, joinUpdates.first)
                    }
                }
                resultsQueue.put({blockPtr->queryContext.blockNum,
                                  InitializeComparisonQueues<size_t, size_t, DistType>(blockPtr->currentGraph, blockPtr->queryContext.blockNum)});
                readyBlocks[blockPtr->queryContext.blockNum] = true;
            };
            return updateTask;
        };

        for(auto& blockNum: resultsToReduce){
            if(!blockNum) continue;
            bool expectTrue = true;

            if(!readyBlocks[*blockNum].compare_exchange_strong(expectTrue, false)) continue;
            pool.DelegateTask(updateGenerator(*blockNum));
            blockNum = std::nullopt;
            ++nullCounter;
        }

        if(nullCounter >= resultsToReduce.size()/2){
            EraseNulls(resultsToReduce);
            nullCounter = 0;
        }

        return resultsToReduce.size()==0;
    }
       


    private:
    std::span<BlockUpdateContext<DistType>> blocks;
    std::span<std::atomic<bool>> readyBlocks;
    std::span<BlockUpdates[]> graphUpdates;


    std::vector<std::optional<size_t>>& resultsToReduce;

    size_t nullCounter;
};


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
struct GraphUpdateConsumer{

    //using BlockUpdates = std::vector<std::pair<size_t, JoinResults<size_t, DistType>>>;
    using TaskResult = std::pair<size_t, ComparisonMap<size_t, size_t>>;
    using NextTaskArgs = TaskResult;
    GraphUpdateConsumer() = default;

    GraphUpdateConsumer(std::span<const BlockUpdateContext<DistType>> blocks):
                            numBlocks(blocks.size()),
                            blocksUpdated(blocks.size()),
                            blocksDone(0),
                            comparisonArr(MakeInvertedComparisonQueues(blocks)){};


    bool operator()(TaskResult result){
        blocksUpdated[result.first] = true;
        //InvertComparisonMap({comparisonArr.get(), numBlocks},
                            //result.first,
                            //std::move(result.second));
        comparisonsToDo.push_back(std::move(result));
        blocksDone++;
        return blocksDone == numBlocks;
    }

    bool CanFeedGenerator() const{
        return comparisonArr.size() > 0;
    }

    bool AllBlocksReady() const{
        return blocksDone == numBlocks;
    }


    private:

    const size_t numBlocks;

    std::vector<bool> blocksUpdated;

    size_t blocksDone;
    std::vector<std::optional<TaskResult>>& comparisonArr;
    //std::unique_ptr<InvertedComparisons<DistType>[]> comparisonArr;

};

template<typename DistType>
DelegatedComparisons<DistType> DelegateQueues(InvertedComparisons<DistType>& comparisonsToMove, std::span<const BlockUpdateContext<DistType>> blocks){
    DelegatedComparisons<DistType> retComparisons = {std::move(comparisonsToMove.comparisonsQueued), blocks};
    comparisonsToMove.comparisonsQueued = std::make_unique<std::vector<BlockIndecies>[]>(comparisonsToMove.mySize);
    return retComparisons;
}


using JoinsToDo = std::pair<size_t, std::unordered_map<size_t, std::vector<size_t>>>;

template<typename DistType, typename COMExtent>
struct GraphComparisonQueuer{

    using TaskArgs = std::pair<size_t, ComparisonMap<size_t, size_t>>;
    using TaskResult = void;
    using BlockUpdates = std::vector<std::pair<size_t, JoinResults<size_t, DistType>>>;

    bool operator()(ThreadPool<ThreadFunctors<DistType, COMExtent>>& pool, AsyncQueue<TaskResult>& resultsQueue){
        auto comparisonGenerator = [&](const size_t blockToUpdate, ComparisonMap<size_t, size_t>&& comparisonsToDo){

            auto comparisonTask = [&, blockPtr = &(blocks[blockToUpdate], comparisons = std::move(comparisonsToDo), blocks, initializedBlocks] mutable{
                //JoinsToDo retMap;
                blockPtr->joinsToDo = InitializeJoinMap<DistType>(blocks, comparisons, blockPtr->blockJoinTracker);
                //resultsQueue.Put();
                initializedBlocks[blockPtr->queryContext.blockNumber];
            };
            return comparisonTask;
        };

        for(auto& comparison: comparisonsToDo){
            if(!comparison) continue;
            //bool expectTrue = true;

            //if(!readyBlocks[*blockNum].compare_exchange_strong(expectTrue, false)) continue;
            if(allUpdated || std::all_of(comparison->second.begin(), comparison->second.end(), [&](const std::pair<size_t, ComparisonVec<size_t>>& targetBlock)->bool{
                return updatedBlocks[targetBlock.first];
            })){
                pool.DelegateTask(comparisonGenerator(comparison->first, comparison->second));
                comparison = std::nullopt;
                ++nullCounter;
            }

            
        }

        if(nullCounter >= resultsToReduce.size()/2){
            EraseNulls(comparisonsToDo);
            nullCounter = 0;
        }

        return resultsToReduce.size()==0;
    }
       


    private:

    std::span<BlockUpdateContext<DistType>> blocks;
    //std::span<std::atomic<bool>> readyBlocks;
    //std::unique_ptr<InvertedComparisons<DistType>[]> comparisonArr;
    std::vector<bool>& updatedBlocks;
    std::span<std::atomic<bool>> initializedBlocks;
    bool allUpdated = false;
    std::vector<std::optional<TaskArgs>>& comparisonsToDo;
    //std::vector<std::optional<size_t>>& resultsToReduce;

    size_t nullCounter;
};





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

    ThreadPool<ThreadFunctors<float, float>> pool(6, euclideanFunctor, comFunctor, splitParams.maxTreeSize, 10);

    std::vector<size_t> sizes;
    sizes.reserve(dataBlocks.size());
    for(const auto& block: dataBlocks){
        sizes.push_back(block.size());
    }
    
    

    pool.StartThreads();

    auto nnFuture = NearestNodesToDo(metaGraph, pool);

    BlocksAndState<float> blocks = InitializeBlockContexts(dataBlocks.size(), sizes, parameters, pool);

    auto nnToDo = nnFuture.get();


    TaskQueuer<NearestNodesGenerator<float, float>, 
               NearestNodesConsumer<float>> testQueuer(std::forward_as_tuple(std::span<BlockUpdateContext<float>>(blocks.blocks.get(), dataBlocks.size()), std::span<std::atomic<bool>>(blocks.isReady.get(), dataBlocks.size()), nnToDo.first),
                                                              std::forward_as_tuple(std::move(nnToDo.second), dataBlocks.size(), parameters.indexParams.nearestNodeNeighbors));

    while(!testQueuer.DoneGenerating()){
        testQueuer.QueueTasks(pool);
        testQueuer.ConsumeResults();
    }
    //auto testRes = futures[sizes.size()-1].second.get();
    

    pool.StopThreads();

    

    //NearestNodesGenerator test{std::span};
    /*
    for (size_t i = 0; i<dataBlocks.size(); i+=1){
        //if (!(futures[i])) 
        assert(blocks[i]);
        auto test = blocks[i].GetUnique();
        std::cout << test->currentGraph.size() << std::endl;
    }
    assert(blocks[0] == false);
    */
    return 0;
}