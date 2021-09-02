/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_NEARESTNODETASK_HPP
#define NND_NEARESTNODETASK_HPP

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>
#include <optional>
#include <atomic>

#include "Parallelization/TaskQueuer.hpp"


namespace nnd {

template<typename DistType, typename COMExtent>
struct NearestNodesGenerator{

    using TaskResult = std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>;

    using BlockPtrPair = std::pair<BlockUpdateContext<DistType>*, BlockUpdateContext<DistType>*>;
    
    using TaskArgs = ComparisonKey<size_t>;
    
    NearestNodesGenerator(std::span<BlockUpdateContext<DistType>> blockSpan,
                          std::span<std::atomic<bool>> blockStates)://,
                          //std::vector<std::optional<ComparisonKey<size_t>>>& nnToDo):
        blocks(blockSpan),
        readyBlocks(blockStates),
        //distancesToCompute(nnToDo),
        nullCounter(0) {};
    
    /*
    std::span<AtomicUniquePtr<BlockUpdateContext<DistType>>> blocks;
    std::vector<std::optional<ComparisonKey<size_t>>>& distancesToCompute;
    size_t nullCounter;
    */
    bool operator()(ThreadPool<ThreadFunctors<DistType, COMExtent>>& pool,
                    AsyncQueue<std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>>& resultsQueue,
                    std::vector<std::optional<ComparisonKey<size_t>>>& distancesToCompute){
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
    //std::vector<std::optional<ComparisonKey<size_t>>>& distancesToCompute;
    size_t nullCounter;
};

template<typename DistType>
struct InitJoinConsumer;

template<typename DistType>
struct NearestNodesConsumer{

    using TaskResult =std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>;
    using StitchHint = std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>;

    NearestNodesConsumer() = default;

    NearestNodesConsumer(std::unique_ptr<size_t[]>&& distanceCounts, const size_t numBlocks, const size_t numNeighbors): 
                            distancesPerBlock(std::move(distanceCounts)),
                            nnNumNeighbors(numNeighbors),
                            nnGraph(numBlocks, numNeighbors),
                            blocksDone(0),
                            joinsPerBlock(std::make_unique<size_t[]>(numBlocks)) {};

    bool operator()(std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>> result){
        joinHints[result.first] = result.second;

        nnGraph[result.first.first].push_back({result.first.second, std::get<2>(result.second)});
        if (distancesPerBlock[result.first.first] == nnGraph[result.first.first].size()){
            GraphVertex<size_t, DistType>& vertex = nnGraph[result.first.first];
            std::partial_sort(vertex.begin(), vertex.begin()+nnNumNeighbors, vertex.end(), NeighborDistanceComparison<size_t, DistType>);
            vertex.resize(nnNumNeighbors);

            for(const auto& neighbor: vertex){
                if(initJoinsQueued.insert({result.first.first, neighbor.first}).second){
                    //std::tuple<size_t, size_t, DistType>> hint = joinHints[{result.first.first, neighbor.first}];
                    initJoinsToDo.push_back(std::make_optional<StitchHint>(*(joinHints.find({result.first.first, neighbor.first}))));
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
                    //initJoinsToDo.push_back(std::make_optional<StitchHint>(ComparisonKey<size_t>{result.first.second, neighbor.first},
                    //                        joinHints[{result.first.second, neighbor.first}]));
                    initJoinsToDo.push_back(std::make_optional<StitchHint>(*(joinHints.find({result.first.second, neighbor.first}))));
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

    std::vector<std::optional<StitchHint>>& GetTaskArgs(){
        return initJoinsToDo;
    }

    
    void OnCompletion(InitJoinConsumer<DistType>& nextStep){
        if(joinsPerBlock) nextStep.UpdateExpectedJoins(std::move(joinsPerBlock));
    }


    private:
    
    std::unique_ptr<size_t[]> distancesPerBlock;
    const size_t nnNumNeighbors;
    Graph<size_t, DistType> nnGraph;
    
    size_t blocksDone;

    std::unordered_map<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>> joinHints;
    std::unordered_set<ComparisonKey<size_t>> initJoinsQueued;
    
    std::vector<std::optional<StitchHint>> initJoinsToDo;

    std::unique_ptr<size_t[]> joinsPerBlock;


};

template<typename DistType, typename COMExtent>
using NNTask = TaskQueuer<NearestNodesGenerator<DistType, COMExtent>, NearestNodesConsumer<DistType>>;

}

#endif