/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_GRAPHCOMPARISONTASK_HPP
#define NND_GRAPHCOMPARISONTASK_HPP

#include <vector>
#include <span>

#include "Parallelization/TaskQueuer.hpp"

#include "Parallelization/ThreadPool.hpp"
#include "ParallelizationObjects.hpp"

namespace nnd {

template<typename DistType, typename COMExtent>
struct GraphComparisonGenerator{

    using TaskArgs = std::pair<BlockNumber_t, ComparisonMap>;
    using TaskResult = void;
    //using BlockUpdates = std::vector<std::pair<size_t, JoinResults<DistType>>>;

    GraphComparisonGenerator(std::span<BlockUpdateContext<DistType>> blocks,
                          std::vector<bool>& updatedBlocks,
                          std::span<std::atomic<bool>> initializedBlocks):
                          blocks(blocks),
                          updatedBlocks(updatedBlocks),
                          initializedBlocks(initializedBlocks){};


    bool operator()(ThreadPool<ThreadFunctors<DistType, COMExtent>>& pool, std::vector<std::optional<TaskArgs>>& comparisonsToDo){
        auto comparisonGenerator = [&](const BlockNumber_t blockToUpdate, ComparisonMap&& comparisonsToDo){

            auto comparisonTask = [&, blockPtr = &(blocks[blockToUpdate]), comparisons = std::move(comparisonsToDo), blocks = this->blocks, initializedBlocks = this->initializedBlocks](ThreadFunctors<DistType, COMExtent>& threadFunctors) mutable{
                blockPtr->joinsToDo = InitializeJoinMap<DistType>(blocks, comparisons, blockPtr->blockJoinTracker);
                //resultsQueue.Put();
                initializedBlocks[blockPtr->queryContext.blockNumber] = true;
            };
            return comparisonTask;
        };
        size_t nullCounter(0);
        for(auto& comparison: comparisonsToDo){
            if(!comparison) {
                nullCounter++;
                continue;
            }
            //bool expectTrue = true;

            //if(!readyBlocks[*blockNum].compare_exchange_strong(expectTrue, false)) continue;
            if(allUpdated || std::all_of(comparison->second.begin(), comparison->second.end(), [&](const std::pair<size_t, ComparisonVec>& targetBlock)->bool{
                return updatedBlocks[targetBlock.first];
            })){
                pool.DelegateTask(comparisonGenerator(comparison->first, std::move(comparison->second)));
                comparison = std::nullopt;
                ++nullCounter;
            }

            
        }

        if(nullCounter >= comparisonsToDo.size()/2){
            EraseNulls(comparisonsToDo);
            nullCounter = 0;
        }

        return comparisonsToDo.size()==0;
    }
       


    private:

    std::span<BlockUpdateContext<DistType>> blocks;
    //std::span<std::atomic<bool>> readyBlocks;
    //std::unique_ptr<InvertedComparisons<DistType>[]> comparisonArr;
    std::vector<bool>& updatedBlocks;
    std::span<std::atomic<bool>> initializedBlocks;
    bool allUpdated = false;
    //std::vector<std::optional<TaskArgs>>& comparisonsToDo;
    //std::vector<std::optional<size_t>>& resultsToReduce;

    //size_t nullCounter;
};

template<typename DistType, typename COMExtent>
using ComparisonTask = TaskQueuer<GraphComparisonGenerator<DistType, COMExtent>, void>;

}

#endif