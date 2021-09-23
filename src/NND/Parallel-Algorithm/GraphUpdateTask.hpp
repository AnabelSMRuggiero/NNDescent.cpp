/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_GRAPHUPDATETASK_HPP
#define NND_GRAPHUPDATETASK_HPP

#include <atomic>
#include <vector>
#include <span>

#include "Parallelization/TaskQueuer.hpp"

#include "ParallelizationObjects.hpp"

#include "../BlockwiseAlgorithm.hpp"

namespace nnd {

template<typename DistType, typename COMExtent>
struct GraphUpdateGenerator{

    using BlockUpdates = std::vector<std::pair<BlockNumber_t, JoinResults<DistType>>>;

    using TaskResult = std::pair<BlockNumber_t, ComparisonMap>;
    using TaskArgs = size_t;
    /*
    std::span<BlockUpdateContext<DistType>> blocks;
    std::span<std::atomic<bool>> readyBlocks;
    std::span<BlockUpdates[]> graphUpdates;
    */
    GraphUpdateGenerator(std::span<BlockUpdateContext<DistType>> blocks,
                     std::span<std::atomic<bool>> readyBlocks,
                     std::span<BlockUpdates> graphUpdates): 
                        blocks(blocks),
                        readyBlocks(readyBlocks),
                        graphUpdates(graphUpdates){};

    bool operator()(ThreadPool<ThreadFunctors<DistType, COMExtent>>& pool, AsyncQueue<TaskResult>& resultsQueue, std::vector<std::optional<size_t>>& resultsToReduce){
        auto updateGenerator = [&](const size_t blockToUpdate){

            auto updateTask = [&, blockPtr = &(blocks[blockToUpdate]), updates = std::move(graphUpdates[blockToUpdate]), readyBlocks = this->readyBlocks](ThreadFunctors<DistType, COMExtent>& threadFunctors) mutable{
                for (auto& joinUpdates: updates){
                    for(auto& vertex: joinUpdates.second){
                        ConsumeVertex(blockPtr->currentGraph[vertex.first], vertex.second, blockPtr->queryContext.graphFragment, joinUpdates.first);
                    }
                }
                if constexpr (debugNND) VerifySubGraphState(blockPtr->currentGraph, blockPtr->queryContext.graphFragment, blockPtr->queryContext.blockNumber);
                resultsQueue.Put({blockPtr->queryContext.blockNumber,
                                  InitializeComparisonQueues<DistType>(blockPtr->currentGraph, blockPtr->queryContext.blockNumber)});
                readyBlocks[blockPtr->queryContext.blockNumber] = true;
            };
            return updateTask;
        };

        size_t nullCounter = 0;
        for(auto& blockNum: resultsToReduce){
            if(!blockNum) {
                nullCounter++;
                continue;
            }
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
    std::span<BlockUpdates> graphUpdates;


    //std::vector<std::optional<size_t>>& resultsToReduce;

    //size_t nullCounter;
};


template<typename DistType>
struct GraphUpdateConsumer{

    //using BlockUpdates = std::vector<std::pair<size_t, JoinResults<size_t, DistType>>>;
    using TaskResult = std::pair<BlockNumber_t, ComparisonMap>;
    using NextTaskArgs = TaskResult;
    GraphUpdateConsumer() = default;

    GraphUpdateConsumer(const size_t numBlocks, std::vector<bool>& updateFlags):
                            numBlocks(numBlocks),
                            blocksUpdated(updateFlags),
                            blocksDone(0){};


    bool operator()(TaskResult result){
        blocksUpdated[result.first] = true;
        //InvertComparisonMap({comparisonArr.get(), numBlocks},
                            //result.first,
                            //std::move(result.second));
        comparisonArr.push_back(std::move(result));
        blocksDone++;
        return blocksDone == numBlocks;
    }

    bool CanFeedGenerator() const{
        return comparisonArr.size() > 0;
    }

    bool AllBlocksReady() const{
        return blocksDone == numBlocks;
    }

    std::vector<std::optional<TaskResult>>& GetTaskArgs(){
        return comparisonArr;
    }


    private:

    const size_t numBlocks;

    std::vector<bool>& blocksUpdated;

    size_t blocksDone;
    std::vector<std::optional<TaskResult>> comparisonArr;
    //std::unique_ptr<InvertedComparisons<DistType>[]> comparisonArr;

};

template<typename DistType, typename COMExtent>
using UpdateTask = TaskQueuer<GraphUpdateGenerator<DistType, COMExtent>, GraphUpdateConsumer<DistType>>;
    
}

#endif