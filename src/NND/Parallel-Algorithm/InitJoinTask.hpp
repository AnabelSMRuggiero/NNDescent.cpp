/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_INITJOINTASK_HPP
#define NND_INITJOINTASK_HPP

#include "Parallelization/TaskQueuer.hpp"

namespace nnd {

template<typename DistType, typename COMExtent>
struct InitJoinGenerator{

    

    using BlockPtrPair = std::pair<BlockUpdateContext<DistType>*, BlockUpdateContext<DistType>*>;
    using StitchHint = std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>;
    using InitJoinResult = std::pair<JoinResults<size_t, DistType>, JoinResults<size_t, DistType>>;

    using TaskArgs = StitchHint;
    using TaskResult = std::pair<ComparisonKey<size_t>, InitJoinResult>;

    
    InitJoinGenerator(std::span<BlockUpdateContext<DistType>> blocks, std::span<std::atomic<bool>> readyBlocks):
        blocks(blocks),
        readyBlocks(readyBlocks) {};
    

    bool operator()(ThreadPool<ThreadFunctors<DistType, COMExtent>>& pool,
                    AsyncQueue<TaskResult>& resultsQueue,
                    std::vector<std::optional<StitchHint>>& initJoinsToDo){
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

        size_t nullCounter;
        for(auto& stitchHint: initJoinsToDo){

            if(!stitchHint) {
                nullCounter++;
                continue;
            }
            if(!readyBlocks[stitchHint->first.first]) continue;
            if(!readyBlocks[stitchHint->first.second]) continue;

            BlockUpdateContext<DistType>* lhsPtr = &(blocks[stitchHint->first.first]);
            BlockUpdateContext<DistType>* rhsPtr = &(blocks[stitchHint->first.second]);
            
            lhsPtr->blockJoinTracker[stitchHint->first.second] = true;
            rhsPtr->blockJoinTracker[stitchHint->first.first] = true;
            //I have two valid unique_ptrs I can use
            pool.DelegateTask(joinGenerator({lhsPtr, rhsPtr}, stitchHint->second));
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

    //size_t nullCounter;
    //using StitchHint = std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>;
    //std::vector<std::optional<StitchHint>>& initJoinsToDo;
};

template<typename DistType>
struct InitJoinConsumer{
    using StitchHint = std::pair<ComparisonKey<size_t>, std::tuple<size_t, size_t, DistType>>;
    using BlockUpdates = std::vector<std::pair<size_t, JoinResults<size_t, DistType>>>;
    using TaskArgs = StitchHint;
    
    using InitJoinResult = std::pair<JoinResults<size_t, DistType>, JoinResults<size_t, DistType>>;
    using TaskResult = std::pair<ComparisonKey<size_t>, InitJoinResult>;

    InitJoinConsumer() = default;

    InitJoinConsumer(std::span<BlockUpdates> updateStorage): 
                            graphUpdates(updateStorage),
                            //graphUpdates(std::make_unique<BlockUpdates[]>(numBlocks)),
                            //numBlocks(numBlocks),
                            joinsPerBlock(),
                            blocksDone(0){};


    using NextTaskArg = size_t;

    bool operator()(std::pair<ComparisonKey<size_t>, InitJoinResult> result){
        graphUpdates[result.first.first].push_back({result.first.second, std::move(result.second.first)});
        graphUpdates[result.first.second].push_back({result.first.first, std::move(result.second.second)});

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
        return blocksDone == graphUpdates.size();
    }

    bool CanFeedGenerator(){
        return resultsToReduce.size() > 0;
    }

    void UpdateExpectedJoins(std::unique_ptr<size_t[]>&& expectedJoins){
        this->joinsPerBlock = std::move(expectedJoins);

        for(size_t i = 0; i<graphUpdates.size(); i+=1){
            if(graphUpdates[i].size() == joinsPerBlock[i]){
                resultsToReduce.push_back(std::make_optional<size_t>(i));
                blocksDone++;
            }
        }
    }

    std::vector<std::optional<size_t>>& GetTaskArgs(){
        return resultsToReduce;
    }
    

    private:
    std::span<BlockUpdates> graphUpdates;
    /*
    std::unique_ptr<BlockUpdates[]> graphUpdates;
    const size_t numBlocks;
    */

    std::unique_ptr<size_t[]> joinsPerBlock;
    size_t blocksDone;

    std::vector<std::optional<size_t>> resultsToReduce;

};

template<typename DistType, typename COMExtent>
using InitJoinTask = TaskQueuer<InitJoinGenerator<DistType, COMExtent>, InitJoinConsumer<DistType>>;

}

#endif