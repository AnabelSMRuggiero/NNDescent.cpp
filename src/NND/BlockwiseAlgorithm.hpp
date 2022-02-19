/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_BLOCKWISEALGORITHM_HPP
#define NND_BLOCKWISEALGORITHM_HPP


#include <algorithm>
#include <vector>
#include <utility>
#include <ranges>
#include <functional>
#include <memory>

#include "../ann/Data.hpp"

#include "MemoryInternals.hpp"
#include "GraphStructures.hpp"
#include "NND/FunctorErasure.hpp"
#include "NND/GraphStructures/CachingFunctor.hpp"
#include "SubGraphQuerying.hpp"

#include "UtilityFunctions.hpp"


namespace nnd{

//template<typename DataIndexType>
using ComparisonVec = std::vector<std::pair<DataIndex_t, DataIndex_t>>;

//template<typename BlockNumberType, typename DataIndexType>
using ComparisonMap = std::unordered_map<BlockNumber_t, ComparisonVec>;

//template<typename DataIndexType>
using JoinHint = std::pair<DataIndex_t, std::vector<DataIndex_t>>;

/*
template<typename DataIndexType>
struct JoinHint{
    DataIndexType joinee;
    std::vector<DataIndexType> joinStart;
};
*/

//template<typename DataIndexType>
using JoinHints = std::unordered_map<DataIndex_t, std::vector<DataIndex_t>>;

//template<template<typename> typename Alloc = std::allocator>                                 
using JoinMap = std::unordered_map<BlockNumber_t, std::unordered_map<DataIndex_t, std::vector<DataIndex_t>>>;


template<typename DistType>
ComparisonMap InitializeComparisonQueues(const Graph<BlockIndecies, DistType>& currentBlock, BlockNumber_t blockNum){

    ComparisonMap retMap;
    for (size_t j = 0; j<currentBlock.size(); j+=1){
        for (const auto& neighbor: currentBlock[j]){
            if (neighbor.first.blockNumber != blockNum) retMap[neighbor.first.blockNumber].push_back({j, neighbor.first.dataIndex});
        }
    }

    return retMap;
}

template<typename DistType>
std::vector<std::pair<DataIndex_t, std::vector<DataIndex_t>>> FlattenHints(const JoinHints& startJoins){
    std::vector<std::pair<DataIndex_t, std::vector<DataIndex_t>>> joinHints(startJoins.size());
    std::ranges::transform(startJoins, joinHints.begin(), [&](auto hint) { return hint;});
    /*
    std::ranges::transform(startJoins, joinHints.begin(), [&] (const auto& hint){
        GraphVertex<DataIndex_t, DistType> queryHint;
        queryHint.resize(hint.second.size());
        std::ranges::transform(hint.second, queryHint.begin(), [&](const auto& index){ 
            return std::pair{index, std::numeric_limits<DistType>::max()};
        });
        return std::pair{hint.first, queryHint};
    });
    */
    return joinHints;
}


template<typename DistType, template<typename> typename Alloc = PolymorphicAllocator>
using JoinResults = std::vector<std::pair<DataIndex_t, GraphVertex<DataIndex_t, DistType, Alloc>>, Alloc<std::pair<DataIndex_t, GraphVertex<DataIndex_t, DistType, Alloc>>>>;

//template<typename BlockNumberType, typename DataIndexType, typename DataEntry, typename DistType>
template<typename DistType, typename QueryFunctor>
JoinResults<DistType> BlockwiseJoin(const JoinHints& startJoins,
                   const Graph<BlockIndecies, DistType>& currentGraphState,
                   const DirectedGraph<DataIndex_t>& searchSubgraph,
                   const QueryContext<DataIndex_t, DistType>& targetBlock,
                   QueryFunctor&& queryFunctor){
    
    std::vector<std::pair<DataIndex_t, std::vector<DataIndex_t>>> joinHints = FlattenHints<DistType>(startJoins);
    
    
    NodeTracker nodesJoined(searchSubgraph.size());
    auto notJoined = [&](const auto& index){return !nodesJoined[index];};

    JoinResults<DistType> retResults;
    std::vector<std::vector<DataIndex_t>> vecCache;
    while(joinHints.size()){
        JoinResults<DistType> joinResults;
        
        for (auto& joinHint: joinHints){
            
            joinResults.push_back({joinHint.first, targetBlock.Query(joinHint.second, joinHint.first, queryFunctor)});
            nodesJoined[joinHint.first] = true;
            joinHint.second.clear();
            vecCache.push_back(std::move(joinHint.second));
        }
        joinHints.clear();

        for(auto& result : joinResults){
            EraseRemove(result.second, currentGraphState[result.first].PushThreshold());
        };
        auto notEmpty = [](const auto& result){
            return result.second.size()>0;
        };
        for(auto& result: joinResults | std::views::filter(notEmpty)){
            
                
            for(const auto& leafNeighbor: searchSubgraph[result.first] | std::views::filter(notJoined)){
                if(vecCache.size()>0){
                    joinHints.push_back({leafNeighbor, std::move(vecCache.back())});
                    vecCache.pop_back();
                } else {
                    joinHints.push_back({leafNeighbor, std::vector<DataIndex_t>{}});
                }
                joinHints.back().second.resize(result.second.size());

                std::ranges::transform(result.second, joinHints.back().second.begin(), [&](const auto& edge){
                    return edge.first;
                });

                nodesJoined[leafNeighbor] = true;
            }

            retResults.push_back({result.first, std::move(result.second)});
            
            
        }
        
    }
    return retResults;
}

template<typename DistType, typename DistanceFunctor>
void ReverseBlockJoin(const JoinHints& startJoins,
                   const Graph<BlockIndecies, DistType>& currentGraphState,
                   const DirectedGraph<DataIndex_t>& searchSubgraph,
                   const QueryContext<DataIndex_t, DistType>& targetBlock,
                   cache_state<DistType>& result_cache,
                   DistanceFunctor&& queryFunctor){
    
    //std::vector<std::pair<size_t, GraphVertex<size_t, DistType>>> joinHints;
    result_cache.reduce_results();
    auto& cache = result_cache.results;

    NodeTracker nodesJoined(searchSubgraph.size());
    std::vector<size_t> successfulJoins;
    for (const auto& hint: startJoins){

        GraphVertex<DataIndex_t, DistType>& vertex = cache.reverseGraph[hint.first];

        
        for (const auto index: hint.second){
            if (!cache.nodesJoined[hint.first][index]){
                if (vertex.size() == targetBlock.queryHint.size()) vertex.PushNeighbor({index, queryFunctor(hint.first, index)});
                else if (vertex.size() == targetBlock.queryHint.size()-1){
                    vertex.push_back({index, queryFunctor(hint.first, index)});
                    vertex.JoinPrep();
                } else vertex.push_back({index, queryFunctor(hint.first, index)});
                cache.nodesJoined[hint.first][index] = true;
            }
        }    
        
        
        nodesJoined[hint.first] = true;
        targetBlock.Query(vertex, hint.first, queryFunctor, cache.nodesJoined[hint.first]);
        EraseRemove(vertex, currentGraphState[hint.first].PushThreshold());
        /*
        NeighborOverDist<size_t, DistType> comparison(currentGraphState[hint.first][0].second);
        vertex.erase(std::remove_if(vertex.begin(),
                                    vertex.end(),
                                    comparison),
                     vertex.end());
        */
        if (vertex.size()!=0) successfulJoins.push_back(hint.first);
        
        /*
        GraphVertex<size_t, DistType> queryHint;
        for (const auto index: hint.second){
            queryHint.push_back({index, std::numeric_limits<DistType>::max()});
        }
        joinHints.push_back({hint.first, std::move(queryHint)});
        */
    }

    std::vector<size_t> joinQueue;
    auto toNeighbor = [&](const auto success){ return searchSubgraph[success]; };
    auto notJoined = [&](const auto leafNeighbor){ return !nodesJoined[leafNeighbor]; };

    auto newJoinView = std::views::transform(toNeighbor)
                      | std::views::join
                      | std::views::filter(notJoined);

    auto addNewJoin = [&](const auto& leafNeighbor){
            joinQueue.push_back(leafNeighbor);
            nodesJoined[leafNeighbor] = true;
    };
    

    std::ranges::for_each(successfulJoins | newJoinView, addNewJoin);
    /*
    for (auto success: successfulJoins){
        for (const auto& leafNeighbor: searchSubgraph[success]){
            if(!nodesJoined[leafNeighbor]){
                joinQueue.push_back(leafNeighbor);

                
                //We can add these to nodesJoined a touch early to prevent dupes
                nodesJoined[leafNeighbor] = true;
            }
        }
    }
    */
    successfulJoins.clear();

    //std::vector<std::pair<size_t, GraphVertex<size_t, DistType>>> retResults;
    while(joinQueue.size()){
        std::vector<std::pair<size_t, GraphVertex<size_t, DistType>>> joinResults;
        for (auto& joinee: joinQueue){

            targetBlock.Query(cache.reverseGraph[joinee], joinee, queryFunctor, cache.nodesJoined[joinee]);
            nodesJoined[joinee] = true;
            EraseRemove(cache.reverseGraph[joinee], currentGraphState[joinee].PushThreshold());
            //NeighborOverDist<DistType> comparison(currentGraphState[joinee][0].second);
            //cache.reverseGraph[joinee].erase(std::remove_if(cache.reverseGraph[joinee].begin(), cache.reverseGraph[joinee].end(), comparison), cache.reverseGraph[joinee].end());
            if (cache.reverseGraph[joinee].size()!=0) successfulJoins.push_back(joinee);
        }
        joinQueue.clear();

        
        std::ranges::for_each(successfulJoins | newJoinView, addNewJoin);
        
        successfulJoins.clear();
        /*
        for(auto& success: successfulJoins){
            
            for(const auto& leafNeighbor: searchSubgraph[success]){
                if(!nodesJoined[leafNeighbor]){
                    joinQueue.push_back(leafNeighbor);

                    //We can add these to nodesJoined a touch early to prevent dupes
                    nodesJoined[leafNeighbor] = true;
                }
            }
        }
        */
        //joinHints = std::move(newJoins);
    }
    return;
}

template<typename DistType, bool checkGraphFragment>
void NewJoinQueues(const JoinResults<DistType>& joinResults,
                   const NodeTracker& blocksJoined,
                   const GraphFragment_t targetFragment,
                   const Graph<BlockIndecies, DistType>& targetGraphState,
                   JoinMap& mapToUpdate){
    
    for (const auto& result: joinResults){
        for (const auto index: result.second){
            for (const auto& targetVertexNeighbor: targetGraphState[index.first]){
                size_t targetBlock = targetVertexNeighbor.first.blockNumber;
                if constexpr(checkGraphFragment){
                    GraphFragment_t neighborFragment = targetVertexNeighbor.first.graphFragment;
                    if (blocksJoined[targetBlock] || neighborFragment != targetFragment) continue;
                } else{
                    if (blocksJoined[targetBlock]) continue;
                }
                auto findItr = std::find(mapToUpdate[targetBlock][result.first].begin(), mapToUpdate[targetBlock][result.first].end(), targetVertexNeighbor.first.dataIndex);
                if (findItr == mapToUpdate[targetBlock][result.first].end()) mapToUpdate[targetBlock][result.first].push_back(targetVertexNeighbor.first.dataIndex);
            } 
            
        }
    }
}

template<typename DistType, bool checkGraphFragment>
void NewJoinQueues(const Graph<DataIndex_t, DistType>& joinResults,
                   const NodeTracker& blocksJoined,
                   const GraphFragment_t targetFragment,
                   const Graph<BlockIndecies, DistType>& targetGraphState,
                   JoinMap& mapToUpdate){
    
    for (size_t i = 0; const auto& result: joinResults){
        for (const auto index: result){
            for (const auto& targetVertexNeighbor: targetGraphState[index.first]){
                BlockNumber_t targetBlock = targetVertexNeighbor.first.blockNumber;
                if constexpr(checkGraphFragment){
                    GraphFragment_t neighborFragment = targetVertexNeighbor.first.graphFragment;
                    if (blocksJoined[targetBlock] || neighborFragment != targetFragment) continue;
                } else{
                    if (blocksJoined[targetBlock]) continue;
                }
                auto findItr = std::find(mapToUpdate[targetBlock][i].begin(), mapToUpdate[targetBlock][i].end(), targetVertexNeighbor.first.dataIndex);
                if (findItr == mapToUpdate[targetBlock][i].end()) mapToUpdate[targetBlock][i].push_back(targetVertexNeighbor.first.dataIndex);
            } 
        }
        i++;
    }
}

template<typename DistType>
struct BlockUpdateContext {

    
    NodeTracker blockJoinTracker;
    JoinMap joinsToDo;
    JoinMap newJoins;
    QueryContext<DataIndex_t, DistType> queryContext;
    Graph<BlockIndecies, DistType> currentGraph;
    DirectedGraph<DataIndex_t> joinPropagation;


    BlockUpdateContext() = default;

    BlockUpdateContext(const Graph<DataIndex_t, DistType>& blockGraph, QueryContext<DataIndex_t, DistType>&& queryContext, const size_t numberOfBlocksToJoin):
        blockJoinTracker(numberOfBlocksToJoin),
        queryContext(std::move(queryContext)),
        currentGraph(ToBlockIndecies(blockGraph, queryContext.graphFragment, queryContext.blockNumber)),
        joinPropagation(blockGraph){
            blockJoinTracker[queryContext.blockNumber] = true;
    };

    void SetNextJoins(){
        
        joinsToDo = std::move(newJoins);
        auto removeExtraneous = [&](const auto& item){
            const auto& [key, value] = item;
            return this->blockJoinTracker[key];
        };
        std::erase_if(joinsToDo, removeExtraneous);
        newJoins = JoinMap();
        
    }
};

template<typename DistType, typename BlockContainer>
JoinMap InitializeJoinMap(const BlockContainer& blockUpdateContexts,
                                          const ComparisonMap& comparisonMap,
                                          const NodeTracker& nodesJoined){
    JoinMap joinMap{};
    for (auto& [targetBlock, queue]: comparisonMap){
        //std::unordered_map<size_t, std::pair<size_t, std::vector<size_t>>> joinHints;
        for (const auto& [sourceIndex, targetIndex]: queue){
            for (const auto& neighbor: blockUpdateContexts[targetBlock].currentGraph[targetIndex]){
                if (neighbor.first.blockNumber == targetBlock || nodesJoined[neighbor.first.blockNumber]) continue;
                auto result = std::ranges::find(joinMap[neighbor.first.blockNumber][sourceIndex], neighbor.first.dataIndex);
                if (result == joinMap[neighbor.first.blockNumber][sourceIndex].end()) joinMap[neighbor.first.blockNumber][sourceIndex].push_back(neighbor.first.dataIndex);
            }
        }
    }
    return joinMap;
}



template<bool checkGraphFragment = false, typename DistType>
int UpdateBlocks(BlockUpdateContext<DistType>& blockLHS,
                 BlockUpdateContext<DistType>& blockRHS,
                 const erased_binary_binder<DistType>& bound_blocks,
                 cache_state<DistType>& result_cache){

    blockLHS.blockJoinTracker[blockRHS.queryContext.blockNumber] = true;
    

    JoinResults<DistType> blockLHSUpdates = BlockwiseJoin(blockLHS.joinsToDo[blockRHS.queryContext.blockNumber],
                                                                        blockLHS.currentGraph,
                                                                        blockLHS.joinPropagation,
                                                                        blockRHS.queryContext,
                                                                        caching_functor<DistType>{result_cache, bound_blocks(blockLHS.queryContext.blockNumber, blockRHS.queryContext.blockNumber)});

    NewJoinQueues<DistType, checkGraphFragment>(blockLHSUpdates, 
                                                blockLHS.blockJoinTracker,
                                                blockRHS.queryContext.graphFragment,
                                                blockRHS.currentGraph,
                                                blockLHS.newJoins);

    
    
    blockRHS.blockJoinTracker[blockLHS.queryContext.blockNumber] = true;
    
    
    ReverseBlockJoin(blockRHS.joinsToDo[blockLHS.queryContext.blockNumber],
                        blockRHS.currentGraph,
                        blockRHS.joinPropagation,
                        blockLHS.queryContext,
                        result_cache,
                        bound_blocks(blockRHS.queryContext.blockNumber, blockLHS.queryContext.blockNumber));
    
    for(size_t i = 0; auto& vertex: result_cache.results){
        EraseRemove(vertex, blockRHS.currentGraph[i].PushThreshold());
        
    }
    NewJoinQueues<DistType, checkGraphFragment>(result_cache.results.reverseGraph, blockRHS.blockJoinTracker, blockLHS.queryContext.graphFragment, blockLHS.currentGraph, blockRHS.newJoins);

    int graphUpdates{0};

    for (size_t i = 0; auto& result: result_cache.results){
        graphUpdates += ConsumeVertex(blockRHS.currentGraph[i], result, blockLHS.queryContext.graphFragment, blockLHS.queryContext.blockNumber);
        i++;
    }
    for (auto& result: blockLHSUpdates){
        graphUpdates += ConsumeVertex(blockLHS.currentGraph[result.first], result.second, blockRHS.queryContext.graphFragment, blockRHS.queryContext.blockNumber);
    }
    
    return graphUpdates;
    
}

}

#endif