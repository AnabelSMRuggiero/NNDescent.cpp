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

#include "GraphStructures.hpp"
#include "SubGraphQuerying.hpp"

#include "UtilityFunctions.hpp"
#include "../Utilities/Data.hpp"

namespace nnd{

template<typename DataIndexType>
using ComparisonVec = std::vector<std::pair<DataIndexType, DataIndexType>>;

template<typename BlockNumberType, typename DataIndexType>
using ComparisonMap = std::unordered_map<BlockNumberType, ComparisonVec<DataIndexType>>;

template<typename DataIndexType>
using JoinHint = std::pair<DataIndexType, std::vector<DataIndexType>>;

/*
template<typename DataIndexType>
struct JoinHint{
    DataIndexType joinee;
    std::vector<DataIndexType> joinStart;
};
*/

template<typename DataIndexType>
using JoinHints = std::unordered_map<DataIndexType, std::vector<DataIndexType>>;

template<typename BlockNumberType, typename DataIndexType>                                   // Consider using a fixed size array
using JoinMap = std::unordered_map<BlockNumberType, std::unordered_map<DataIndexType, std::vector<DataIndexType>>>;
// I could also do a struct where the actual data is vectors, and I use unordered_maps to remap indicies

template<typename BlockNumberType, typename DataIndexType, typename DistType>
ComparisonMap<BlockNumberType, DataIndexType> InitializeComparisonQueues(const Graph<BlockIndecies, DistType>& currentBlock, BlockNumberType blockNum){

    ComparisonMap<BlockNumberType, DataIndexType> retMap;
    for (size_t j = 0; j<currentBlock.size(); j+=1){
        for (const auto& neighbor: currentBlock[j]){
            if (neighbor.first.blockNumber != blockNum) retMap[neighbor.first.blockNumber].push_back({j, neighbor.first.dataIndex});
        }
    }

    return retMap;
}



template<typename DataIndexType, typename DistType>
using JoinResults = std::vector<std::pair<DataIndexType, GraphVertex<DataIndexType, DistType>>>;

//template<typename BlockNumberType, typename DataIndexType, typename DataEntry, typename DistType>
template<typename DistType, typename QueryFunctor>
JoinResults<size_t, DistType> BlockwiseJoin(const JoinHints<size_t>& startJoins,
                   const Graph<BlockIndecies, DistType>& currentGraphState,
                   const Graph<size_t, DistType>& searchSubgraph,
                   const QueryContext<DistType>& targetBlock,
                   QueryFunctor& queryFunctor){
    
    std::vector<std::pair<size_t, GraphVertex<size_t, DistType>>> joinHints;
    for (const auto& hint: startJoins){
        GraphVertex<size_t, DistType> queryHint;
        for (const auto index: hint.second){
            queryHint.push_back({index, std::numeric_limits<DistType>::max()});
        }
        joinHints.push_back({hint.first, std::move(queryHint)});
    }
    NodeTracker nodesJoined(searchSubgraph.size());
    
    std::vector<std::pair<size_t, GraphVertex<size_t, DistType>>> retResults;
    while(joinHints.size()){
        std::vector<std::pair<size_t, GraphVertex<size_t, DistType>>> joinResults;
        for (auto& joinHint: joinHints){
            //GraphVertex<size_t, DistType> joinResult = targetBlock || QueryPoint{joinHint.second, blockData[joinHint.first]};
            //const QueryPoint<size_t, DataEntry, DistType> query(joinHint.second, blockData[joinHint.first], joinHint.first);
            joinResults.push_back({joinHint.first, targetBlock.Query(joinHint.second, joinHint.first, queryFunctor)});
            nodesJoined[joinHint.first] = true;
        }
        joinHints.clear();
        //std::vector<std::pair<size_t, GraphVertex<size_t, DistType>>> newJoins;
        for(auto& result: joinResults){
            //std::heap_sort(result.second.begin(), result.second.end(), NeighborDistanceComparison<size_t, DistType>);
            bool newNeighbor = false;
            GraphVertex<size_t, DistType> updatedResult;
            for (const auto& neighborCandidate: result.second){
                if (neighborCandidate.second < currentGraphState[result.first][0].second){
                    newNeighbor = true;
                    updatedResult.push_back(neighborCandidate);
                }
            }
            if (newNeighbor){   
                
                for(const auto& leafNeighbor: searchSubgraph[result.first]){
                    if(!nodesJoined[leafNeighbor.first]){
                        joinHints.push_back({leafNeighbor.first, result.second});
                        //We can add these to nodesJoined a touch early to prevent dupes
                        nodesJoined[leafNeighbor.first] = true;
                    }
                }
                retResults.push_back({result.first, std::move(updatedResult)});
            }
            //result.second = updatedResult;
        }
        //joinHints = std::move(newJoins);
    }
    return retResults;
}

template<typename DistType, typename DistanceFunctor>
void ReverseBlockJoin(const JoinHints<size_t>& startJoins,
                   const Graph<BlockIndecies, DistType>& currentGraphState,
                   const Graph<size_t, DistType>& searchSubgraph,
                   const QueryContext<DistType>& targetBlock,
                   CachingFunctor<DistType>& cache,
                   DistanceFunctor& queryFunctor){
    
    //std::vector<std::pair<size_t, GraphVertex<size_t, DistType>>> joinHints;

    NodeTracker nodesJoined(searchSubgraph.size());
    std::vector<size_t> successfulJoins;
    for (const auto& hint: startJoins){

        GraphVertex<size_t, DistType>& vertex = cache.reverseGraph[hint.first];

        
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
        EraseRemove(vertex, currentGraphState[hint.first][0].second);
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
    for (auto success: successfulJoins){
        for (const auto& leafNeighbor: searchSubgraph[success]){
            if(!nodesJoined[leafNeighbor.first]){
                joinQueue.push_back(leafNeighbor.first);

                
                //We can add these to nodesJoined a touch early to prevent dupes
                nodesJoined[leafNeighbor.first] = true;
            }
        }
    }
    successfulJoins.clear();

    //std::vector<std::pair<size_t, GraphVertex<size_t, DistType>>> retResults;
    while(joinQueue.size()){
        std::vector<std::pair<size_t, GraphVertex<size_t, DistType>>> joinResults;
        for (auto& joinee: joinQueue){

            targetBlock.Query(cache.reverseGraph[joinee], joinee, queryFunctor, cache.nodesJoined[joinee]);
            nodesJoined[joinee] = true;
            EraseRemove(cache.reverseGraph[joinee], currentGraphState[joinee][0].second);
            //NeighborOverDist<DistType> comparison(currentGraphState[joinee][0].second);
            //cache.reverseGraph[joinee].erase(std::remove_if(cache.reverseGraph[joinee].begin(), cache.reverseGraph[joinee].end(), comparison), cache.reverseGraph[joinee].end());
            if (cache.reverseGraph[joinee].size()!=0) successfulJoins.push_back(joinee);
        }
        joinQueue.clear();
        
        for(auto& success: successfulJoins){
            
            for(const auto& leafNeighbor: searchSubgraph[success]){
                if(!nodesJoined[leafNeighbor.first]){
                    joinQueue.push_back(leafNeighbor.first);

                    //We can add these to nodesJoined a touch early to prevent dupes
                    nodesJoined[leafNeighbor.first] = true;
                }
            }
        }
        //joinHints = std::move(newJoins);
    }
    return;
}

template<typename DistType>
void NewJoinQueues(const std::vector<std::pair<size_t, GraphVertex<size_t, DistType>>>& joinResults,
                   const NodeTracker& blocksJoined,
                   const Graph<BlockIndecies, DistType>& targetGraphState,
                   JoinMap<size_t, size_t>& mapToUpdate){
    
    for (const auto& result: joinResults){
        for (const auto index: result.second){
            for (const auto& targetVertexNeighbor: targetGraphState[index.first]){
                size_t targetBlock = targetVertexNeighbor.first.blockNumber;
                if (blocksJoined[targetBlock]) continue;
                auto findItr = std::find(mapToUpdate[targetBlock][result.first].begin(), mapToUpdate[targetBlock][result.first].end(), targetVertexNeighbor.first.dataIndex);
                if (findItr == mapToUpdate[targetBlock][result.first].end()) mapToUpdate[targetBlock][result.first].push_back(targetVertexNeighbor.first.dataIndex);
            } 
            
        }
    }
}

template<typename DistType>
void NewJoinQueues(const Graph<size_t, DistType>& joinResults,
                   const NodeTracker& blocksJoined,
                   const Graph<BlockIndecies, DistType>& targetGraphState,
                   JoinMap<size_t, size_t>& mapToUpdate){
    
    for (size_t i = 0; const auto& result: joinResults){
        for (const auto index: result){
            for (const auto& targetVertexNeighbor: targetGraphState[index.first]){
                size_t targetBlock = targetVertexNeighbor.first.blockNumber;
                if (blocksJoined[targetBlock]) continue;
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
    JoinMap<size_t, size_t> joinsToDo;
    JoinMap<size_t, size_t> newJoins;
    QueryContext<DistType> queryContext;
    Graph<size_t, DistType> joinPropagation;
    Graph<BlockIndecies, DistType> currentGraph;

    BlockUpdateContext() = default;

    BlockUpdateContext(Graph<size_t, DistType>&& blockGraph, QueryContext<DistType>&& queryContext, const size_t numberOfBlocksToJoin):
        blockJoinTracker(numberOfBlocksToJoin),
        queryContext(std::move(queryContext)),
        joinPropagation(std::forward<Graph<size_t, DistType>>(blockGraph)),
        currentGraph(ToBlockIndecies(joinPropagation, queryContext.blockNumber)) {
            blockJoinTracker[queryContext.blockNumber] = true;
    };

    void SetNextJoins(){
        joinsToDo = std::move(newJoins);
        auto removeExtraneous = [&](const auto& item){
            const auto& [key, value] = item;
            return this->blockJoinTracker[key];
        };
        std::erase_if(joinsToDo, removeExtraneous);
        newJoins = JoinMap<size_t, size_t>();
    }
};

template<typename DistType, typename BlockContainer>
JoinMap<size_t, size_t> InitializeJoinMap(const BlockContainer& blockUpdateContexts,
                                          const ComparisonMap<size_t, size_t>& comparisonMap,
                                          const NodeTracker& nodesJoined){
    JoinMap<size_t, size_t> joinMap;
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



template<typename DistType>
int UpdateBlocks(BlockUpdateContext<DistType>& blockLHS,
                 BlockUpdateContext<DistType>& blockRHS,
                 CachingFunctor<DistType>& cachingFunctor){


    //bool doRHSJoin = blockRHS.joinsToDo.find(blockLHS.queryContext.blockNumber) != blockRHS.joinsToDo.end();

    int graphUpdates(0);

    //if(doRHSJoin){
        blockLHS.blockJoinTracker[blockRHS.queryContext.blockNumber] = true;
        
        cachingFunctor.SetBlocks(blockLHS.queryContext.blockNumber, blockRHS.queryContext.blockNumber);
        
        //auto cachingDistanceFunctor = blockLHS.queryContext.defaultQueryFunctor.CachingFunctor(distanceCache);
        //auto cachedDistanceFunctor = blockRHS.queryContext.defaultQueryFunctor.CachedFunctor(distanceCache);
        

        JoinResults<size_t, DistType> blockLHSUpdates = BlockwiseJoin(blockLHS.joinsToDo[blockRHS.queryContext.blockNumber],
                                                                            blockLHS.currentGraph,
                                                                            blockLHS.joinPropagation,
                                                                            blockRHS.queryContext,
                                                                            cachingFunctor);
        NewJoinQueues<float>(blockLHSUpdates, blockLHS.blockJoinTracker, blockRHS.currentGraph, blockLHS.newJoins);

        
        
        blockRHS.blockJoinTracker[blockLHS.queryContext.blockNumber] = true;
        /*
        if(blockRHS.joinsToDo.find(blockLHS.queryContext.blockNumber) == blockRHS.joinsToDo.end()){
            if (blockLHSUpdates.size() == 0) return 0;
            size_t lhsBlockNum = blockLHS.queryContext.blockNumber;
            for (const auto& result: blockLHSUpdates){
                for (const auto& newNeighbor: result.second){
                    auto findItr = std::find(blockRHS.joinsToDo[lhsBlockNum][result.first].begin(), 
                                            blockRHS.joinsToDo[lhsBlockNum][result.first].end(),
                                            newNeighbor.first);
                    if (findItr == blockRHS.joinsToDo[lhsBlockNum][result.first].end()) {
                        
                        blockRHS.joinsToDo[blockLHS.queryContext.blockNumber][newNeighbor.first].push_back(result.first);
                    }
                }
            }

        }
        */
        
        cachingFunctor.metricFunctor.SetBlocks(blockRHS.queryContext.blockNumber, blockLHS.queryContext.blockNumber);

        ReverseBlockJoin(blockRHS.joinsToDo[blockLHS.queryContext.blockNumber],
                            blockRHS.currentGraph,
                            blockRHS.joinPropagation,
                            blockLHS.queryContext,
                            cachingFunctor,
                            cachingFunctor.metricFunctor);
        
        for(size_t i = 0; auto& vertex: cachingFunctor.reverseGraph){
            NeighborOverDist<DistType> comparison(blockRHS.currentGraph[i][0].second);
            vertex.erase(std::remove_if(vertex.begin(),
                                        vertex.end(),
                                        comparison),
                        vertex.end());
        }
        NewJoinQueues<float>(cachingFunctor.reverseGraph, blockRHS.blockJoinTracker, blockLHS.currentGraph, blockRHS.newJoins);

        for (size_t i = 0; auto& result: cachingFunctor.reverseGraph){
            graphUpdates += ConsumeVertex(blockRHS.currentGraph[i], result, blockLHS.queryContext.blockNumber);
            i++;
        }
        for (auto& result: blockLHSUpdates){
            graphUpdates += ConsumeVertex(blockLHS.currentGraph[result.first], result.second, blockRHS.queryContext.blockNumber);
        }
        
        return graphUpdates;
    /*
    } else {
        //This feels like som jank control flow
        blockLHS.blockJoinTracker[blockRHS.queryContext.blockNumber] = true;
        
        cachingFunctor.metricFunctor.SetBlocks(blockLHS.queryContext.blockNumber, blockRHS.queryContext.blockNumber);
        

        JoinResults<size_t, DistType> blockLHSUpdates = BlockwiseJoin(blockLHS.joinsToDo[blockRHS.queryContext.blockNumber],
                                                                            blockLHS.currentGraph,
                                                                            blockLHS.joinPropagation,
                                                                            blockRHS.queryContext,
                                                                            cachingFunctor.metricFunctor);
        NewJoinQueues<float>(blockLHSUpdates, blockLHS.blockJoinTracker, blockRHS.currentGraph, blockLHS.newJoins);

        for (auto& result: blockLHSUpdates){
            graphUpdates += ConsumeVertex(blockLHS.currentGraph[result.first], result.second, blockRHS.queryContext.blockNumber);
        }
        
        return graphUpdates;

    }
    */
}

}

#endif