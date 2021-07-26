#ifndef NND_BLOCKWISEALGORITHM_HPP
#define NND_BLOCKWISEALGORITHM_HPP


#include "GraphStructures.hpp"
#include "SubGraphQuerying.hpp"

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
                   const UndirectedGraph<size_t>& searchSubgraph,
                   const QueryContext<DistType>& targetBlock,
                   QueryFunctor queryFunctor){
    
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
        for (const auto& joinHint: joinHints){
            //GraphVertex<size_t, DistType> joinResult = targetBlock || QueryPoint{joinHint.second, blockData[joinHint.first]};
            //const QueryPoint<size_t, DataEntry, DistType> query(joinHint.second, blockData[joinHint.first], joinHint.first);
            joinResults.push_back({joinHint.first, targetBlock.QueryHotPath(joinHint.second, joinHint.first, queryFunctor)});
            nodesJoined[joinHint.first] = true;
        }
        std::vector<std::pair<size_t, GraphVertex<size_t, DistType>>> newJoins;
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
                    if(!nodesJoined[leafNeighbor]){
                        newJoins.push_back({leafNeighbor, result.second});
                        //We can add these to nodesJoined a touch early to prevent dupes
                        nodesJoined[leafNeighbor] = true;
                    }
                }
                retResults.push_back({result.first, std::move(updatedResult)});
            }
            //result.second = updatedResult;
        }
        joinHints = std::move(newJoins);
    }
    return retResults;
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
struct BlockUpdateContext {

    

    NodeTracker blockJoinTracker;
    JoinMap<size_t, size_t> joinsToDo;
    JoinMap<size_t, size_t> newJoins;
    QueryContext<DistType> queryContext;
    Graph<BlockIndecies, DistType> currentGraph;


    BlockUpdateContext(const Graph<size_t, DistType>& blockGraph, QueryContext<DistType>&& queryContext, const size_t numberOfBlocksToJoin):
        queryContext(std::move(queryContext)),
        currentGraph(blockGraph.size(), blockGraph[0].size()),
        blockJoinTracker(numberOfBlocksToJoin){
            /*
            for(size_t i = 0; auto& vertex: currentGraph){
                for (const auto& neighbor: leafGraph[i]){
                    vertex.push_back({{dataBlock.blockNumber, neighbor.first}, neighbor.second});
                }
                i++;
            }
            */
    }

    void SetNextJoins(){
        joinsToDo = std::move(newJoins);
        newJoins = JoinMap<size_t, size_t>();
    }
};

template<typename DistType>
JoinMap<size_t, size_t> InitializeJoinMap(const std::vector<BlockUpdateContext<DistType>>& blockUpdateContexts,
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
                 BlockUpdateContext<DistType>& blockRHS){
/*(JoinHints<DataIndexType>&& joinsToDo,
                  NodeTracker& blockJoinTracker,
                  Graph<BlockIndecies, DistType>& updatedBlockGraph, 
                  const Graph<BlockIndecies, DistType>& targetBlockGraph,
                  const DataBlock<DataEntry>& targetDataBlock,
                  const BlockNumberType targetBlockNumber,
                  QueryContext<DataIndexType, DataEntry, DistType>& targetContext){
*/
    
        
    //JoinMap<size_t, size_t> LHSNewJoinHints;
    
    //JoinMap<size_t, size_t> RHSNewJoinHints;

    bool doRHSJoin = blockRHS.joinsToDo.find(blockLHS.dataBlock.blockNumber) != blockRHS.joinsToDo.end();

    int graphUpdates(0);

    if(doRHSJoin){
        blockLHS.blockJoinTracker[blockRHS.dataBlock.blockNumber] = true;

        DistanceCache<DistType> distanceCache;
        
        auto cachingDistanceFunctor = blockRHS.queryContext.defaultQueryFunctor.CachingFunctor(distanceCache);
        auto cachedDistanceFunctor = blockLHS.queryContext.defaultQueryFunctor.CachedFunctor(distanceCache);
        

        JoinResults<size_t, DistType> blockLHSUpdates = BlockwiseJoin(blockLHS.joinsToDo[blockRHS.dataBlock.blockNumber],
                                                                            blockLHS.currentGraph,
                                                                            blockLHS.leafGraph,
                                                                            blockRHS.queryContext,
                                                                            cachingDistanceFunctor);
        NewJoinQueues<size_t, size_t, float>(blockLHSUpdates, blockLHS.blockJoinTracker, blockRHS.currentGraph, blockLHS.newJoins);

        

        blockRHS.blockJoinTracker[blockLHS.dataBlock.blockNumber] = true;

        JoinResults<size_t, DistType> blockRHSUpdates = BlockwiseJoin(blockRHS.joinsToDo[blockLHS.dataBlock.blockNumber],
                                                                            blockRHS.currentGraph,
                                                                            blockRHS.leafGraph,
                                                                            blockLHS.queryContext,
                                                                            cachedDistanceFunctor);

        NewJoinQueues<size_t, size_t, float>(blockRHSUpdates, blockRHS.blockJoinTracker, blockLHS.currentGraph, blockRHS.newJoins);

        for (auto& result: blockRHSUpdates){
            graphUpdates += ConsumeVertex(blockRHS.currentGraph[result.first], result.second, blockLHS.dataBlock.blockNumber);
        }
        for (auto& result: blockLHSUpdates){
            graphUpdates += ConsumeVertex(blockLHS.currentGraph[result.first], result.second, blockRHS.dataBlock.blockNumber);
        }
        
        return graphUpdates;

    } else {
        //This feels like som jank control flow
        blockLHS.blockJoinTracker[blockRHS.dataBlock.blockNumber] = true;
        
        

        JoinResults<size_t, DistType> blockLHSUpdates = BlockwiseJoin(blockLHS.joinsToDo[blockRHS.dataBlock.blockNumber],
                                                                            blockLHS.currentGraph,
                                                                            blockLHS.leafGraph,
                                                                            blockRHS.queryContext,
                                                                            blockRHS.queryContext.defaultQueryFunctor);
        NewJoinQueues<size_t, size_t, float>(blockLHSUpdates, blockLHS.blockJoinTracker, blockRHS.currentGraph, blockLHS.newJoins);

        for (auto& result: blockLHSUpdates){
            graphUpdates += ConsumeVertex(blockLHS.currentGraph[result.first], result.second, blockRHS.dataBlock.blockNumber);
        }
        
        return graphUpdates;

    }
}

}

#endif