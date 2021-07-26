/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_SEARCH_HPP
#define NND_SEARCH_HPP

#include <vector>



#include "GraphStructures.hpp"
#include "BlockwiseAlgorithm.hpp"
#include "Type.hpp"

#include "../Utilities/Data.hpp"

namespace nnd{

template<typename BlockNumberType, typename DataIndexType, typename DataEntry, typename DistType>
struct SearchContext{

    GraphVertex<BlockIndecies, DistType> currentNeighbors;
    NodeTracker blocksJoined;
    const DataEntry& data;
    
    SearchContext(const size_t numNeighbors, const size_t numBlocks, const DataEntry& dataEntry):
        currentNeighbors(numNeighbors), blocksJoined(numBlocks), data(dataEntry){};

};



//template<typename BlockNumberType, typename DataIndexType, typename DataEntry, typename DistType>
template<typename BlockNumberType, typename DataIndexType, typename DataEntry, typename DataView, typename DistType, typename QueryFunctor>
GraphVertex<DataIndexType, DistType> BlockwiseSearch(SearchContext<BlockNumberType, DataIndexType, DataEntry, DistType>& searchingPoint,
                   const QueryContext<BlockNumberType, DataIndexType, DataEntry, DataView, DistType>& targetBlock,
                   const DataIndexType hint,
                   QueryFunctor queryFunctor){
    
    
    
    GraphVertex<DataIndexType, DistType> queryHint;
    
    queryHint.push_back({hint, std::numeric_limits<DistType>::max()});

    //searchingPoint.joinHints.erase(targetBlock.dataBlock.blockNumber);
    

    //std::vector<std::pair<DataIndexType, GraphVertex<DataIndexType, DistType>>> retResults;
    
    GraphVertex<DataIndexType, DistType> joinResults = targetBlock.QueryHotPath(queryHint, searchingPoint.data, 0, queryFunctor);
    searchingPoint.blocksJoined[targetBlock.dataBlock.blockNumber] = true;
    size_t resultsAdded = ConsumeVertex(searchingPoint.currentNeighbors, joinResults, targetBlock.dataBlock.blockNumber);
    
    joinResults.resize(resultsAdded);
    
    
    return joinResults;
}


template<typename BlockNumberType, typename DataIndexType, typename DataEntry, typename DataView, typename DistType>
void QueueSearches(const BlockUpdateContext<BlockNumberType, DataIndexType, DataEntry, DataView, DistType>& graphFragment,
                   SearchContext<BlockNumberType, DataIndexType, DataEntry, DistType>& searchingPoint,
                   const BlockIndecies searchToQueue,
                   GraphVertex<DataIndexType, DistType>& joinResults,
                   std::vector<std::unordered_map<BlockIndecies, BlockNumberType>>& searchQueues){
    for (const auto& result: joinResults){
        for (const auto& resultNeighbor: graphFragment.currentGraph[result.first]){
            if (!searchingPoint.blocksJoined[resultNeighbor.first.blockNumber]) searchQueues[resultNeighbor.first.blockNumber][searchToQueue] = resultNeighbor.first.dataIndex;
        }
    }
}

}

#endif