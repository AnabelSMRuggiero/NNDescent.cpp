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
#include "../Utilities/Metrics/FunctorErasure.hpp"

namespace nnd{

template<typename DistType>
struct SearchContext{

    GraphVertex<BlockIndecies, DistType> currentNeighbors;
    NodeTracker blocksJoined;
    SinglePointFunctor<DistType> distFunctor;
    
    SearchContext(const size_t numNeighbors, const size_t numBlocks, SinglePointFunctor<DistType> distFunctor):
        currentNeighbors(numNeighbors), blocksJoined(numBlocks), distFunctor(distFunctor){};

};



//template<typename BlockNumberType, typename DataIndexType, typename DataEntry, typename DistType>
template<typename DistType, typename QueryFunctor>
GraphVertex<size_t, DistType> BlockwiseSearch(SearchContext<DistType>& searchingPoint,
                   const QueryContext<DistType>& targetBlock,
                   const size_t hint,
                   QueryFunctor queryFunctor){
    
    
    
    GraphVertex<size_t, DistType> queryHint;
    
    queryHint.push_back({hint, std::numeric_limits<DistType>::max()});

    //searchingPoint.joinHints.erase(targetBlock.dataBlock.blockNumber);
    

    //std::vector<std::pair<DataIndexType, GraphVertex<DataIndexType, DistType>>> retResults;
    queryFunctor.SetBlock(targetBlock.blockNumber);
    GraphVertex<size_t, DistType> joinResults = targetBlock.QueryHotPath(queryHint, 0, queryFunctor);
    searchingPoint.blocksJoined[targetBlock.blockNumber] = true;
    size_t resultsAdded = ConsumeVertex(searchingPoint.currentNeighbors, joinResults, targetBlock.blockNumber);
    
    joinResults.resize(resultsAdded);
    
    
    return joinResults;
}


template<typename DistType>
void QueueSearches(const BlockUpdateContext<DistType>& graphFragment,
                   SearchContext<DistType>& searchingPoint,
                   const BlockIndecies searchToQueue,
                   GraphVertex<size_t, DistType>& joinResults,
                   std::vector<std::unordered_map<BlockIndecies, size_t>>& searchQueues){
    for (const auto& result: joinResults){
        for (const auto& resultNeighbor: graphFragment.currentGraph[result.first]){
            if (!searchingPoint.blocksJoined[resultNeighbor.first.blockNumber]) searchQueues[resultNeighbor.first.blockNumber][searchToQueue] = resultNeighbor.first.dataIndex;
        }
    }
}

}

#endif