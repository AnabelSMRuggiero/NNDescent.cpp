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
    size_t dataIndex;
    std::unordered_map<size_t, std::vector<size_t>> searchesToDo;
    
    SearchContext(const size_t numNeighbors, const size_t numBlocks, const size_t dataIndex):
        currentNeighbors(numNeighbors), blocksJoined(numBlocks), dataIndex(dataIndex){};

};



//template<typename BlockNumberType, typename DataIndexType, typename DataEntry, typename DistType>
template<typename DistType, typename QueryFunctor>
GraphVertex<size_t, DistType> BlockwiseSearch(SearchContext<DistType>& searchingPoint,
                   const QueryContext<DistType>& targetBlock,
                   const std::vector<size_t>& hint,
                   QueryFunctor queryFunctor){
    
    
    
    GraphVertex<size_t, DistType> queryHint(targetBlock.queryHint.size());
    
    for(auto index: hint){
        queryHint.push_back({index, std::numeric_limits<DistType>::max()});
    }

    int sizeDif = targetBlock.queryHint.size() - queryHint.size();

    for(int i = 0; i<sizeDif; i+=1){
        if(!std::any_of(hint.begin(), hint.end(), [&](const size_t index){ return index == targetBlock.subGraph[hint[0]][i];})){
             queryHint.push_back({targetBlock.subGraph[hint[0]][i], std::numeric_limits<DistType>::max()});
        }
    }
    
    searchingPoint.blocksJoined[targetBlock.blockNumber] = true;
    queryFunctor.SetBlock(targetBlock.blockNumber);
    targetBlock.Query(queryHint, searchingPoint.dataIndex, queryFunctor);
    size_t resultsAdded = ConsumeVertex(searchingPoint.currentNeighbors, queryHint, targetBlock.blockNumber);
    
    queryHint.resize(resultsAdded);
    
    
    return queryHint;
}

/*
template<typename DistType, typename QueryFunctor, typename DistanceFunctor>
unsigned int BlockwiseSearch(SearchContext<DistType>& searchingPoint,
                   const QueryContext<DistType, DistanceFunctor>& targetBlock,
                   const size_t hint,
                   QueryFunctor queryFunctor){
    
    
    
    GraphVertex<size_t, DistType> queryHint(searchingPoint.currentNeighbors.size());
    queryHint.push_back({hint, std::numeric_limits<DistType>::max()});

    queryFunctor.SetBlock(targetBlock.blockNumber);
    targetBlock.Query(queryHint, searchingPoint.dataIndex, queryFunctor);
    //searchingPoint.blocksJoined[targetBlock.blockNumber] = true;
    size_t resultsAdded = ConsumeVertex(searchingPoint.currentNeighbors, joinResults, targetBlock.blockNumber);
    
    joinResults.resize(resultsAdded);
    
    
    return joinResults;
}
*/


template<typename DistType>
void QueueSearches(const BlockUpdateContext<DistType>& graphFragment,
                   SearchContext<DistType>& searchingPoint,
                   const BlockIndecies searchToQueue,
                   GraphVertex<size_t, DistType>& joinResults,
                   //SearchQueue& searchQueues,
                   const size_t maxNewSearches){
    unsigned int hintsAdded(0);
    for (const auto& result: joinResults){
        for (const auto& resultNeighbor: graphFragment.currentGraph[result.first]){
            if (!searchingPoint.blocksJoined[resultNeighbor.first.blockNumber]){
                std::vector<size_t>& queue = searchingPoint.searchesToDo[resultNeighbor.first.blockNumber];
                auto result = std::find(queue.begin(), queue.end(), resultNeighbor.first.dataIndex);
                if(result == queue.end()) queue.push_back(resultNeighbor.first.dataIndex);
                //searchingPoint.blocksJoined[resultNeighbor.first.blockNumber] = true;
                //searchQueues[resultNeighbor.first.blockNumber].push_back({searchToQueue, resultNeighbor.first.dataIndex});
                hintsAdded++;
            }
        }
        if (hintsAdded >= maxNewSearches) break;
    }
}

template<typename DistType>
GraphVertex<BlockIndecies, DistType> InitialSearch(SinglePointFunctor<DistType>& distFunctor,
                                                   const QueryContext<DistType>& blockToSearch,
                                                   const size_t searchIndex){
    distFunctor.SetBlock(blockToSearch.blockNumber);
    GraphVertex<size_t, DistType> initNeighbors = blockToSearch.queryHint;
    blockToSearch.Query(initNeighbors, searchIndex, distFunctor);

    //context.blocksJoined[blockToSearch.blockNumber] = true;
    return ToBlockIndecies(initNeighbors, blockToSearch.blockNumber);
    // This shouldn't be needed?

}

//template<typename DistType>
std::unordered_map<size_t, std::vector<size_t>> InitialQueue(SearchContext<float>& searchPoint, const Graph<BlockIndecies, float>& graphFragment){
    
    std::unordered_map<size_t, std::vector<size_t>> searchesToDo;

    for (const auto& result: searchPoint.currentNeighbors){
        for (const auto& resultNeighbor: graphFragment[result.first]){
            if (!searchPoint.blocksJoined[resultNeighbor.first.blockNumber]) {
                std::vector<size_t>& queue = searchesToDo[resultNeighbor.first.blockNumber];
                auto result = std::find(queue.begin(), queue.end(), resultNeighbor.first.dataIndex);
                if(result == queue.end()) queue.push_back(resultNeighbor.first.dataIndex);
            }
        }
    }

    return searchesToDo;
}
//searchContexts, searchFunctor,                 blockUpdateContexts,                
//std::vector<std::vector<SearchContext<float>>>  SinglePointFunctor<DistType>&, std::span<const BlockUpdateContexts>
template<typename DistType>
SearchQueue FirstBlockSearch(std::vector<std::vector<SearchContext<DistType>>>& searchContexts,
                             SinglePointFunctor<DistType>& searchFunctor,
                             std::span<BlockUpdateContext<DistType>> blockUpdateContexts){

    SearchQueue searchHints(searchContexts.size());
    


    for (size_t i = 0; auto& testBlock: searchContexts){
            
            
        for (size_t j = 0; auto& context: testBlock){


            context.blocksJoined[i] = true;
            context.currentNeighbors = InitialSearch(searchFunctor, blockUpdateContexts[i].queryContext, context.dataIndex);
            
            
            std::unordered_map<size_t, std::vector<size_t>> searchesToDo = InitialQueue(context, blockUpdateContexts[i].currentGraph);

            for (auto& [blockNum, dataIndecies]: searchesToDo){
                searchHints[blockNum].push_back({{i,j}, std::move(dataIndecies)});
            }
        
            
            j++;
        }
        i++;
    }
    return searchHints;
}

//searchFunctor                 searchHints         searchContexts                              blockUpdateContexts             maxNewSearches
//SinglePointFunctor<float>&    searchQueue& (&&?)  std::vector<std::vector<SearchContext>>&,   std::span<BlockUpdateContext>   size_t
void SearchLoop(SinglePointFunctor<float>& searchFunctor,
                SearchQueue& searchHints,
                std::vector<std::vector<SearchContext<float>>>& searchContexts,
                std::span<BlockUpdateContext<float>> blockUpdateContexts,
                const size_t maxNewSearches){
    size_t searchUpdates = 1;
    while(searchUpdates){
        searchUpdates = 0;
        for (size_t i = 0; auto& hintMap: searchHints){
            for (size_t j = 0; const auto& hint: hintMap){
                searchFunctor.SetBlock(i);

                GraphVertex<size_t, float> newNodes = BlockwiseSearch(searchContexts[hint.first.blockNumber][hint.first.dataIndex],
                                                                                blockUpdateContexts[i].queryContext,
                                                                                hint.second,
                                                                                searchFunctor);
                
                searchUpdates += newNodes.size();

                QueueSearches(blockUpdateContexts[i],
                                searchContexts[hint.first.blockNumber][hint.first.dataIndex],
                                hint.first,
                                newNodes,
                                maxNewSearches);

            }
            hintMap.clear();
            i++;
        }


        for (size_t i = 0; auto& vector: searchContexts){
            for (size_t j = 0; auto& context: vector){
                for (auto& [blockNum, indecies]: context.searchesToDo){
                    if (!context.blocksJoined[blockNum]){
                        searchHints[blockNum].push_back({{i,j}, std::move(indecies)});
                    }
                }
                context.searchesToDo.clear();
                j++;
            }
            i++;
        }
    }
}

}

#endif