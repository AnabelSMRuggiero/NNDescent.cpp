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
#include <algorithm>
#include <execution>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <span>

#include "GraphStructures.hpp"
#include "BlockwiseAlgorithm.hpp"
#include "Type.hpp"

#include "../Utilities/Data.hpp"
#include "../Utilities/Metrics/FunctorErasure.hpp"

namespace nnd{

using SearchQueue = std::vector<std::vector<std::pair<BlockIndecies, std::vector<size_t>>>>;
using SearchSet = std::unordered_map<size_t, std::vector<size_t>>;

template<typename IndexType, typename DistType>
std::pair<std::vector<IndexType>, std::vector<IndexType>> AddNeighbors(GraphVertex<IndexType, DistType>& vertexToUpdate,
                                                         GraphVertex<IndexType, DistType>& updates){
    EraseRemove(updates, vertexToUpdate.PushThreshold());
    std::unordered_set<IndexType> addedNeighbors;
    std::unordered_set<IndexType> removedNeighbors;
    for (auto& neighbor: updates){
        addedNeighbors.insert(neighbor.first);
        removedNeighbors.insert(vertexToUpdate.PushNeighbor(neighbor, returnRemovedTag).first);
    }
    std::vector<IndexType> flattenedAdds;
    for(auto& neighbor : addedNeighbors){
        if (removedNeighbors.contains(neighbor)) removedNeighbors.erase(neighbor);
        else flattenedAdds.push_back(neighbor);
    }
    std::vector<IndexType> flattenedRemoves;
    for (auto& neighbor : removedNeighbors){
        flattenedRemoves.push_back(neighbor);
    }

    return {std::move(flattenedAdds), std::move(flattenedRemoves)};
}

template<typename IndexType, typename DistType>
std::pair<std::vector<IndexType>, std::vector<IndexType>> AddNeighbors(GraphVertex<IndexType, DistType>& vertexToUpdate,
                                                         std::vector<GraphVertex<IndexType, DistType>>& updates){
    std::unordered_set<IndexType> addedNeighbors;
    std::unordered_set<IndexType> removedNeighbors;
    for (auto& update: updates){
        EraseRemove(update, vertexToUpdate.PushThreshold());
        for (auto& neighbor: update){
            addedNeighbors.insert(neighbor.first);
            removedNeighbors.insert(vertexToUpdate.PushNeighbor(neighbor, returnRemovedTag).first);
        }
    }
    std::vector<IndexType> flattenedAdds;
    for(auto& neighbor : addedNeighbors){
        if (removedNeighbors.contains(neighbor)) removedNeighbors.erase(neighbor);
        else flattenedAdds.push_back(neighbor);
    }
    std::vector<IndexType> flattenedRemoves;
    for (auto& neighbor : removedNeighbors){
        flattenedRemoves.push_back(neighbor);
    }

    return {std::move(flattenedAdds), std::move(flattenedRemoves)};
}

template<typename DistType>
struct SearchContext{

    GraphVertex<BlockIndecies, DistType> currentNeighbors;
    NodeTracker blocksJoined;
    size_t dataIndex;
    std::vector<GraphVertex<BlockIndecies, DistType>> searchResults;
    //std::unordered_map<size_t, std::vector<size_t>> searchesToDo;
    std::unordered_map<BlockIndecies, std::vector<BlockIndecies>> comparisonResults;
    bool done;
    //std::unordered_map<size_t, size_t> targetBlocks;
    
    SearchContext(const size_t numNeighbors, const size_t numBlocks, const size_t dataIndex):
        currentNeighbors(numNeighbors), blocksJoined(numBlocks), dataIndex(dataIndex), done(false){};


    void AddSearchResult(GraphVertex<BlockIndecies, DistType>&& result){
        searchResults.push_back(std::move(result));
    }

    
    void ConsumeUpdates(std::span<const IndexBlock> graphFragments){
        auto [added, removed] = AddNeighbors(currentNeighbors, searchResults);
        searchResults.clear();
        //searchUpdates += newNodes.size();
        for (auto& remove: removed){
            comparisonResults.erase(remove);
        }

        AddComparisons(graphFragments, *this, added);
    }
    

    std::optional<SearchSet> NextSearches(const size_t maxQueueDepth){
        if(done) return std::nullopt;
        currentNeighbors.UnPrep();
        SearchSet searchesToDo;
        for (size_t i = 0; auto& neighbor: currentNeighbors){
            if (comparisonResults.contains(neighbor.first)){
                
                for (auto& comparison: comparisonResults[neighbor.first]){
                    if(!blocksJoined[comparison.blockNumber]){
                    
                    auto result = std::find(searchesToDo[comparison.blockNumber].begin(),
                                            searchesToDo[comparison.blockNumber].end(),
                                            comparison.dataIndex);
                    if(result == searchesToDo[comparison.blockNumber].end())
                        searchesToDo[comparison.blockNumber].push_back(comparison.dataIndex);
                    }
                }
                comparisonResults.erase(neighbor.first);
                //if (searchesToDo.size() == 0) continue;
                
                //currentNeighbors.JoinPrep();
                //return searchesToDo;
            }
            if(i == maxQueueDepth) break;
            i++;
        }
        currentNeighbors.JoinPrep();
        if (searchesToDo.size() > 0){
            for(const auto& [blockNum, ignore]: searchesToDo){
                blocksJoined[blockNum] = true;
            }
            return searchesToDo;
        }
        done = true;
        return std::nullopt;
    }

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
    //size_t resultsAdded = ConsumeVertex(searchingPoint.currentNeighbors, queryHint, targetBlock.blockNumber);
    
    //queryHint.resize(resultsAdded);
    
    
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

/*
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
*/
template<typename DistType>
void AddComparisons(const Graph<BlockIndecies, DistType>& graphFragment,
                    SearchContext<DistType>& context,
                    std::vector<BlockIndecies>& addedIndecies){
    for (const auto& index: addedIndecies){
        const GraphVertex<BlockIndecies, DistType>& vertex = graphFragment[index];
        for (const auto& neighbor: vertex){
            if(!context.blocksJoined[neighbor.first.blockNumber]){
                context.comparisonResults[index].push_back(neighbor.first);
            }
        }

    }

}

//std::span<BlockUpdateContext<float>> blockUpdateContexts

template<typename DistType>
void AddComparisons(std::span<const IndexBlock> graphFragments,
                    SearchContext<DistType>& context,
                    std::vector<BlockIndecies>& addedIndecies){
    for (const auto& index: addedIndecies){
        const IndexBlock& graphFragment = graphFragments[index.blockNumber];
        const std::vector<BlockIndecies>& vertex = graphFragment[index.dataIndex];
        for (const auto& neighbor: vertex){
            if(!context.blocksJoined[neighbor.blockNumber]){
                context.comparisonResults[index].push_back(neighbor);
            }
        }

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
/*
template<typename DistType>
std::unordered_map<size_t, std::vector<size_t>> InitialQueue(const SearchContext<DistType>& searchPoint,
                                                             const GraphVertex<BlockIndecies, DistType>& vertex,
                                                             const Graph<BlockIndecies, DistType>& graphFragment){
    
    std::unordered_map<size_t, std::vector<size_t>> searchesToDo;

    for (const auto& result: vertex){
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
*/
template<typename DistType>
std::unordered_map<BlockIndecies, std::vector<BlockIndecies>> InitialComparisons(const SearchContext<DistType>& searchPoint,
                                                                                const IndexBlock& graphFragment){
    
    std::unordered_map<BlockIndecies, std::vector<BlockIndecies>> comparisons;

    for (const auto& result: searchPoint.currentNeighbors){
        //const GraphVertex<BlockIndecies, DistType>& neighborNeighbors = graphFragment[result.first];
        //std::vector<BlockIndecies> searchTargets;
        /*
        searchTargets.reserve(neighborNeighbors.size());
        for (const auto& neighbor: neighborNeighbors){
            if (!searchPoint.blocksJoined[neighbor.blockNumber]) searchTargets.push_back(neighbor);
        }
        */
        comparisons[result.first] = graphFragment[result.first.dataIndex];
    }

    return comparisons;
}

template<typename DistType>
void UpdateInitialQueue(std::unordered_map<size_t, std::vector<size_t>>& searchesToDo,
                        const SearchContext<DistType>& searchPoint,
                        const GraphVertex<BlockIndecies, DistType>& vertex,
                        const Graph<BlockIndecies, DistType>& graphFragment){
    
    //std::unordered_map<size_t, std::vector<size_t>> searchesToDo;

    for (const auto& result: vertex){
        for (const auto& resultNeighbor: graphFragment[result.first]){
            if (!searchPoint.blocksJoined[resultNeighbor.first.blockNumber]) {
                std::vector<size_t>& queue = searchesToDo[resultNeighbor.first.blockNumber];
                auto result = std::find(queue.begin(), queue.end(), resultNeighbor.first.dataIndex);
                if(result == queue.end()) queue.push_back(resultNeighbor.first.dataIndex);
            }
        }
    }
}

template<typename DistType>
void UpdateInitialComparisons(std::unordered_map<BlockIndecies, std::vector<BlockIndecies>>& comparisons,
                        const SearchContext<DistType>& searchPoint,
                        const GraphVertex<BlockIndecies, DistType>& vertex,
                        const Graph<BlockIndecies, DistType>& graphFragment){
    
    //std::unordered_map<size_t, std::vector<size_t>> searchesToDo;

    for (const auto& result: vertex){
        const GraphVertex<BlockIndecies, DistType>& neighborNeighbors = graphFragment[result.first];
        std::vector<BlockIndecies> searchTargets;
        searchTargets.reserve(neighborNeighbors.size());
        for (const auto& neighbor: neighborNeighbors){
            if (!searchPoint.blocksJoined[neighbor.first.blockNumber]) searchTargets.push_back(neighbor.first);
        }
        comparisons[result.first] = std::move(searchTargets);
    }

}

//searchContexts, searchFunctor,                 blockUpdateContexts,                
//std::vector<std::vector<SearchContext<float>>>  SinglePointFunctor<DistType>&, std::span<const BlockUpdateContexts>

std::vector<std::vector<size_t>> BlocksToSearch(std::vector<std::vector<SearchContext<float>>>& searchContexts,
                                                const MetaGraph<float>& metaGraph,
                                                const size_t numInitSearches){
    std::vector<std::vector<size_t>> blocksToSearch;
    blocksToSearch.reserve(metaGraph.verticies.size());
    for (const auto& vertex: metaGraph.verticies){
        blocksToSearch.push_back({});
        for (size_t i = 0; i<numInitSearches; i+=1){
            blocksToSearch.back().push_back(vertex[i].first);
        }
    }

    return blocksToSearch;
}




template<typename DistType>
SearchQueue FirstBlockSearch(std::vector<std::vector<SearchContext<DistType>>>& searchContexts,
                             const std::vector<std::vector<size_t>>& blocksToSearch,
                             SinglePointFunctor<DistType>& searchFunctor,
                             std::span<BlockUpdateContext<DistType>> blockUpdateContexts,
                             std::span<const IndexBlock> indexBlocks,
                             const size_t maxNewSearches){

    SearchQueue searchHints(searchContexts.size());
    
    //size_t extraNeighborsAdded(0);

    for (size_t i = 0; auto& testBlock: searchContexts){
            
            
        for (size_t j = 0; auto& context: testBlock){

            std::vector<GraphVertex<BlockIndecies, DistType>> potentialNeighbors;
            context.blocksJoined[i] = true;
            context.currentNeighbors = InitialSearch(searchFunctor, blockUpdateContexts[i].queryContext, context.dataIndex);
            context.comparisonResults = InitialComparisons(context, indexBlocks[i]);
            
            for(auto& blockNum: blocksToSearch[i]){
                context.blocksJoined[blockNum] = true;
                context.AddSearchResult(InitialSearch(searchFunctor, blockUpdateContexts[blockNum].queryContext, context.dataIndex));
                //EraseRemove(potentialNeighbors.back(), potentialNeighbors.front().PushThreshold());
                //extraNeighborsAdded += potentialNeighbors.back().size();
                /*
                extraNeighborsAdded += ConsumeVertex(
                    context.currentNeighbors,
                    InitialSearch(searchFunctor, blockUpdateContexts[blockNum].queryContext, context.dataIndex)
                );
                */
            }
            context.ConsumeUpdates(indexBlocks);
            /*
            vertex.erase(std::remove_if(vertex.begin(),
                                    vertex.end(),
                                    comparison),
                     vertex.end());
                     */
            //std::unordered_map<BlockIndecies, std::vector<BlockIndecies>> comparisons = InitialComparisons(context, potentialNeighbors.front(), blockUpdateContexts[i].currentGraph);
            /*
            for (size_t k = 0; k<potentialNeighbors.size(); k+=1){
                UpdateInitialComparisons(comparisons, context, potentialNeighbors[k], blockUpdateContexts[blocksToSearch[i][k]].currentGraph);
            }

            //std::vector<std::pair<BlockIndecies, DistType>> removedValues;
            for (size_t k = 0; k<potentialNeighbors.size(); k+=1){
                for(const auto& neighbor: potentialNeighbors[k]){
                    comparisons.erase(potentialNeighbors.front().PushNeighbor(neighbor, returnRemovedTag).first);
                }
            }
            
            context.currentNeighbors = std::move(potentialNeighbors.front());
            context.comparisonResults = std::move(comparisons);
            */
            std::optional<SearchSet> searches = context.NextSearches(maxNewSearches);
            /*
            for (const auto& [ignore, comparison]: context.comparisonResults){
                for (const auto& index: comparison){
                    context.targetBlocks[index.blockNumber] += 1;
                }
            }
            */
            //auto min = std::min_element(std::execution::unseq, context.currentNeighbors.begin(), context.currentNeighbors.end(), NeighborDistanceComparison<BlockIndecies, DistType>);
            /*
            
            */
            if(searches){
                for (auto& [blockNum, dataIndecies]: *searches){
                    searchHints[blockNum].push_back({{i,j}, std::move(dataIndecies)});
                }
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
                std::span<const IndexBlock> indexBlocks,
                const size_t maxNewSearches,
                const size_t searchesToDo){
    size_t doneSearches = 0;
    while(doneSearches<searchesToDo){
        doneSearches = 0;
        for (size_t i = 0; auto& hintMap: searchHints){
            for (size_t j = 0; const auto& hint: hintMap){
                searchFunctor.SetBlock(i);
                SearchContext<float>& context = searchContexts[hint.first.blockNumber][hint.first.dataIndex];
                GraphVertex<size_t, float> newNodes = BlockwiseSearch(context,
                                                                      blockUpdateContexts[i].queryContext,
                                                                      hint.second,
                                                                      searchFunctor);
                
                //EraseRemove(newNodes, context.currentNeighbors[0].second);
                GraphVertex<BlockIndecies, float> convertex = ToBlockIndecies(newNodes, i);
                context.AddSearchResult(std::move(convertex));
                /*
                auto [added, removed] = AddNeighbors(context.currentNeighbors, convertex);
                //searchUpdates += newNodes.size();
                for (auto& remove: removed){
                    context.comparisonResults.erase(remove);
                }

                AddComparisons(blockUpdateContexts[i].currentGraph, context, added);
                */
                /*
                QueueSearches(blockUpdateContexts[i],
                                searchContexts[hint.first.blockNumber][hint.first.dataIndex],
                                hint.first,
                                newNodes,
                                maxNewSearches);
                */
                
            }
            hintMap.clear();
            i++;
        }

        for (size_t i = 0; auto& vector: searchContexts){
            for (size_t j = 0; auto& context: vector){
                context.ConsumeUpdates(indexBlocks);
                std::optional<SearchSet> searches = context.NextSearches(maxNewSearches);
                if(searches){
                    for (auto& [blockNum, dataIndecies]: *searches){
                        searchHints[blockNum].push_back({{i,j}, std::move(dataIndecies)});
                    }
                } else doneSearches++;
                j++;
            }
            i++;
        }

        /*
        for (size_t i = 0; auto& vector: searchContexts){
            for (size_t j = 0; auto& context: vector){
                for (auto& [blockNum, indecies]: context.searchesToDo){
                    if (!context.blocksJoined[blockNum]){
                        //searchHints[blockNum].push_back({{i,j}, std::move(indecies)});
                    }
                }
                context.searchesToDo.clear();
                j++;
            }
            i++;
        }
        */
    }
}

}

#endif