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
//include <atomic>
//#include <list>

#include "GraphStructures.hpp"
#include "BlockwiseAlgorithm.hpp"
#include "Type.hpp"
#include "SearchContexts.hpp"

#include "../ann/Data.hpp"
#include "../ann/Metrics/FunctorErasure.hpp"

namespace nnd{

using SearchQueue = std::vector<std::vector<std::pair<BlockIndecies, std::vector<DataIndex_t>>>>;
using QueueView = std::span<std::vector<std::pair<BlockIndecies, std::vector<DataIndex_t>>>>;


template<typename SearchContext, typename DistType, typename QueryFunctor>
GraphVertex<DataIndex_t, DistType> BlockwiseSearch(SearchContext& searchingPoint,
                   const QueryContext<DataIndex_t, DistType>& targetBlock,
                   const std::vector<DataIndex_t>& hint,
                   QueryFunctor& queryFunctor){
    
    
    
    queryFunctor.SetBlock(targetBlock.blockNumber);

    return targetBlock.Query(hint, searchingPoint.dataIndex, queryFunctor);
}



template<typename DistType>
GraphVertex<DataIndex_t, DistType> InitialSearch(SinglePointFunctor<DistType>& distFunctor,
                                                   const QueryContext<DataIndex_t, DistType>& blockToSearch,
                                                   const size_t searchIndex){
    distFunctor.SetBlock(blockToSearch.blockNumber);
    
    //GraphVertex<DataIndex_t, DistType> initNeighbors = blockToSearch.Query(std::vector<DataIndex_t>{}, searchIndex, distFunctor);

    
    return blockToSearch.Query(std::vector<DataIndex_t>{}, searchIndex, distFunctor);

}

template<typename SearchContext>
std::unordered_map<BlockIndecies, std::span<const BlockIndecies>> InitialComparisons(const SearchContext& searchPoint,
                                                                                const IndexBlock& graphFragment){
    
    std::unordered_map<BlockIndecies, std::span<const BlockIndecies>> comparisons;

    for (const auto& result: searchPoint.currentNeighbors){
        comparisons[result.first] = graphFragment[result.first.dataIndex];
    }

    return comparisons;
}



/*
template<typename Functor>
DataBlock<BlockNumber_t, alignof(BlockNumber_t)> BlocksToSearch(MetaGraph<float>& metaGraph,
                                                                const size_t numInitSearches,
                                                                const size_t searchSetSize,
                                                                Functor searchFunctor){

    using DistType = float;
    DataBlock<BlockNumber_t, alignof(BlockNumber_t)> blocksToSearch(searchSetSize, numInitSearches, 0);

    metaGraph.queryContext.querySize = std::max(numInitSearches, metaGraph.queryContext.querySearchDepth);
    
    for (size_t i = 0; i<searchSetSize; i+=1){
        //GraphVertex<BlockNumber_t, DistType> initHint;
        GraphVertex<BlockNumber_t, DistType> initHint = metaGraph.queryContext.Query(std::vector<BlockNumber_t>{} , i, searchFunctor);
        for (size_t j = 0; j<numInitSearches; j+=1){
            blocksToSearch[i][j] = initHint[j].first;
        }
        
    }

    return blocksToSearch;
}
*/

std::vector<std::vector<size_t>> QueueSearches(const DataBlock<BlockNumber_t, alignof(BlockNumber_t)>& blocksToSearch, const size_t numberOfBlocks){

    std::vector<std::vector<size_t>> searchesQueued(numberOfBlocks);
    for (size_t i = 0; i<blocksToSearch.size(); i+=1){
        for (size_t j = 0; j<blocksToSearch[0].size(); j+=1){
            searchesQueued[blocksToSearch[i][j]].push_back(i);
        }
    }

    return searchesQueued;
}

template<typename DistType>
using ContextBlock = std::pair<size_t, std::vector<SearchContext<DistType>>>;

template<typename DistType>
using ParallelContextBlock = std::pair<size_t, std::vector<ParallelSearchContext<DistType>>>;

template<typename DistType>
SearchQueue FirstBlockSearch(std::vector<ContextBlock<DistType>>& searchContexts,
                             //const std::vector<std::vector<size_t>>& blocksToSearch,
                             SinglePointFunctor<DistType>& searchFunctor,
                             std::span<QueryContext<DataIndex_t, DistType>> queryContexts,
                             std::span<const IndexBlock> indexBlocks,
                             const size_t maxNewSearches){

    SearchQueue searchHints(indexBlocks.size());
    
    //size_t extraNeighborsAdded(0);

    for (size_t i = 0; auto& [blockNumber, testBlock]: searchContexts){
            
            
        for (size_t j = 0; auto& context: testBlock){

            std::vector<GraphVertex<BlockIndecies, DistType>> potentialNeighbors;
            context.blocksJoined[blockNumber] = true;
            //context.currentNeighbors = InitialSearch(searchFunctor, queryContexts[i], context.dataIndex);
            context.AddInitialResult(queryContexts[blockNumber].graphFragment,
                                     blockNumber,
                                     InitialSearch(searchFunctor, queryContexts[blockNumber], context.dataIndex));
            
            context.comparisonResults = InitialComparisons(context, indexBlocks[i]);
            
            /*
            for(auto& extraBlock: blocksToSearch[blockNumber]){
                context.blocksJoined[extraBlock] = true;
                context.AddSearchResult(InitialSearch(searchFunctor, blockUpdateContexts[extraBlock].queryContext, context.dataIndex));
            }
            */
            //context.ConsumeUpdates(indexBlocks);
            
            std::optional<SearchSet> searches = context.NextSearches(maxNewSearches);
            
            if(searches){
                for (auto& [targetBlock, dataIndecies]: *searches){
                    searchHints[targetBlock].push_back({{0u, i, j}, std::move(dataIndecies)});
                }
            }
            
            
            j++;
        }
        i++;
    }
    return searchHints;
}

template<typename DistType>
void SingleSearch(std::span<QueryContext<DataIndex_t, DistType>> queryContexts,
                  std::span<const IndexBlock> indexBlocks,
                  SearchContext<DistType>& context,
                  const GraphFragment_t currentFragment,
                  const BlockNumber_t startBlock,
                  SinglePointFunctor<DistType>& searchFunctor,
                  const size_t maxNewSearches){

    context.blocksJoined[startBlock] = true;
    context.AddInitialResult(currentFragment,
                             startBlock,
                             InitialSearch(searchFunctor, queryContexts[startBlock], context.dataIndex));
    
    context.comparisonResults = InitialComparisons(context, indexBlocks[startBlock]);
    
    
    std::optional<SearchSet> searches = context.NextSearches(maxNewSearches);

    while(searches){
        for (const auto& search: *searches){
            searchFunctor.SetBlock(search.first);
            
            context.AddSearchResult(queryContexts[search.first].graphFragment,
                                    search.first,
                                    BlockwiseSearch(context,
                                                    queryContexts[search.first],
                                                    search.second,
                                                    searchFunctor));
        }

        context.ConsumeUpdates(indexBlocks);
        searches = context.NextSearches(maxNewSearches);
    }
}

template<typename DistType>
struct InitialSearchTask{
    std::span<QueryContext<DataIndex_t, DistType>> queryContexts;
    std::span<const IndexBlock> indexBlocks;
    //const std::vector<std::vector<size_t>> blocksToSearch;
    const size_t maxNewSearches;

    AsyncQueue<std::pair<BlockIndecies, SearchSet>> searchesToQueue;

    void operator()(std::vector<ParallelContextBlock<DistType>>& searchContexts, ThreadPool<SinglePointFunctor<DistType>>& pool){
        auto blockSearchGenerator = [&](const size_t i, ParallelContextBlock<DistType>& testBlock)->auto{
            auto blockSearchTask = [&, i](SinglePointFunctor<DistType>& searchFunctor) mutable -> void{
                
                BlockNumber_t contextBlockNum = testBlock.first;
                GraphFragment_t graphFragment = this->queryContexts[contextBlockNum].graphFragment;
                size_t j=0;
                for (auto& context: testBlock.second){

                    context.blocksJoined[contextBlockNum] = true;
                    context.AddInitialResult(graphFragment,
                                             contextBlockNum,
                                             InitialSearch(searchFunctor, this->queryContexts[contextBlockNum], context.dataIndex));
                    
                    context.comparisonResults = InitialComparisons(context, this->indexBlocks[contextBlockNum]);
                    
                    /*
                    for(auto& blockNum: this->blocksToSearch[i]){
                        context.blocksJoined[blockNum] = true;
                        context.AddInitialResult(InitialSearch(searchFunctor, this->blockUpdateContexts[blockNum].queryContext, context.dataIndex));
                    }
                    */
                    //context.ConsumeUpdates(this->indexBlocks);
                    
                    std::optional<SearchSet> searches = context.NextSearches(this->maxNewSearches);
                    
                    if(searches){
                        this->searchesToQueue.Put({{0u, i,j}, std::move(*searches)});
                    }
                    
                    
                    j++;
                }
            };

            return blockSearchTask;
        };
        for (size_t i = 0; auto& testBlock: searchContexts){        
            pool.DelegateTask(blockSearchGenerator(i, testBlock));
            i++;
        }
    }

};


template<typename DistType>
SearchQueue ParaFirstBlockSearch(InitialSearchTask<DistType>& searchQueuer,
                             std::vector<ParallelContextBlock<DistType>>& searchContexts,         
                             ThreadPool<SinglePointFunctor<DistType>>& pool){

    SearchQueue searchHints(searchQueuer.indexBlocks.size());
    searchQueuer(searchContexts, pool);

    std::list<std::pair<BlockIndecies, SearchSet>> newSearchSets = searchQueuer.searchesToQueue.TryTakeAll();

    for(auto& [index, searches]: newSearchSets){
        for (auto& [blockNum, dataIndecies]: searches){
            searchHints[blockNum].push_back({index, std::move(dataIndecies)});
        }
    }
    pool.Latch();

    newSearchSets = searchQueuer.searchesToQueue.TryTakeAll();

    for(auto& [index, searches]: newSearchSets){
        for (auto& [blockNum, dataIndecies]: searches){
            searchHints[blockNum].push_back({index, std::move(dataIndecies)});
        }
    }

    return searchHints;
}


template<typename DistType>
void SearchLoop(SinglePointFunctor<DistType>& searchFunctor,
                QueueView searchHints,
                std::vector<ContextBlock<DistType>>& searchContexts,
                std::span<QueryContext<DataIndex_t, DistType>> queryContexts,
                std::span<const IndexBlock> indexBlocks,
                const size_t maxNewSearches,
                const size_t searchesToDo){
    size_t doneSearches = 0;
    while(doneSearches<searchesToDo){
        doneSearches = 0;
        for (size_t i = 0; auto& hintMap: searchHints){
            for (size_t j = 0; const auto& hint: hintMap){
                searchFunctor.SetBlock(i);
                SearchContext<DistType>& context = searchContexts[hint.first.blockNumber].second[hint.first.dataIndex];
                
                context.AddSearchResult(queryContexts[i].graphFragment,
                                        i,
                                        BlockwiseSearch(context,
                                                        queryContexts[i],
                                                        hint.second,
                                                        searchFunctor));

            }
            hintMap.clear();
            i++;
        }

        for (size_t i = 0; auto& [ignore, vector]: searchContexts){
            for (size_t j = 0; auto& context: vector){
                context.ConsumeUpdates(indexBlocks);
                std::optional<SearchSet> searches = context.NextSearches(maxNewSearches);
                if(searches){
                    for (auto& [blockNum, dataIndecies]: *searches){
                        searchHints[blockNum].push_back({{0u,i,j}, std::move(dataIndecies)});
                    }
                } else doneSearches++;
                j++;
            }
            i++;
        }

    }
}

//using QueueView = 

template<typename DistType>
void ParaSearchLoop(ThreadPool<SinglePointFunctor<DistType>>& pool,
                QueueView searchHints,
                std::vector<ParallelContextBlock<DistType>>& searchContexts,
                std::span<QueryContext<DataIndex_t, DistType>> queryContexts,
                std::span<const IndexBlock> indexBlocks,
                const size_t maxNewSearches,
                const size_t searchesToDo){

    AsyncQueue<BlockIndecies> searchesToUpdate;
    AsyncQueue<std::pair<BlockIndecies, SearchSet>> searchesToQueue;



    
    
    auto searchGenerator = [&](const size_t i, std::vector<std::pair<BlockIndecies, std::vector<DataIndex_t>>>&& searchSet)->auto{

        auto searchTask = [&, i, hintMap = std::move(searchSet)](SinglePointFunctor<DistType>& searchFunctor)->void{
            searchFunctor.SetBlock(i);
            for (size_t j = 0; const auto& hint: hintMap){
                
                ParallelSearchContext<DistType>& context = searchContexts[hint.first.blockNumber].second[hint.first.dataIndex];
                GraphVertex<DataIndex_t, DistType> newNodes = BlockwiseSearch(context,
                                                                      queryContexts[i],
                                                                      hint.second,
                                                                      searchFunctor);
                

                if(context.AddSearchResult(queryContexts[i].graphFragment, queryContexts[i].blockNumber, std::move(newNodes))){
                    searchesToUpdate.Put(BlockIndecies(hint.first));
                };
                //j++;
            }
               
        };
        return searchTask;
    };

    auto comparisonGenerator  = [&](BlockIndecies searchIndex)->auto{
            
        auto comparisonTask = [&, &context = searchContexts[searchIndex.blockNumber].second[searchIndex.dataIndex], searchIndex]
            (SinglePointFunctor<DistType>&)->void{
                context.ConsumeUpdates(indexBlocks);
                std::optional<SearchSet> searches = context.NextSearches(maxNewSearches);
                if(searches){
                    searchesToQueue.Put({searchIndex, std::move(*searches)});
                }
        };

        return comparisonTask;
    };
    
    bool latchAndBreak = false;

    while(true){

        for (size_t i = 0; auto& hintMap: searchHints){
            pool.DelegateTask(searchGenerator(i, std::move(hintMap)));
            
            hintMap = std::vector<std::pair<BlockIndecies, std::vector<DataIndex_t>>>();
            i++;
        }
        
        
        std::list<BlockIndecies> searches = searchesToUpdate.TryTakeAll();
        for (BlockIndecies& index : searches){
            pool.DelegateTask(comparisonGenerator(index));
        }

        std::list<std::pair<BlockIndecies, SearchSet>> newSearchesToQueue = searchesToQueue.TryTakeAll();
        for (auto& [index, newSearches]: newSearchesToQueue){
            for (auto& [blockNum, dataIndecies]: newSearches){
                searchHints[blockNum].push_back({index, std::move(dataIndecies)});
            }
        }
        if(latchAndBreak && newSearchesToQueue.size()==0){
            pool.Latch();
            break;
        }
        if(newSearchesToQueue.size() == 0){
            latchAndBreak = true;
        }

    }
}


}

#endif