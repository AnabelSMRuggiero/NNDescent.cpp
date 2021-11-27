/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_SEARCHCONTEXTS_HPP
#define NND_SEARCHCONTEXTS_HPP

#include <unordered_set>
#include <unordered_map>
#include <atomic>
#include <optional>
#include <ranges>

#include "GraphStructures/GraphVertex.hpp"

namespace nnd{

using SearchSet = std::unordered_map<BlockNumber_t, std::vector<DataIndex_t>>;

template<typename DistType>
struct SearchResult{
    GraphFragment_t fragmentNumber;
    BlockNumber_t blockNumber;
    GraphVertex<DataIndex_t, DistType> searchResult; 
};

template<typename IndexType, typename DistType>
std::pair<std::vector<IndexType>, std::vector<IndexType>> AddNeighbors(GraphVertex<IndexType, DistType>& vertexToUpdate,
                                                                       SearchResult<DistType>& updates){
    EraseRemove(updates.searchResult, vertexToUpdate.PushThreshold());
    std::unordered_set<IndexType> addedNeighbors;
    std::unordered_set<IndexType> removedNeighbors;
    for (auto& neighbor: updates.searchResult){
        BlockIndecies result{updates.fragmentNumber, updates.blockNumber, neighbor.first};
        addedNeighbors.insert(result);
        removedNeighbors.insert(vertexToUpdate.PushNeighbor({result, neighbor.second}, returnRemovedTag).first);
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
                                                         std::vector<SearchResult<DistType>>& updates){
    std::unordered_set<IndexType> addedNeighbors;
    std::unordered_set<IndexType> removedNeighbors;
    for (auto& update: updates){
        EraseRemove(update.searchResult, vertexToUpdate.PushThreshold());
        for (auto& neighbor: update.searchResult){
            
            BlockIndecies result{update.fragmentNumber, update.blockNumber, neighbor.first};
            addedNeighbors.insert(result);
            removedNeighbors.insert(vertexToUpdate.PushNeighbor({result, neighbor.second}, returnRemovedTag).first);
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

template<typename SearchContext>
void AddComparisons(std::span<const IndexBlock> graphFragments,
                    SearchContext& context,
                    std::vector<BlockIndecies>& addedIndecies){
    for (const auto& index: addedIndecies){
        const IndexBlock& graphFragment = graphFragments[index.blockNumber];
        context.comparisonResults[index] = graphFragment[index.dataIndex];
        
    }

}

template<typename DistType>
struct SearchContext{

    GraphVertex<BlockIndecies, DistType> currentNeighbors;
    NodeTracker blocksJoined;
    size_t dataIndex;
    std::vector<SearchResult<DistType>> searchResults;
    //std::unordered_map<size_t, std::vector<size_t>> searchesToDo;
    std::unordered_map<BlockIndecies, std::span<const BlockIndecies>> comparisonResults;
    bool done;
    size_t numNeighbors;
    //std::unordered_map<size_t, size_t> targetBlocks;
    
    SearchContext(const size_t numNeighbors, const size_t numBlocks, const size_t dataIndex):
        currentNeighbors(numNeighbors), blocksJoined(numBlocks), dataIndex(dataIndex), done(false), numNeighbors(numNeighbors){};

    void AddInitialResult(GraphFragment_t fragment, BlockNumber_t block, GraphVertex<DataIndex_t, DistType>&& result){
        for (const auto& neighbor : result){
            currentNeighbors.push_back({{fragment, block, neighbor.first}, neighbor.second});
        }
        currentNeighbors.JoinPrep();
    }

    void AddSearchResult(GraphFragment_t fragment, BlockNumber_t block, GraphVertex<DataIndex_t, DistType>&& result){
        searchResults.push_back({fragment, block, std::move(result)});
    }

    
    void ConsumeUpdates(std::span<const IndexBlock> graphFragments){
        auto [added, removed] = AddNeighbors(currentNeighbors, searchResults);
        searchResults.clear();

        for (auto& remove: removed){
            comparisonResults.erase(remove);
        }

        AddComparisons(graphFragments, *this, added);
    }
    

    std::optional<SearchSet> NextSearches(const size_t maxQueueDepth){
        if(done) return std::nullopt;
        currentNeighbors.UnPrep();
        SearchSet searchesToDo;

        auto haveComparison = [&](const auto& edge){return this->comparisonResults.contains(edge.first);};

        auto extractComparison = [&](const auto& edge){
            auto span = comparisonResults[edge.first];
            comparisonResults.erase(edge.first);
            return span;
        };

        auto notQueued = [&](const auto& comparison){
            return !blocksJoined[comparison.blockNumber] && 
                    (searchesToDo[comparison.blockNumber].end() ==
                        std::find(searchesToDo[comparison.blockNumber].begin(),
                                  searchesToDo[comparison.blockNumber].end(),
                                  comparison.dataIndex));
        };

        for(const auto& comparison : currentNeighbors
                                 | std::views::take(maxQueueDepth)
                                 | std::views::filter(haveComparison)
                                 | std::views::transform(extractComparison)
                                 | std::views::join
                                 | std::views::filter(notQueued)){
            searchesToDo[comparison.blockNumber].push_back(comparison.dataIndex);
        }
        
        //currentNeighbors.JoinPrep();
        if (searchesToDo.size() == 0){
            done = true;
            return std::nullopt;
        }
        
        for(const auto& [blockNum, ignore]: searchesToDo){
            blocksJoined[blockNum] = true;
        }
        return searchesToDo;
    }

};


template<typename DistType>
struct ParallelSearchContext{

    GraphVertex<BlockIndecies, DistType> currentNeighbors;
    NodeTracker blocksJoined;
    size_t dataIndex;
    std::vector<SearchResult<DistType>> searchResults;
    
    //std::unordered_map<size_t, std::vector<size_t>> searchesToDo;
    std::unordered_map<BlockIndecies, std::span<const BlockIndecies>> comparisonResults;
    bool done;
    //std::unordered_map<size_t, size_t> targetBlocks;

    ParallelSearchContext() = default;
    
    ParallelSearchContext(const size_t numNeighbors, const size_t numBlocks, const size_t dataIndex):
        currentNeighbors(numNeighbors), blocksJoined(numBlocks), dataIndex(dataIndex), done(false){};

    ParallelSearchContext(ParallelSearchContext&&) = default;

    ParallelSearchContext& operator=(ParallelSearchContext&&) = default;

    void AddInitialResult(GraphFragment_t fragment, BlockNumber_t block, GraphVertex<DataIndex_t, DistType>&& result){
        //searchResults.push_back({fragment, block, std::move(result)});
        for (const auto& neighbor : result){
            currentNeighbors.push_back({{fragment, block, neighbor.first}, neighbor.second});
        }
        currentNeighbors.JoinPrep();
    }


    bool AddSearchResult(GraphFragment_t fragment, BlockNumber_t block, GraphVertex<DataIndex_t, DistType>&& result){
        size_t index = resultIndex.fetch_add(1);
        searchResults[index] = {fragment, block, std::move(result)};
        return resultsAdded.fetch_add(1)+1 == searchResults.size();
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
        //currentNeighbors.UnPrep();
        SearchSet searchesToDo;
        auto haveComparison = [&](const auto& edge){return this->comparisonResults.contains(edge.first);};

        auto extractComparison = [&](const auto& edge){
            auto span = comparisonResults[edge.first];
            comparisonResults.erase(edge.first);
            return span;
        };

        auto notQueued = [&](const auto& comparison){
            return !blocksJoined[comparison.blockNumber] && 
                    (searchesToDo[comparison.blockNumber].end() ==
                        std::find(searchesToDo[comparison.blockNumber].begin(),
                                  searchesToDo[comparison.blockNumber].end(),
                                  comparison.dataIndex));
        };

        for(const auto& comparison : currentNeighbors
                                 | std::views::take(maxQueueDepth)
                                 | std::views::filter(haveComparison)
                                 | std::views::transform(extractComparison)
                                 | std::views::join
                                 | std::views::filter(notQueued)){
            searchesToDo[comparison.blockNumber].push_back(comparison.dataIndex);
        }
        
        //currentNeighbors.JoinPrep();
        if (searchesToDo.size() == 0){
            done = true;
            return std::nullopt;
        }
        

        for(const auto& [blockNum, ignore]: searchesToDo){
            blocksJoined[blockNum] = true;
        }
        searchResults.resize(searchesToDo.size());
        resultIndex = 0;
        resultsAdded = 0;
        return searchesToDo;
    }
    private:
    std::atomic<size_t> resultIndex;
    std::atomic<size_t> resultsAdded;
};

}

#endif