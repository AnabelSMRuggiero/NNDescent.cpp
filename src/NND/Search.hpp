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

#include <new>
#include <variant>
#include <vector>
#include <algorithm>
#include <execution>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <span>

#include "BlockwiseAlgorithm.hpp"
#include "GraphStructures.hpp"
#include "Index.hpp"
#include "MetaGraph.hpp"
#include "NND/MetricHelpers.hpp"
#include "RPTrees/Forest.hpp"
#include "RPTrees/SplittingScheme.hpp"
#include "SearchContexts.hpp"
#include "Type.hpp"
#include "Parallelization/AsyncQueue.hpp"
#include "Parallelization/ThreadPool.hpp"

#include "ann/Data.hpp"
#include "FunctorErasure.hpp"

namespace nnd{

using SearchQueue = std::vector<std::vector<std::pair<BlockIndecies, std::vector<DataIndex_t>>>>;
using QueueView = std::span<std::vector<std::pair<BlockIndecies, std::vector<DataIndex_t>>>>;



template<typename DistType>
GraphVertex<DataIndex_t, DistType> InitialSearch(erased_unary_binder<DistType> distFunctor,
                                                   const QueryContext<DataIndex_t, DistType>& blockToSearch,
                                                   const size_t searchIndex){

    
    return blockToSearch.Query(std::vector<DataIndex_t>{}, searchIndex, distFunctor(blockToSearch.blockNumber));

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

inline std::vector<std::vector<size_t>> QueueSearches(const DataBlock<BlockNumber_t, std::align_val_t{alignof(BlockNumber_t)}>& blocksToSearch, const size_t numberOfBlocks){

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
                             const erased_unary_binder<DistType>& searchFunctor,
                             std::span<const QueryContext<DataIndex_t, DistType>> queryContexts,
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
void SingleSearch(std::span<const QueryContext<DataIndex_t, DistType>> queryContexts,
                  std::span<const IndexBlock> indexBlocks,
                  SearchContext<DistType>& context,
                  const GraphFragment_t currentFragment,
                  const BlockNumber_t startBlock,
                  erased_unary_binder<DistType> searchFunctor,
                  const size_t maxNewSearches){

    context.blocksJoined[startBlock] = true;
    context.AddInitialResult(currentFragment,
                             startBlock,
                             InitialSearch(searchFunctor, queryContexts[startBlock], context.dataIndex));
    
    context.comparisonResults = InitialComparisons(context, indexBlocks[startBlock]);
    
    
    std::optional<SearchSet> searches = context.NextSearches(maxNewSearches);

    while(searches){
        for (const auto& search: *searches){
            
            
            context.AddSearchResult(queryContexts[search.first].graphFragment,
                                    search.first,
                                    queryContexts[search.first].Query(search.second, context.dataIndex, searchFunctor(search.first)));
        }

        context.ConsumeUpdates(indexBlocks);
        searches = context.NextSearches(maxNewSearches);
    }
}

template<typename DistType>
struct InitialSearchTask{
    std::span<const QueryContext<DataIndex_t, DistType>> queryContexts;
    std::span<const IndexBlock> indexBlocks;
    //const std::vector<std::vector<size_t>> blocksToSearch;
    const size_t maxNewSearches;

    AsyncQueue<std::pair<BlockIndecies, SearchSet>> searchesToQueue;

    void operator()(std::span<ParallelContextBlock<DistType>> searchContexts, ThreadPool<erased_unary_binder<DistType>>& pool){
        auto blockSearchGenerator = [&](const size_t i, ParallelContextBlock<DistType>& testBlock)->auto{
            auto blockSearchTask = [&, i](erased_unary_binder<DistType> searchFunctor) mutable -> void{
                
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
                             std::span<ParallelContextBlock<DistType>> searchContexts,         
                             ThreadPool<erased_unary_binder<DistType>>& pool){

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
void SearchLoop(const erased_unary_binder<DistType>& searchFunctor,
                QueueView searchHints,
                std::vector<ContextBlock<DistType>>& searchContexts,
                std::span<const QueryContext<DataIndex_t, DistType>> queryContexts,
                std::span<const IndexBlock> indexBlocks,
                const size_t maxNewSearches,
                const size_t searchesToDo){
    size_t doneSearches = 0;
    while(doneSearches<searchesToDo){
        doneSearches = 0;
        for (size_t i = 0; auto& hintMap: searchHints){
            for (size_t j = 0; const auto& hint: hintMap){

                SearchContext<DistType>& context = searchContexts[hint.first.blockNumber].second[hint.first.dataIndex];
                
                context.AddSearchResult(queryContexts[i].graphFragment,
                                        i,
                                        queryContexts[i].Query(hint.second, context.dataIndex, searchFunctor(i)));

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
void ParaSearchLoop(ThreadPool<erased_unary_binder<DistType>>& pool,
                QueueView searchHints,
                std::span<ParallelContextBlock<DistType>> searchContexts,
                std::span<const QueryContext<DataIndex_t, DistType>> queryContexts,
                std::span<const IndexBlock> indexBlocks,
                const size_t maxNewSearches,
                const size_t searchesToDo){

    AsyncQueue<BlockIndecies> searchesToUpdate;
    AsyncQueue<std::pair<BlockIndecies, SearchSet>> searchesToQueue;



    
    
    auto searchGenerator = [&](const size_t i, std::vector<std::pair<BlockIndecies, std::vector<DataIndex_t>>>&& searchSet)->auto{

        auto searchTask = [&, i, hintMap = std::move(searchSet)](erased_unary_binder<DistType>& searchFunctor)->void{
            for (size_t j = 0; const auto& hint: hintMap){
                
                ParallelSearchContext<DistType>& context = searchContexts[hint.first.blockNumber].second[hint.first.dataIndex];
                GraphVertex<DataIndex_t, DistType> newNodes = queryContexts[i].Query(hint.second, context.dataIndex, searchFunctor(i));

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
            (erased_unary_binder<DistType>&)->void{
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

struct search_set_mappings {
    std::vector<ParallelContextBlock<float>> contexts;
    IndexMaps<size_t> mappings;
};
template<typename DistanceType>
search_set_mappings block_search_set(
    const DataSet<DistanceType>& searchSet, const RandomProjectionForest& searchForest, const SearchParameters& searchParams,
    std::size_t num_index_blocks, const std::unordered_map<unsigned long, unsigned long>& splitToBlockNum) {

    DataMapper<DistanceType, void, void> testMapper(searchSet);
    std::vector<ParallelContextBlock<DistanceType>> searchContexts;

    auto searcherConstructor = [&, &splitToBlockNum = splitToBlockNum](size_t splittingIndex, std::span<const size_t> indicies) -> void {
        ParallelContextBlock<DistanceType> retBlock{ splitToBlockNum.at(splittingIndex),
                                              std::vector<ParallelSearchContext<float>>(indicies.size()) };

        for (size_t i = 0; size_t index : indicies) {

            ParallelSearchContext<DistanceType>* contextPtr = &retBlock.second[i];
            contextPtr->~ParallelSearchContext<DistanceType>();
            new (contextPtr) ParallelSearchContext<DistanceType>(searchParams.searchNeighbors, num_index_blocks, index);

            i++;
        }
        searchContexts.push_back(std::move(retBlock));
        testMapper(splittingIndex, indicies);
    };

    CrawlTerminalLeaves(searchForest, searcherConstructor);


    IndexMaps<size_t> testMappings = { std::move(testMapper.splitToBlockNum),
                                       std::move(testMapper.blockIndexToSource),
                                       std::move(testMapper.sourceToBlockIndex),
                                       std::move(testMapper.sourceToSplitIndex) };

    return { std::move(searchContexts), std::move(testMappings) };
}
template< typename DistanceType >
void ParallelSearch(
    ThreadPool<erased_unary_binder<DistanceType>>& threadPool, std::size_t max_searches_queued,
    std::span<ParallelContextBlock<DistanceType>> searchContexts, std::span<const QueryContext<DataIndex_t, DistanceType>> queryContexts,
    std::span<const IndexBlock> indexView) {

    InitialSearchTask<DistanceType> searchGenerator = {
        queryContexts, indexView, max_searches_queued, AsyncQueue<std::pair<BlockIndecies, SearchSet>>()
    };

    threadPool.StartThreads();
    SearchQueue searchHints = ParaFirstBlockSearch(searchGenerator, searchContexts, threadPool);

    QueueView hintView = { searchHints.data(), searchHints.size() };

    ParaSearchLoop(threadPool, hintView, searchContexts, queryContexts, indexView, max_searches_queued, searchContexts.size());
    threadPool.StopThreads();
}

template<typename DistanceType>
constexpr auto select_parallel_transform = []<typename VectorMap>(const DataSet<DistanceType>& search_data,
                                                                  const std::unordered_set<size_t>& splits_to_do,
                                                                  auto num_threads, 
                                                                  const VectorMap& splitting_vectors){

    constexpr splitting_scheme scheme = std::same_as<VectorMap, euclidean_splitting_vectors<DistanceType>> 
                                        ? splitting_scheme::euclidean
                                        : splitting_scheme::angular;

    using splitting_scheme = borrowed_splitting_scheme<scheme, DistanceType, ann::aligned_array<DistanceType>>;

    return RPTransformData(search_data.size(),
                           splits_to_do,
                           splitting_scheme::bind(search_data, splitting_vectors),
                           num_threads);
    
};

template<typename DistanceType>
std::vector<std::vector<std::size_t>> search(const DataSet<DistanceType>& search_data, const index<DistanceType>& index, std::size_t num_threads){
    

    
    
    auto bound_transform = std::bind_front(select_parallel_transform<DistanceType>, std::ref(search_data), std::ref(index.splits_to_do), num_threads);
    RandomProjectionForest rp_trees = std::visit(bound_transform, index.splits);
    /*
    RandomProjectionForest rpTreesTest = RPTransformData(search_data.size(),
                                                         index.splits_to_do,
                                                         borrowed_euclidean(search_data, index.splitting_vectors),
                                                         num_threads);
    */
    size_t numberSearchBlocks = index.data_points.size();

    
    IndexMaps<size_t> testMappings;

     

    ThreadPool<erased_unary_binder<DistanceType>> searchPool(num_threads, index.distance_metric);

    std::span<const ann::dynamic_array<size_t>> indexMappingView(index.block_idx_to_source_idx);

    auto [searchContexts, mappings] =
        block_search_set(search_data, rp_trees, index.search_parameters, index.graph_neighbors.size(), index.split_idx_to_block_idx);

    ParallelSearch(searchPool, index.search_parameters.maxSearchesQueued, std::span{searchContexts}, as_const_span(index.query_contexts), as_const_span(index.graph_neighbors));

    std::vector<std::vector<size_t>> results(search_data.size());

    for (size_t i = 0; auto& [blockNum, testBlock] : searchContexts) {
        for (size_t j = 0; auto& context : testBlock) {
            GraphVertex<BlockIndecies, float>& result = context.currentNeighbors;
            
            size_t testIndex = context.dataIndex;
            
            for (const auto& neighbor : result) {
                results[testIndex].push_back(indexMappingView[neighbor.first.blockNumber][neighbor.first.dataIndex]);
            }
            j++;
        }
        i++;
    }

    return results;
}

template<typename DistanceType>
constexpr auto select_transform = []<typename VectorMap>(const DataSet<DistanceType>& search_data,
                                                         const std::unordered_set<size_t>& splits_to_do,
                                                         const VectorMap& splitting_vectors){

    constexpr splitting_scheme scheme = std::same_as<VectorMap, euclidean_splitting_vectors<DistanceType>> 
                                        ? splitting_scheme::euclidean
                                        : splitting_scheme::angular;

    using splitting_scheme = borrowed_splitting_scheme<scheme, DistanceType, ann::aligned_array<DistanceType>>;

    return RPTransformData(search_data,
                               splits_to_do,
                               splitting_scheme::bind(search_data, splitting_vectors));
    
};


template< typename DistanceType>
std::vector<std::vector<std::size_t>> search(const DataSet<DistanceType>& search_data, const nnd::index<DistanceType>& index){

    
    //using splitting_scheme = borrowed_splitting_scheme<choose_scheme<Metric>, DistanceType, ann::aligned_array<DistanceType>>;
    

    //fixed_block_binder searchDist(Metric{}, search_data, as_const_span(index.data_points));
    //erased_unary_binder<float> searchFunctor(searchDist);

    std::vector<std::vector<size_t>> results(search_data.size());
    IndexMaps<size_t> testMappings;

    auto bound_transform = std::bind_front(select_transform<DistanceType>, std::ref(search_data), std::ref(index.splits_to_do));
    RandomProjectionForest rp_trees = std::visit(bound_transform, index.splits);
    //RandomProjectionForest rpTreesTest = RPTransformData(search_data, index.splits_to_do, splitting_scheme::bind(search_data, index.splitting_vectors));
    DataMapper<float, void, void> testMapper(search_data);
    std::vector<ContextBlock<float>> searchContexts;
    auto searcherConstructor =
        [&, &splitToBlockNum = index.split_idx_to_block_idx](size_t splittingIndex, std::span<const size_t> indicies) -> void {
        ContextBlock<float> retBlock;
        retBlock.first = splitToBlockNum.at(splittingIndex);
        retBlock.second.reserve(indicies.size());
        for (size_t idx : indicies) {

            retBlock.second.push_back({ index.search_parameters.searchNeighbors, index.data_points.size(), idx });
        }
        searchContexts.push_back(std::move(retBlock));
        testMapper(splittingIndex, indicies);
    };

    CrawlTerminalLeaves(rp_trees, searcherConstructor);


    testMappings = { std::move(testMapper.splitToBlockNum),
                        std::move(testMapper.blockIndexToSource),
                        std::move(testMapper.sourceToBlockIndex),
                        std::move(testMapper.sourceToSplitIndex) };

    std::span<const IndexBlock> indexView{ std::as_const(index.graph_neighbors) };

    SearchQueue searchHints =
        FirstBlockSearch(searchContexts, index.distance_metric, as_const_span(index.query_contexts), indexView, index.search_parameters.maxSearchesQueued);


    QueueView hintView = { searchHints.data(), searchHints.size() };

    SearchLoop(
        index.distance_metric, hintView, searchContexts, as_const_span(index.query_contexts), indexView, index.search_parameters.maxSearchesQueued, search_data.size());



    std::span<const ann::dynamic_array<size_t>> indexMappingView{index.block_idx_to_source_idx};

    for (size_t i = 0; auto& [ignore, testBlock] : searchContexts) {
        for (size_t j = 0; auto& context : testBlock) {
            GraphVertex<BlockIndecies, float>& result = context.currentNeighbors;
            
            size_t testIndex = context.dataIndex;
            
            for (const auto& neighbor : result) {
                results[testIndex].push_back(indexMappingView[neighbor.first.blockNumber][neighbor.first.dataIndex]);
            }
            j++;
        }
        i++;
    }
    return results;
}


}

#endif