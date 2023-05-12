/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_INDEX_HPP
#define NND_INDEX_HPP


#include "Parallelization/ThreadPool.hpp"
#include "ann/AlignedMemory/DynamicArray.hpp"
#include "ann/Data.hpp"


#include "Parallel-Algorithm/FreeFunctions.hpp"
#include "Parallel-Algorithm/ParallelizationObjects.hpp"
#include "RPTrees/Forest.hpp"
#include "RPTrees/SplittingScheme.hpp"

#include "GraphInitialization.hpp"
#include "SubGraphQuerying.hpp"
#include "Type.hpp"

#include <filesystem>
#include <vector>
#include <unordered_map>
#include <unordered_set>


namespace nnd{
struct FragmentMetaData {
    size_t numBlocks;
};

template<typename DistType>
ann::dynamic_array<DataBlock<DistType>> OpenDataBlocks(std::filesystem::path fragmentDirectory) {
    std::ifstream metaDataFile{ fragmentDirectory / "MetaData.bin", std::ios_base::binary };
    FragmentMetaData metadata = Extract<FragmentMetaData>(metaDataFile);

    ann::dynamic_array<DataBlock<DistType>> dataBlocks{metadata.numBlocks};
    

    for (size_t i = 0; i < metadata.numBlocks; i += 1) {
        std::filesystem::path dataBlockPath = fragmentDirectory / ("DataBlock-" + std::to_string(i) + ".bin");
        std::ifstream dataBlockFile{ dataBlockPath, std::ios_base::binary };
        dataBlocks[i] = extract<DataBlock<DistType>>(dataBlockFile, i);
    }
    return dataBlocks;
}

template<typename IndexType, typename DistType>
ann::dynamic_array<QueryContext<IndexType, DistType>> OpenQueryContexts(std::filesystem::path fragmentDirectory) {
    std::ifstream metaDataFile{ fragmentDirectory / "MetaData.bin", std::ios_base::binary };
    FragmentMetaData metadata = Extract<FragmentMetaData>(metaDataFile);

    ann::dynamic_array<QueryContext<IndexType, DistType>> queryContexts{metadata.numBlocks};

    for (size_t i = 0; i < metadata.numBlocks; i += 1) {
        std::filesystem::path dataBlockPath = fragmentDirectory / ("QueryContext-" + std::to_string(i) + ".bin");
        std::ifstream contextFile{ dataBlockPath, std::ios_base::binary };
        queryContexts[i] = extract<QueryContext<IndexType, DistType>>(contextFile);
    }
    return queryContexts;
}

inline ann::dynamic_array<IndexBlock> OpenIndexBlocks(std::filesystem::path fragmentDirectory) {
    std::ifstream metaDataFile{ fragmentDirectory / "MetaData.bin", std::ios_base::binary };
    FragmentMetaData metadata = Extract<FragmentMetaData>(metaDataFile);

    ann::dynamic_array<IndexBlock> index_blocks{metadata.numBlocks};

    for (size_t i = 0; i < metadata.numBlocks; i += 1) {
        std::filesystem::path dataBlockPath = fragmentDirectory / ("IndexBlock-" + std::to_string(i) + ".bin");
        std::ifstream contextFile{ dataBlockPath, std::ios_base::binary };
        index_blocks[i] = extract<IndexBlock>(contextFile);
    }
    return index_blocks;
}

template<typename DistanceType>
struct index {
    ann::dynamic_array<DataBlock<DistanceType>> data_points;
    ann::dynamic_array<QueryContext<DataIndex_t, DistanceType>> query_contexts;
    ann::dynamic_array<IndexBlock> graph_neighbors;
    ann::dynamic_array<ann::dynamic_array<size_t>> block_idx_to_source_idx;

    std::unordered_set<size_t> splits_to_do;
    splitting_vectors<DistanceType> splits;
    std::unordered_map<size_t, size_t> split_idx_to_block_idx;

    erased_unary_binder<DistanceType> distance_metric;

    search_parameters search_params;
};

template<typename DistanceType>
index<DistanceType> open_index(const std::filesystem::path& index_directory){

    auto splits = [&]() {
        std::ifstream vecFile(index_directory / "SplittingVectors.bin", std::ios_base::binary);
        return extract<splitting_vectors<DistanceType>>(vecFile);
    }();

    auto split_idx_to_block_idx = [&]() {
        std::ifstream mappingFile(index_directory / "SplittingIndexToBlockNumber.bin", std::ios_base::binary);
        return extract<std::unordered_map<size_t, size_t>>(mappingFile);
    }();

    auto block_idx_to_source_idx = [&]{
        std::ifstream mappingFile(index_directory / "BlockIndexToSourceIndex.bin", std::ios_base::binary);
        return extract<ann::dynamic_array<ann::dynamic_array<size_t>>>(mappingFile);
    }();

    auto splits_to_do = std::visit([&](const auto& vectors){
        std::unordered_set<size_t> splitting_indicies;
        for (const auto& [key, value] : vectors)
            splitting_indicies.insert(key);

        for (const auto& [key, value] : split_idx_to_block_idx)
            splitting_indicies.erase(key);
        
        return splitting_indicies;
    }, splits);

    return index<DistanceType> {
        .data_points                = OpenDataBlocks<float>(index_directory),
        .query_contexts             = OpenQueryContexts<DataIndex_t, float>(index_directory),
        .graph_neighbors            = OpenIndexBlocks(index_directory),
        .block_idx_to_source_idx    = std::move(block_idx_to_source_idx),
        .splits_to_do               = std::move(splits_to_do),
        .splits                     = std::move(splits),
        .split_idx_to_block_idx     = std::move(split_idx_to_block_idx),
        .distance_metric            = {},
        .search_parameters          = {}
    };
}

template<typename DistType>
ann::dynamic_array<IndexBlock> index_finalization(std::span<BlockUpdateContext<DistType>> blocks){

    ann::dynamic_array<UnevenBlock<BlockIndecies>> index(blocks.size());
    
    std::transform(blocks.begin(), blocks.end(), index.begin(), [&](const auto& block){    

        auto outOfBlock = [&](const auto& neighbor){
            return neighbor.first.graphFragment != block.queryContext.graphFragment ||
                   neighbor.first.blockNumber != block.queryContext.blockNumber;
        };

        std::vector<size_t> filteredSizes(block.currentGraph.size());

        std::transform(block.currentGraph.begin(),
                       block.currentGraph.end(),
                       filteredSizes.begin(),
                       [&](const auto& vertex){

            return std::transform_reduce(vertex.begin(), 
                                         vertex.end(),
                                         std::size_t{0},
                                         std::plus<>{},
                                         outOfBlock);
        });

        size_t totalSize = std::accumulate(filteredSizes.begin(), filteredSizes.end(), size_t{0});

        UnevenBlock<BlockIndecies> graphFragment = UninitUnevenBlock<BlockIndecies>(filteredSizes.size(), totalSize);
        
        size_t* headerStart = static_cast<size_t*>(static_cast<void*>(graphFragment.data()));
        std::inclusive_scan(filteredSizes.begin(), filteredSizes.end(), headerStart+1, std::plus<size_t>{}, 0);
        
        BlockIndecies* indexStart = graphFragment.firstIndex;
        for (const auto& vertex: block.currentGraph){
            for (const auto& neighbor: vertex){
                if (outOfBlock(neighbor)){
                    *indexStart = neighbor.first;
                    ++indexStart;
                }
            }
        }


        return graphFragment;
    });

    return index;
}

template<typename DistType>
void SerializeFragmentIndex(std::span<const DataBlock<DistType>> dataBlocks, std::span<const QueryContext<DataIndex_t, DistType>> graphBlocks, std::span<const IndexBlock> indexBlocks, std::filesystem::path fragmentDirectory){

    FragmentMetaData metadata{dataBlocks.size()};

    std::ofstream metaDataFile{fragmentDirectory / "MetaData.bin", std::ios_base::binary | std::ios_base::trunc};
    serialize(metadata, metaDataFile);

    for (const auto& block: dataBlocks){
        std::filesystem::path dataBlockPath = fragmentDirectory / ("DataBlock-" + std::to_string(block.blockNumber) + ".bin");
        std::ofstream dataBlockFile{dataBlockPath, std::ios_base::binary | std::ios_base::trunc};
        serialize(block, dataBlockFile);
    }


    for (const auto& block: graphBlocks){
        std::filesystem::path queryContextPath = fragmentDirectory / ("QueryContext-" + std::to_string(block.blockNumber) + ".bin");
        std::ofstream contextFile{queryContextPath, std::ios_base::binary | std::ios_base::trunc};
        serialize(block, contextFile);
    }

    for (size_t i = 0; const auto& block: indexBlocks){
        std::filesystem::path indexBlockPath = fragmentDirectory / ("IndexBlock-" + std::to_string(i) + ".bin");
        std::ofstream indexBlockFile{indexBlockPath, std::ios_base::binary | std::ios_base::trunc};
        serialize(block, indexBlockFile);
        i++;
    }
}

template<typename DistanceType>
void SerializeSplittingVectors(const splitting_vectors<DistanceType>& splittingVectors, std::filesystem::path filePath){
    std::ofstream vectorFile{filePath, std::ios_base::binary | std::ios_base::trunc};
    serialize(splittingVectors, vectorFile);
}


template<typename DistanceType, typename DistanceMetric>
nnd::index<DistanceType> build_index(const DataSet<DistanceType>& training_data, const DistanceMetric& distance, std::size_t num_threads, const hyper_parameters& index_parameters = {}){

    using splitting_scheme = pick_parallel_scheme<choose_scheme<DistanceMetric>, DistanceType, ann::aligned_array<DistanceType>>;
    
    auto [rp_trees, splitting_vectors] = BuildRPForest<splitting_scheme>(training_data, index_parameters.split_params, num_threads);
                                            
    ThreadPool<void> block_builder(num_threads);

    auto [index_mappings, data_blocks] = PartitionData<DistanceType>(rp_trees, training_data, block_builder);

    auto candidates = seed_candidates<splitting_scheme>(training_data, index_mappings, num_threads);

    //block_builder.StopThreads();
    
    block_binder euclideanBinder(distance, std::span{std::as_const(data_blocks)}, std::span{std::as_const(data_blocks)});
    erased_binary_binder<DistanceType> distance_metric{euclideanBinder};

    ThreadPool<thread_functors<float>> pool(num_threads, euclideanBinder, index_parameters.split_params.max_tree_size, index_parameters.index_params.block_graph_neighbors);
    pool.StartThreads();
    //ann::dynamic_array<BlockUpdateContext<DistanceType>> blockContextArr = BuildGraph(meta_graph, index_parameters, pool);
    ann::dynamic_array<BlockUpdateContext<DistanceType>> blockContextArr = BuildGraphRedux(candidates, index_parameters, pool);
    pool.StopThreads();
    
    std::span<BlockUpdateContext<DistanceType>> blockUpdateContexts{blockContextArr.data(), data_blocks.size()};
    
    
    ann::dynamic_array<IndexBlock> index = index_finalization(blockUpdateContexts);

    auto splits_to_do = [&splits = splitting_vectors, &mappings = index_mappings]{
        std::unordered_set<size_t> splitting_indicies;
        for (const auto& [key, value] : splits)
            splitting_indicies.insert(key);

        for (const auto& [key, value] : mappings.splitToBlockNum)
            splitting_indicies.erase(key);
        
        return splitting_indicies;
    }();
    //auto move_query_view = blockContextArr | std::views::transform([](auto&& element)->auto&& {return std::move(element).queryContext;});
    return nnd::index<DistanceType>{
        .data_points                = std::move(data_blocks),
        .query_contexts             = blockContextArr | std::views::transform([](auto&& element)->auto&& {return std::move(element).queryContext;}),
        .graph_neighbors            = std::move(index),
        .block_idx_to_source_idx    = std::move(index_mappings).blockIndexToSource,
        .splits_to_do               = std::move(splits_to_do),
        .splits                     = std::move(splitting_vectors),
        .split_idx_to_block_idx     = std::move(index_mappings).splitToBlockNum,
        .distance_metric            = {},
        .search_params          = {}
    };
    
}



template<typename DistanceType, typename DistanceMetric>
nnd::index<DistanceType> build_index(const DataSet<DistanceType>& training_data, const DistanceMetric& distance, const hyper_parameters& index_parameters = {}){
    
    using splitting_scheme = pick_serial_scheme<choose_scheme<DistanceMetric>, DistanceType, ann::aligned_array<DistanceType>>;

    auto [rp_trees, splitting_vectors] = BuildRPForest<splitting_scheme>(training_data, index_parameters.split_params);
                                            
    auto [index_mappings, data_blocks] = PartitionData<DistanceType>(rp_trees, training_data);

    
    block_binder euclideanBinder(distance, std::span{std::as_const(data_blocks)}, std::span{std::as_const(data_blocks)});
    erased_binary_binder<DistanceType> distance_metric{euclideanBinder};

    auto candidates = seed_candidates<splitting_scheme>(training_data, index_mappings);
    //MetaGraph<DistanceType> meta_graph = BuildMetaGraphFragment<DistanceType>(data_blocks, index_parameters.indexParams, 0, EuclideanMetricSet{}, EuclideanCOM<float, float>);
    //fixed_block_binder com_functor(distance, meta_graph.points, std::span{std::as_const(data_blocks)});

    
    //ann::dynamic_array<BlockUpdateContext<DistanceType>> blockContextArr = BuildGraphRedux(meta_graph, distance_metric, index_parameters, erased_unary_binder{com_functor});
    ann::dynamic_array<BlockUpdateContext<DistanceType>> blockContextArr = BuildGraphRedux(candidates, distance_metric, index_parameters);
    std::span<BlockUpdateContext<DistanceType>> blockUpdateContexts{blockContextArr.data(), data_blocks.size()};
    
    
    ann::dynamic_array<IndexBlock> index = index_finalization(blockUpdateContexts);

    auto splits_to_do = [&splits = splitting_vectors, &mappings = index_mappings]{
        std::unordered_set<size_t> splitting_indicies;
        for (const auto& [key, value] : splits)
            splitting_indicies.insert(key);

        for (const auto& [key, value] : mappings.splitToBlockNum)
            splitting_indicies.erase(key);
        
        return splitting_indicies;
    }();
    
    return nnd::index<DistanceType>{
        .data_points                = std::move(data_blocks),
        .query_contexts             = blockContextArr | std::views::transform([](auto&& element)->auto&& {return std::move(element).queryContext;}),
        .graph_neighbors            = std::move(index),
        .block_idx_to_source_idx    = std::move(index_mappings).blockIndexToSource,
        .splits_to_do               = std::move(splits_to_do),
        .splits                     = std::move(splitting_vectors),
        .split_idx_to_block_idx     = std::move(index_mappings).splitToBlockNum,
        .distance_metric            = {},
        .search_params          = {}
    };
    
}

}

#endif