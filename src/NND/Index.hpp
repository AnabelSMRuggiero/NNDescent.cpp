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

#include "RPTrees/SplittingScheme.hpp"
#include "ann/AlignedMemory/DynamicArray.hpp"
#include "NND/Type.hpp"
#include "ann/Data.hpp"

#include "SubGraphQuerying.hpp"


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
    splitting_vectors splits;
    std::unordered_map<size_t, size_t> split_idx_to_block_idx;

    erased_unary_binder<DistanceType> distance_metric;

    SearchParameters search_parameters;
};

template<typename DistanceType>
index<DistanceType> open_index(const std::filesystem::path& index_directory){

    auto splitting_vectors = [&]() {
        std::ifstream vecFile(index_directory / "SplittingVectors.bin", std::ios_base::binary);
        return extract<std::unordered_map<size_t, std::pair<ann::aligned_array<DistanceType>, DistanceType>>>(vecFile);
    }();

    auto split_idx_to_block_idx = [&]() {
        std::ifstream mappingFile(index_directory / "SplittingIndexToBlockNumber.bin", std::ios_base::binary);
        return extract<std::unordered_map<size_t, size_t>>(mappingFile);
    }();

    auto block_idx_to_source_idx = [&]{
        std::ifstream mappingFile(index_directory / "BlockIndexToSourceIndex.bin", std::ios_base::binary);
        return extract<ann::dynamic_array<ann::dynamic_array<size_t>>>(mappingFile);
    }();

    auto splits_to_do = [&]{
        std::unordered_set<size_t> splitting_indicies;
        for (const auto& [key, value] : splitting_vectors)
            splitting_indicies.insert(key);

        for (const auto& [key, value] : split_idx_to_block_idx)
            splitting_indicies.erase(key);
        
        return splitting_indicies;
    }();

    return index<DistanceType> {
        .data_points                = OpenDataBlocks<float>(index_directory),
        .query_contexts             = OpenQueryContexts<DataIndex_t, float>(index_directory),
        .graph_neighbors            = OpenIndexBlocks(index_directory),
        .block_idx_to_source_idx    = std::move(block_idx_to_source_idx),
        .splits_to_do               = std::move(splits_to_do),
        .splitting_vectors          = std::move(splitting_vectors),
        .split_idx_to_block_idx     = std::move(split_idx_to_block_idx),
        .distance_metric            = {},
        .search_parameters          = {}
    };
}
}

#endif