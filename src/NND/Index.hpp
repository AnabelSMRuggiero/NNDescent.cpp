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
std::vector<DataBlock<DistType>> OpenDataBlocks(std::filesystem::path fragmentDirectory) {
    std::ifstream metaDataFile{ fragmentDirectory / "MetaData.bin", std::ios_base::binary };
    FragmentMetaData metadata = Extract<FragmentMetaData>(metaDataFile);

    std::vector<DataBlock<DistType>> dataBlocks;
    dataBlocks.reserve(metadata.numBlocks);

    for (size_t i = 0; i < metadata.numBlocks; i += 1) {
        std::filesystem::path dataBlockPath = fragmentDirectory / ("DataBlock-" + std::to_string(i) + ".bin");
        std::ifstream dataBlockFile{ dataBlockPath, std::ios_base::binary };
        auto extractor = [&]() { return Extract<DataBlock<DistType>>(dataBlockFile, i); };
        dataBlocks.emplace_back(DelayConstruct<DataBlock<DistType>>(extractor));
    }
    return dataBlocks;
}

template<typename IndexType, typename DistType>
std::vector<QueryContext<IndexType, DistType>> OpenQueryContexts(std::filesystem::path fragmentDirectory) {
    std::ifstream metaDataFile{ fragmentDirectory / "MetaData.bin", std::ios_base::binary };
    FragmentMetaData metadata = Extract<FragmentMetaData>(metaDataFile);

    std::vector<QueryContext<IndexType, DistType>> queryContexts;
    queryContexts.reserve(metadata.numBlocks);

    for (size_t i = 0; i < metadata.numBlocks; i += 1) {
        std::filesystem::path dataBlockPath = fragmentDirectory / ("QueryContext-" + std::to_string(i) + ".bin");
        std::ifstream contextFile{ dataBlockPath, std::ios_base::binary };
        auto extractor = [&]() { return Extract<QueryContext<IndexType, DistType>>(contextFile); };
        queryContexts.emplace_back(DelayConstruct<QueryContext<IndexType, DistType>>(extractor));
        // queryContexts.emplace_back(QueryContext<IndexType, DistType>(contextFile));
    }
    return queryContexts;
}

inline std::vector<IndexBlock> OpenIndexBlocks(std::filesystem::path fragmentDirectory) {
    std::ifstream metaDataFile{ fragmentDirectory / "MetaData.bin", std::ios_base::binary };
    FragmentMetaData metadata = Extract<FragmentMetaData>(metaDataFile);

    std::vector<IndexBlock> queryContexts;
    queryContexts.reserve(metadata.numBlocks);

    for (size_t i = 0; i < metadata.numBlocks; i += 1) {
        std::filesystem::path dataBlockPath = fragmentDirectory / ("IndexBlock-" + std::to_string(i) + ".bin");
        std::ifstream contextFile{ dataBlockPath, std::ios_base::binary };
        auto extractor = [&]() { return Extract<IndexBlock>(contextFile); };
        queryContexts.emplace_back(DelayConstruct<IndexBlock>(extractor));
    }
    return queryContexts;
}

template<typename DistanceType>
struct index {
    std::vector<DataBlock<DistanceType>> data_points;
    std::vector<QueryContext<DataIndex_t, DistanceType>> query_contexts;
    std::vector<IndexBlock> graph_neighbors;
    std::vector<std::vector<size_t>> block_idx_to_source_idx;

    std::unordered_set<size_t> splits_to_do;
    std::unordered_map<size_t, std::pair<ann::aligned_array<float>, float>> splitting_vectors;
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
        return extract<std::vector<std::vector<size_t>>>(mappingFile);
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