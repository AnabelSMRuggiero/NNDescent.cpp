/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ann/AlignedMemory/DynamicArray.hpp"
#include "ann/Data.hpp"
#include "ann/DataDeserialization.hpp"
#include "ann/DelayConstruct.hpp"
#include "ann/Metrics/Euclidean.hpp"
#include "ann/Metrics/SpaceMetrics.hpp"
#include "ann/Type.hpp"

#include "NND/FunctorErasure.hpp"
#include "NND/MetaGraph.hpp"
#include "NND/MetricHelpers.hpp"
#include "NND/Parallel-Algorithm/FreeFunctions.hpp"
#include "NND/Search.hpp"
#include "NND/SubGraphQuerying.hpp"
#include "NND/Type.hpp"

#include "RPTrees/Forest.hpp"
#include "RPTrees/SplittingScheme.hpp"

using namespace nnd;

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

std::vector<IndexBlock> OpenIndexBlocks(std::filesystem::path fragmentDirectory) {
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

auto chain_steps(auto&&... funcs) {
    return [&](auto&&... args) { (funcs(args...), ...); };
}

namespace nnd {
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


template<typename DistanceType>
struct index_view {
    // std::vector<DataBlock<DistanceType>> data_points;
    std::span<const QueryContext<DataIndex_t, DistanceType>> query_contexts;
    std::span<const IndexBlock> graph_neighbors;
};

template<typename DistanceType>
std::vector<std::vector<std::size_t>> search(const DataSet<DistanceType>& search_data, const index<DistanceType>& index, std::size_t num_threads){
    
    RandomProjectionForest rpTreesTest = RPTransformData(search_data.size(),
                                                         index.splits_to_do,
                                                         borrowed_euclidean(search_data, index.splitting_vectors),
                                                         num_threads);

    size_t numberSearchBlocks = index.data_points.size();

    for (auto& context : index.query_contexts) {
        context.querySearchDepth = index.search_parameters.searchDepth;
        context.querySize = index.search_parameters.searchNeighbors;
    }

    
    IndexMaps<size_t> testMappings;

     

    ThreadPool<erased_unary_binder<DistanceType>> searchPool(num_threads, index.distance_metric);

    std::span<const std::vector<size_t>> indexMappingView(index.block_idx_to_source_idx);

    auto [searchContexts, mappings] =
        block_search_set(search_data, rpTreesTest, index.search_parameters, index.graph_neighbors.size(), index.split_idx_to_block_idx);

    ParallelSearch(searchPool, index.search_parameters.maxSearchesQueued, searchContexts, index.query_contexts, index.graph_neighbors);

    std::vector<std::vector<size_t>> results(search_data.size());

    for (size_t i = 0; auto& [blockNum, testBlock] : searchContexts) {
        for (size_t j = 0; auto& context : testBlock) {
            GraphVertex<BlockIndecies, float>& result = context.currentNeighbors;
            // size_t testIndex = testMappings.blockIndexToSource[i][j];
            size_t testIndex = context.dataIndex;
            // std::sort_heap(result.begin(), result.end(), NeighborDistanceComparison<BlockIndecies, float>);
            for (const auto& neighbor : result) {
                results[testIndex].push_back(indexMappingView[neighbor.first.blockNumber][neighbor.first.dataIndex]);
            }
            j++;
        }
        i++;
    }

    return results;
}

} // namespace nnd

struct search_set_mappings {
    std::vector<ParallelContextBlock<float>> contexts;
    IndexMaps<size_t> mappings;
};

search_set_mappings block_search_set(
    const DataSet<float>& searchSet, const RandomProjectionForest& searchForest, const SearchParameters& searchParams,
    std::size_t num_index_blocks, const std::unordered_map<unsigned long, unsigned long>& splitToBlockNum) {

    DataMapper<float, void, void> testMapper(searchSet);
    std::vector<ParallelContextBlock<float>> searchContexts;

    auto searcherConstructor = [&, &splitToBlockNum = splitToBlockNum](size_t splittingIndex, std::span<const size_t> indicies) -> void {
        ParallelContextBlock<float> retBlock{ splitToBlockNum.at(splittingIndex),
                                              std::vector<ParallelSearchContext<float>>(indicies.size()) };

        for (size_t i = 0; size_t index : indicies) {
            // const DataView searchPoint, const std::vector<DataBlock<DataEntry>>& blocks
            ParallelSearchContext<float>* contextPtr = &retBlock.second[i];
            contextPtr->~ParallelSearchContext<float>();
            new (contextPtr) ParallelSearchContext<float>(searchParams.searchNeighbors, num_index_blocks, index);

            // retVec.emplace_back(numberSearchNeighbors, numberSearchBlocks, index);
            i++;
        }
        searchContexts.push_back(std::move(retBlock));
        testMapper(splittingIndex, indicies);
    };
    // DataMapper<AlignedArray<float>, std::vector<ParallelSearchContext<float>>, decltype(searcherConstructor)>
    // testMapper(mnistFashionTest, searcherConstructor);
    CrawlTerminalLeaves(searchForest, searcherConstructor);

    // auto searchContexts = std::move(testMapper.dataBlocks);

    IndexMaps<size_t> testMappings = { std::move(testMapper.splitToBlockNum),
                                       std::move(testMapper.blockIndexToSource),
                                       std::move(testMapper.sourceToBlockIndex),
                                       std::move(testMapper.sourceToSplitIndex) };

    return { std::move(searchContexts), std::move(testMappings) };
}

void ParallelSearch(
    ThreadPool<erased_unary_binder<float>>& threadPool, std::size_t max_searches_queued,
    std::span<ParallelContextBlock<float>> searchContexts, std::span<nnd::QueryContext<unsigned int, float>> queryContexts,
    std::span<const nnd::IndexBlock> indexView) {

    InitialSearchTask<float> searchGenerator = {
        queryContexts, indexView, max_searches_queued, AsyncQueue<std::pair<BlockIndecies, SearchSet>>()
    };

    threadPool.StartThreads();
    SearchQueue searchHints = ParaFirstBlockSearch(searchGenerator, searchContexts, threadPool);

    QueueView hintView = { searchHints.data(), searchHints.size() };

    ParaSearchLoop(threadPool, hintView, searchContexts, queryContexts, indexView, max_searches_queued, searchContexts.size());
    threadPool.StopThreads();
}

int main(int argc, char* argv[]) {

    constexpr size_t numThreads = 12;

    // IndexParamters indexParams{12, 40, 35, 6};
    IndexParameters indexParams{ 12, 20, 15, 6 };

    size_t numBlockGraphNeighbors = 12;
    // size_t numCOMNeighbors = 40;
    // size_t maxNearestNodes = 35;
    size_t numCOMNeighbors = 20;
    size_t maxNearestNodes = 15;
    size_t queryDepth = 6;

    SearchParameters searchParams{ 10, 6, 10 };
    size_t numberSearchNeighbors = 10;
    size_t searchQueryDepth = 6;
    size_t maxNewSearches = 10;

    // SplittingHeurisitcs splitParams= {2500, 1500, 3500, 0.0f};
    SplittingHeurisitcs splitParams = { 205, 123, 287, 0.0f };

    // SplittingHeurisitcs splitParams= {20, 12, 28, 0.0f};

    size_t additionalInitSearches = 8;

    HyperParameterValues parameters{ splitParams, indexParams, searchParams };

    bool parallelIndexBuild = true;
    bool parallelSearch = true;

    /*
    std::filesystem::path indexLocation("./Saved-Indecies/MNIST-Fashion");

    std::string testDataFilePath("./TestData/MNIST-Fashion-Test.bin");
    std::string testNeighborsFilePath("./TestData/MNIST-Fashion-Neighbors.bin");
    DataSet<float> test_data_set(testDataFilePath, 28 * 28, 10'000);
    DataSet<uint32_t, ann::align_val_of<uint32_t>> test_neighbors(testNeighborsFilePath, 100, 10'000);
    */

    std::filesystem::path indexLocation("./Saved-Indecies/SIFT");

    std::string testDataFilePath("./TestData/SIFT-Test.bin");
    std::string testNeighborsFilePath("./TestData/SIFT-Neighbors.bin");
    DataSet<float> test_data_set(testDataFilePath, 128, 10'000);
    DataSet<std::uint32_t, ann::align_val_of<std::uint32_t>> test_neighbors(testNeighborsFilePath, 100, 10'000);

    auto dataBlocks = OpenDataBlocks<float>(indexLocation);

    auto queryContexts = OpenQueryContexts<DataIndex_t, float>(indexLocation);

    auto indexBlocks = OpenIndexBlocks(indexLocation);

    auto splittingVectors = [&]() {
        std::filesystem::path splittingVectorsPath = indexLocation / "SplittingVectors.bin";
        std::ifstream vecFile(splittingVectorsPath, std::ios_base::binary);
        return extract<std::unordered_map<size_t, std::pair<ann::aligned_array<float>, float>>>(vecFile);
    }();

    auto splitToBlockNum = [&]() {
        std::ifstream mappingFile(indexLocation / "SplittingIndexToBlockNumber.bin", std::ios_base::binary);
        return extract<std::unordered_map<size_t, size_t>>(mappingFile);
    }();

    auto blockIndexToSource = [&]() {
        std::ifstream mappingFile(indexLocation / "BlockIndexToSourceIndex.bin", std::ios_base::binary);
        return extract<std::vector<std::vector<size_t>>>(mappingFile);
    }();
    std::unordered_set<size_t> splittingIndicies;
    for (const auto& [key, value] : splittingVectors)
        splittingIndicies.insert(key);

    for (const auto& [key, value] : splitToBlockNum)
        splittingIndicies.erase(key);

    EuclidianScheme<float, ann::aligned_array<float>> transformingScheme(test_data_set);

    fixed_block_binder searchDist(EuclideanMetricPair{}, test_data_set, std::span{ std::as_const(dataBlocks) });
    erased_unary_binder<float> searchFunctor(searchDist);

    transformingScheme.splittingVectors = std::move(splittingVectors);

    for (size_t i = 0; i < 10; i += 1) {
        std::chrono::time_point<std::chrono::steady_clock> runStart2 = std::chrono::steady_clock::now();

        RandomProjectionForest rpTreesTest =
            (parallelSearch) ? RPTransformData(test_data_set, splittingIndicies, transformingScheme, numThreads)
                             : RPTransformData(test_data_set, splittingIndicies, transformingScheme);

        size_t numberSearchBlocks = dataBlocks.size();

        for (auto& context : queryContexts) {
            context.querySearchDepth = searchQueryDepth;
            context.querySize = numberSearchNeighbors;
        }

        fixed_block_binder searchDist(EuclideanMetricPair{}, test_data_set, std::span{ std::as_const(dataBlocks) });
        erased_unary_binder<float> searchFunctor(searchDist);

        std::vector<std::vector<size_t>> results(test_data_set.size());
        IndexMaps<size_t> testMappings;

        if (parallelSearch) {

            ThreadPool<erased_unary_binder<float>> searchPool(numThreads, searchDist);

            std::span<std::vector<size_t>> indexMappingView(blockIndexToSource.data(), blockIndexToSource.size());

            auto [searchContexts, mappings] =
                block_search_set(test_data_set, rpTreesTest, searchParams, indexBlocks.size(), splitToBlockNum);

            ParallelSearch(searchPool, searchParams.maxSearchesQueued, searchContexts, queryContexts, indexBlocks);

            std::chrono::time_point<std::chrono::steady_clock> runEnd2 = std::chrono::steady_clock::now();
            // std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd2 - runStart2).count() << "s test set search " <<
            // std::endl;
            std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd2 - runStart2).count() << std::endl;

            for (size_t i = 0; auto& [blockNum, testBlock] : searchContexts) {
                for (size_t j = 0; auto& context : testBlock) {
                    GraphVertex<BlockIndecies, float>& result = context.currentNeighbors;
                    // size_t testIndex = testMappings.blockIndexToSource[i][j];
                    size_t testIndex = context.dataIndex;
                    // std::sort_heap(result.begin(), result.end(), NeighborDistanceComparison<BlockIndecies, float>);
                    for (const auto& neighbor : result) {
                        results[testIndex].push_back(indexMappingView[neighbor.first.blockNumber][neighbor.first.dataIndex]);
                    }
                    j++;
                }
                i++;
            }
        } else {

            DataMapper<float, void, void> testMapper(test_data_set);
            std::vector<ContextBlock<float>> searchContexts;
            auto searcherConstructor =
                [&, &splitToBlockNum = splitToBlockNum](size_t splittingIndex, std::span<const size_t> indicies) -> void {
                ContextBlock<float> retBlock;
                retBlock.first = splitToBlockNum[splittingIndex];
                retBlock.second.reserve(indicies.size());
                for (size_t index : indicies) {
                    // const DataView searchPoint, const std::vector<DataBlock<DataEntry>>& blocks

                    retBlock.second.push_back({ numberSearchNeighbors, numberSearchBlocks, index });
                }
                searchContexts.push_back(std::move(retBlock));
                testMapper(splittingIndex, indicies);
            };

            CrawlTerminalLeaves(rpTreesTest, searcherConstructor);

            // auto searchContexts = std::move(testMapper.dataBlocks);

            testMappings = { std::move(testMapper.splitToBlockNum),
                             std::move(testMapper.blockIndexToSource),
                             std::move(testMapper.sourceToBlockIndex),
                             std::move(testMapper.sourceToSplitIndex) };

            // OffsetSpan<const IndexBlock> indexSpan(index.data(), index.size(), metaGraph.GetBlockOffset());
            std::span<const IndexBlock> indexView{ indexBlocks.data(), indexBlocks.size() };

            SearchQueue searchHints =
                FirstBlockSearch(searchContexts, searchFunctor, std::span{ queryContexts }, indexView, maxNewSearches);
            // std::vector<IndexBlock> index = IndexFinalization(blockUpdateContexts)

            QueueView hintView = { searchHints.data(), searchHints.size() };

            SearchLoop(
                searchFunctor, hintView, searchContexts, std::span{ queryContexts }, indexView, maxNewSearches, test_data_set.size());

            /*
            auto task = [queryContexts = std::span{queryContexts}, indexView, maxNewSearches](const BlockNumber_t startBlock,
            SearchContext<float>& context, SinglePointFunctor<float>& searchFunctor){

                    SingleSearch(queryContexts, indexView, context,
                                 0, startBlock, searchFunctor, maxNewSearches);

            };


            for (auto& [blockNum, contexts]: searchContexts){
                for (auto& context: contexts){
                    task(blockNum, context, searchFunctor);
                }
            }
            */

            std::chrono::time_point<std::chrono::steady_clock> runEnd2 = std::chrono::steady_clock::now();
            // std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd2 - runStart2).count() << "s test set search " <<
            // std::endl;
            std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd2 - runStart2).count() << std::endl;

            std::span<std::vector<size_t>> indexMappingView(blockIndexToSource.data(), blockIndexToSource.size());

            for (size_t i = 0; auto& [ignore, testBlock] : searchContexts) {
                for (size_t j = 0; auto& context : testBlock) {
                    GraphVertex<BlockIndecies, float>& result = context.currentNeighbors;
                    size_t testIndex = context.dataIndex;
                    // std::sort_heap(result.begin(), result.end(), NeighborDistanceComparison<BlockIndecies, float>);
                    for (const auto& neighbor : result) {
                        results[testIndex].push_back(indexMappingView[neighbor.first.blockNumber][neighbor.first.dataIndex]);
                    }
                    j++;
                }
                i++;
            }
        }

        size_t numNeighborsCorrect(0);
        std::vector<size_t> correctNeighborsPerIndex(results.size());
        for (size_t i = 0; const auto& result : results) {
            for (size_t j = 0; const auto& neighbor : result) {
                auto findItr = std::find(std::begin(test_neighbors[i]), std::begin(test_neighbors[i]) + 10, neighbor);
                if (findItr != (std::begin(test_neighbors[i]) + 10)) {
                    numNeighborsCorrect++;
                    correctNeighborsPerIndex[i]++;
                }
                j++;
            }
            i++;
        }

        double recall = double(numNeighborsCorrect) / double(10 * test_neighbors.size());
        std::cout << (recall * 100) << std::endl;
    }
    return 0;
}