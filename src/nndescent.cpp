/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

//This is primarily for testing an debugging

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory_resource>
#include <ranges>
#include <string>
#include <utility>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <cmath>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <bit>
#include <memory>
#include <execution>
#include <string_view>
#include <cstdlib>
#include <cstdint>

//#include <type_traits>

#include "NND/Type.hpp"
#include "ann/MemoryResources.hpp"
#include "ann/Type.hpp"
#include "ann/Data.hpp"
#include "ann/Metrics/SpaceMetrics.hpp"
#include "ann/Metrics/Euclidean.hpp"
#include "ann/AlignedMemory/DynamicArray.hpp"


#include "NND/BlockwiseAlgorithm.hpp"
#include "NND/FunctorErasure.hpp"
#include "NND/GraphInitialization.hpp"
#include "NND/GraphStructures.hpp"
#include "NND/Index.hpp"
#include "NND/MetaGraph.hpp"
#include "NND/MetricHelpers.hpp"
#include "NND/Parallel-Algorithm/FreeFunctions.hpp"
#include "NND/Search.hpp"
#include "NND/SubGraphQuerying.hpp"


#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"

#include "ann/DataSerialization.hpp"
#include "ann/DataDeserialization.hpp"


using namespace nnd;


template<typename DistType>
std::vector<BlockIndecies> VertexToIndex(const GraphVertex<BlockIndecies, DistType>& vertex, const size_t blockNumber){
    std::vector<std::pair<BlockIndecies, DistType>> neighborsOOB(vertex.size());

    auto lastCopied = std::remove_copy_if(vertex.begin(),
                                            vertex.end(),
                                            neighborsOOB.begin(),
                                            [blockNumber] (const std::pair<BlockIndecies, DistType>& neighbor) {
                                                return neighbor.first.blockNumber == blockNumber;
    });

    neighborsOOB.erase(lastCopied, neighborsOOB.end());

    std::vector<BlockIndecies> result(neighborsOOB.size());
    std::transform(neighborsOOB.begin(), neighborsOOB.end(), result.begin(), [](const std::pair<BlockIndecies, DistType> neighbor){return neighbor.first;});
    
    return result;
}



template<typename DistType>
ann::dynamic_array<IndexBlock> IndexFinalization(std::span<BlockUpdateContext<DistType>> blocks, std::pmr::memory_resource* resource = std::pmr::get_default_resource()){

    ann::dynamic_array<UnevenBlock<BlockIndecies>> index(blocks.size());
    //index.reserve(blocks.size());
    
    //maybe I should write my own "multi_transform"
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
                                         size_t{0},
                                         std::plus<size_t>{},
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

void SerializeSplittingVectors(const splitting_vectors& splittingVectors, std::filesystem::path filePath){
    std::ofstream vectorFile{filePath, std::ios_base::binary | std::ios_base::trunc};
    serialize(splittingVectors, vectorFile);
}

constexpr SplittingHeurisitcs seed_parameters{
    .splitThreshold = 10,
    .childThreshold = 4,
    .maxTreeSize = 22,
    .maxSplitFraction = 0.0f,
};

constexpr auto add_candidates = [] (auto&& candidates, auto&& index_maps, std::size_t splitIdx, std::span<const size_t> indecies){
    constexpr std::size_t buffer_size = sizeof(BlockIndecies) * seed_parameters.maxTreeSize * 4;
    std::byte stack_buffer[buffer_size];
    std::pmr::monotonic_buffer_resource stack_resource{stack_buffer, buffer_size};

    ann::pmr::dynamic_array<BlockIndecies> block_indecies{ 
        indecies | std::views::transform([&](const auto& index){
            return index_maps.sourceToBlockIndex[index];
        }), 
        &stack_resource
    };

    for (const auto& element : block_indecies){
        auto& candidates_vec = candidates[element.blockNumber][element.dataIndex];
        candidates_vec.reserve(block_indecies.size());
        std::ranges::copy_if(block_indecies, std::back_inserter(candidates_vec), [&](const auto current_candidate){
            return current_candidate.blockNumber != element.blockNumber;
        });
        if (candidates_vec.size()>seed_parameters.childThreshold){
            candidates_vec.resize(seed_parameters.childThreshold);
        }
    }
};

template<typename SplittingScheme, typename DistanceType>
auto seed_candidates(const DataSet<DistanceType>& training_data, const IndexMaps<std::size_t>& index_maps){
    
    auto [rp_trees, splitting_vectors] = BuildRPForest<SplittingScheme>(training_data, seed_parameters);

    auto seed_arrays = index_maps.blockIndexToSource 
                       | std::views::transform(
                           [](const auto& block){
                               return ann::dynamic_array<std::vector<BlockIndecies>>{block.size()};
                         });

    ann::dynamic_array<ann::dynamic_array<std::vector<BlockIndecies>>> candidates{seed_arrays};

    

    auto candidate_task = [&](std::size_t splitIdx, std::span<const size_t> indecies){
        add_candidates(candidates, index_maps, splitIdx, indecies);
    };

    CrawlTerminalLeaves(rp_trees, candidate_task);

    return candidates;
}

template<typename SplittingScheme, typename DistanceType>
auto seed_candidates(const DataSet<DistanceType>& training_data, const IndexMaps<std::size_t>& index_maps, std::size_t num_threads){
    
    auto [rp_trees, splitting_vectors] = BuildRPForest<SplittingScheme>(training_data, seed_parameters, num_threads);

    auto seed_arrays = index_maps.blockIndexToSource 
                       | std::views::transform(
                           [](const auto& block){
                               return ann::dynamic_array<std::vector<BlockIndecies>>{block.size()};
                         });

    ann::dynamic_array<ann::dynamic_array<std::vector<BlockIndecies>>> candidates{seed_arrays};

    ThreadPool<void> assemble_candidates(num_threads);

    auto add_task = [&](std::size_t splitIdx, std::span<const size_t> indecies){
        assemble_candidates.DelegateTask([&, splitIdx, indecies]{
            add_candidates(candidates, index_maps, splitIdx, indecies);
        });
    };

    threaded_region(assemble_candidates, [&](){CrawlTerminalLeaves(rp_trees, add_task);});

    return candidates;
}

template<typename DistanceType, typename DistanceMetric>
nnd::index<DistanceType> build_index(const DataSet<DistanceType>& training_data, const DistanceMetric& distance, std::size_t num_threads, const hyper_parameters& index_parameters = {}){

    using splitting_scheme = pick_parallel_scheme<choose_scheme<DistanceMetric>, DistanceType, ann::aligned_array<DistanceType>>;
    
    auto [rp_trees, splitting_vectors] = BuildRPForest<splitting_scheme>(training_data, index_parameters.splitParams, num_threads);
                                            
    ThreadPool<void> block_builder(num_threads);

    auto [index_mappings, data_blocks] = PartitionData<DistanceType>(rp_trees, training_data, block_builder);

    auto candidates = seed_candidates<splitting_scheme>(training_data, index_mappings, num_threads);

    //block_builder.StopThreads();
    
    block_binder euclideanBinder(distance, std::span{std::as_const(data_blocks)}, std::span{std::as_const(data_blocks)});
    erased_binary_binder<DistanceType> distance_metric{euclideanBinder};

    ThreadPool<thread_functors<float>> pool(num_threads, euclideanBinder, index_parameters.splitParams.maxTreeSize, index_parameters.indexParams.blockGraphNeighbors);
    pool.StartThreads();
    //ann::dynamic_array<BlockUpdateContext<DistanceType>> blockContextArr = BuildGraph(meta_graph, index_parameters, pool);
    ann::dynamic_array<BlockUpdateContext<DistanceType>> blockContextArr = BuildGraphRedux(candidates, index_parameters, pool);
    pool.StopThreads();
    
    std::span<BlockUpdateContext<DistanceType>> blockUpdateContexts{blockContextArr.data(), data_blocks.size()};
    
    
    ann::dynamic_array<IndexBlock> index = IndexFinalization(blockUpdateContexts);

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
        .search_parameters          = {}
    };
    
}



template<typename DistanceType, typename DistanceMetric>
nnd::index<DistanceType> build_index(const DataSet<DistanceType>& training_data, const DistanceMetric& distance, const hyper_parameters& index_parameters = {}){
    
    using splitting_scheme = pick_serial_scheme<choose_scheme<DistanceMetric>, DistanceType, ann::aligned_array<DistanceType>>;

    auto [rp_trees, splitting_vectors] = BuildRPForest<splitting_scheme>(training_data, index_parameters.splitParams);
                                            
    auto [index_mappings, data_blocks] = PartitionData<DistanceType>(rp_trees, training_data);

    
    block_binder euclideanBinder(distance, std::span{std::as_const(data_blocks)}, std::span{std::as_const(data_blocks)});
    erased_binary_binder<DistanceType> distance_metric{euclideanBinder};

    auto candidates = seed_candidates<splitting_scheme>(training_data, index_mappings);
    //MetaGraph<DistanceType> meta_graph = BuildMetaGraphFragment<DistanceType>(data_blocks, index_parameters.indexParams, 0, EuclideanMetricSet{}, EuclideanCOM<float, float>);
    //fixed_block_binder com_functor(distance, meta_graph.points, std::span{std::as_const(data_blocks)});

    
    //ann::dynamic_array<BlockUpdateContext<DistanceType>> blockContextArr = BuildGraphRedux(meta_graph, distance_metric, index_parameters, erased_unary_binder{com_functor});
    ann::dynamic_array<BlockUpdateContext<DistanceType>> blockContextArr = BuildGraphRedux(candidates, distance_metric, index_parameters);
    std::span<BlockUpdateContext<DistanceType>> blockUpdateContexts{blockContextArr.data(), data_blocks.size()};
    
    
    ann::dynamic_array<IndexBlock> index = IndexFinalization(blockUpdateContexts);

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
        .search_parameters          = {}
    };
    
}


enum class Options{
    blockGraphNeighbors,
    COMNeighbors,
    nearestNodeNeighbors,
    queryDepth,
    targetSplitSize,
    minSplitSize,
    maxSplitSize,
    searchNeighbors,
    searchDepth,
    maxSearchesQueued,
    additionalInitSearches,
    parallelIndexBuild,
    parallelSearch
};


using std::operator""s;
static const std::unordered_map<std::string, Options> optionNumber = {
    {"-blockGraphNeighbors"s,    Options::blockGraphNeighbors},
    {"-COMNeighbors"s,           Options::COMNeighbors},
    {"-nearestNodeNeighbors"s,   Options::nearestNodeNeighbors},
    {"-queryDepth"s,             Options::queryDepth},
    {"-targetSplitSize"s,        Options::targetSplitSize},
    {"-minSplitSize"s,           Options::minSplitSize},
    {"-maxSplitSize"s,           Options::maxSplitSize},
    {"-searchNeighbors"s,        Options::searchNeighbors},
    {"-searchDepth"s,            Options::searchDepth},
    {"-maxSearchesQueued"s,      Options::maxSearchesQueued},
    {"-additionalInitSearches"s, Options::additionalInitSearches},
    {"-parallelIndexBuild"s,     Options::parallelIndexBuild},
    {"-parallelSearch"s,         Options::parallelSearch}
};



int main(int argc, char *argv[]){
    
    

    /*
    IndexParamters indexParams{5, 10, 3, 2};

    size_t numBlockGraphNeighbors = 5;
    size_t numCOMNeighbors = 10;
    size_t maxNearestNodes = 3;
    size_t queryDepth = 2;

    SearchParameters searchParams{10, 10, 10};
    size_t numberSearchNeighbors = 10;
    size_t searchQueryDepth = 10;
    size_t maxNewSearches = 10;

    SplittingHeurisitcs splitParams= {16, 140, 60, 180};
    */

    constexpr size_t numThreads = 12;

    //IndexParameters indexParams{12, 40, 35, 8};
    IndexParameters indexParams{12, 20, 15, 6};

    size_t numBlockGraphNeighbors = 12;
    //size_t numCOMNeighbors = 40;
    //size_t maxNearestNodes = 35;
    size_t numCOMNeighbors = 20;
    size_t maxNearestNodes = 15;
    size_t queryDepth = 8;

    SearchParameters searchParams{10, 6, 5};
    size_t numberSearchNeighbors = 10;
    size_t searchQueryDepth = 6;
    size_t maxNewSearches = 10;

    //SplittingHeurisitcs splitParams= {1250, 750, 1750, 0.0f};
    SplittingHeurisitcs splitParams= {2500, 1500, 3500, 0.0f};
    //SplittingHeurisitcs splitParams= {205, 123, 287, 0.0f};

    //SplittingHeurisitcs splitParams= {20, 12, 28, 0.0f};

    size_t additionalInitSearches = 8;

    //maxNearestNodes <= numCOMNeighbors
    //additionalInitSearches <= numCOMNeighbors
    //searchDepths <= numBlockGraphsNeighbors
    // something about splitParams
    //COMNeighbors<NumBlocks

    // data types affect split params now
    // max block size must be < <DataIndex_t>::max()
    // max fragment size must be <  dataSet size (min num fragments * min block size)

    bool parallelIndexBuild = false;
    bool parallelSearch = true;


    std::vector<std::string> options;
    options.reserve(argc-1);
    for (size_t i = 1; i < argc; i+=1){
        options.emplace_back(argv[i]);
    }

    for (const auto& option: options){
        size_t nameEnd = option.find('=');
        if (nameEnd == std::string::npos){
            std::cout << "Could not split option from value; no '=' in: " << option << std::endl;
            return EXIT_FAILURE;
        }
        Options optionEnum;
        try {
            optionEnum = optionNumber.at(option.substr(0,nameEnd));
        } catch (...){
            std::cout << "Unrecognized option: " << option.substr(0,nameEnd) << std::endl;
            return EXIT_FAILURE;
        }

        switch (optionEnum){
            
            case Options::blockGraphNeighbors:
                indexParams.blockGraphNeighbors = numBlockGraphNeighbors = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::COMNeighbors:
                indexParams.COMNeighbors = numCOMNeighbors = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::nearestNodeNeighbors:
                indexParams.nearestNodeNeighbors = maxNearestNodes = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::queryDepth:
                indexParams.queryDepth = queryDepth = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::targetSplitSize:
                splitParams.splitThreshold = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::minSplitSize:
                splitParams.childThreshold = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::maxSplitSize:
                splitParams.maxTreeSize = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::searchNeighbors:
                searchParams.searchNeighbors = numberSearchNeighbors = stoul(std::string(option.substr(nameEnd+1)));
                break;
                
            case Options::searchDepth:
                searchParams.searchDepth = searchQueryDepth = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::maxSearchesQueued:
                searchParams.maxSearchesQueued = maxNewSearches = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::additionalInitSearches:
                additionalInitSearches = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::parallelIndexBuild:
                if (option.substr(nameEnd+1) == "true"){
                    parallelIndexBuild = true;
                } else if (option.substr(nameEnd+1) == "false"){
                    parallelIndexBuild = false;
                } else{
                    std::cout << "parallelIndexBuild input (" << option.substr(nameEnd+1) << ") does not evaluate to 'true' or 'false'" << std::endl;
                }
                break;
            case Options::parallelSearch:
                if (option.substr(nameEnd+1) == "true"){
                    parallelSearch = true;
                } else if (option.substr(nameEnd+1) == "false"){
                    parallelSearch = false;
                } else{
                    std::cout << "parallelSearch input (" << option.substr(nameEnd+1) << ") does not evaluate to 'true' or 'false'" << std::endl;
                }
        }
    }

    hyper_parameters parameters{splitParams, indexParams, searchParams};
    {
        /*
        std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");
        DataSet<float> mnistFashionTrain(trainDataFilePath, 28*28, 60'000);

        std::filesystem::path indexLocation("./Saved-Indecies/MNIST-Fashion");
        using metric = euclidean_metric_pair;
        */

        /*
        std::string testDataFilePath("./TestData/MNIST-Fashion-Test.bin");
        std::string testNeighborsFilePath("./TestData/MNIST-Fashion-Neighbors.bin");
        DataSet<float> mnistFashionTest(testDataFilePath, 28*28, 10'000);
        DataSet<uint32_t, alignof(uint32_t)> mnistFashionTestNeighbors(testNeighborsFilePath, 100, 10'000);
        */

        /*
        std::string trainDataFilePath("./TestData/SIFT-Train.bin");
        DataSet<float> mnistFashionTrain(trainDataFilePath, 128, 1'000'000);
        std::filesystem::path indexLocation("./Saved-Indecies/SIFT");
        using metric = euclidean_metric_pair;
        */
        
        std::string trainDataFilePath("./TestData/NYTimes-Angular-Train.bin");
        DataSet<float> training_data(trainDataFilePath, 256, 290'000);
        std::filesystem::path indexLocation("./Saved-Indecies/NYTimes");
        using metric = inner_product_pair;
        

        /*
        std::string testDataFilePath("./TestData/SIFT-Test.bin");
        std::string testNeighborsFilePath("./TestData/SIFT-Neighbors.bin");
        DataSet<float> mnistFashionTest(testDataFilePath, 128, 10'000);
        DataSet<uint32_t, alignof(uint32_t)> mnistFashionTestNeighbors(testNeighborsFilePath, 100, 10'000);
        */

        /*
        std::string trainDataFilePath("./TestData/NYTimes-Angular-Train.bin");
        DataSet<AlignedArray<float>> mnistFashionTrain(trainDataFilePath, 256, 290'000, &ExtractNumericArray<AlignedArray<float>,dataEndianness>);

        std::string testDataFilePath("./TestData/NYTimes-Angular-Test.bin");
        //std::string testNeighborsFilePath("./TestData/NYTimes-Angular-Neighbors.bin");
        DataSet<float> mnistFashionTest(testDataFilePath, 256, 10'000, &ExtractNumericArray<AlignedArray<float>,dataEndianness>);
        //DataSet<AlignedArray<uint32_t>> mnistFashionTestNeighbors(testNeighborsFilePath, 100, 10'000, &ExtractNumericArray<AlignedArray<uint32_t>,dataEndianness>);
        */
        
        //std::cout << "I/O done." << std::endl;


        std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();
        
        nnd::index built_index = parallelIndexBuild 
                                    ? build_index(training_data, metric{}, numThreads, parameters)
                                    : build_index(training_data, metric{}, parameters);
        std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for index building " << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << std::endl;
        //std::chrono::time_point<std::chrono::steady_clock> finalizationStart = std::chrono::steady_clock::now();

        SerializeSplittingVectors(built_index.splits, indexLocation / "SplittingVectors.bin");

        [&, &split_idx_to_block_idx = built_index.split_idx_to_block_idx](){
            std::ofstream mappingFile{indexLocation/"SplittingIndexToBlockNumber.bin", std::ios_base::binary | std::ios_base::trunc};
            serialize(split_idx_to_block_idx, mappingFile);
        }();

        [&, &block_idx_to_source_idx = built_index.block_idx_to_source_idx](){
            std::ofstream mappingFile{indexLocation/"BlockIndexToSourceIndex.bin", std::ios_base::binary | std::ios_base::trunc};
            serialize(block_idx_to_source_idx, mappingFile);
        }();
        
        
        //std::span<const IndexBlock> indexView{index.data(), index.size()};
        //std::chrono::time_point<std::chrono::steady_clock> finalizationEnd = std::chrono::steady_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(finalizationEnd - finalizationStart).count() << "s total for index finalization " << std::endl;
        
        
        SerializeFragmentIndex( as_const_span(built_index.data_points),
                            as_const_span(built_index.query_contexts),
                            as_const_span(built_index.graph_neighbors),
                            indexLocation);
        
    }

    return 0;
}