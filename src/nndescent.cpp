/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/



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
    index_parameters indexParams{12, 20, 15, 8};

    size_t numBlockGraphNeighbors = 12;
    //size_t numCOMNeighbors = 40;
    //size_t maxNearestNodes = 35;
    size_t numCOMNeighbors = 20;
    size_t maxNearestNodes = 15;
    size_t queryDepth = 8;

    search_parameters searchParams{10, 6, 5};
    size_t numberSearchNeighbors = 10;
    size_t searchQueryDepth = 6;
    size_t maxNewSearches = 10;

    //SplittingHeurisitcs splitParams= {1250, 750, 1750, 0.0f};
    splitting_heurisitcs splitParams= {2500, 1500, 3500, 50, 0.0f};
    //SplittingHeurisitcs splitParams= {725, 435, 1160, 50, 0.4f};
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

    bool parallelIndexBuild = true;
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
                indexParams.block_graph_neighbors = numBlockGraphNeighbors = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::COMNeighbors:
                indexParams.COM_neighbors = numCOMNeighbors = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::nearestNodeNeighbors:
                indexParams.nearest_node_neighbors = maxNearestNodes = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::queryDepth:
                indexParams.query_depth = queryDepth = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::targetSplitSize:
                splitParams.split_threshold = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::minSplitSize:
                splitParams.child_threshold = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::maxSplitSize:
                splitParams.max_tree_size = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::searchNeighbors:
                searchParams.search_neighbors = numberSearchNeighbors = stoul(std::string(option.substr(nameEnd+1)));
                break;
                
            case Options::searchDepth:
                searchParams.search_depth = searchQueryDepth = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case Options::maxSearchesQueued:
                searchParams.max_searches_queued = maxNewSearches = stoul(std::string(option.substr(nameEnd+1)));
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

        
        std::string trainDataFilePath("./TestData/SIFT-Train.bin");
        DataSet<float> training_data(trainDataFilePath, 128, 1'000'000);
        std::filesystem::path indexLocation("./Saved-Indecies/SIFT");
        using metric = euclidean_metric_pair;
        
        /*
        std::string trainDataFilePath("./TestData/NYTimes-Angular-Train.bin");
        DataSet<float> training_data(trainDataFilePath, 256, 290'000);
        std::filesystem::path indexLocation("./Saved-Indecies/NYTimes");
        using metric = inner_product_pair;
        */
        //NormalizeDataSet(training_data);

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

        //for (std::size_t i = 0; i< 50; ++i)
        {
            std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();
            
            nnd::index built_index = parallelIndexBuild 
                                        ? build_index(training_data, metric{}, numThreads, parameters)
                                        : build_index(training_data, metric{}, parameters);
            std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
            //std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for index building " << std::endl;
            std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << std::endl;
        }
        //std::chrono::time_point<std::chrono::steady_clock> finalizationStart = std::chrono::steady_clock::now();
        /*
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
        */  
    }

    return 0;
}