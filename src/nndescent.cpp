/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

//This is primarily for testing an debugging

#include <string>
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

#include "Utilities/Type.hpp"
#include "Utilities/Data.hpp"
#include "Utilities/Metrics/SpaceMetrics.hpp"
#include "Utilities/Metrics/FunctorErasure.hpp"

#include "NND/GraphStructures.hpp"
#include "NND/MetaGraph.hpp"
#include "NND/SubGraphQuerying.hpp"
#include "NND/BlockwiseAlgorithm.hpp"
#include "NND/GraphInitialization.hpp"
#include "NND/Search.hpp"

#include "NND/Parallel-Algorithm/FreeFunctions.hpp"

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"

#include "Utilities/DataSerialization.hpp"
#include "Utilities/DataDeserialization.hpp"

using namespace nnd;
/*
struct IndexParamters{
    size_t blockGraphNeighbors;
    size_t COMNeighbors;
    size_t nearestNodeNeighbors;
    size_t queryDepth;
};

struct SearchParameters{
    size_t searchNeighbors;
    size_t searchDepth;
    size_t maxSearchesQueued;
};

struct OptionsValues{
    SplittingHeurisitcs splitParams;
    IndexParamters indexParams;
    SearchParameters searchParams;
};
*/

template<typename DataEntry, typename DistType, typename COMExtent>
std::vector<BlockUpdateContext<float>> BuildGraph(const std::vector<DataBlock<DataEntry>>& dataBlocks,
                                                                          const MetaGraph<COMExtent>& metaGraph,
                                                                          DispatchFunctor<DistType>& dispatch,
                                                                          std::vector<size_t>&& sizes,
                                                                          const HyperParameterValues& hyperParams,
                                                                          std::execution::sequenced_policy){


    std::vector<Graph<size_t, float>> blockGraphs = InitializeBlockGraphs<float>(dataBlocks.size(), sizes, hyperParams.indexParams.blockGraphNeighbors, dispatch);

    DataComDistance<DataEntry, COMExtent, EuclideanMetricPair> comFunctor(metaGraph, dataBlocks);

    Graph<size_t, float> queryHints = GenerateQueryHints<float, float>(blockGraphs, metaGraph, hyperParams.indexParams.blockGraphNeighbors, comFunctor);


    std::vector<BlockUpdateContext<float>> blockUpdateContexts = InitializeBlockContexts<DistType, COMExtent>(blockGraphs, 
                                                                                         metaGraph,
                                                                                         queryHints,
                                                                                         hyperParams.indexParams.queryDepth);
    

    CachingFunctor<float> cacher(dispatch, hyperParams.splitParams.maxTreeSize, hyperParams.indexParams.blockGraphNeighbors);

    auto [nearestNodeDistances, stitchHints] = NearestNodeDistances(blockUpdateContexts, metaGraph, hyperParams.indexParams.nearestNodeNeighbors, dispatch);
    StitchBlocks(nearestNodeDistances, stitchHints, blockUpdateContexts, cacher);
    
    
    //int iteration(1);
    int graphUpdates(1);
    while(graphUpdates>0){
        graphUpdates = 0;
        for(size_t i = 0; i<blockUpdateContexts.size(); i+=1){
            for (auto& joinList: blockUpdateContexts[i].joinsToDo){
                graphUpdates += UpdateBlocks(blockUpdateContexts[i], blockUpdateContexts[joinList.first], cacher);
                blockUpdateContexts[joinList.first].joinsToDo.erase(i);
            }
        }
        for (auto& context: blockUpdateContexts){
            context.SetNextJoins();
        }

    }

    return blockUpdateContexts;
}

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

using IndexBlock = std::vector<std::vector<BlockIndecies>>;

template<typename DistType>
std::vector<IndexBlock> IndexFinalization(std::span<BlockUpdateContext<DistType>> blocks){

    std::vector<std::vector<std::vector<BlockIndecies>>> index(blocks.size());
    //index.reserve(blocks.size());
    
    //maybe I should write my own "multi_transform"
    std::transform(blocks.begin(), blocks.end(), index.begin(), [](const auto& block){    

        std::vector<std::vector<BlockIndecies>> graphFragment(block.currentGraph.size());
        std::transform(block.currentGraph.begin(), block.currentGraph.end(), graphFragment.begin(),
            [blockNumber = block.queryContext.blockNumber](const auto& vertex){
                return VertexToIndex(vertex, blockNumber);
            });
        return graphFragment;
    });

    return index;
}
        //


    //}








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
    parallelIndexBuild
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
    {"-parallelIndexBuild"s,     Options::parallelIndexBuild}
};



int main(int argc, char *argv[]){
    
    static const std::endian dataEndianness = std::endian::big;

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

    IndexParamters indexParams{12, 40, 35, 4};

    size_t numBlockGraphNeighbors = 12;
    size_t numCOMNeighbors = 40;
    size_t maxNearestNodes = 35;
    size_t queryDepth = 5;

    SearchParameters searchParams{15, 5, 10};
    size_t numberSearchNeighbors = 10;
    size_t searchQueryDepth = 5;
    size_t maxNewSearches = 5;

    SplittingHeurisitcs splitParams= {16, 205, 123, 287};

    size_t additionalInitSearches = 3;

    //maxNearestNodes <= numCOMNeighbors
    //additionalInitSearches <= numCOMNeighbors
    //searchDepths <= numBlockGraphsNeighbors
    // something about splitParams

    bool parallelIndexBuild = true;


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
                /*
                    searchNeighbors,
                    searchDepth,
                    maxSearchesQueued
                */

        }
    }

    HyperParameterValues parameters{splitParams, indexParams, searchParams};

    

    std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");
    DataSet<AlignedArray<float>> mnistFashionTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<AlignedArray<float>,dataEndianness>);


    std::string testDataFilePath("./TestData/MNIST-Fashion-Data.bin");
    std::string testNeighborsFilePath("./TestData/MNIST-Fashion-Neighbors.bin");
    DataSet<AlignedArray<float>> mnistFashionTest(testDataFilePath, 28*28, 10'000, &ExtractNumericArray<AlignedArray<float>,dataEndianness>);
    DataSet<AlignedArray<uint32_t>> mnistFashionTestNeighbors(testNeighborsFilePath, 100, 10'000, &ExtractNumericArray<AlignedArray<uint32_t>,dataEndianness>);
    


    //std::cout << "I/O done." << std::endl;


    std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();



    RngFunctor rngFunctor(0, mnistFashionTrain.size() - 1);

    

    auto [rpTrees, splittingVectors] = (parallelIndexBuild) ? 
                                        BuildRPForest<ParallelEuclidianScheme<AlignedArray<float>, AlignedArray<float>>>(std::execution::par_unseq, mnistFashionTrain, parameters.splitParams, 12) :
                                        BuildRPForest<EuclidianScheme<AlignedArray<float>, AlignedArray<float>>>(std::execution::seq, mnistFashionTrain, parameters.splitParams);
                                        

    //std::chrono::time_point<std::chrono::steady_clock> rpTrainEnd = std::chrono::steady_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(rpTrainEnd - runStart).count() << "s total for test set rpTrees " << std::endl;


    //std::vector<size_t> trainClassifications(mnistFashionTrain.numberOfSamples);
    


    auto [indexMappings, dataBlocks] = PartitionData<AlignedArray<float>>(rpTrees, mnistFashionTrain);

    
    MetricFunctor<AlignedArray<float>, EuclideanMetricPair> euclideanFunctor(dataBlocks);
    DispatchFunctor<float> testDispatch(euclideanFunctor);

    std::vector<size_t> sizes;
    sizes.reserve(dataBlocks.size());
    for(const auto& block: dataBlocks){
        sizes.push_back(block.size());
    }

    
    //MetricFunctor<AlignedArray<float>, EuclideanMetricPair> euclideanFunctor(dataBlocks);
    
    
    
    
    MetaGraph<float> metaGraph(dataBlocks, parameters.indexParams.COMNeighbors, EuclideanMetricPair());
    DataComDistance<AlignedArray<float>, float, EuclideanMetricPair> comFunctor(metaGraph, dataBlocks);
    
    //hacky but not a long term thing
    std::vector<BlockUpdateContext<float>> blockContextVec;
    std::unique_ptr<BlockUpdateContext<float>[]> blockContextArr;
    std::span<BlockUpdateContext<float>> blockUpdateContexts;

    if (parallelIndexBuild){
        ThreadPool<ThreadFunctors<float, float>> pool(12, euclideanFunctor, comFunctor, splitParams.maxTreeSize, parameters.indexParams.blockGraphNeighbors);
        pool.StartThreads();
        blockContextArr = BuildGraph(std::move(sizes), metaGraph, parameters, pool);
        blockUpdateContexts = {blockContextArr.get(), dataBlocks.size()};
        pool.StopThreads();
    } else {
        blockContextVec = BuildGraph<AlignedArray<float>, float, float>(dataBlocks, metaGraph, testDispatch, std::move(sizes), parameters, std::execution::seq);
        blockUpdateContexts = {blockContextVec.data(), blockContextVec.size()};
    }
    
    
    std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for index building " << std::endl;
    //std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << std::endl;
    std::chrono::time_point<std::chrono::steady_clock> finalizationStart = std::chrono::steady_clock::now();

    std::vector<IndexBlock> index = IndexFinalization(blockUpdateContexts);

    std::chrono::time_point<std::chrono::steady_clock> finalizationEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(finalizationEnd - finalizationStart).count() << "s total for index finalization " << std::endl;


    std::chrono::time_point<std::chrono::steady_clock> runStart2 = std::chrono::steady_clock::now();

    EuclidianScheme<AlignedArray<float>, AlignedArray<float>> transformingScheme(mnistFashionTest);

    transformingScheme.splittingVectors = std::move(splittingVectors);

    std::unique_ptr<size_t[]> testIndecies = std::make_unique<size_t[]>(mnistFashionTest.size());
    std::iota(testIndecies.get(), testIndecies.get()+mnistFashionTest.size(), 0);

    std::unordered_set<size_t> splittingIndicies;
    auto accumulateSplits = [&](const TreeLeaf& node, std::span<const size_t> indicies){
        if(node.children.first != nullptr && node.children.second != nullptr) splittingIndicies.insert(node.splittingIndex);
    };
    CrawlLeaves(rpTrees, accumulateSplits);
    /*
    for (auto& leaf: rpTreesTrain.treeLeaves){
        if(leaf.children.first == 0 && leaf.children.second == 0) continue;
        splittingIndicies.insert(leaf.splittingIndex);
    }
    */

    //TransformingSplittingScheme transformingFunc(transformingScheme);

    //std::mt19937_64 testRngEngine(0);
    //std::uniform_int_distribution<size_t> testDist(size_t(0), mnistFashionTest.size() - 1);
    RngFunctor testFunctor(size_t(0), mnistFashionTest.size() - 1);

    ForestBuilder testBuilder{std::move(testFunctor), splitParams, transformingScheme};


    RandomProjectionForest rpTreesTest = testBuilder(std::move(testIndecies), mnistFashionTest.size(), splittingIndicies);


    //std::chrono::time_point<std::chrono::steady_clock> rpTestEnd = std::chrono::steady_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(rpTestEnd - runStart2).count() << "s total for test set rpTrees " << std::endl;

    //RandomProjectionForest rpTreesTest(mnistFashionTest.numberOfSamples, transformingFunc, splittingIndicies);

    


    size_t numberSearchBlocks = dataBlocks.size();

    GraphVertex<BlockIndecies, float> nullVertex;
    for(size_t i = 0; i<numberSearchNeighbors; i+=1){
        nullVertex.push_back({{size_t(0),size_t(0)}, std::numeric_limits<float>::max()});
    }

    for (auto& context: blockUpdateContexts){
        context.queryContext.querySearchDepth = searchQueryDepth;
        context.queryContext.querySize = numberSearchNeighbors;
    }

    SearchFunctor<AlignedArray<float>, EuclideanMetricPair> searchDist(dataBlocks, mnistFashionTest);
    SinglePointFunctor<float> searchFunctor(searchDist);

    auto searcherConstructor = [&](const DataSet<AlignedArray<float>>& dataSource, std::span<const size_t> indicies, size_t blockCounter)->
                                  std::vector<SearchContext<float>>{
        std::vector<SearchContext<float>> retVec;
        retVec.reserve(indicies.size());
        for(size_t index: indicies){
            //const DataView searchPoint, const std::vector<DataBlock<DataEntry>>& blocks
            
            retVec.push_back({numberSearchNeighbors, numberSearchBlocks, index});
        }
        return retVec;
    };


    DataMapper<AlignedArray<float>, std::vector<SearchContext<float>>, decltype(searcherConstructor)> testMapper(mnistFashionTest, searcherConstructor);
    CrawlTerminalLeaves(rpTreesTest, testMapper);

    auto searchContexts = std::move(testMapper.dataBlocks);

    IndexMaps<size_t> testMappings = {
        std::move(testMapper.splitToBlockNum),
        std::move(testMapper.blockIndexToSource),
        std::move(testMapper.sourceToBlockIndex),
        std::move(testMapper.sourceToSplitIndex)
    };
    
    
    auto blocksToSearch = BlocksToSearch(searchContexts, metaGraph, additionalInitSearches);
    
    SearchQueue searchHints = FirstBlockSearch(searchContexts, blocksToSearch, searchFunctor, blockUpdateContexts, std::span(std::as_const(index)), maxNewSearches);
    
    SearchLoop(searchFunctor, searchHints, searchContexts, blockUpdateContexts, std::span(std::as_const(index)), maxNewSearches, mnistFashionTest.size());


    std::chrono::time_point<std::chrono::steady_clock> runEnd2 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd2 - runStart2).count() << "s test set search " << std::endl;
    //std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd2 - runStart2).count() << std::endl;


    

    std::vector<std::vector<size_t>> results(mnistFashionTest.samples.size());
    for (size_t i = 0; auto& testBlock: searchContexts){
        for (size_t j = 0; auto& context: testBlock){
            GraphVertex<BlockIndecies, float>& result = context.currentNeighbors;
            size_t testIndex = testMappings.blockIndexToSource[{i, j}];
            std::sort_heap(result.begin(), result.end(), NeighborDistanceComparison<BlockIndecies, float>);
            for (const auto& neighbor: result){
                results[testIndex].push_back(indexMappings.blockIndexToSource[neighbor.first]);
            }
            j++;
        }
        i++;
    }
    size_t numNeighborsCorrect(0);
    std::vector<size_t> correctNeighborsPerIndex(results.size());
    for(size_t i = 0; const auto& result: results){
        for(size_t j = 0; const auto& neighbor: result){
            auto findItr = std::find(std::begin(mnistFashionTestNeighbors.samples[i]), std::begin(mnistFashionTestNeighbors.samples[i]) + 10, neighbor);
            if (findItr != (std::begin(mnistFashionTestNeighbors.samples[i]) + 10)){
                numNeighborsCorrect++;
                correctNeighborsPerIndex[i]++;
            }
            j++;
        }
        i++;
    }

    std::vector<size_t> correctNeighborsPerBlock(searchContexts.size());
    for (size_t i = 0; i< correctNeighborsPerIndex.size(); i+=1){
        correctNeighborsPerBlock[testMappings.sourceToBlockIndex[i].blockNumber] += correctNeighborsPerIndex[i];
    }
    std::vector<float> correctPerBlockFloat(searchContexts.size());
    for (size_t i =0; i<correctNeighborsPerBlock.size(); i+=1){
        correctPerBlockFloat[i] = float(correctNeighborsPerBlock[i]*10)/float(searchContexts[i].size());
    }
    double recall = double(numNeighborsCorrect)/ double(10*mnistFashionTestNeighbors.samples.size());
    //std::cout << (recall * 100) << std::endl;
    std::cout << "Recall: " << (recall * 100) << "%" << std::endl;
    
    //WeightedGraphEdges graphEdges = NeighborsOutOfBlock(mnistFashionTestNeighbors, trainMapper.sourceToBlockIndex, testClassifications);

    //for (size_t i = 0; i < trainMapper.sourceToBlockIndex.size(); i += 1){
    //    trainClassifications[i] = trainMapper.sourceToBlockIndex[i].blockNumber;
    //}

    //SerializeCOMS(metaGraph.points, "./TestData/MNIST-Fashion-Train-COMs.bin");
    //SerializeMetaGraph(graphEdges, "./TestData/MNIST-Fashion-Test-MetaGraphEdges.bin");
    //SerializeVector<size_t>(trainClassifications, "./TestData/MNIST-Fashion-Train-SplittingIndicies.bin");
    return 0;
}