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

struct HyperParameterValues{
    SplittingHeurisitcs splitParams;
    IndexParamters indexParams;
    SearchParameters searchParams;
};
*/

template<typename DataEntry, typename DistType, typename COMExtent>
std::vector<BlockUpdateContext<float>> BuildGraph(const std::vector<DataBlock<DataEntry>>& dataBlocks,
                                                                          const MetaGraph<DistType>& metaGraph,
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






enum class HyperParameter{
    blockGraphNeighbors,
    COMNeighbors,
    nearestNodeNeighbors,
    queryDepth,
    targetSplitSize,
    minSplitSize,
    maxSplitSize,
    searchNeighbors,
    searchDepth,
    maxSearchesQueued
};


using std::operator""s;
static const std::unordered_map<std::string, HyperParameter> optionNumber = {
    {"-blockGraphNeighbors"s,   HyperParameter::blockGraphNeighbors},
    {"-COMNeighbors"s,          HyperParameter::COMNeighbors},
    {"-nearestNodeNeighbors"s,  HyperParameter::nearestNodeNeighbors},
    {"-queryDepth"s,            HyperParameter::queryDepth},
    {"-targetSplitSize"s,       HyperParameter::targetSplitSize},
    {"-minSplitSize"s,          HyperParameter::minSplitSize},
    {"-maxSplitSize"s,          HyperParameter::maxSplitSize},
    {"-searchNeighbors"s,       HyperParameter::searchNeighbors},
    {"-searchDepth"s,           HyperParameter::searchDepth},
    {"-maxSearchesQueued"s,     HyperParameter::maxSearchesQueued}
};



int main(int argc, char *argv[]){
    
    static const std::endian dataEndianness = std::endian::big;

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


    std::vector<std::string> options;
    options.reserve(argc-1);
    for (size_t i = 1; i < argc; i+=1){
        options.emplace_back(argv[i]);
    }

    for (const auto option: options){
        size_t nameEnd = option.find('=');
        if (nameEnd == std::string::npos){
            std::cout << "Could not split option from value; no '=' in: " << option << std::endl;
            return EXIT_FAILURE;
        }
        HyperParameter optionEnum;
        try {
            optionEnum = optionNumber.at(option.substr(0,nameEnd));
        } catch (...){
            std::cout << "Unrecognized option: " << option.substr(0,nameEnd) << std::endl;
            return EXIT_FAILURE;
        }

        switch (optionEnum){
            
            case HyperParameter::blockGraphNeighbors:
                indexParams.blockGraphNeighbors = numBlockGraphNeighbors = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case HyperParameter::COMNeighbors:
                indexParams.COMNeighbors = numCOMNeighbors = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case HyperParameter::nearestNodeNeighbors:
                indexParams.nearestNodeNeighbors = maxNearestNodes = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case HyperParameter::queryDepth:
                indexParams.queryDepth = queryDepth = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case HyperParameter::targetSplitSize:
                splitParams.splitThreshold = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case HyperParameter::minSplitSize:
                splitParams.childThreshold = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case HyperParameter::maxSplitSize:
                splitParams.maxTreeSize = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case HyperParameter::searchNeighbors:
                searchParams.searchNeighbors = numberSearchNeighbors = stoul(std::string(option.substr(nameEnd+1)));
                break;
                
            case HyperParameter::searchDepth:
                searchParams.searchDepth = searchQueryDepth = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case HyperParameter::maxSearchesQueued:
                searchParams.maxSearchesQueued = maxNewSearches = stoul(std::string(option.substr(nameEnd+1)));
                break;
                /*
                    searchNeighbors,
                    searchDepth,
                    maxSearchesQueued
                */

        }
    }

    HyperParameterValues parameters{splitParams, indexParams, searchParams};

    //std::string trainDataFilePath("./TestData/train-images.idx3-ubyte");

    

    //DataSet<std::valarray<unsigned char>> mnistDigitsTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<unsigned char,dataEndianness>);

    //std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();

    //std::string uint8FashionPath("./TestData/MNIST-Train-Images-UInt8.bin");
    //DataSet<std::valarray<uint8_t>> uint8FashionData(uint8FashionPath, 28*28, 60'000, &ExtractNumericArray<uint8_t,dataEndianness>);

    std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");
    DataSet<AlignedArray<float>> mnistFashionTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<AlignedArray<float>,dataEndianness>);

    //std::string reserializedTrainFilePath("./TestData/MNIST-Fashion-Train-Uint8.bin");
    //SerializeDataSet<std::valarray<float>, uint8_t, dataEndianness>(mnistFashionTrain, reserializedTrainFilePath);

    std::string testDataFilePath("./TestData/MNIST-Fashion-Data.bin");
    std::string testNeighborsFilePath("./TestData/MNIST-Fashion-Neighbors.bin");
    DataSet<AlignedArray<float>> mnistFashionTest(testDataFilePath, 28*28, 10'000, &ExtractNumericArray<AlignedArray<float>,dataEndianness>);
    DataSet<AlignedArray<uint32_t>> mnistFashionTestNeighbors(testNeighborsFilePath, 100, 10'000, &ExtractNumericArray<AlignedArray<uint32_t>,dataEndianness>);
    
    //std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::seconds>(runEnd - runStart).count() << "s Pointwise Join Calcs " << std::endl;

    //std::cout << "I/O done." << std::endl;


    std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();

    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), mnistFashionTrain.numberOfSamples - 1);
    StlRngFunctor<std::mt19937_64, std::uniform_int_distribution, size_t> rngFunctor(std::move(rngEngine), std::move(rngDist));

    EuclidianTrain<AlignedArray<float>, AlignedArray<float>> splittingScheme(mnistFashionTrain);
    TrainingSplittingScheme splitterFunc(splittingScheme);
    
    RandomProjectionForest rpTreesTrain(size_t(mnistFashionTrain.numberOfSamples), rngFunctor, splitterFunc, parameters.splitParams);


    //std::vector<size_t> trainClassifications(mnistFashionTrain.numberOfSamples);
    


    auto [indexMappings, dataBlocks] = PartitionData<AlignedArray<float>>(rpTreesTrain, mnistFashionTrain);

    
    MetricFunctor<AlignedArray<float>, EuclideanMetricPair> testFunctor(dataBlocks);
    DispatchFunctor<float> testDispatch(testFunctor);

    std::vector<size_t> sizes;
    sizes.reserve(dataBlocks.size());
    for(const auto& block: dataBlocks){
        sizes.push_back(block.size());
    }
    
    
    MetaGraph<float> metaGraph(dataBlocks, parameters.indexParams.COMNeighbors, EuclideanMetricPair());

    std::vector<BlockUpdateContext<float>> blockUpdateContexts = BuildGraph<AlignedArray<float>, float, float>(dataBlocks, metaGraph, testDispatch, std::move(sizes), parameters, std::execution::seq);

    
    
    std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for index building " << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << std::endl;

    std::chrono::time_point<std::chrono::steady_clock> runStart2 = std::chrono::steady_clock::now();

    EuclidianTransform<AlignedArray<float>, AlignedArray<float>> transformingScheme(mnistFashionTest, splitterFunc.target<decltype(splittingScheme)>()->splittingVectors);
    
    std::unordered_set<size_t> splittingIndicies;
    for (auto& leaf: rpTreesTrain.treeLeaves){
        if(leaf.children.first == 0 && leaf.children.second == 0) continue;
        splittingIndicies.insert(leaf.splittingIndex);
    }

    TransformingSplittingScheme transformingFunc(transformingScheme);

    RandomProjectionForest rpTreesTest(mnistFashionTest.numberOfSamples, transformingFunc, splittingIndicies);

    


    size_t numberSearchBlocks = dataBlocks.size();

    GraphVertex<BlockIndecies, float> nullVertex;
    for(size_t i = 0; i<numberSearchNeighbors; i+=1){
        nullVertex.push_back({{0,0}, std::numeric_limits<float>::max()});
    }

    for (auto& context: blockUpdateContexts){
        context.queryContext.querySearchDepth = searchQueryDepth;
        context.queryContext.querySize = numberSearchNeighbors;
    }

    SearchFunctor<AlignedArray<float>, EuclideanMetricPair> searchDist(dataBlocks, mnistFashionTest);
    //SinglePointFunctor<float> searchFunctor(searchDist);

    auto searcherConstructor = [&, blocks = &dataBlocks](const DataSet<AlignedArray<float>>& dataSource, std::span<const size_t> indicies, size_t blockCounter)->
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
    
    SearchQueue searchHints(dataBlocks.size());
    
    for (auto& queue: searchHints){
        queue.reserve(maxNewSearches * 10'000 / dataBlocks.size());
    }
    
    for (size_t i = 0; auto& testBlock: searchContexts){
        
        
        for (size_t j = 0; auto& context: testBlock){
            context.blocksJoined[i] = true;
            searchDist.SetBlock(i);
            GraphVertex<size_t, float> initNeighbors = blockUpdateContexts[i].queryContext.queryHint;
            blockUpdateContexts[i].queryContext.Query(initNeighbors, context.dataIndex, searchDist);
            size_t hintsAdded = 0;
            context.blocksJoined[i] = true;
            context.currentNeighbors = nullVertex;
            ConsumeVertex(context.currentNeighbors, initNeighbors, i);
            std::sort(context.currentNeighbors.begin(), context.currentNeighbors.end(), NeighborDistanceComparison<BlockIndecies, float>);
            for (const auto& result: context.currentNeighbors){
                
                for (const auto& resultNeighbor: blockUpdateContexts[i].currentGraph[result.first]){
                    if (!context.blocksJoined[resultNeighbor.first.blockNumber]) {
                        context.blocksJoined[resultNeighbor.first.blockNumber] = true;
                        searchHints[resultNeighbor.first.blockNumber].push_back({{i,j}, resultNeighbor.first.dataIndex});
                        hintsAdded++;
                        
                    }
                }
                if (hintsAdded >= maxNewSearches) break;
            }
            context.currentNeighbors.JoinPrep();
            //std::cout << hintsAdded << std::endl;
            j++;
        }
        i++;
    }
    size_t searchUpdates = 1;
    while(searchUpdates){
        searchUpdates = 0;
        for (size_t i = 0; auto& hintMap: searchHints){
            for (size_t j = 0; const auto& hint: hintMap){
                
                GraphVertex<size_t, float> newNodes = BlockwiseSearch(searchContexts[hint.first.blockNumber][hint.first.dataIndex],
                                                                                blockUpdateContexts[i].queryContext,
                                                                                hint.second,
                                                                                searchDist);
                
                searchUpdates += newNodes.size();

                QueueSearches(blockUpdateContexts[i],
                                searchContexts[hint.first.blockNumber][hint.first.dataIndex],
                                hint.first,
                                newNodes,
                                searchHints,
                                maxNewSearches);


                
            }
            hintMap.clear();
            i++;
        }
    }
    std::chrono::time_point<std::chrono::steady_clock> runEnd2 = std::chrono::steady_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd2 - runStart2).count() << "s test set search " << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd2 - runStart2).count() << std::endl;


    

    std::vector<std::vector<size_t>> results(mnistFashionTest.samples.size());
    for (size_t i = 0; auto& testBlock: searchContexts){
        for (size_t j = 0; auto& context: testBlock){
            GraphVertex<BlockIndecies, float>& result = context.currentNeighbors;
            size_t testIndex = testMappings.blockIndexToSource[{i,j}];
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
    std::cout << (recall * 100) << std::endl;
    //std::cout << "Recall: " << (recall * 100) << "%" << std::endl;
    
    //WeightedGraphEdges graphEdges = NeighborsOutOfBlock(mnistFashionTestNeighbors, trainMapper.sourceToBlockIndex, testClassifications);

    //for (size_t i = 0; i < trainMapper.sourceToBlockIndex.size(); i += 1){
    //    trainClassifications[i] = trainMapper.sourceToBlockIndex[i].blockNumber;
    //}

    //SerializeCOMS(metaGraph.points, "./TestData/MNIST-Fashion-Train-COMs.bin");
    //SerializeMetaGraph(graphEdges, "./TestData/MNIST-Fashion-Test-MetaGraphEdges.bin");
    //SerializeVector<size_t>(trainClassifications, "./TestData/MNIST-Fashion-Train-SplittingIndicies.bin");
    return 0;
}