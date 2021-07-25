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
#include <valarray>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <bit>
#include <ranges>
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

#include "NND/Search.hpp"

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"

#include "Utilities/DataSerialization.hpp"
#include "Utilities/DataDeserialization.hpp"


//namespace chrono = std::chrono;
using namespace nnd;


// Two member struct with the following properties. hash({x,y}) == hash({y,x}) and {x,y} == {y,x}
// This way a set can be used to queue up an operation between two blocks without worrying which is first or second.
template<typename IndexType>
struct ComparisonKey{
    IndexType first;
    IndexType second;
};

template<typename IndexType>
bool operator==(ComparisonKey<IndexType> lhs, ComparisonKey<IndexType> rhs){
    return (lhs.first == rhs.first && lhs.second == rhs.second) ||
           (lhs.first == rhs.second && lhs.second == rhs.first);
}

// If first = second, I screwed up before calling this
template<typename IndexType>
struct std::hash<ComparisonKey<IndexType>>{

    size_t operator()(const ComparisonKey<IndexType>& key) const noexcept{
        return std::hash<size_t>()(key.first) ^ std::hash<size_t>()(key.second);
    };

};

//(const DataSet<DataEntry>& dataSource, std::span<const size_t> dataPoints, size_t blockNumber)

template<std::integral BlockNumberType,  typename DataEntry>
std::pair<IndexMaps<BlockNumberType>, std::vector<DataBlock<DataEntry>>> PartitionData(const RandomProjectionForest& treeToMap, const DataSet<DataEntry>& sourceData){
    //There might be a more elegant way to do this with templates. I tried.
    auto boundContructor = [](const DataSet<DataEntry>& dataSource, std::span<const size_t> dataPoints, size_t blockNumber){ 
        return DataBlock<DataEntry>(dataSource, dataPoints, blockNumber);
    };
    
    DataMapper<DataEntry, DataBlock<DataEntry>, decltype(boundContructor)> dataMapper(sourceData, boundContructor);
    CrawlTerminalLeaves(treeToMap, dataMapper);
    
    std::vector<DataBlock<DataEntry>> retBlocks = std::move(dataMapper.dataBlocks);
    IndexMaps<BlockNumberType> retMaps = {
        std::move(dataMapper.splitToBlockNum),
        std::move(dataMapper.blockIndexToSource),
        std::move(dataMapper.sourceToBlockIndex),
        std::move(dataMapper.sourceToSplitIndex)
    };

    return {retMaps, std::move(retBlocks)};
}

template<std::integral DataIndexType, typename DataEntry, typename DataView, typename DistType>
std::vector<Graph<DataIndexType, DistType>> InitializeBlockGraphs(const std::vector<DataBlock<DataEntry>>& dataBlocks, const size_t numNeighbors, SpaceMetric<DataView, DataView, DistType> distanceFunctor){
    
    std::vector<Graph<DataIndexType, DistType>> blockGraphs(0);
    blockGraphs.reserve(dataBlocks.size());
    for (const auto& dataBlock : dataBlocks){
        blockGraphs.push_back(BruteForceBlock<DataIndexType, DataEntry, DataView, DistType>(numNeighbors, dataBlock, distanceFunctor));
    }

    return blockGraphs;
}

template<typename BlockNumberType, typename DataIndexType, typename DistType>
Graph<BlockIndecies, DistType> ToBlockIndecies(const Graph<DataIndexType, DistType>& blockGraph, const BlockNumberType blockNum){
    Graph<BlockIndecies, DistType> newGraph(blockGraph.size(), blockGraph[0].size());
    for (size_t j = 0; const auto& vertex: blockGraph){
        newGraph[j].resize(blockGraph[j].size());
        for(size_t k = 0; const auto& neighbor: vertex){
            newGraph[j][k] = {{blockNum, neighbor.first}, neighbor.second};
            k++;
        }   
        j++;
    }
    return newGraph;
}

template <std::integral BlockNumberType, std::integral DataIndexType, typename DataEntry, typename DataView, typename DistType, typename COMExtent>
Graph<DataIndexType, DistType> GenerateQueryHints(const std::vector<DataBlock<DataEntry>>& dataBlocks,
                                                                    const std::vector<Graph<DataIndexType, DistType>>& blockGraphs,
                                                                    const MetaGraph<BlockNumberType, DistType>& metaGraph,
                                                                    const size_t numNeighbors,
                                                                    SpaceMetric<AlignedSpan<const COMExtent>, DataView, DistType> distanceFunctor){
    
    Graph<DataIndexType, DistType> retGraph;
    for(size_t i = 0; i<metaGraph.points.size(); i+=1){
        retGraph.push_back(QueryHintFromCOM<DataIndexType, DataEntry, DataView, DistType, COMExtent>(metaGraph.points[i].centerOfMass, 
                                                                                           {blockGraphs[i], dataBlocks[i]}, 
                                                                                           numNeighbors, 
                                                                                           distanceFunctor));
    }

    return retGraph;

}


template<std::integral BlockNumberType, std::integral DataIndexType, typename DataEntry, typename DataView, typename DistType, typename COMExtent>
std::vector<BlockUpdateContext<BlockNumberType, DataIndexType, DataEntry, DataView, DistType>> InitializeBlockContexts(const std::vector<DataBlock<DataEntry>>& dataBlocks,
                                                                                                             const std::vector<Graph<DataIndexType, DistType>>& blockGraphs,
                                                                                                             const MetaGraph<BlockNumberType, COMExtent>& metaGraph,
                                                                                                             Graph<DataIndexType, DistType>& queryHints,
                                                                                                             const int queryDepth,
                                                                                                             SpaceMetric<DataView, DataView, DistType> distanceFunctor,
                                                                                                             BatchMetric<DataView, DataView, std::vector<DistType>> batchingFunctor){
                                                                        
    std::vector<BlockUpdateContext<BlockNumberType, DataIndexType, DataEntry, DataView, DistType>> blockUpdateContexts;
    blockUpdateContexts.reserve(dataBlocks.size());
    //template<typename DataIndexType, typename DataEntry, typename DistType, typename COMExtentType>
    for (size_t i = 0; i<dataBlocks.size(); i+=1){


        blockUpdateContexts.emplace_back(SubProblemData{blockGraphs[i], dataBlocks[i]},
                                         QueryContextInitArgs<DataIndexType, DataView, DistType>(queryHints[i], distanceFunctor, batchingFunctor),
                                         metaGraph.verticies.size(), queryDepth);

        blockUpdateContexts.back().currentGraph = ToBlockIndecies(blockGraphs[i], i);
        blockUpdateContexts.back().blockJoinTracker[i] = true;
    }

    return blockUpdateContexts;
}

template<std::integral BlockNumberType, std::integral DataIndexType, typename DistType>
using InitialJoinHints = std::unordered_map<ComparisonKey<BlockNumberType>, std::tuple<DataIndexType, DataIndexType, DistType>>;

template<std::integral BlockNumberType, std::integral DataIndexType, typename DataEntry, typename DataView, typename DistType>
std::pair<Graph<BlockNumberType, DistType>, InitialJoinHints<BlockNumberType, DataIndexType, DistType>> NearestNodeDistances(const std::vector<BlockUpdateContext<BlockNumberType, DataIndexType, DataEntry, DataView, DistType>>& blockUpdateContexts,
                                                        const MetaGraph<BlockNumberType, DistType>& metaGraph,
                                                        const size_t maxNearestNodeNeighbors){

    std::unordered_set<ComparisonKey<BlockNumberType>> nearestNodeDistQueue;

    for (size_t i = 0; const auto& vertex: metaGraph.verticies){
        for (const auto& neighbor: vertex){
            nearestNodeDistQueue.insert({i, neighbor.first});
        }
        i++;
    }
    
    std::vector<ComparisonKey<BlockNumberType>> distancesToCompute;
    distancesToCompute.reserve(nearestNodeDistQueue.size());
    for (const auto& pair: nearestNodeDistQueue){
        distancesToCompute.push_back(pair);
    }
    
    std::vector<std::tuple<DataIndexType, DataIndexType, DistType>> nnDistanceResults(nearestNodeDistQueue.size());
    auto nnDistanceFunctor = [&](const ComparisonKey<BlockNumberType> blockNumbers) -> std::tuple<DataIndexType, DataIndexType, DistType>{
        return blockUpdateContexts[blockNumbers.first].queryContext * blockUpdateContexts[blockNumbers.second].queryContext;
    };

    std::transform(std::execution::unseq, distancesToCompute.begin(), distancesToCompute.end(), nnDistanceResults.begin(), nnDistanceFunctor);

    std::unordered_map<ComparisonKey<BlockNumberType>, std::tuple<DataIndexType, DataIndexType, DistType>> blockJoinHints;

    for (size_t i = 0; i<distancesToCompute.size(); i += 1){
        blockJoinHints[distancesToCompute[i]] = nnDistanceResults[i];
    }

    Graph<BlockNumberType, DistType> nearestNodeDistances(metaGraph.verticies.size(), maxNearestNodeNeighbors);
    for(size_t i = 0; const auto& result: nnDistanceResults){
        
        nearestNodeDistances[distancesToCompute[i].first].push_back({distancesToCompute[i].second,
                                                                     std::get<2>(result)});
        nearestNodeDistances[distancesToCompute[i].second].push_back({distancesToCompute[i].first,
                                                                      std::get<2>(result)});
        //nearestNeighbors.push_back({pair.first, std::get<2>(pair.second)});
        i++;
    }

    auto sortFunctor = [=] (GraphVertex<BlockNumberType, DistType>& vertex){
        std::sort(std::execution::unseq, vertex.begin(), vertex.end(), NeighborDistanceComparison<BlockNumberType, DistType>);
        vertex.resize(maxNearestNodeNeighbors);
    };
    std::for_each(std::execution::unseq, nearestNodeDistances.begin(), nearestNodeDistances.end(), sortFunctor);

    return {std::move(nearestNodeDistances), std::move(blockJoinHints)};
}


template<std::integral BlockNumberType, std::integral DataIndexType, typename DataEntry, typename DataView, typename DistType>
void StitchBlocks(const Graph<BlockNumberType, DistType>& nearestNodeDistances,
                  const InitialJoinHints<BlockNumberType, DataIndexType, DistType>& stitchHints,
                  std::vector<BlockUpdateContext<BlockNumberType, DataIndexType, DataEntry, DataView, DistType>>& blockUpdateContexts){

    std::unordered_set<ComparisonKey<size_t>> initBlockJoinQueue;
    for(size_t i = 0; const auto& vertex: nearestNodeDistances){
        for(size_t j = 0; j<nearestNodeDistances[0].size(); j+=1){
            initBlockJoinQueue.insert({i, vertex[j].first});
        }
        i++;
    }

    std::vector<ComparisonKey<size_t>> initBlockJoins;
    initBlockJoins.reserve(initBlockJoinQueue.size());
    for (const auto& pair: initBlockJoinQueue){
        blockUpdateContexts[pair.first].blockJoinTracker[pair.second] = true;
        blockUpdateContexts[pair.second].blockJoinTracker[pair.first] = true;
        initBlockJoins.push_back(pair);
    }
    

    std::vector<std::pair<JoinResults<DataIndexType, DistType>, JoinResults<DataIndexType, DistType>>> initUpdates(initBlockJoins.size());
    

    auto initBlockJoin = [&](const ComparisonKey<size_t> blockNumbers) -> std::pair<JoinResults<DataIndexType, DistType>, JoinResults<DataIndexType, DistType>>{
        
        auto [blockNums, stitchHint] = *(stitchHints.find(blockNumbers));
        if (blockNums.first != blockNumbers.first) stitchHint = {std::get<1>(stitchHint), std::get<0>(stitchHint), std::get<2>(stitchHint)};
        auto blockLHS = blockUpdateContexts[blockNumbers.first];
        auto blockRHS = blockUpdateContexts[blockNumbers.second];
        JoinHints<DataIndexType> LHShint;
        LHShint[std::get<0>(stitchHint)] = {std::get<1>(stitchHint)};
        JoinHints<DataIndexType> RHShint;
        RHShint[std::get<1>(stitchHint)] = {std::get<0>(stitchHint)};

        DistanceCache<DistType> distanceCache;
        auto cachingDistanceFunctor = blockRHS.queryContext.defaultQueryFunctor.CachingFunctor(distanceCache);
        auto cachedDistanceFunctor = blockLHS.queryContext.defaultQueryFunctor.CachedFunctor(distanceCache);
        
        std::pair<JoinResults<DataIndexType, DistType>, JoinResults<DataIndexType, DistType>> retPair;
        retPair.first = BlockwiseJoin(LHShint,
                                      blockLHS.currentGraph,
                                      blockLHS.leafGraph,
                                      blockLHS.dataBlock,
                                      blockRHS.queryContext,
                                      cachingDistanceFunctor);


        retPair.second = BlockwiseJoin(RHShint,
                                      blockRHS.currentGraph,
                                      blockRHS.leafGraph,
                                      blockRHS.dataBlock,
                                      blockLHS.queryContext,
                                      cachedDistanceFunctor);

        return retPair;
    };

    std::transform(std::execution::seq, initBlockJoins.begin(), initBlockJoins.end(), initUpdates.begin(), initBlockJoin);
    int initGraphUpdates(0);
    for (size_t i = 0; i<initUpdates.size(); i += 1){
        ComparisonKey<size_t> blocks = initBlockJoins[i];
        std::pair<JoinResults<DataIndexType, DistType>, JoinResults<DataIndexType, DistType>>& updates = initUpdates[i];
        for (auto& result: updates.first){
            initGraphUpdates += ConsumeVertex(blockUpdateContexts[blocks.first].currentGraph[result.first], result.second, blocks.second);
        }
        for (auto& result: updates.second){
            initGraphUpdates += ConsumeVertex(blockUpdateContexts[blocks.second].currentGraph[result.first], result.second, blocks.first);
        }

    }

    
    

    
    //Initial filling of comparison targets.
    std::vector<ComparisonMap<size_t, size_t>> queueMaps;
    queueMaps.reserve(blockUpdateContexts.size());
    for (size_t i = 0; i<blockUpdateContexts.size(); i+=1){
        queueMaps.push_back(InitializeComparisonQueues<size_t, size_t, float>(blockUpdateContexts[i].currentGraph, i));
    }
    
    //std::vector<JoinMap<size_t, size_t>> joinHints;

    for(size_t i = 0; i<queueMaps.size(); i+=1){
        ComparisonMap<size_t, size_t>& comparisonMap = queueMaps[i];
        
        blockUpdateContexts[i].joinsToDo = InitializeJoinMap<BlockNumberType, DataIndexType, DataEntry, DataView, DistType>(blockUpdateContexts, comparisonMap, blockUpdateContexts[i].blockJoinTracker);
    }
}



enum class HyperParameter{
    blockGraphNeighbors,
    COMNeighbors,
    nearestNodeNeighbors,
    queryDepth,
    targetSplitSize,
    minSplitSize,
    maxSplitSize
};


using std::operator""s;
static const std::unordered_map<std::string, HyperParameter> optionNumber = {
    {"-blockGraphNeighbors"s,   HyperParameter::blockGraphNeighbors},
    {"-COMNeighbors"s,          HyperParameter::COMNeighbors},
    {"-nearestNodeNeighbors"s,  HyperParameter::nearestNodeNeighbors},
    {"-queryDepth"s,            HyperParameter::queryDepth},
    {"-targetSplitSize"s,       HyperParameter::targetSplitSize},
    {"-minSplitSize"s,          HyperParameter::minSplitSize},
    {"-maxSplitSize"s,          HyperParameter::maxSplitSize}
};



int main(int argc, char *argv[]){

    static const std::endian dataEndianness = std::endian::big;

    size_t numBlockGraphNeighbors = 10;
    size_t numCOMNeighbors = 30;
    size_t maxNearestNodes = 3;
    size_t queryDepth = 2;

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
                numBlockGraphNeighbors = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case HyperParameter::COMNeighbors:
                numCOMNeighbors = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case HyperParameter::nearestNodeNeighbors:
                maxNearestNodes = stoul(std::string(option.substr(nameEnd+1)));
                break;

            case HyperParameter::queryDepth:
                queryDepth = stoul(std::string(option.substr(nameEnd+1)));
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

        }
    }

    /*
    struct SplittingHeurisitcs{
    int splits = 16;
    int splitThreshold = 80;
    int childThreshold = 32;
    int maxTreeSize = 130;
    };

    */

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
    
    MetricFunctor<AlignedArray<float>, EuclideanMetricPair, float> testFunctor;
    DispatchFunctor<float> testDispatch(testFunctor);
    testDispatch.SetBlocks(2,3);

    //std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::seconds>(runEnd - runStart).count() << "s Pointwise Join Calcs " << std::endl;

    std::cout << "I/O done." << std::endl;


    std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();

    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), mnistFashionTrain.numberOfSamples - 1);
    StlRngFunctor<std::mt19937_64, std::uniform_int_distribution, size_t> rngFunctor(std::move(rngEngine), std::move(rngDist));

    EuclidianTrain<AlignedArray<float>, AlignedArray<float>> splittingScheme(mnistFashionTrain);
    TrainingSplittingScheme splitterFunc(splittingScheme);
    
    RandomProjectionForest rpTreesTrain(size_t(mnistFashionTrain.numberOfSamples), rngFunctor, splitterFunc, splitParams);


    //std::vector<size_t> trainClassifications(mnistFashionTrain.numberOfSamples);
    


    auto [indexMappings, dataBlocks] = PartitionData<size_t, AlignedArray<float>>(rpTreesTrain, mnistFashionTrain);

    
    
    std::vector<Graph<size_t, float>> blockGraphs = InitializeBlockGraphs<size_t, AlignedArray<float>, AlignedSpan<const float>, float>(dataBlocks, numBlockGraphNeighbors, EuclideanNorm<AlignedSpan<const float>, AlignedSpan<const float>, float>);

    MetaGraph<size_t, float> metaGraph(dataBlocks, numCOMNeighbors);

    Graph<size_t, float> queryHints = GenerateQueryHints<size_t, size_t, AlignedArray<float>, AlignedSpan<const float>, float, float>(dataBlocks, blockGraphs, metaGraph, numBlockGraphNeighbors, EuclideanNorm<AlignedSpan<const float>, AlignedSpan<const float>, float>);

    std::vector<BlockUpdateContext<size_t, size_t, AlignedArray<float>, AlignedSpan<const float>, float>> blockUpdateContexts = InitializeBlockContexts(dataBlocks, 
                                                                                                                                                        blockGraphs, 
                                                                                                                                                        metaGraph,
                                                                                                                                                        queryHints,
                                                                                                                                                        queryDepth,
                                                                                                                                                        EuclideanNorm<AlignedSpan<const float>, AlignedSpan<const float>, float>,
                                                                                                                                                        EuclideanBatcher);
    
    auto [nearestNodeDistances, stitchHints] = NearestNodeDistances(blockUpdateContexts, metaGraph, maxNearestNodes);
    StitchBlocks(nearestNodeDistances, stitchHints, blockUpdateContexts);
    

    GraphVertex<BlockIndecies, float> nullVertex;
    for(size_t i = 0; i<10; i+=1){
        nullVertex.push_back({{0,0}, std::numeric_limits<float>::max()});
    }
    int iteration(1);
    int graphUpdates(1);
    while(graphUpdates>0){
        graphUpdates = 0;
        for(size_t i = 0; i<blockUpdateContexts.size(); i+=1){
            for (auto& joinList: blockUpdateContexts[i].joinsToDo){
                graphUpdates += UpdateBlocks(blockUpdateContexts[i], blockUpdateContexts[joinList.first]);
                blockUpdateContexts[joinList.first].joinsToDo.erase(i);
            }
        }
        for (auto& context: blockUpdateContexts){
            context.SetNextJoins();
        }

        iteration += 1;
    }

    std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd - runStart).count() << "s total for index building " << std::endl;


    std::chrono::time_point<std::chrono::steady_clock> runStart2 = std::chrono::steady_clock::now();

    EuclidianTransform<AlignedArray<float>, AlignedArray<float>> transformingScheme(mnistFashionTest, splitterFunc.target<decltype(splittingScheme)>()->splittingVectors);
    
    std::unordered_set<size_t> splittingIndicies;
    for (auto& leaf: rpTreesTrain.treeLeaves){
        if(leaf.children.first == 0 && leaf.children.second == 0) continue;
        splittingIndicies.insert(leaf.splittingIndex);
    }

    TransformingSplittingScheme transformingFunc(transformingScheme);

    RandomProjectionForest rpTreesTest(mnistFashionTest.numberOfSamples, transformingFunc, splittingIndicies);

    
    //std::vector<size_t> testClassifications(mnistFashionTest.numberOfSamples);
    /*
    auto testClassificationFunction = [&testClassifications, &trainMapper](size_t splittingIndex, std::span<const size_t> indicies){
        for (const auto& index : indicies){
            testClassifications[index] = trainMapper.splitToBlockNum.at(splittingIndex);
        }
    };
    */


    const size_t numberSearchNeighbors = 15;
    size_t numberSearchBlocks = dataBlocks.size();

    auto searcherConstructor = [=](const DataSet<AlignedArray<float>>& dataSource, std::span<const size_t> indicies, size_t blockCounter)->
                                  std::vector<SearchContext<size_t, size_t, AlignedArray<float>, float>>{
        std::vector<SearchContext<size_t, size_t, AlignedArray<float>, float>> retVec;
        retVec.reserve(indicies.size());
        for(size_t index: indicies){
            retVec.emplace_back(numberSearchNeighbors, numberSearchBlocks, dataSource[index]);
        }
        return retVec;
    };


    DataMapper<AlignedArray<float>, std::vector<SearchContext<size_t, size_t, AlignedArray<float>, float>>, decltype(searcherConstructor)> testMapper(mnistFashionTest, searcherConstructor);
    CrawlTerminalLeaves(rpTreesTest, testMapper);

    auto searchContexts = std::move(testMapper.dataBlocks);

    IndexMaps<size_t> testMappings = {
        std::move(testMapper.splitToBlockNum),
        std::move(testMapper.blockIndexToSource),
        std::move(testMapper.sourceToBlockIndex),
        std::move(testMapper.sourceToSplitIndex)
    };
    
    std::vector<std::unordered_map<BlockIndecies, size_t>> searchHints(dataBlocks.size());

    for (size_t i = 0; auto& testBlock: searchContexts){
        const GraphVertex<size_t, float>& hint = blockUpdateContexts[i].queryContext.queryHint;
        const auto& queryFunctor = blockUpdateContexts[i].queryContext.defaultQueryFunctor;
        for (size_t j = 0; auto& context: testBlock){
            context.blocksJoined[i] = true;
            GraphVertex<size_t, float> initNeighbors = blockUpdateContexts[i].queryContext.QueryHotPath(hint, context.data, 0, queryFunctor);
            for (const auto& result: initNeighbors){
                context.currentNeighbors.push_back({{i, result.first}, result.second});
                for (const auto& resultNeighbor: blockUpdateContexts[i].currentGraph[result.first]){
                    if (resultNeighbor.first.blockNumber != i) searchHints[resultNeighbor.first.blockNumber][{i,j}] = resultNeighbor.first.dataIndex;
                }
            }
            j++;
        }
        i++;
    }
    size_t searchUpdates = 1;
    while(searchUpdates){
        searchUpdates = 0;
        for (size_t i = 0; auto& hintMap: searchHints){
            for (const auto& hint: hintMap){
                
                GraphVertex<size_t, float> newNodes = BlockwiseSearch(searchContexts[hint.first.blockNumber][hint.first.dataIndex],
                                                                                blockUpdateContexts[i].queryContext,
                                                                                hint.second,
                                                                                blockUpdateContexts[i].queryContext.defaultQueryFunctor);
                
                searchUpdates += newNodes.size();

                QueueSearches(blockUpdateContexts[i],
                                searchContexts[hint.first.blockNumber][hint.first.dataIndex],
                                hint.first,
                                newNodes,
                                searchHints);

                
            }
            hintMap = std::unordered_map<BlockIndecies, size_t>();
            i++;
        }
    }
    std::chrono::time_point<std::chrono::steady_clock> runEnd2 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd2 - runStart2).count() << "s test set search " << std::endl;


    /*
    std::vector<Graph<size_t, float>> reflexiveGraphs(0);
    reflexiveGraphs.reserve(testMapper.dataBlocks.size());
    std::vector<Graph<BlockIndecies,float>> nearestNeighbors;
    nearestNeighbors.reserve(testMapper.dataBlocks.size());

    std::vector<JoinMap<size_t, size_t>> testJoinHints(testMapper.dataBlocks.size());

    auto hintBuilder = [](const QueryContext<size_t, size_t, std::valarray<float>, float>& context) -> std::vector<size_t>{
        std::vector<size_t> retVec;
        for (const auto& neighbor: context.queryHint){
            retVec.push_back(neighbor.first);
        }
        return retVec;
    };
    
    //std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();

    for (size_t i=0; const auto& dataBlock : testMapper.dataBlocks){
        Graph<size_t, float> blockGraph = BruteForceBlock<size_t, std::valarray<float>, float>(10, dataBlock, EuclideanNorm<float, float, float>);
        nearestNeighbors.push_back(Graph<BlockIndecies, float>(dataBlock.size(), 10));
        for (size_t j = 0; auto& vertex: nearestNeighbors[i]){
            for(size_t k = 0; k<15; k+=1){
                vertex.push_back({{0,0}, std::numeric_limits<float>::max()});
            }
            testJoinHints[i][i][j] = std::vector<size_t>();
            j++;
        }
        
        i++;
    }

    std::vector<NodeTracker> testJoinTrackers(blockGraphs.size(), NodeTracker(blockGraphs.size()));
    
    iteration = 1;
    graphUpdates = 1;
    while(graphUpdates>0){
        graphUpdates = 0;
        std::vector<std::unordered_map<size_t, JoinResults<size_t, float>>> blockUpdates(nearestNeighbors.size());
        for(size_t i = 0; i<nearestNeighbors.size(); i+=1){
            JoinMap<size_t, size_t>& joinsToDo = testJoinHints[i];
            JoinMap<size_t, size_t> newJoinHints;
            for (auto& joinList: joinsToDo){
                testJoinTrackers[i][joinList.first] = true;
            }
            for (auto& joinList: joinsToDo){
                blockUpdates[i][joinList.first] = BlockwiseSearch(joinList.second, nearestNeighbors[i], testMapper.dataBlocks[i], blockUpdateContexts[joinList.first].queryContext, blockUpdateContexts[joinList.first].queryContext.defaultQueryFunctor);
                NewJoinQueues<size_t, size_t, float>(blockUpdates[i][joinList.first], testJoinTrackers[i], blockUpdateContexts[joinList.first].currentGraph, newJoinHints);
            }
            testJoinHints[i] = std::move(newJoinHints);  
        }

        for (size_t i = 0; i<blockUpdates.size(); i+=1){
            std::unordered_map<size_t, GraphVertex<BlockIndecies, float>> consolidatedResults;
            for (auto& blockResult: blockUpdates[i]){
                for (auto& result: blockResult.second){
                    //GraphVertex<BlockIndecies, double> newVertex;
                    //for (const auto& resultEntry: result.second){
                    //    newVertex.push_back({{blockResult.first, resultEntry.first}, resultEntry.second});
                    //}
                    if(consolidatedResults.find(result.first) == consolidatedResults.end()){
                        consolidatedResults[result.first] = nullVertex;
                    }
                    ConsumeVertex(consolidatedResults[result.first], result.second, blockResult.first);
                }
            }

            for(auto& consolidatedResult: consolidatedResults){
                graphUpdates += ConsumeVertex(nearestNeighbors[i][consolidatedResult.first], consolidatedResult.second);
            }
        }
        //std::cout << graphUpdates << " updates in iteration " << iteration << std::endl;
        //iteration += 1;
    }
    */

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
    std::cout << "Recall: " << (recall * 100) << "%" << std::endl;
    
    //WeightedGraphEdges graphEdges = NeighborsOutOfBlock(mnistFashionTestNeighbors, trainMapper.sourceToBlockIndex, testClassifications);

    //for (size_t i = 0; i < trainMapper.sourceToBlockIndex.size(); i += 1){
    //    trainClassifications[i] = trainMapper.sourceToBlockIndex[i].blockNumber;
    //}

    //SerializeCOMS(metaGraph.points, "./TestData/MNIST-Fashion-Train-COMs.bin");
    //SerializeMetaGraph(graphEdges, "./TestData/MNIST-Fashion-Test-MetaGraphEdges.bin");
    //SerializeVector<size_t>(trainClassifications, "./TestData/MNIST-Fashion-Train-SplittingIndicies.bin");

    //std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::seconds>(runEnd - runStart).count() << "s Pointwise Join Calcs " << std::endl;
    


    //std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::seconds>(runEnd - runStart).count() << "s Nearest Node Calcs " << std::endl;



    return 0;
}