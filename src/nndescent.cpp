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
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <bit>
#include <fstream>
#include <limits>
#include <span>
#include <ranges>
#include <cassert>


#include "Utilities/Data.hpp"
#include "NND/SpaceMetrics.hpp"
#include "NND/GraphStructures.hpp"
#include "NND/Algorithm.hpp"
#include "NND/MetaGraph.hpp"
#include "NND/SubGraphQuerying.hpp"

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"

#include "Utilities/DataSerialization.hpp"
#include "Utilities/DataDeserialization.hpp"


//namespace chrono = std::chrono;
using namespace nnd;





int main(){

    static const std::endian dataEndianness = std::endian::big;

    //std::string trainDataFilePath("./TestData/train-images.idx3-ubyte");

    

    //DataSet<std::valarray<unsigned char>> mnistDigitsTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<unsigned char,dataEndianness>);

    
    std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");
    DataSet<std::valarray<float>> mnistFashionTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<float,dataEndianness>);

    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), mnistFashionTrain.numberOfSamples - 1);
    StlRngFunctor<std::mt19937_64, std::uniform_int_distribution, size_t> rngFunctor(std::move(rngEngine), std::move(rngDist));

    EuclidianTrain<float, double> splittingScheme(mnistFashionTrain);
    TrainingSplittingScheme splitterFunc(splittingScheme);
    
    RandomProjectionForest rpTreesTrain(size_t(mnistFashionTrain.numberOfSamples), rngFunctor, splitterFunc);


    std::vector<size_t> trainClassifications(mnistFashionTrain.numberOfSamples);
    

    

    
    DataMapper<std::valarray<float>> trainMapper(mnistFashionTrain);
    CrawlTerminalLeaves(rpTreesTrain, trainMapper);


    /*
    std::string testDataFilePath("./TestData/MNIST-Fashion-Data.bin");
    std::string testNeighborsFilePath("./TestData/MNIST-Fashion-Neighbors.bin");
    DataSet<std::valarray<float>> mnistFashionTest(testDataFilePath, 28*28, 10'000, &ExtractNumericArray<float,dataEndianness>);
    DataSet<std::valarray<int32_t>> mnistFashionTestNeighbors(testNeighborsFilePath, 100, 10'000, &ExtractNumericArray<int32_t,dataEndianness>);

    EuclidianTransform<float, double> transformingScheme(mnistFashionTest, splitterFunc.target<EuclidianTrain<float, double>>()->splittingVectors);
    
    std::unordered_set<size_t> splittingIndicies;
    for (auto& leaf: rpTreesTrain.treeLeaves){
        if(leaf.children.first == 0 && leaf.children.second == 0) continue;
        splittingIndicies.insert(leaf.splittingIndex);
    }

    TransformingSplittingScheme transformingFunc(transformingScheme);

    RandomProjectionForest rpTreesTest(mnistFashionTest.numberOfSamples, transformingFunc, splittingIndicies);


    std::vector<size_t> testClassifications(mnistFashionTest.numberOfSamples);

    auto testClassificationFunction = [&testClassifications, &trainMapper](size_t splittingIndex, std::span<const size_t> indicies){
        for (const auto& index : indicies){
            testClassifications[index] = trainMapper.splitToBlockNum.at(splittingIndex);
        }
    };

    DataMapper<std::valarray<float>> testMapper(mnistFashionTest);
    CrawlTerminalLeaves(rpTreesTest, testClassificationFunction);
    */

    
    std::vector<Graph<size_t, double>> blockGraphs(0);
    blockGraphs.reserve(trainMapper.dataBlocks.size());
    for (const auto& dataBlock : trainMapper.dataBlocks){
        Graph<size_t, double> blockGraph(dataBlock.blockData.size(), size_t(5));
        BruteForceBlock<size_t, std::valarray<float>, double>(blockGraph, 5, dataBlock, EuclideanNorm<float, float, double>);
        blockGraphs.push_back(std::move(blockGraph));
    }
    
    MetaGraph metaGraph(trainMapper.dataBlocks, 5);

    
    std::vector<QueryContext<size_t, std::valarray<float>, double>> queryContexts;

    for (size_t i = 0; i<metaGraph.verticies.size(); i+=1){
        queryContexts.push_back(QueryContext<size_t, std::valarray<float>, double>::QueryContext(blockGraphs[i],
                                trainMapper.dataBlocks[i],
                                metaGraph.points[i].centerOfMass,
                                5,
                                EuclideanNorm<float, float, double>,
                                EuclideanNorm<double, float, double>));
    }

    for (size_t i = 0; i<metaGraph.verticies.size(); i+=1){
        GraphVertex<size_t, double>& vertex = metaGraph.verticies[i];
        std::sort(vertex.begin(), vertex.end(), NeighborDistanceComparison<size_t, double>);
        for(size_t j = 0; j<vertex.size(); j+=1){
            //Nearest Node distance; results are cached within objects. In fact, this op currently returns void. Maybe return a const ref to... something?
            queryContexts[i] * queryContexts[vertex[j].first];
        }
    }

    Graph<size_t, double> nearestNodeDistances;
    for(auto& context: queryContexts){
        GraphVertex<size_t, double> nearestNeighbors;
        std::unordered_map<size_t, double> distanceMap = std::move(context.distances);
        for(const auto& pair: distanceMap){
            nearestNeighbors.push_back(pair);
        }
        std::sort(nearestNeighbors.begin(), nearestNeighbors.end(), NeighborDistanceComparison<size_t, double>);
        nearestNodeDistances.push_back(std::move(nearestNeighbors));
    }

    //std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();

    for(size_t i = 0; i<nearestNodeDistances.size(); i+=1){
        //Do all of them? Let's try for now. I can add in hyperparam here.
        for(const auto& neighbor: nearestNodeDistances[i]){
            //Compute the nodewise join of the two blocks, results cached
            queryContexts[i] || queryContexts[neighbor.first];
        }
    }

    std::vector<Graph<BlockIndex, double>> updatedBlockGraphs;
    for(auto& context: queryContexts){
        std::unordered_map<size_t, Graph<size_t, double>> candidates(std::move(context.neighborCandidates));
        Graph<BlockIndex, double> blockGraph;
        for (const auto& vertex: blockGraphs[context.dataBlock.blockNumber]){
            GraphVertex<BlockIndex, double> newVert;
            for (const auto& neighbor: vertex){
                newVert.push_back({{context.dataBlock.blockNumber, neighbor.first}, neighbor.second});
            }
            blockGraph.push_back(std::move(newVert));
        }

        for (auto& [blockNum, updateGraph]: candidates){
            for (size_t i = 0; i<updateGraph.size(); i+=1){
                ConsumeVertex(blockGraph[i], updateGraph[i], blockNum);
            }
        }
        updatedBlockGraphs.push_back(std::move(blockGraph));
    };


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








    /*
    Graph<unsigned char> initGraph = ConstructInitialGraph<unsigned char>(digits, 5, rngFunctor, &EuclideanNorm<unsigned char>);
    std::vector<ComparisonQueue> joinQueues = ConstructQueues(digits.numberOfSamples, 100);
    std::vector<ComparisonQueue> candidateQueues = ConstructQueues(digits.numberOfSamples, 10);

    std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();
    std::vector<std::chrono::time_point<std::chrono::steady_clock>> timePoints(0);
    timePoints.push_back(runStart);

    PopulateInitialQueueStates(initGraph, joinQueues);

    std::vector<int> joinsPerCycle;
    int totalJoins = ComputeLocalJoins(digits, initGraph, joinQueues, candidateQueues, &EuclideanNorm<unsigned char>);
    
    joinsPerCycle.push_back(totalJoins);
    timePoints.push_back(std::chrono::steady_clock::now());
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(timePoints[timePoints.size()-1]-timePoints[timePoints.size()-2]).count() << "s for iteration 0." << std::endl;
    std::cout << "Number of joins this iteration: " << totalJoins << std::endl;
    //VerifyGraphState(initGraph);

    for (size_t i = 0; i < 149; i++){
        PopulateJoinQueueStates(initGraph, candidateQueues, joinQueues);
        totalJoins = ComputeLocalJoins(digits, initGraph, joinQueues, candidateQueues, &EuclideanNorm<unsigned char>);
        timePoints.push_back(std::chrono::steady_clock::now());
        std::cout << std::chrono::duration_cast<std::chrono::seconds>(timePoints[timePoints.size()-1]-timePoints[timePoints.size()-2]).count() << "s for iteration "<< i+1 << "." << std::endl;
        joinsPerCycle.push_back(totalJoins);
        std::cout << "Number of joins this iteration: " << totalJoins << std::endl;
        //VerifyGraphState(initGraph);
    }
    */

    // compQueues(0);

    return 0;
}