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


#include "MNISTData.hpp"
#include "NND/SpaceMetrics.hpp"
#include "NND/GraphStructures.hpp"
#include "NND/Algorithm.hpp"
#include "NND/MetaGraph.hpp"

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"

#include "Utilities/DataSerialization.hpp"
#include "Utilities/DataDeserialization.hpp"


//namespace chrono = std::chrono;
using namespace nnd;

template<TriviallyCopyable IndexType, typename DataEntry, typename FloatType>
struct SubProblemData{
    Graph<IndexType, FloatType>& subGraph;
    const DataBlock<DataEntry>& dataBlock;
};

struct NodeTracker{

    using reference = std::vector<bool>::reference;
    using size_type = std::vector<bool>::size_type;

    std::vector<bool> flags;

    NodeTracker(size_t graphSize): flags(graphSize, false){};

    reference operator[](size_type i){
        return flags[i];
    }

    reference operator[](BlockIndex i){
        //Assuming block index lines up here;
        return flags[i.dataIndex];
    }

};


template<TriviallyCopyable IndexType, typename DataEntry, typename FloatType>
Graph<IndexType, FloatType> QuerySubGraph(SubProblemData<IndexType, DataEntry, FloatType> subGraphA,
                                          SubProblemData<IndexType, DataEntry, FloatType> subGraphB,
                                          const std::vector<IndexType>& queryPoints,
                                          const GraphVertex<BlockIndex, FloatType>& queryHint,
                                          int numCandidates,
                                          SpaceMetric<DataEntry, FloatType>  distanceFunctor){

    //Initialize results with queryHint
    Graph<IndexType, FloatType> retGraph(queryPoints.size(), numCandidates);
    for (int i = 0; i < numCandidates; i+=1){
        for(const auto& vertex: retGraph){
            vertex.neighbors.push_back(queryHint.neighbors[i]);
        }
    }
    
    for (size_t i = 0; i<queryPoints.size(); i += 1){
        GraphVertex<IndexType, FloatType>& vertex = retGraph[i];
        DataEntry queryData = subGraphB.dataBlock[queryPoints[i]];
        
        NodeTracker nodesVisited(subGraphA.dataBlock.size(), false);
        GraphVertex<IndexType, FloatType> newState(vertex);
        bool breakVar = false;
        while (!breakVar){
            breakVar = true;
            for (const auto& neighbor: vertex){
                const GraphVertex<IndexType, FloatType>& currentNeighbor = subGraphA.subGraph[neighbor.first];
                for (const auto& joinTarget: currentNeighbor){
                    if (nodesVisited[joinTarget.first] == true) continue;
                    nodesVisited[joinTarget.first] = true;
                    FloatType distance = distanceFunctor(queryData, subGraphA.dataBlock[joinTarget.first]);
                    if (distance < newState[0].second){
                        newState.PushNeigbor({joinTarget.first, distance});
                        breakVar = false;
                    }
                }
            }
            std::swap(vertex, newState);
        }
    }
}


// Think about adding neighbor updates into this. I need to tweak neighbor storing.
//NND to find an approximation of the closest pair of points between clusters.
template<TriviallyCopyable IndexType, typename DataEntry, typename FloatType>
void NearestNodes(SubProblemData<IndexType, DataEntry, FloatType> subGraphA,
                  SubProblemData<IndexType, DataEntry, FloatType> subGraphB,
                  const GraphVertex<BlockIndex, FloatType>& queryHintA,
                  const GraphVertex<BlockIndex, FloatType>& queryHintB,
                  SpaceMetric<DataEntry, FloatType>  distanceFunctor){

    std::pair<size_t, size_t> bestPair;
    FloatType bestDistance;
    NodeTracker nodesVisitedA(subGraphA.dataBlock.size());
    NodeTracker nodesVisitedB(subGraphA.dataBlock.size());

    for(const auto& starterA: queryHintA.neighbors){
        nodesVisitedA[starterA.first] = true;
        for (const auto& starterB: queryHintB.neighbors){
            nodesVisitedB[starterB.first] = true;
            FloatType distance = distanceFunctor(subGraphA.dataBlock[starterA.first], subGraphB.dataBlock[starterB.first]);
            if (distance < bestDistance){
                bestDistance = distance;
                bestPair = {starterA.first.dataIndex, starterB.first.dataIndex};
            }
        }
    }

    bool breakVar = false;
    while (!breakVar){
        breakVar = true;
        for (const auto& neighborA: subGraphA.dataBlock[bestPair.first]){
            if (nodesVisitedA[neighborA.first]) continue;
            nodesVisitedA[neighborA.first] = true;
            FloatType distance = distanceFunctor(subGraphA.dataBlock[neighborA.first.dataIndex], subGraphB.dataBlock[bestDistance.second]);
            if (distance < bestDistance){
                bestDistance = distance;
                bestPair.first = neighborA.dataIndex;
                breakVar = false;
            }
        }
        for (const auto& neighborB: subGraphB.dataBlock[bestPair.second]){
            if (nodesVisitedB[neighborB.first]) continue;
            nodesVisitedB[neighborB.first] = true;
            FloatType distance = distanceFunctor(subGraphA.dataBlock[bestPair.first], subGraphB.dataBlock[neighborB.first.dataIndex]);
            if (distance < bestDistance){
                bestDistance = distance;
                bestPair.second = neighborB.dataIndex;
                breakVar = false;
            }
        }
    }
    
}

template<TriviallyCopyable IndexType, typename DataEntry, typename FloatType, size_t queueMargin>
GraphVertex<IndexType, FloatType> QueryCOMNeighbors(const std::valarray<FloatType>& centerOfMass,
                                                     SubProblemData<IndexType, DataEntry, FloatType> subProb, 
                                                     int numCandidates,
                                                     COMMetric<DataEntry, std::valarray<FloatType>, FloatType> distanceFunctor){

    GraphVertex<IndexType, FloatType> COMneighbors(numCandidates);
    ComparisonQueue<IndexType> cmpQueue(numCandidates*queueMargin);
    //Just gonna dummy it and select the first few nodes.
    for (size_t i = 0; i < numCandidates; i+=1){
        COMneighbors.neighbors.push_back({i,
                                          distanceFunctor(centerOfMass, subProb.dataBlock.blockData[i])});
        cmpQueue.PushQueue(i);
    }
    ComparisonQueue<size_t> joinQueue(numCandidates*queueMargin*2);
    //std::vector<FloatType> distances(0);
    //distances.reserve(numCandidates*queueMargin*2);
    NeighborSearchFunctor searchFunctor;
    while(cmpQueue.size()>0){
        for (const auto& compareCandidate: cmpQueue.queue){
            for (const auto& joinTarget: subProb.subGraph[compareCandidate].neighbors){
                searchFunctor.searchValue = joinTarget.first;
                //Check to see if A is already a neighbor of B, if so, bingo
                auto result = std::find_if(COMneighbors.neighbors.begin(), COMneighbors.neighbors.end(), searchFunctor);
                if(result != COMneighbors.neighbors.end()) joinQueue.PushQueue(joinTarget.first);
            }
        }
        cmpQueue.FlushQueue();

        for (const auto& joinTarget: joinQueue.queue){
            FloatType distance = distanceFunctor(centerOfMass, subProb.dataBlock.blockData[joinTarget]);
            
            if (distance <  COMneighbors.neighbors[0].second){
                COMneighbors.PushNeigbor({joinTarget, distance});
                cmpQueue.PushQueue(joinTarget);
            }
        }
        joinQueue.FlushQueue();
    }

    return COMneighbors;
}
/*
template<typename DataEntry, typename FloatType>
auto CreateBlockDistanceStrategy(const DataBlock<DataEntry>& dataBlockA,
                                 const DataBlock<DataEntry>& dataBlockB,
                                 SpaceMetric<DataEntry, FloatType>  distanceFunctor){
    auto retLambda = [&](const std::vector<size_t>& pointAIndecies, const std::vector<size_t>& pointBIndecies)->std::vector<FloatType>{
        
    }
}
*/



//DataBlock(const DataSet<DataEntry>& dataSource, std::span<size_t> dataPoints, SpaceMetric<DataEntry, FloatType> metric, size_t blockNumber):

int main(){

    static const std::endian dataEndianness = std::endian::big;

    //std::string trainDataFilePath("./TestData/train-images.idx3-ubyte");

    std::string testDataFilePath("./TestData/MNIST-Fashion-Data.bin");
    std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");
    std::string testNeighborsFilePath("./TestData/MNIST-Fashion-Neighbors.bin");
    

    //DataSet<std::valarray<unsigned char>> mnistDigitsTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<unsigned char,dataEndianness>);

    //DataSet<std::valarray<float>> mnistFashionTest(testDataFilePath, 28*28, 10'000, &ExtractNumericArray<float,dataEndianness>);
    DataSet<std::valarray<float>> mnistFashionTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<float,dataEndianness>);
    //DataSet<std::valarray<int32_t>> mnistFashionTestNeighbors(testNeighborsFilePath, 100, 10'000, &ExtractNumericArray<int32_t,dataEndianness>);
    

    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), mnistFashionTrain.numberOfSamples - 1);
    StlRngFunctor<std::mt19937_64, std::uniform_int_distribution, size_t> rngFunctor(std::move(rngEngine), std::move(rngDist));

    EuclidianTrain<float, double> splittingScheme(mnistFashionTrain);
    TrainingSplittingScheme splitterFunc(splittingScheme);
    
    RandomProjectionForest rpTreesTrain(size_t(mnistFashionTrain.numberOfSamples), rngFunctor, splitterFunc);


    std::vector<size_t> trainClassifications(mnistFashionTrain.numberOfSamples);
    

    


    DataMapper<std::valarray<float>> trainMapper(mnistFashionTrain);
    CrawlTerminalLeaves(rpTreesTrain, trainMapper);



    //EuclidianTransform<float, double> transformingScheme(mnistFashionTest, splitterFunc.target<EuclidianTrain<float, double>>()->splittingVectors);
    //std::unordered_set<size_t> splittingIndicies;
    //for (auto& leaf: rpTreesTrain.treeLeaves){
    //    if(leaf.children.first == 0 && leaf.children.second == 0) continue;
    //    splittingIndicies.insert(leaf.splittingIndex);
    //}

    //TransformingSplittingScheme transformingFunc(transformingScheme);

    //RandomProjectionForest rpTreesTest(mnistFashionTest.numberOfSamples, transformingFunc, splittingIndicies);


    //std::vector<size_t> testClassifications(mnistFashionTest.numberOfSamples);

    //auto testClassificationFunction = [&testClassifications, &trainMapper](size_t splittingIndex, std::span<const size_t> indicies){
    //    for (const auto& index : indicies){
    //        testClassifications[index] = trainMapper.splitToBlockNum.at(splittingIndex);
    //    }
    //};

    //DataMapper<std::valarray<float>> testMapper(mnistFashionTest);
    //CrawlTerminalLeaves(rpTreesTest, testClassificationFunction);

    std::vector<Graph<BlockIndex, double>> blockGraphs(0);
    blockGraphs.reserve(trainMapper.dataBlocks.size());
    for (const auto& dataBlock : trainMapper.dataBlocks){
        Graph<BlockIndex, double> blockGraph(dataBlock.blockData.size(), size_t(5));
        BruteForceBlock<std::valarray<float>, double>(blockGraph, 5, dataBlock, EuclideanNorm<float, double>);
        blockGraphs.push_back(std::move(blockGraph));
    }
    
    MetaGraph metaGraph(trainMapper.dataBlocks, 10);
    std::vector<GraphVertex<BlockIndex, double>> queryHints;
    for (size_t i = 0; i<metaGraph.points.size(); i += 1){
        SubProblemData subProb{blockGraphs[i], trainMapper.dataBlocks[i]};
        queryHints.push_back(QueryCOMNeighbors<BlockIndex, std::valarray<float>, double, 2>(metaGraph.points[i].centerOfMass,
                                                                          subProb,
                                                                          5,
                                                                          EuclideanNorm<float, double>));

    }

    /*
    struct SubProblemData{
        Graph<IndexType, FloatType>& subGraph;
        const DataBlock<DataEntry>& dataBlock;
    };
    template<TriviallyCopyable IndexType, typename DataEntry, typename FloatType, size_t queueMargin>
    GraphVertex<IndexType, FloatType> QueryCOMNeighbors(const std::valarray<FloatType>& centerOfMass,
                                                     SubProblemData<IndexType, DataEntry, FloatType> subProb, 
                                                     int numCandidates,
                                                     COMMetric<DataEntry, FloatType> distanceFunctor)

    */



    //WeightedGraphEdges graphEdges = NeighborsOutOfBlock(mnistFashionTestNeighbors, trainMapper.sourceToBlockIndex, testClassifications);

    //for (size_t i = 0; i < trainMapper.sourceToBlockIndex.size(); i += 1){
    //    trainClassifications[i] = trainMapper.sourceToBlockIndex[i].blockNumber;
    //}

    //SerializeCOMS(metaGraph.points, "./TestData/MNIST-Fashion-Train-COMs.bin");
    //SerializeMetaGraph(graphEdges, "./TestData/MNIST-Fashion-Test-MetaGraphEdges.bin");
    //SerializeVector<size_t>(trainClassifications, "./TestData/MNIST-Fashion-Train-SplittingIndicies.bin");


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