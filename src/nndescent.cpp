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

template<TriviallyCopyable IndexType, typename DataEntry, std::floating_point FloatType>
struct SubProblemData{
    Graph<IndexType, FloatType>& subGraph;
    const DataBlock<DataEntry>& dataBlock;
};

template<TriviallyCopyable IndexType, typename DataEntry, std::floating_point FloatType>
struct QueryContext{
    const Graph<IndexType, FloatType>& subGraph;
    const DataBlock<DataEntry>& dataBlock;
    const GraphVertex<IndexType, DataEntry> queryHint;
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

//Figure out to make this template take a range as a template parameter
template<TriviallyCopyable IndexType, typename DataEntry, std::floating_point FloatType>
Graph<IndexType, FloatType> QuerySubGraph(SubProblemData<IndexType, DataEntry, FloatType> subGraphA,
                                          SubProblemData<IndexType, DataEntry, FloatType> subGraphB,
                                          const std::vector<IndexType>& queryPoints,
                                          const GraphVertex<IndexType, FloatType>& queryHint,
                                          int numCandidates,
                                          SpaceMetric<DataEntry, DataEntry, FloatType>  distanceFunctor){

    //Initialize results with queryHint
    Graph<IndexType, FloatType> retGraph(queryPoints.size(), numCandidates);

    for(auto& vertex: retGraph){
        for (int i = 0; i < numCandidates; i+=1){
            vertex.neighbors.push_back({queryHint.neighbors[i].first, std::numeric_limits<FloatType>::max()});
        }
    }
    
    for (size_t i = 0; i<queryPoints.size(); i += 1){
        GraphVertex<IndexType, FloatType>& vertex = retGraph[i];
        DataEntry queryData = subGraphB.dataBlock[queryPoints[i]];
        
        NodeTracker nodesVisited(subGraphA.dataBlock.size());
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
            vertex = newState;
        }
    }
    return retGraph;
}


// Think about adding neighbor updates into this. I need to tweak neighbor storing.
//NND to find an approximation of the closest pair of points between clusters.
template<TriviallyCopyable IndexType, typename DataEntry, std::floating_point FloatType>
std::tuple<size_t, size_t, FloatType> NearestNodes(SubProblemData<IndexType, DataEntry, FloatType> subGraphA,
                  SubProblemData<IndexType, DataEntry, FloatType> subGraphB,
                  const GraphVertex<IndexType, FloatType>& queryHintA,
                  const GraphVertex<IndexType, FloatType>& queryHintB,
                  SpaceMetric<DataEntry, DataEntry, FloatType>  distanceFunctor){

    std::pair<size_t, size_t> bestPair;
    FloatType bestDistance(std::numeric_limits<FloatType>::max());
    //NodeTracker nodesVisitedA(subGraphA.dataBlock.size());
    //NodeTracker nodesVisitedB(subGraphB.dataBlock.size());

    for(const auto& starterA: queryHintA.neighbors){
        //nodesVisitedA[starterA.first] = true;
        for (const auto& starterB: queryHintB.neighbors){
            //nodesVisitedB[starterB.first] = true;
            FloatType distance = distanceFunctor(subGraphA.dataBlock[starterA.first], subGraphB.dataBlock[starterB.first]);
            if (distance < bestDistance){
                bestDistance = distance;
                bestPair = std::pair<size_t, size_t>(starterA.first, starterB.first);
            }
        }
    }

    bool breakVar = false;
    while (!breakVar){
        breakVar = true;
        std::pair<size_t, size_t> tmpPair = bestPair;
        for (const auto& neighborA: subGraphA.subGraph[bestPair.first]){
            //if (!nodesVisitedA[neighborA.first]){
            FloatType distance = distanceFunctor(subGraphA.dataBlock[neighborA.first], subGraphB.dataBlock[tmpPair.second]);
            if (distance < bestDistance){
                bestDistance = distance;
                tmpPair.first = neighborA.first;
                breakVar = false;
            }
                //nodesVisitedA[neighborA.first] = true;
            //}  
            
            for (const auto& neighborOfNeighborA: subGraphA.subGraph[neighborA.first]){
                //if (nodesVisitedA[neighborOfNeighborA.first]) continue;
                //nodesVisitedA[neighborOfNeighborA.first] = true;
                FloatType distance = distanceFunctor(subGraphA.dataBlock[neighborOfNeighborA.first], subGraphB.dataBlock[tmpPair.second]);
                if (distance < bestDistance){
                    bestDistance = distance;
                    tmpPair.first = neighborOfNeighborA.first;
                    breakVar = false;
                }
            }
        }
        for (const auto& neighborB: subGraphB.subGraph[bestPair.second]){
            //if (!nodesVisitedB[neighborB.first]){
                FloatType distance = distanceFunctor(subGraphA.dataBlock[tmpPair.first], subGraphB.dataBlock[neighborB.first]);
            if (distance < bestDistance){
                bestDistance = distance;
                tmpPair.second = neighborB.first;
                breakVar = false;
            }
              //  nodesVisitedB[neighborB.first] = true;
            //}
            for (const auto& neighborOfNeighborB: subGraphB.subGraph[neighborB.first]){
                //nodesVisitedB[neighborOfNeighborB.first] = true;
                FloatType distance = distanceFunctor(subGraphA.dataBlock[tmpPair.first], subGraphB.dataBlock[neighborOfNeighborB.first]);
                if (distance < bestDistance){
                    bestDistance = distance;
                    tmpPair.second = neighborOfNeighborB.first;
                    breakVar = false;
                }
            }
        }
        bestPair = tmpPair;
    }
    
    return {bestPair.first, bestPair.second, bestDistance};
}

template<TriviallyCopyable IndexType, typename DataEntry, std::floating_point FloatType>
std::tuple<size_t, size_t, FloatType> BruteNearestNodes(SubProblemData<IndexType, DataEntry, FloatType> subGraphA,
                  SubProblemData<IndexType, DataEntry, FloatType> subGraphB,
                  SpaceMetric<DataEntry, DataEntry, FloatType>  distanceFunctor){

    std::pair<size_t, size_t> bestPair;
    FloatType bestDistance(std::numeric_limits<FloatType>::max());


    for(size_t i = 0; i<subGraphA.dataBlock.size(); i+=1){
        for (size_t j = 0; j<subGraphB.dataBlock.size(); j+=1){
            FloatType distance = distanceFunctor(subGraphA.dataBlock[i], subGraphB.dataBlock[j]);
            if (distance < bestDistance){
                bestDistance = distance;
                bestPair = std::pair<size_t, size_t>(i,j);
            }
        }
    }
    
    return {bestPair.first, bestPair.second, bestDistance};
}

template<TriviallyCopyable IndexType, typename DataEntry, std::floating_point FloatType>
GraphVertex<IndexType, FloatType> QueryCOMNeighbors(const std::valarray<FloatType>& centerOfMass,
                                                     SubProblemData<IndexType, DataEntry, FloatType> subProb, 
                                                     int numCandidates,
                                                     SpaceMetric<std::valarray<FloatType>, DataEntry, FloatType> distanceFunctor){

    GraphVertex<IndexType, FloatType> COMneighbors(numCandidates);
    //ComparisonQueue<IndexType> cmpQueue(numCandidates*queueMargin);
    //Just gonna dummy it and select the first few nodes.
    NodeTracker nodesVisited(subProb.dataBlock.size());
    for (size_t i = 0; i < numCandidates; i+=1){
        COMneighbors.neighbors.push_back(std::pair<IndexType, FloatType>(i,
                                          distanceFunctor(centerOfMass, subProb.dataBlock.blockData[i])));
        //cmpQueue.PushQueue(static_cast<IndexType>(i));
        nodesVisited[i] = true;
    }
    std::make_heap(COMneighbors.neighbors.begin(), COMneighbors.neighbors.end(), NeighborDistanceComparison<IndexType, FloatType>);
    //ComparisonQueue<IndexType> joinQueue(numCandidates*queueMargin*2);
    //std::vector<FloatType> distances(0);
    //distances.reserve(numCandidates*queueMargin*2);

    /*
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
    */
    bool breakVar = false;
    GraphVertex<IndexType, FloatType> newState(COMneighbors);
    while (!breakVar){
        breakVar = true;   
        for (const auto& curCandidate: COMneighbors){
            for (const auto& joinTarget: subProb.subGraph[curCandidate.first]){
                if(nodesVisited[joinTarget.first]) continue;
                nodesVisited[joinTarget.first] = true;
                FloatType distance = distanceFunctor(centerOfMass, subProb.dataBlock[joinTarget.first]);
                if (distance < newState[0].second){
                    newState.PushNeigbor({joinTarget.first, distance});
                    breakVar = false;
                }
                
            }
        }

        COMneighbors = newState;
        /*
        cmpQueue.FlushQueue();

        for (const auto& joinTarget: joinQueue.queue){

            FloatType distance = distanceFunctor(centerOfMass, subProb.dataBlock[joinTarget]);
            
            if (distance <  COMneighbors.neighbors[0].second){
                COMneighbors.PushNeigbor(std::pair<IndexType, FloatType>(joinTarget, distance));
                cmpQueue.PushQueue(static_cast<IndexType>(joinTarget));
            }
        }
        joinQueue.FlushQueue();
        */
    }

    return COMneighbors;
}
/*
template<typename DataEntry, typename FloatType, size_t queueMargin>
GraphVertex<BlockIndex, FloatType> QueryCOMNeighbors(const std::valarray<FloatType>& centerOfMass,
                                                     SubProblemData<BlockIndex, DataEntry, FloatType> subProb, 
                                                     int numCandidates,
                                                     COMMetric<DataEntry, std::valarray<FloatType>, FloatType> distanceFunctor){

    GraphVertex<BlockIndex, FloatType> COMneighbors(numCandidates);
    ComparisonQueue<BlockIndex> cmpQueue(numCandidates*queueMargin);
    //Just gonna dummy it and select the first few nodes.
    for (size_t i = 0; i < numCandidates; i+=1){
        COMneighbors.neighbors.push_back({{subProb.dataBlock.blockNumber, i},
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
                COMneighbors.PushNeigbor({{subProb.dataBlock.blockNumber, joinTarget}, distance});
                cmpQueue.PushQueue(joinTarget);
            }
        }
        joinQueue.FlushQueue();
    }

    return COMneighbors;
}
*/
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
    
    MetaGraph metaGraph(trainMapper.dataBlocks, 20);
    std::vector<GraphVertex<size_t, double>> queryHints;
    for (size_t i = 0; i<metaGraph.points.size(); i += 1){
        SubProblemData subProb{blockGraphs[i], trainMapper.dataBlocks[i]};
        queryHints.push_back(QueryCOMNeighbors<size_t, std::valarray<float>, double>(metaGraph.points[i].centerOfMass,
                                                                          subProb,
                                                                          5,
                                                                          EuclideanNorm<double, float, double>));

    }
    //SubProblemData subProbA{blockGraphs[0], trainMapper.dataBlocks[0]};
    //SubProblemData subProbB{blockGraphs[1], trainMapper.dataBlocks[1]};

    //std::tuple<size_t, size_t, double> testBest = NearestNodes<size_t, std::valarray<float>, double>(subProbA, subProbB, queryHints[0], queryHints[1], EuclideanNorm<float, double>);

    Graph<size_t, double> nearestNodeDistances(metaGraph.verticies.size(), 20);
    //std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();
    for (size_t i = 0; i < metaGraph.verticies.size(); i += 1){

        for(const auto& neighbor: metaGraph.verticies[i]){
            SubProblemData subProbA{blockGraphs[i], trainMapper.dataBlocks[i]};
            SubProblemData subProbB{blockGraphs[neighbor.first], trainMapper.dataBlocks[neighbor.first]};
            double nnDist;
            std::tie(std::ignore, std::ignore, nnDist) = NearestNodes<size_t, std::valarray<float>, double>(subProbA, subProbB, queryHints[i], queryHints[neighbor.first], EuclideanNorm<float, float, double>);
            nearestNodeDistances[i].neighbors.push_back(std::pair(neighbor.first, nnDist));
        }

        std::sort(nearestNodeDistances[i].begin(), nearestNodeDistances[i].end(), NeighborDistanceComparison<size_t, double>);
        std::vector<Graph<size_t, double>> candidates;
        for (size_t j = 0; j<10; j+=1){
            SubProblemData subProbA{blockGraphs[nearestNodeDistances[i][j].first], trainMapper.dataBlocks[nearestNodeDistances[i][j].first]};
            SubProblemData subProbB{blockGraphs[i], trainMapper.dataBlocks[i]};
            std::vector<size_t> indicies(subProbB.dataBlock.size());
            std::iota(indicies.begin(), indicies.end(), 0);
            candidates.push_back(QuerySubGraph(subProbA, subProbB, indicies, queryHints[nearestNodeDistances[i][j].first], 5, EuclideanNorm<float, float, double>));
        }
    }

    

    /*

    template<TriviallyCopyable IndexType, typename DataEntry, typename FloatType>
    Graph<IndexType, FloatType> QuerySubGraph(SubProblemData<IndexType, DataEntry, FloatType> subGraphA,
                                          SubProblemData<IndexType, DataEntry, FloatType> subGraphB,
                                          const std::vector<IndexType>& queryPoints,
                                          const GraphVertex<BlockIndex, FloatType>& queryHint,
                                          int numCandidates,
                                          SpaceMetric<DataEntry, FloatType>  distanceFunctor)

    */

    //std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::seconds>(runEnd - runStart).count() << "s Nearest Node Calcs " << std::endl;

    //std::tuple<size_t, size_t, double> bfTest = BruteNearestNodes<size_t, std::valarray<float>, double>(subProbA, subProbB, EuclideanNorm<float, double>);


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