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
    const Graph<IndexType, FloatType>& subGraph;
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
                        newState.PushNeighbor({joinTarget.first, distance});
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
/*
template<TriviallyCopyable IndexType, typename DataEntry, std::floating_point FloatType>
std::tuple<IndexType, IndexType, FloatType> NearestNodes(const SubProblemData<IndexType, DataEntry, FloatType> subGraphA,
                  const SubProblemData<IndexType, DataEntry, FloatType> subGraphB,
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
*/

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
                                                     const SubProblemData<IndexType, DataEntry, FloatType> subProb, 
                                                     const int numCandidates,
                                                     SpaceMetric<std::valarray<FloatType>, DataEntry, FloatType> distanceFunctor){

    GraphVertex<IndexType, FloatType> COMneighbors(numCandidates);
    
    //Just gonna dummy it and select the first few nodes. Since the splitting process is randomized, this is a totally random selection, right? /s
    NodeTracker nodesVisited(subProb.dataBlock.size());
    for (size_t i = 0; i < numCandidates; i+=1){
        COMneighbors.neighbors.push_back(std::pair<IndexType, FloatType>(i,
                                          distanceFunctor(centerOfMass, subProb.dataBlock.blockData[i])));
        nodesVisited[i] = true;
    }
    std::make_heap(COMneighbors.neighbors.begin(), COMneighbors.neighbors.end(), NeighborDistanceComparison<IndexType, FloatType>);

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
                    newState.PushNeighbor({joinTarget.first, distance});
                    breakVar = false;
                }
                
            }
        }

        COMneighbors = newState;
    }

    return COMneighbors;
}




template<TriviallyCopyable IndexType, typename DataEntry, std::floating_point FloatType>
struct QueryContext{
    const Graph<IndexType, FloatType>& subGraph;
    const DataBlock<DataEntry>& dataBlock;
    GraphVertex<IndexType, FloatType> queryHint;
    const int numCandidates;
    std::unordered_map<size_t, Graph<IndexType, FloatType>> neighborCandidates;
    std::unordered_map<size_t, FloatType> distances;
    SpaceMetric<DataEntry, DataEntry, FloatType> distanceFunctor;

    QueryContext(const Graph<IndexType, FloatType>& subGraph,
                 const DataBlock<DataEntry>& dataBlock,
                 GraphVertex<IndexType, DataEntry> queryHint,
                 SpaceMetric<DataEntry, DataEntry, FloatType> distanceFunctor,
                 const int numCandidates): subGraph(subGraph), dataBlock(dataBlock), queryHint(std::move(queryHint)), numCandidates(numCandidates), neighborCandidates(), distanceFunctor(distanceFunctor){};

    QueryContext(const Graph<IndexType, FloatType>& subGraph,
                 const DataBlock<DataEntry>& dataBlock,
                 const std::valarray<FloatType>& centerOfMass,
                 const int numCandidates,
                 SpaceMetric<DataEntry, DataEntry, FloatType> distanceFunctor,
                 SpaceMetric<std::valarray<FloatType>, DataEntry, FloatType> comDistanceFunctor):
                    subGraph(subGraph), dataBlock(dataBlock), numCandidates(numCandidates), neighborCandidates(), distanceFunctor(distanceFunctor){
        const SubProblemData thisSub{subGraph, dataBlock};
        queryHint = QueryCOMNeighbors<IndexType, DataEntry, FloatType>(centerOfMass, thisSub, numCandidates, comDistanceFunctor);
        for (auto& hint: queryHint){
            hint.second = std::numeric_limits<FloatType>::max();
        }
    };

    //Nearest Node Distance
    //make checking this in parallel safe
    FloatType operator*(QueryContext& rhs){
        auto result = this->distances.find(rhs.dataBlock.blockNumber);
        if(result == distances.end()){
            FloatType distance = NearestNodes(rhs);
            this->distances[rhs.dataBlock.blockNumber] = distance;
            rhs.distances[this->dataBlock.blockNumber] = distance;
        }
        return this->distances[rhs.dataBlock.blockNumber];

    }

    void operator||(QueryContext& rhs){
        auto result = this->neighborCandidates.find(rhs.dataBlock.blockNumber);
        if (result != neighborCandidates.end()) return;
        //Query RHS data against mine
        rhs.neighborCandidates[this->dataBlock.blockNumber] = this->QuerySubGraph(rhs);
        //Query My data against RHS
        this->neighborCandidates[rhs.dataBlock.blockNumber] = rhs.QuerySubGraph(*this);
    }
    //

    template<typename QueryType>
    GraphVertex<IndexType, FloatType> QueryHotPath(GraphVertex<IndexType, FloatType> initVertex,
                                                   const QueryType& queryData,
                                                   NodeTracker nodesVisited){
        GraphVertex<IndexType, FloatType> newState(initVertex);
        bool breakVar = false;
        while (!breakVar){
            breakVar = true;
            for (const auto& neighbor: initVertex){
                const GraphVertex<IndexType, FloatType>& currentNeighbor = subGraph[neighbor.first];
                for (const auto& joinTarget: currentNeighbor){
                    if (nodesVisited[joinTarget.first] == true) continue;
                    nodesVisited[joinTarget.first] = true;
                    FloatType distance = this->distanceFunctor(queryData, dataBlock[joinTarget.first]);
                    if (distance < newState[0].second){
                        newState.PushNeighbor({joinTarget.first, distance});
                        breakVar = false;
                    }
                }
            }
            initVertex = newState;
        }
        return initVertex;
    }

    private:

    Graph<IndexType, FloatType> QuerySubGraph(const QueryContext& rhs){

        //Initialize results with queryHint
        Graph<IndexType, FloatType> retGraph;

        
        for (size_t i = 0; i<rhs.dataBlock.size(); i += 1){
            DataEntry queryData = rhs.dataBlock[i];
            /*
            template<typename QueryType>
    GraphVertex<IndexType, FloatType> QueryHotPath(GraphVertex<IndexType, FloatType> initVertex,
                                                   const QueryType& queryData,
            */
            retGraph.push_back(QueryHotPath(queryHint, queryData, NodeTracker(dataBlock.size())));
        }
        return retGraph;
    }

    FloatType NearestNodes(const QueryContext& rhs){

        assert(this->distanceFunctor == rhs.distanceFunctor);

        std::pair<size_t, size_t> bestPair;
        FloatType bestDistance(std::numeric_limits<FloatType>::max());
        //NodeTracker nodesVisitedA(subGraphA.dataBlock.size());
        //NodeTracker nodesVisitedB(subGraphB.dataBlock.size());

        for(const auto& starterA: this->queryHint.neighbors){
            //nodesVisitedA[starterA.first] = true;
            for (const auto& starterB: rhs.queryHint.neighbors){
                //nodesVisitedB[starterB.first] = true;
                FloatType distance = distanceFunctor(this->dataBlock[starterA.first], rhs.dataBlock[starterB.first]);
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
            for (const auto& neighborA: this->subGraph[bestPair.first]){
                //if (!nodesVisitedA[neighborA.first]){
                FloatType distance = distanceFunctor(this->dataBlock[neighborA.first], rhs.dataBlock[tmpPair.second]);
                if (distance < bestDistance){
                    bestDistance = distance;
                    tmpPair.first = neighborA.first;
                    breakVar = false;
                }
                    //nodesVisitedA[neighborA.first] = true;
                //}  
                
                for (const auto& neighborOfNeighborA: this->subGraph[neighborA.first]){
                    //if (nodesVisitedA[neighborOfNeighborA.first]) continue;
                    //nodesVisitedA[neighborOfNeighborA.first] = true;
                    FloatType distance = this->distanceFunctor(this->dataBlock[neighborOfNeighborA.first], rhs.dataBlock[tmpPair.second]);
                    if (distance < bestDistance){
                        bestDistance = distance;
                        tmpPair.first = neighborOfNeighborA.first;
                        breakVar = false;
                    }
                }
            }
            for (const auto& neighborB: rhs.subGraph[bestPair.second]){
                //if (!nodesVisitedB[neighborB.first]){
                    FloatType distance = this->distanceFunctor(this->dataBlock[tmpPair.first], rhs.dataBlock[neighborB.first]);
                if (distance < bestDistance){
                    bestDistance = distance;
                    tmpPair.second = neighborB.first;
                    breakVar = false;
                }
                //  nodesVisitedB[neighborB.first] = true;
                //}
                for (const auto& neighborOfNeighborB: rhs.subGraph[neighborB.first]){
                    //nodesVisitedB[neighborOfNeighborB.first] = true;
                    FloatType distance = this->distanceFunctor(this->dataBlock[tmpPair.first], rhs.dataBlock[neighborOfNeighborB.first]);
                    if (distance < bestDistance){
                        bestDistance = distance;
                        tmpPair.second = neighborOfNeighborB.first;
                        breakVar = false;
                    }
                }
            }
            bestPair = tmpPair;
        }
        
        return bestDistance;
    }

};




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
    
    MetaGraph metaGraph(trainMapper.dataBlocks, 5);
    /*
    std::vector<GraphVertex<size_t, double>> queryHints;
    for (size_t i = 0; i<metaGraph.points.size(); i += 1){
        SubProblemData subProb{blockGraphs[i], trainMapper.dataBlocks[i]};
        queryHints.push_back(QueryCOMNeighbors<size_t, std::valarray<float>, double>(metaGraph.points[i].centerOfMass,
                                                                          subProb,
                                                                          5,
                                                                          EuclideanNorm<double, float, double>));

    }
    */
    //SubProblemData subProbA{blockGraphs[0], trainMapper.dataBlocks[0]};
    //SubProblemData subProbB{blockGraphs[1], trainMapper.dataBlocks[1]};

    //std::tuple<size_t, size_t, double> testBest = NearestNodes<size_t, std::valarray<float>, double>(subProbA, subProbB, queryHints[0], queryHints[1], EuclideanNorm<float, double>);

    
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

    //std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::seconds>(runEnd - runStart).count() << "s Pointwise Join Calcs " << std::endl;
    /*

    QueryContext(const Graph<IndexType, FloatType>& subGraph,
                 const DataBlock<DataEntry>& dataBlock,
                 const std::valarray<FloatType>& centerOfMass,
                 SpaceMetric<std::valarray<FloatType>, DataEntry, FloatType> distanceFunctor,
                 const int numCandidates)

    */
    //std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();
    /*
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