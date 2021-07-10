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
/*
template<typename DataIndexType>
struct BlockQueue{
    // If I routinely fill this up 24bytes worth duplicates from within block, use a vector instead of multiple pairs.
    std::vector<std::pair<DataIndexType, DataIndexType>> comparisonTargets;
    std::vector<std::pair<DataIndexType, std::vector<DataIndexType>>> joinTargets;
};
*/

template<typename DataIndexType>
using ComparisonVec = std::vector<std::pair<DataIndexType, DataIndexType>>;

template<typename BlockNumberType, typename DataIndexType>
using ComparisonMap = std::unordered_map<BlockNumberType, ComparisonVec<DataIndexType>>;

template<typename DataIndexType>
using JoinHint = std::pair<DataIndexType, std::vector<DataIndexType>>;

/*
template<typename DataIndexType>
struct JoinHint{
    DataIndexType joinee;
    std::vector<DataIndexType> joinStart;
};
*/

template<typename DataIndexType>
using JoinHints = std::unordered_map<DataIndexType, std::vector<DataIndexType>>;

template<typename BlockNumberType, typename DataIndexType>                                   // Consider using a fixed size array
using JoinMap = std::unordered_map<BlockNumberType, std::unordered_map<DataIndexType, std::vector<DataIndexType>>>;
// I could also do a struct where the actual data is vectors, and I use unordered_maps to remap indicies

template<typename BlockNumberType, typename DataIndexType, typename DistType>
ComparisonMap<BlockNumberType, DataIndexType> InitializeComparisonQueues(const Graph<BlockIndecies, DistType>& currentBlock, BlockNumberType blockNum){

    ComparisonMap<BlockNumberType, DataIndexType> retMap;
    for (size_t j = 0; j<currentBlock.size(); j+=1){
        for (const auto& neighbor: currentBlock[j]){
            if (neighbor.first.blockNumber != blockNum) retMap[neighbor.first.blockNumber].push_back({j, neighbor.first.dataIndex});
        }
    }

    return retMap;
}

template<typename BlockNumberType, typename DataIndexType, typename DistType>
JoinMap<BlockNumberType, DataIndexType> InitializeJoinMap(const std::vector<Graph<BlockIndecies, DistType>>& updatedBlockGraphs,
                                                          const ComparisonMap<BlockNumberType, DataIndexType>& comparisonMap,
                                                          const NodeTracker& nodesJoined){
    JoinMap<BlockNumberType, DataIndexType> joinMap;
    for (auto& [targetBlock, queue]: comparisonMap){
        //std::unordered_map<size_t, std::pair<size_t, std::vector<size_t>>> joinHints;
        for (const auto& [sourceIndex, targetIndex]: queue){
            for (const auto& neighbor: updatedBlockGraphs[targetBlock][targetIndex]){
                if (neighbor.first.blockNumber == targetBlock || nodesJoined[neighbor.first.blockNumber]) continue;
                auto result = std::ranges::find(joinMap[neighbor.first.blockNumber][sourceIndex], neighbor.first.dataIndex);
                if (result == joinMap[neighbor.first.blockNumber][sourceIndex].end()) joinMap[neighbor.first.blockNumber][sourceIndex].push_back(neighbor.first.dataIndex);
            }
        }
    }
    return joinMap;
}

template<typename DataIndexType, typename DistType>
using JoinResults = std::vector<std::pair<DataIndexType, GraphVertex<DataIndexType, DistType>>>;

//template<typename BlockNumberType, typename DataIndexType, typename DataEntry, typename DistType>
template<typename DataIndexType, typename DataEntry, typename DistType>
JoinResults<DataIndexType, DistType> BlockwiseJoin(const JoinHints<DataIndexType>& startJoins,
                   const Graph<BlockIndecies, DistType>& currentGraphState,
                   Graph<DataIndexType, DistType>& searchSubgraph,
                   const DataBlock<DataEntry>& blockData,
                   const QueryContext<DataIndexType, DataEntry, DistType>& targetBlock){
    
    std::vector<std::pair<DataIndexType, GraphVertex<DataIndexType, DistType>>> joinHints;
    for (const auto& hint: startJoins){
        GraphVertex<DataIndexType, DistType> queryHint;
        for (const auto index: hint.second){
            queryHint.push_back({index, std::numeric_limits<DistType>::max()});
        }
        joinHints.push_back({hint.first, std::move(queryHint)});
    }
    NodeTracker nodesJoined(searchSubgraph.size());
    
    std::vector<std::pair<DataIndexType, GraphVertex<DataIndexType, DistType>>> retResults;
    while(joinHints.size()){
        std::vector<std::pair<DataIndexType, GraphVertex<DataIndexType, DistType>>> joinResults;
        for (const auto& joinHint: joinHints){
            //GraphVertex<DataIndexType, DistType> joinResult = targetBlock || QueryPoint{joinHint.second, blockData[joinHint.first]};
            const QueryPoint<DataIndexType, DataEntry, DistType> query(joinHint.second, blockData[joinHint.first]);
            joinResults.push_back({joinHint.first, targetBlock || query});
            nodesJoined[joinHint.first] = true;
        }
        std::vector<std::pair<DataIndexType, GraphVertex<DataIndexType, DistType>>> newJoins;
        for(auto& result: joinResults){
            //std::heap_sort(result.second.begin(), result.second.end(), NeighborDistanceComparison<DataIndexType, DistType>);
            bool newNeighbor = false;
            GraphVertex<DataIndexType, DistType> updatedResult;
            for (const auto& neighborCandidate: result.second){
                if (neighborCandidate.second < currentGraphState[result.first][0].second){
                    newNeighbor = true;
                    updatedResult.push_back(neighborCandidate);
                }
            }
            if (newNeighbor){   
                
                for(const auto& leafNeighbor: searchSubgraph[result.first]){
                    if(!nodesJoined[leafNeighbor.first]){
                        newJoins.push_back({leafNeighbor.first, result.second});
                        //We can add these to nodesJoined a touch early to prevent dupes
                        nodesJoined[leafNeighbor.first] = true;
                    }
                }
                retResults.push_back({result.first, std::move(updatedResult)});
            }
            //result.second = updatedResult;
        }
        joinHints = std::move(newJoins);
    }
    return retResults;
}
template<typename BlockNumberType, typename DataIndexType, typename DistType>
void NewJoinQueues(const std::vector<std::pair<DataIndexType, GraphVertex<DataIndexType, DistType>>>& joinResults,
                   const NodeTracker& blocksJoined,
                   const Graph<BlockIndecies, DistType>& targetGraphState,
                   JoinMap<BlockNumberType, DataIndexType>& mapToUpdate){
    
    for (const auto& result: joinResults){
        for (const auto index: result.second){
            for (const auto& targetVertexNeighbor: targetGraphState[index.first]){
                BlockNumberType targetBlock = targetVertexNeighbor.first.blockNumber;
                if (blocksJoined[targetBlock]) continue;
                auto findItr = std::find(mapToUpdate[targetBlock][result.first].begin(), mapToUpdate[targetBlock][result.first].end(), targetVertexNeighbor.first.dataIndex);
                if (findItr == mapToUpdate[targetBlock][result.first].end()) mapToUpdate[targetBlock][result.first].push_back(targetVertexNeighbor.first.dataIndex);
            } 
            
        }
    }
}


int main(){

    std::cout << "NNDescent.cpp prototype - 7/10/2021" << std::endl;
    std::cout << "This binary was compiled with Microsoft (R) C/C++ Optimizing Compiler Version 19.29.30038.1 for x64" << std::endl;
    std::cout << "With the following settings:" << std::endl;
    std::cout << "/Zi /EHsc /nologo /std:c++latest /O2 /fp:fast /arch:AVX2" << std::endl;
    std::cout << "This binary is a single threaded, extremely early prototype that has some preconditions for running." << std::endl;
    std::cout << "It needs the MNIST fashion data set or other similarly size data set, like MNIST digits." << std::endl;
    std::cout << "All of the data must be in big endian format (odd, I know, for my testing I just ended up going with big because I forgot what the common x64 endianess was)." << std::endl;
    std::cout << "The training and test data must be 32-bit floating point numbers, the ground truth must be 32-bit ints with 100 entries per test point." << std::endl;
    std::cout << "The data must use the following paths:" << std::endl;
    std::cout << "./TestData/MNIST-Fashion-Train.bin" << std::endl;
    std::cout << "./TestData/MNIST-Fashion-Data.bin" << std::endl;
    std::cout << "./TestData/MNIST-Fashion-Neighbors.bin" << std::endl;
    std::cout << std::endl;
    std::cout << "The source code for this binary may be found at:" << std::endl;
    std::cout << "https://github.com/AnabelSMRuggiero/NNDescent.cpp/tree/initial-prototype-archive" << std::endl;
    std::cout << std::endl;

    
    static const std::endian dataEndianness = std::endian::big;

    //std::string trainDataFilePath("./TestData/train-images.idx3-ubyte");

    

    //DataSet<std::valarray<unsigned char>> mnistDigitsTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<unsigned char,dataEndianness>);

    //std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();

    std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");
    DataSet<std::valarray<float>> mnistFashionTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<float,dataEndianness>);

    std::string testDataFilePath("./TestData/MNIST-Fashion-Data.bin");
    std::string testNeighborsFilePath("./TestData/MNIST-Fashion-Neighbors.bin");
    DataSet<std::valarray<float>> mnistFashionTest(testDataFilePath, 28*28, 10'000, &ExtractNumericArray<float,dataEndianness>);
    DataSet<std::valarray<int32_t>> mnistFashionTestNeighbors(testNeighborsFilePath, 100, 10'000, &ExtractNumericArray<int32_t,dataEndianness>);
    /*
    volatile float resultA, resultB, resultC;
    for (size_t i =0; i<10'000; i +=1){
        resultA = Dot<float, float>(mnistFashionTrain.samples[0], mnistFashionTrain.samples[1]);
    }

    for (size_t i =0; i<1'000'000; i +=1){
        resultB = EuclideanNorm<float, float>(mnistFashionTrain.samples[0], mnistFashionTrain.samples[1]);
    }

    for (size_t i =0; i<1'000'000; i +=1){
        resultC = TestEuclideanNorm<float, float>(mnistFashionTrain.samples[0], mnistFashionTrain.samples[1]);
    }
    */
    //std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::seconds>(runEnd - runStart).count() << "s Pointwise Join Calcs " << std::endl;

    std::cout << "I/O done: I know it's pretty scuffed at the moment." << std::endl;


    std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();

    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), mnistFashionTrain.numberOfSamples - 1);
    StlRngFunctor<std::mt19937_64, std::uniform_int_distribution, size_t> rngFunctor(std::move(rngEngine), std::move(rngDist));

    EuclidianTrain<float, float> splittingScheme(mnistFashionTrain);
    TrainingSplittingScheme splitterFunc(splittingScheme);
    
    RandomProjectionForest rpTreesTrain(size_t(mnistFashionTrain.numberOfSamples), rngFunctor, splitterFunc);


    //std::vector<size_t> trainClassifications(mnistFashionTrain.numberOfSamples);
    

    

    
    DataMapper<std::valarray<float>> trainMapper(mnistFashionTrain);
    CrawlTerminalLeaves(rpTreesTrain, trainMapper);
    

    
    std::vector<Graph<size_t, float>> blockGraphs(0);
    blockGraphs.reserve(trainMapper.dataBlocks.size());
    for (const auto& dataBlock : trainMapper.dataBlocks){
        Graph<size_t, float> blockGraph(dataBlock.blockData.size(), size_t(10));
        BruteForceBlock<size_t, std::valarray<float>, float>(blockGraph, 10, dataBlock, EuclideanNorm<float, float, float>);
        blockGraphs.push_back(std::move(blockGraph));
    }
    
    MetaGraph<size_t, float> metaGraph(trainMapper.dataBlocks, 10);

    
    std::vector<QueryContext<size_t, std::valarray<float>, float>> queryContexts;

    for (size_t i = 0; i<metaGraph.verticies.size(); i+=1){
        queryContexts.push_back(QueryContext<size_t, std::valarray<float>, float>::QueryContext(blockGraphs[i],
                                trainMapper.dataBlocks[i],
                                metaGraph.points[i].centerOfMass,
                                10,
                                EuclideanNorm<float, float, float>,
                                EuclideanNorm<float, float, float>));
    }

    for (size_t i = 0; i<metaGraph.verticies.size(); i+=1){
        GraphVertex<size_t, float>& vertex = metaGraph.verticies[i];
        std::sort(vertex.begin(), vertex.end(), NeighborDistanceComparison<size_t, float>);
        for(size_t j = 0; j<vertex.size(); j+=1){
            //Nearest Node distance; results are cached within objects. In fact, this op currently returns void. Maybe return a const ref to... something?
            queryContexts[i] * queryContexts[vertex[j].first];
        }
    }

    Graph<size_t, float> nearestNodeDistances;
    for(auto& context: queryContexts){
        GraphVertex<size_t, float> nearestNeighbors;
        std::unordered_map<size_t, float> distanceMap = std::move(context.nearestNodeDistances);
        for(const auto& pair: distanceMap){
            nearestNeighbors.push_back(pair);
        }
        std::sort(nearestNeighbors.begin(), nearestNeighbors.end(), NeighborDistanceComparison<size_t, float>);
        nearestNodeDistances.push_back(std::move(nearestNeighbors));
    }

    //std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();

    std::vector<NodeTracker> blockJoinTrackers(blockGraphs.size(), NodeTracker(blockGraphs.size()));

    for(size_t i = 0; i<nearestNodeDistances.size(); i+=1){
        //Do all of them? Let's try for now. I can add in hyperparam here.
        //for(const auto& neighbor: nearestNodeDistances[i]){
        blockJoinTrackers[i][i] = true;
        for (size_t j = 0; j<(nearestNodeDistances[i].size()/2); j+=1){
            blockJoinTrackers[i][nearestNodeDistances[i][j].first] = true;
            blockJoinTrackers[nearestNodeDistances[i][j].first][i] = true;
            //Compute the nodewise join of the two blocks, results cached
            queryContexts[i] || queryContexts[nearestNodeDistances[i][j].first];
        }
    }

    std::vector<Graph<BlockIndecies, float>> updatedBlockGraphs;
    for(auto& context: queryContexts){
        std::unordered_map<size_t, Graph<size_t, float>> candidates(std::move(context.neighborCandidates));
        Graph<BlockIndecies, float> blockGraph;
        for (const auto& vertex: blockGraphs[context.dataBlock.blockNumber]){
            GraphVertex<BlockIndecies, float> newVert;
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
    
    //Initial filling of comparison targets.
    std::vector<ComparisonMap<size_t, size_t>> queueMaps;
    queueMaps.reserve(updatedBlockGraphs.size());
    for (size_t i = 0; i<updatedBlockGraphs.size(); i+=1){
        queueMaps.push_back(InitializeComparisonQueues<size_t, size_t, float>(updatedBlockGraphs[i], i));
    }
    
    std::vector<JoinMap<size_t, size_t>> joinHints;

    for(size_t i = 0; i<queueMaps.size(); i+=1){
        ComparisonMap<size_t, size_t>& comparisonMap = queueMaps[i];
        
        joinHints.push_back(InitializeJoinMap<size_t, size_t, float>(updatedBlockGraphs, comparisonMap, blockJoinTrackers[i]));
    }


    std::vector<std::unordered_map<size_t, JoinResults<size_t, float>>> blockUpdates(updatedBlockGraphs.size());
    

    GraphVertex<BlockIndecies, float> nullVertex;
    for(size_t i = 0; i<10; i+=1){
        nullVertex.push_back({{0,0}, std::numeric_limits<float>::max()});
    }
    int iteration(1);
    int graphUpdates(1);
    while(graphUpdates>0){
        graphUpdates = 0;
        std::vector<std::unordered_map<size_t, JoinResults<size_t, float>>> blockUpdates(updatedBlockGraphs.size());
        for(size_t i = 0; i<updatedBlockGraphs.size(); i+=1){
            JoinMap<size_t, size_t>& joinsToDo = joinHints[i];
            JoinMap<size_t, size_t> newJoinHints;
            for (auto& joinList: joinsToDo){
                blockJoinTrackers[i][joinList.first] = true;
            }
            for (auto& joinList: joinsToDo){
                blockUpdates[i][joinList.first] = BlockwiseJoin(joinList.second, updatedBlockGraphs[i], blockGraphs[i], trainMapper.dataBlocks[i], queryContexts[joinList.first]);
                NewJoinQueues<size_t, size_t, float>(blockUpdates[i][joinList.first], blockJoinTrackers[i], updatedBlockGraphs[joinList.first], newJoinHints);
            }
            joinHints[i] = std::move(newJoinHints);  
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
                graphUpdates += ConsumeVertex(updatedBlockGraphs[i][consolidatedResult.first], consolidatedResult.second);
            }
        }
        //std::cout << graphUpdates << " updates in iteration " << iteration << std::endl;
        iteration += 1;
    }

    std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(runEnd - runStart).count() << "s total for index building " << std::endl;

    //for (size_t i = 0; i<updatedBlockGraphs.size(); i+=1){
    //    VerifySubGraphState(updatedBlockGraphs[i], i);
    //}
    
    // Lets try and get searching up and running
    
    std::chrono::time_point<std::chrono::steady_clock> runStart2 = std::chrono::steady_clock::now();

    EuclidianTransform<float, float> transformingScheme(mnistFashionTest, splitterFunc.target<EuclidianTrain<float, float>>()->splittingVectors);
    
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
    
    DataMapper<std::valarray<float>> testMapper(mnistFashionTest);
    CrawlTerminalLeaves(rpTreesTest, testMapper);
    
    std::vector<Graph<size_t, float>> reflexiveGraphs(0);
    reflexiveGraphs.reserve(testMapper.dataBlocks.size());
    std::vector<Graph<BlockIndecies,float>> nearestNeighbors;
    nearestNeighbors.reserve(testMapper.dataBlocks.size());

    std::vector<JoinMap<size_t, size_t>> testJoinHints(testMapper.dataBlocks.size());

    auto hintBuilder = [](const QueryContext<size_t, std::valarray<float>, float>& context) -> std::vector<size_t>{
        std::vector<size_t> retVec;
        for (const auto& neighbor: context.queryHint){
            retVec.push_back(neighbor.first);
        }
        return retVec;
    };
    
    //std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();

    for (size_t i=0; const auto& dataBlock : testMapper.dataBlocks){
        Graph<size_t, float> blockGraph(dataBlock.blockData.size(), size_t(10));
        BruteForceBlock<size_t, std::valarray<float>, float>(blockGraph, 10, dataBlock, EuclideanNorm<float, float, float>);
        reflexiveGraphs.push_back(std::move(blockGraph));
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
                blockUpdates[i][joinList.first] = BlockwiseJoin(joinList.second, nearestNeighbors[i], reflexiveGraphs[i], testMapper.dataBlocks[i], queryContexts[joinList.first]);
                NewJoinQueues<size_t, size_t, float>(blockUpdates[i][joinList.first], testJoinTrackers[i], updatedBlockGraphs[joinList.first], newJoinHints);
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
    std::chrono::time_point<std::chrono::steady_clock> runEnd2 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(runEnd2 - runStart2).count() << "s test set search " << std::endl;

    std::vector<std::vector<size_t>> results(mnistFashionTest.samples.size());
    for (size_t i = 0; auto& testBlock: nearestNeighbors){
        for (size_t j = 0; auto& result: testBlock){
            size_t testIndex = testMapper.blockIndexToSource[{i,j}];
            std::sort_heap(result.begin(), result.end(), NeighborDistanceComparison<BlockIndecies, float>);
            for (const auto& neighbor: result){
                results[testIndex].push_back(trainMapper.blockIndexToSource[neighbor.first]);
            }
            j++;
        }
        i++;
    }
    size_t numNeighborsCorrect(0);
    for(size_t i = 0; const auto& result: results){
        for(size_t j = 0; const auto& neighbor: result){
            auto findItr = std::find(std::begin(mnistFashionTestNeighbors.samples[i]), std::begin(mnistFashionTestNeighbors.samples[i]) + 10, neighbor);
            if (findItr != (std::begin(mnistFashionTestNeighbors.samples[i]) + 10)){
                numNeighborsCorrect++;
            }
            j++;
        }
        i++;
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