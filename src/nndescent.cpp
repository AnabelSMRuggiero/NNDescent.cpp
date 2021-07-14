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



template<typename DataIndexType, typename DistType>
using JoinResults = std::vector<std::pair<DataIndexType, GraphVertex<DataIndexType, DistType>>>;

//template<typename BlockNumberType, typename DataIndexType, typename DataEntry, typename DistType>
template<typename BlockNumberType, typename DataIndexType, typename DataEntry, typename DistType, typename QueryFunctor>
JoinResults<DataIndexType, DistType> BlockwiseJoin(const JoinHints<DataIndexType>& startJoins,
                   const Graph<BlockIndecies, DistType>& currentGraphState,
                   const Graph<DataIndexType, DistType>& searchSubgraph,
                   const DataBlock<DataEntry>& blockData,
                   const QueryContext<BlockNumberType, DataIndexType, DataEntry, DistType>& targetBlock,
                   QueryFunctor queryFunctor){
    
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
            //const QueryPoint<DataIndexType, DataEntry, DistType> query(joinHint.second, blockData[joinHint.first], joinHint.first);
            joinResults.push_back({joinHint.first, targetBlock.QueryHotPath(joinHint.second, blockData[joinHint.first], joinHint.first, queryFunctor)});
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


template<typename DataIndexType, typename DataEntry, typename DistType>
struct QueryContextInitArgs{
    GraphVertex<DataIndexType, DistType>& queryHint;
    SpaceMetric<DataEntry, DataEntry, DistType> distanceFunctor;

    QueryContextInitArgs(GraphVertex<DataIndexType, DistType>& queryHint, SpaceMetric<DataEntry, DataEntry, DistType> distanceFunctor):
        queryHint(queryHint),
        distanceFunctor(distanceFunctor) {};
};

template<typename BlockNumberType, typename DataIndexType, typename DataEntry, typename DistType>
struct BlockUpdateContext {

    
    using TemplatedSubProblem = SubProblemData<DataIndexType, DataEntry, DistType>;
    using TemplatedContextInitArgs = QueryContextInitArgs<DataIndexType, DataEntry, DistType>;

    NodeTracker blockJoinTracker;
    const DataBlock<DataEntry>& dataBlock;
    JoinMap<BlockNumberType, DataIndexType> joinsToDo;
    JoinMap<BlockNumberType, DataIndexType> newJoins;
    QueryContext<BlockNumberType, DataIndexType, DataEntry, DistType> queryContext;
    const Graph<DataIndexType, DistType>& leafGraph;
    Graph<BlockIndecies, DistType> currentGraph;


    BlockUpdateContext(const TemplatedSubProblem subProbResults, const TemplatedContextInitArgs contextArgs, const BlockNumberType numberOfBlocksToJoin):
        leafGraph(subProbResults.subGraph),
        dataBlock(subProbResults.dataBlock),
        queryContext(subProbResults.subGraph, subProbResults.dataBlock, contextArgs.queryHint, contextArgs.distanceFunctor, contextArgs.queryHint.size()),
        currentGraph(subProbResults.dataBlock.size(), contextArgs.queryHint.size()),
        joinsToDo(),
        blockJoinTracker(numberOfBlocksToJoin){
            for(size_t i = 0; auto& vertex: currentGraph){
                for (const auto& neighbor: leafGraph[i]){
                    vertex.push_back({{dataBlock.blockNumber, neighbor.first}, neighbor.second});
                }
                i++;
            }
    }

    void SetNextJoins(){
        joinsToDo = std::move(newJoins);
        newJoins = JoinMap<BlockNumberType, DataIndexType>();
    }
};

template<typename BlockNumberType, typename DataIndexType, typename DataEntry, typename DistType>
JoinMap<BlockNumberType, DataIndexType> InitializeJoinMap(const std::vector<BlockUpdateContext<BlockNumberType, DataIndexType, DataEntry, DistType>>& blockUpdateContexts,
                                                          const ComparisonMap<BlockNumberType, DataIndexType>& comparisonMap,
                                                          const NodeTracker& nodesJoined){
    JoinMap<BlockNumberType, DataIndexType> joinMap;
    for (auto& [targetBlock, queue]: comparisonMap){
        //std::unordered_map<size_t, std::pair<size_t, std::vector<size_t>>> joinHints;
        for (const auto& [sourceIndex, targetIndex]: queue){
            for (const auto& neighbor: blockUpdateContexts[targetBlock].currentGraph[targetIndex]){
                if (neighbor.first.blockNumber == targetBlock || nodesJoined[neighbor.first.blockNumber]) continue;
                auto result = std::ranges::find(joinMap[neighbor.first.blockNumber][sourceIndex], neighbor.first.dataIndex);
                if (result == joinMap[neighbor.first.blockNumber][sourceIndex].end()) joinMap[neighbor.first.blockNumber][sourceIndex].push_back(neighbor.first.dataIndex);
            }
        }
    }
    return joinMap;
}


template<typename BlockNumberType, typename DataIndexType, typename DataEntry, typename DistType>
int UpdateBlocks(BlockUpdateContext<BlockNumberType, DataIndexType, DataEntry, DistType>& blockLHS,
                 BlockUpdateContext<BlockNumberType, DataIndexType, DataEntry, DistType>& blockRHS){
/*(JoinHints<DataIndexType>&& joinsToDo,
                  NodeTracker& blockJoinTracker,
                  Graph<BlockIndecies, DistType>& updatedBlockGraph, 
                  const Graph<BlockIndecies, DistType>& targetBlockGraph,
                  const DataBlock<DataEntry>& targetDataBlock,
                  const BlockNumberType targetBlockNumber,
                  QueryContext<DataIndexType, DataEntry, DistType>& targetContext){
*/
    
        
    //JoinMap<size_t, size_t> LHSNewJoinHints;
    
    //JoinMap<size_t, size_t> RHSNewJoinHints;

    bool doRHSJoin = blockRHS.joinsToDo.find(blockLHS.dataBlock.blockNumber) != blockRHS.joinsToDo.end();

    int graphUpdates(0);

    if(doRHSJoin){
        blockLHS.blockJoinTracker[blockRHS.dataBlock.blockNumber] = true;

        std::unordered_map<std::pair<DataIndexType, DataIndexType>, DistType, IntegralPairHasher<DataIndexType>> distanceCache;
        /*
        DistType operator()(DataIndexType LHSIndex, DataIndexType RHSIndex, const DataEntry& queryData) const{
        return this->distanceFunctor(dataBlock[LHSIndex], queryData);
        }
        */
        auto cachingDistanceFunctor = [&](DataIndexType LHSIndex, DataIndexType RHSIndex, const DataEntry& queryData) -> DistType{
            DistType distance = blockRHS.queryContext.defaultQueryFunctor(LHSIndex, RHSIndex, queryData);
            distanceCache[std::pair{LHSIndex, RHSIndex}] = distance;
            return distance;
        };
        

        JoinResults<DataIndexType, DistType> blockLHSUpdates = BlockwiseJoin(blockLHS.joinsToDo[blockRHS.dataBlock.blockNumber],
                                                                            blockLHS.currentGraph,
                                                                            blockLHS.leafGraph,
                                                                            blockLHS.dataBlock,
                                                                            blockRHS.queryContext,
                                                                            cachingDistanceFunctor);
        NewJoinQueues<size_t, size_t, float>(blockLHSUpdates, blockLHS.blockJoinTracker, blockRHS.currentGraph, blockLHS.newJoins);

        
        auto cachedDistanceFunctor = [&](DataIndexType LHSIndex, DataIndexType RHSIndex, const DataEntry& queryData) -> DistType{
            auto result = distanceCache.find(std::pair{RHSIndex, LHSIndex});
            if(result != distanceCache.end()) return result->second;
            else return blockLHS.queryContext.defaultQueryFunctor(LHSIndex, RHSIndex, queryData);
        };
    

        blockRHS.blockJoinTracker[blockLHS.dataBlock.blockNumber] = true;

        JoinResults<DataIndexType, DistType> blockRHSUpdates = BlockwiseJoin(blockRHS.joinsToDo[blockLHS.dataBlock.blockNumber],
                                                                            blockRHS.currentGraph,
                                                                            blockRHS.leafGraph,
                                                                            blockRHS.dataBlock,
                                                                            blockLHS.queryContext,
                                                                            cachedDistanceFunctor);

        NewJoinQueues<size_t, size_t, float>(blockRHSUpdates, blockRHS.blockJoinTracker, blockLHS.currentGraph, blockRHS.newJoins);

        for (auto& result: blockRHSUpdates){
            graphUpdates += ConsumeVertex(blockRHS.currentGraph[result.first], result.second, blockRHS.dataBlock.blockNumber);
        }
        for (auto& result: blockLHSUpdates){
            graphUpdates += ConsumeVertex(blockLHS.currentGraph[result.first], result.second, blockRHS.dataBlock.blockNumber);
        }
        
        return graphUpdates;

    } else {
        //This feels like som jank control flow
        blockLHS.blockJoinTracker[blockRHS.dataBlock.blockNumber] = true;
        
        

        JoinResults<DataIndexType, DistType> blockLHSUpdates = BlockwiseJoin(blockLHS.joinsToDo[blockRHS.dataBlock.blockNumber],
                                                                            blockLHS.currentGraph,
                                                                            blockLHS.leafGraph,
                                                                            blockLHS.dataBlock,
                                                                            blockRHS.queryContext,
                                                                            blockRHS.queryContext.defaultQueryFunctor);
        NewJoinQueues<size_t, size_t, float>(blockLHSUpdates, blockLHS.blockJoinTracker, blockRHS.currentGraph, blockLHS.newJoins);

        for (auto& result: blockLHSUpdates){
            graphUpdates += ConsumeVertex(blockLHS.currentGraph[result.first], result.second, blockRHS.dataBlock.blockNumber);
        }
        
        return graphUpdates;

    }
    //blockRHS.joinsToDo.erase(blockLHS.dataBlock.blockNumber);
    //blockLHS.joinsToDo.erase(blockRHS.dataBlock.blockNumber);

    
    

    
    
}

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



int main(){

    static const std::endian dataEndianness = std::endian::big;

    //std::string trainDataFilePath("./TestData/train-images.idx3-ubyte");

    

    //DataSet<std::valarray<unsigned char>> mnistDigitsTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<unsigned char,dataEndianness>);

    //std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();

    //std::string uint8FashionPath("./TestData/MNIST-Train-Images-UInt8.bin");
    //DataSet<std::valarray<uint8_t>> uint8FashionData(uint8FashionPath, 28*28, 60'000, &ExtractNumericArray<uint8_t,dataEndianness>);

    std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");
    DataSet<std::valarray<float>> mnistFashionTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<float,dataEndianness>);

    //std::string reserializedTrainFilePath("./TestData/MNIST-Fashion-Train-Uint8.bin");
    //SerializeDataSet<std::valarray<float>, uint8_t, dataEndianness>(mnistFashionTrain, reserializedTrainFilePath);

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
    
    MetaGraph<size_t, float> metaGraph(trainMapper.dataBlocks, 10);
    
    std::vector<Graph<size_t, float>> blockGraphs(0);
    blockGraphs.reserve(trainMapper.dataBlocks.size());
    for (const auto& dataBlock : trainMapper.dataBlocks){
        blockGraphs.push_back(BruteForceBlock<size_t, std::valarray<float>, float>(10, dataBlock, EuclideanNorm<float, float, float>));
    }
    

    std::vector<BlockUpdateContext<size_t, size_t, std::valarray<float>, float>> blockUpdateContexts;
    


    for (size_t i = 0; i<metaGraph.verticies.size(); i+=1){
        GraphVertex<size_t, float> queryHint = QueryHintFromCOM<size_t, std::valarray<float>, float, float>(metaGraph.points[i].centerOfMass, 
                                                                                                            {blockGraphs[i], trainMapper.dataBlocks[i]}, 
                                                                                                            10, 
                                                                                                            EuclideanNorm<float, float, float>);

        blockUpdateContexts.emplace_back(SubProblemData{blockGraphs[i], trainMapper.dataBlocks[i]},
                                         QueryContextInitArgs<size_t, std::valarray<float>, float>(queryHint, EuclideanNorm<float, float, float>),
                                         metaGraph.verticies.size());

    }
    
    for (size_t i = 0; const auto& graph: blockGraphs){
        Graph<BlockIndecies, float> newGraph(graph.size(), graph[0].size());
        for (size_t j = 0; const auto& vertex: graph){
            newGraph[j].resize(graph[j].size());
            auto emplaceFunctor = [&](size_t index){
                newGraph[j][index] = {{i, vertex[index].first}, vertex[index].second};
            };
            std::ranges::iota_view indecies(size_t(0), vertex.size());
            std::for_each(indecies.begin(), indecies.end(), emplaceFunctor);
            j++;
        }
        blockUpdateContexts[i].currentGraph = std::move(newGraph);
        blockUpdateContexts[i].blockJoinTracker[i] = true;
        i++;
    }

    std::unordered_set<ComparisonKey<size_t>> nearestNodeDistQueue;

    for (size_t i = 0; const auto& vertex: metaGraph.verticies){
        for (const auto& neighbor: vertex){
            nearestNodeDistQueue.insert({i, neighbor.first});
        }
        i++;
    }

    std::vector<ComparisonKey<size_t>> distancesToCompute;
    distancesToCompute.reserve(nearestNodeDistQueue.size());
    for (const auto& pair: nearestNodeDistQueue){
        distancesToCompute.push_back(pair);
    }

    std::vector<std::tuple<size_t, size_t, float>> nnDistanceResults(nearestNodeDistQueue.size());
    auto nnDistanceFunctor = [&](const ComparisonKey<size_t> blockNumbers) -> std::tuple<size_t, size_t, float>{
        return blockUpdateContexts[blockNumbers.first].queryContext * blockUpdateContexts[blockNumbers.second].queryContext;
    };

    std::transform(std::execution::seq, distancesToCompute.begin(), distancesToCompute.end(), nnDistanceResults.begin(), nnDistanceFunctor);


    Graph<size_t, float> nearestNodeDistances(metaGraph.verticies.size(), 10);
    for(size_t i = 0; const auto& result: nnDistanceResults){
        
        nearestNodeDistances[distancesToCompute[i].first].push_back({distancesToCompute[i].second, std::get<2>(result)});
        nearestNodeDistances[distancesToCompute[i].second].push_back({distancesToCompute[i].first, std::get<2>(result)});
        //nearestNeighbors.push_back({pair.first, std::get<2>(pair.second)});
        i++;
    }

    auto sortFunctor = [] (GraphVertex<size_t, float>& vertex){
        std::sort(std::execution::unseq, vertex.begin(), vertex.end(), NeighborDistanceComparison<size_t, float>);
    };
    std::for_each(std::execution::unseq, nearestNodeDistances.begin(), nearestNodeDistances.end(), sortFunctor);


    std::unordered_set<ComparisonKey<size_t>> initBlockJoinQueue;
    for(size_t i = 0; const auto& vertex: nearestNodeDistances){
        for(size_t j = 0; j<3; j+=1){
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

    std::vector<std::pair<Graph<size_t, float>, Graph<size_t, float>>> initUpdates(initBlockJoins.size());

    auto initBlockJoin = [&](const ComparisonKey<size_t> blockNumbers) -> std::pair<Graph<size_t, float>, Graph<size_t, float>>{
        return blockUpdateContexts[blockNumbers.first].queryContext || blockUpdateContexts[blockNumbers.second].queryContext;
    };

    std::transform(std::execution::seq, initBlockJoins.begin(), initBlockJoins.end(), initUpdates.begin(), initBlockJoin);
    int initGraphUpdates(0);
    for (size_t i = 0; i<initUpdates.size(); i += 1){
        ComparisonKey<size_t> blocks = initBlockJoins[i];
        std::pair<Graph<size_t, float>, Graph<size_t, float>>& updates = initUpdates[i];
        for (size_t j = 0; j<blockUpdateContexts[blocks.first].currentGraph.size(); j+=1){
            initGraphUpdates += ConsumeVertex(blockUpdateContexts[blocks.first].currentGraph[j], updates.first[j], blocks.second);
        }

        for (size_t j = 0; j<blockUpdateContexts[blocks.second].currentGraph.size(); j+=1){
            initGraphUpdates += ConsumeVertex(blockUpdateContexts[blocks.second].currentGraph[j], updates.second[j], blocks.first);
        }
    }
    //std::vector<Graph<BlockIndecies, float>> currentGraphs(blockGraphs.size());

    

    //std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();

    //std::vector<NodeTracker> blockJoinTrackers(blockGraphs.size(), NodeTracker(blockGraphs.size()));

    
    //Initial filling of comparison targets.
    std::vector<ComparisonMap<size_t, size_t>> queueMaps;
    queueMaps.reserve(blockUpdateContexts.size());
    for (size_t i = 0; i<blockUpdateContexts.size(); i+=1){
        queueMaps.push_back(InitializeComparisonQueues<size_t, size_t, float>(blockUpdateContexts[i].currentGraph, i));
    }
    
    //std::vector<JoinMap<size_t, size_t>> joinHints;

    for(size_t i = 0; i<queueMaps.size(); i+=1){
        ComparisonMap<size_t, size_t>& comparisonMap = queueMaps[i];
        
        blockUpdateContexts[i].joinsToDo = InitializeJoinMap<size_t, size_t, std::valarray<float>, float>(blockUpdateContexts, comparisonMap, blockUpdateContexts[i].blockJoinTracker);
    }


    //std::vector<std::unordered_map<size_t, JoinResults<size_t, float>>> blockUpdates(blockUpdateContexts.size());
    

    GraphVertex<BlockIndecies, float> nullVertex;
    for(size_t i = 0; i<10; i+=1){
        nullVertex.push_back({{0,0}, std::numeric_limits<float>::max()});
    }
    int iteration(1);
    int graphUpdates(1);
    while(graphUpdates>0){
        graphUpdates = 0;
       // std::vector<std::unordered_map<size_t, JoinResults<size_t, float>>> blockUpdates(updatedBlockGraphs.size());
        for(size_t i = 0; i<blockUpdateContexts.size(); i+=1){
            for (auto& joinList: blockUpdateContexts[i].joinsToDo){
                graphUpdates += UpdateBlocks(blockUpdateContexts[i], blockUpdateContexts[joinList.first]);
                blockUpdateContexts[joinList.first].joinsToDo.erase(i);
            }
            /*
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
            */
        }
        for (auto& context: blockUpdateContexts){
            context.SetNextJoins();
        }
        /*
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
        */
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
                blockUpdates[i][joinList.first] = BlockwiseJoin(joinList.second, nearestNeighbors[i], reflexiveGraphs[i], testMapper.dataBlocks[i], blockUpdateContexts[joinList.first].queryContext, blockUpdateContexts[joinList.first].queryContext.defaultQueryFunctor);
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