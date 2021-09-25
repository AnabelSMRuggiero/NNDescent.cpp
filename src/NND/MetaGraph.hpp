/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_METAGRAPH_HPP
#define NND_METAGRAPH_HPP

#include <vector>
#include <valarray>
#include <ranges>
#include <span>
#include <concepts>
#include <execution>
#include <algorithm>

#include "../Utilities/Data.hpp"

#include "Type.hpp"
#include "GraphStructures.hpp"
#include "SubGraphQuerying.hpp"

namespace nnd{

// Todo might need to use something other than valarray if I need to handle truly arbitrary data types.
template<typename DistType>
struct MetaPoint{
    unsigned int weight;
    AlignedArray<DistType> centerOfMass;
    //SinglePointFunctor<DistType> distanceFunctor;
};



template<typename DataType, typename COMExtent>
AlignedArray<COMExtent> CalculateCOM(const DataBlock<DataType>& dataBlock){

    
    //retPoint.weight = dataBlock.size();
    AlignedArray<COMExtent> retPoint(std::ranges::size(dataBlock[0]));
    //retPoint.distanceFunctor = distanceFunctor;
    
    for (size_t i = 0; i<dataBlock.size(); i += 1){
        for(size_t j = 0; j<dataBlock[i].size(); j += 1){
            retPoint[j] += static_cast<COMExtent>(dataBlock[i][j]);
        }
    }

    for(auto& extent: retPoint) extent /= dataBlock.size();

    return retPoint;
}

template<typename DataType, typename COMExtent>
void EuclideanCOM(const DataBlock<DataType>& dataBlock, AlignedSpan<COMExtent> writeLocation){
    
    for (size_t i = 0; i<dataBlock.size(); i += 1){
        for(size_t j = 0; j<dataBlock[i].size(); j += 1){
            writeLocation[j] += static_cast<COMExtent>(dataBlock[i][j]);
        }
    }

    for(auto& extent: writeLocation) extent /= dataBlock.size();

    //return retPoint;
}

template<typename FloatType, typename Metric>
void BruteForceGraph(Graph<BlockNumber_t, FloatType>& uninitGraph, size_t numNeighbors, const DataBlock<FloatType>& dataVector, Metric distanceFunctor){

    for (size_t i = 0; i < uninitGraph.size(); i += 1){
        for (size_t j = i+1; j < uninitGraph.size(); j += 1){
            FloatType distance = distanceFunctor(dataVector[i], dataVector[j]);
            if (uninitGraph[i].size() < numNeighbors){
                uninitGraph[i].push_back(std::pair<size_t, FloatType>(j, distance));
                if (uninitGraph[i].size() == numNeighbors){
                    uninitGraph[i].JoinPrep();
                    //std::make_heap(uninitGraph[i].neighbors.begin(), uninitGraph[i].neighbors.end(), NeighborDistanceComparison<size_t, FloatType>);
                }
            } else if (distance < uninitGraph[i].PushThreshold()){
                uninitGraph[i].PushNeighbor(std::pair<size_t, FloatType>(j, distance));
            }
            if (uninitGraph[j].size() < numNeighbors){
                uninitGraph[j].push_back(std::pair<size_t, FloatType>(i, distance));
                if (uninitGraph[j].size() == numNeighbors){
                    uninitGraph[j].JoinPrep();
                    //std::make_heap(uninitGraph[j].begin(), uninitGraph[j].end(), NeighborDistanceComparison<size_t, FloatType>);
                }
            } else if (distance < uninitGraph[j].PushThreshold()){
                uninitGraph[j].PushNeighbor(std::pair<size_t, FloatType>(i, distance));
            }
        }
    }
}

template<typename COMExtent>
struct MetaGraph{
    //std::vector<MetaPoint<COMExtent>> points;
    std::vector<size_t> weights;
    DataBlock<COMExtent> points;
    Graph<BlockNumber_t, COMExtent> verticies;
    QueryContext<BlockNumber_t, COMExtent> queryContext;
    NodeTracker fragmentsJoined = {};
    /*
    template<typename DataType, typename Metric, typename COMFunctor>
    MetaGraph(const std::vector<DataBlock<DataType>>& dataBlocks, const size_t numNeighbors, Metric metricFunctor, COMFunctor COMCalculator):
        weights(0), 
        points(dataBlocks.size(), dataBlocks[0].entryLength),
        verticies(dataBlocks.size(), numNeighbors){
        //SinglePointFunctor<COMExtent> functor(DataComDistance<DataEntry, COMExtent, MetricPair>(*this, dataBlocks, metricFunctor));
        weights.reserve(dataBlocks.size());
        for (size_t i = 0; const auto& dataBlock: dataBlocks){
            weights.push_back(dataBlock.size());
            COMCalculator(dataBlock, points[i]);
            i++;
        }
        BruteForceGraph<COMExtent, Metric>(verticies, numNeighbors, points, metricFunctor);
        for(auto& vertex: verticies){
            std::sort(vertex.begin(), vertex.end(), NeighborDistanceComparison<size_t, COMExtent>);
        }
    }
    */

    size_t size() const{
        return points.numEntries;
    }

    size_t FragmentNumber() const{
        return points.blockNumber;
    }
    /*
    DataComDistance(const ComView& centerOfMass, const std::vector<DataBlock<DataEntry>>& blocks): centerOfMass(centerOfMass), blocks(blocks), functor(){};

    DataComDistance(const ComView& centerOfMass, const std::vector<DataBlock<DataEntry>>& blocks, MetricPair functor)
    */
};

template<typename COMExtent, typename DataType, typename MetricSet, typename COMFunctor>
MetaGraph<COMExtent> BuildMetaGraphFragment(const std::vector<DataBlock<DataType>>& dataBlocks, const IndexParameters& params, const size_t fragmentNumber, MetricSet metricSet, COMFunctor COMCalculator){
    std::vector<size_t> weights(0);
    weights.reserve(dataBlocks.size());
    DataBlock<COMExtent> points(dataBlocks.size(), dataBlocks[0].entryLength, fragmentNumber);
    Graph<BlockNumber_t, COMExtent> verticies(dataBlocks.size(), params.COMNeighbors);
    //SinglePointFunctor<COMExtent> functor(DataComDistance<DataEntry, COMExtent, MetricPair>(*this, dataBlocks, metricFunctor));
    weights.reserve(dataBlocks.size());
    for (size_t i = 0; const auto& dataBlock: dataBlocks){
        weights.push_back(dataBlock.size());
        COMCalculator(dataBlock, points[i]);
        i++;
    }

    AlignedArray<COMExtent> centerOfMass(dataBlocks[0].entryLength);
    COMCalculator(points, {centerOfMass.GetAlignedPtr(0), centerOfMass.size()});

    BruteForceGraph<COMExtent>(verticies, params.COMNeighbors, points, metricSet.dataToData);
    for(auto& vertex: verticies){
        std::sort(vertex.begin(), vertex.end(), NeighborDistanceComparison<size_t, COMExtent>);
    }

    auto neighborFunctor = [&](size_t, size_t pointIndex){
        return metricSet.comToCom({centerOfMass.GetAlignedPtr(0), centerOfMass.size()}, points[pointIndex]);
    };
    GraphVertex<BlockNumber_t, COMExtent> queryHint = QueryCOMNeighbors<COMExtent, DataType, BlockNumber_t>(0, verticies, params.COMNeighbors, neighborFunctor);

    QueryContext<BlockNumber_t,COMExtent> queryContext(verticies,
                                         std::move(queryHint),
                                         params.queryDepth,
                                         fragmentNumber,
                                         BlockNumber_t{std::numeric_limits<BlockNumber_t>::max()},
                                         points.size());

    return MetaGraph<COMExtent>{weights, std::move(points), verticies, std::move(queryContext)};
}


/*
    File format
    [offset] [type]         [value]                         [description]
    0000     8 byte int     COMs.size()                     number of node positions
    0008     8 byte int     firstEntry.centerOfMass.size()  length of COM vectors (all vectors have the same dimensionality)
    0024     8 byte int     firstEntry.weight               weight of first COM
    0032     8 byte double  firstEntry.COM[0]               0th (first) dimension value
    0040     8 byte double  firstEntry.COM[1]               1st dimension value
    ........ 
    xxxx     8 byte double  firstEntry.COM[n-1]       (n-1)th (last) dimension value
    xxxx     8 byte int     secondEntry.weight        weight of second COM
    xxxx     8 byte double  secondEntry.COM[0]        0th (first) dimension value

*/
template<typename DistType>
void SerializeCOMS(const std::vector<MetaPoint<DistType>>& COMs, const std::string& outputFile){
    std::ofstream outStream(outputFile, std::ios_base::binary);

    SerializeData<size_t, std::endian::big>(outStream, COMs.size());
    SerializeData<size_t, std::endian::big>(outStream, COMs.begin()->centerOfMass.size());
    
    for(const auto& point : COMs){
        SerializeData<size_t, std::endian::big>(outStream, point.weight);
        for(const auto& extent : point.centerOfMass){
            SerializeData<DistType, std::endian::big>(outStream, extent);
        }

    }

}

/*
template<typename DataEntry>
auto GenerateDataBlockingFunc(const DataSet<DataEntry>& dataSource,
                                  std::vector<DataBlock<DataEntry>>& accumulationVector,
                                  std::unordered_map<size_t, size_t>& splitToBlockIndex,
                                  std::vector<BlockIndex>& indexRemapping){

    size_t blockCounter(0);
    auto retLambda = [&](size_t splittingIndex, std::span<const size_t> indicies){
        splitToBlockIndex[splittingIndex] = blockCounter;
        accumulationVector.push_back(DataBlock(dataSource, indicies, blockCounter++, indexRemapping));
    };
    return retLambda;
}
*/

template<std::integral BlockNumberType>//, std::integral DataIndexType>
struct IndexMaps{

    std::unordered_map<size_t, BlockNumberType> splitToBlockNum;
    std::vector<std::vector<size_t>> blockIndexToSource;
    std::vector<BlockIndecies> sourceToBlockIndex;
    std::vector<size_t> sourceToSplitIndex;

};

template<typename DataType, typename DataStructure, typename BoundConstructor>
struct DataMapper{

    const DataSet<DataType>& dataSource;
    size_t blockCounter;
    size_t graphFragment;
    std::vector<DataStructure> dataBlocks;
    std::unordered_map<size_t, size_t> splitToBlockNum;
    std::vector<std::vector<size_t>> blockIndexToSource;
    std::vector<BlockIndecies> sourceToBlockIndex;
    std::vector<size_t> sourceToSplitIndex;
    BoundConstructor construct;

    DataMapper(const DataSet<DataType>& source, BoundConstructor constructor, const size_t fragmentNumber = 0, const size_t startIndex = 0):
        dataSource(source),
        blockCounter(startIndex),
        graphFragment(fragmentNumber),
        sourceToBlockIndex(dataSource.size()),
        sourceToSplitIndex(dataSource.size()),
        construct(constructor) {};

    void operator()(size_t splittingIndex, std::span<const size_t> indicies){
        //[[unlikely]]if (indicies.size() == 0) return;
        splitToBlockNum[splittingIndex] = blockCounter;
        std::vector<size_t> indeciesInBlock(indicies.size());
        std::copy(indicies.begin(), indicies.end(), indeciesInBlock.begin());
        blockIndexToSource.push_back(std::move(indeciesInBlock));
        for (size_t i = 0; i<indicies.size(); i += 1){
            size_t index = indicies[i];
            sourceToBlockIndex[index] = BlockIndecies{graphFragment, blockCounter, i};
            sourceToSplitIndex[index] = splittingIndex;
        }
        dataBlocks.push_back(construct(dataSource, indicies, blockCounter++));
    };
};

template<typename DataEntry>
struct DataMapper<DataEntry, void, void>{

    const DataSet<DataEntry>& dataSource;
    size_t blockCounter;
    size_t graphFragment;
    std::unordered_map<size_t, size_t> splitToBlockNum;
    std::vector<std::vector<size_t>> blockIndexToSource;
    std::vector<BlockIndecies> sourceToBlockIndex;
    std::vector<size_t> sourceToSplitIndex;
    

    DataMapper(const DataSet<DataEntry>& source, const size_t fragmentNumber = 0, const size_t startIndex = 0):
        dataSource(source),
        blockCounter(startIndex),
        graphFragment(fragmentNumber),
        sourceToBlockIndex(dataSource.size()),
        sourceToSplitIndex(dataSource.size()) {};

    void operator()(size_t splittingIndex, std::span<const size_t> indicies){
        //[[unlikely]]if (indicies.size() == 0) return;
        splitToBlockNum[splittingIndex] = blockCounter;
        std::vector<size_t> indeciesInBlock(indicies.size());
        std::copy(indicies.begin(), indicies.end(), indeciesInBlock.begin());
        blockIndexToSource.push_back(std::move(indeciesInBlock));
        for (size_t i = 0; i<indicies.size(); i += 1){
            size_t index = indicies[i];
            sourceToBlockIndex[index] = BlockIndecies{graphFragment, blockCounter, i};
            sourceToSplitIndex[index] = splittingIndex;
            //blockIndexToSource[BlockIndecies{blockCounter, i}] = index;
            
        }
        blockCounter++;
    };
};

/*
template<typename DataEntry, typename DataStructure>
struct DataMapper<DataEntry, DataStructure>{

    const DataSet<DataEntry>& dataSource;
    size_t blockCounter;
    std::vector<DataStructure> dataBlocks;
    std::unordered_map<size_t, size_t> splitToBlockNum;
    std::unordered_map<BlockIndecies, size_t> blockIndexToSource;
    std::vector<BlockIndecies> sourceToBlockIndex;
    std::vector<size_t> sourceToSplitIndex;

    DataMapper(const DataSet<DataEntry>& source):
        dataSource(source), sourceToBlockIndex(dataSource.numberOfSamples), sourceToSplitIndex(dataSource.numberOfSamples), blockCounter(0) {};

    void operator()(size_t splittingIndex, std::span<const size_t> indicies){
        //[[unlikely]]if (indicies.size() == 0) return;
        splitToBlockNum[splittingIndex] = blockCounter;
        for (size_t i = 0; i<indicies.size(); i += 1){
            size_t index = indicies[i];
            sourceToBlockIndex[index] = BlockIndecies{blockCounter, i};
            sourceToSplitIndex[index] = splittingIndex;
            blockIndexToSource[BlockIndecies{blockCounter, i}] = index;
        }
        dataBlocks.push_back(DataStructure(dataSource, indicies, blockCounter++));
    };
};
*/




using UnweightedGraphEdges = std::unordered_map<size_t, std::unordered_map<size_t, size_t>>;
using WeightedGraphEdges = std::unordered_map<size_t, std::vector<std::pair<size_t, double>>>;

WeightedGraphEdges NeighborsOutOfBlock(const DataSet<int32_t>& groundTruth,
    const std::vector<BlockIndecies>& trainClassifications,
    const std::vector<size_t>& testClassifications){
        UnweightedGraphEdges unweightedGraph;
        for(size_t i = 0; i<groundTruth.size(); i += 1){
            size_t treeIndex = testClassifications[i];
            //for(const auto& neighbor: groundTruth.samples[i]){
            for (size_t j = 0; j<10; j +=1){
                int32_t neighbor = groundTruth[i][j];
                (unweightedGraph[treeIndex])[trainClassifications[neighbor].blockNumber] += 1;
            }
        }

        WeightedGraphEdges retGraph;

        auto weightedEdgeCmp = [](std::pair<size_t, double> lhs, std::pair<size_t, double> rhs){
            return lhs.second > rhs.second;
        };

        for (const auto& [originIndex, edges] : unweightedGraph){
            size_t totWeight(0);
            for (const auto [targetIndex, weight] : edges){
                totWeight += weight;
            }
            for (const auto [targetIndex, weight] : edges){
                (retGraph[originIndex]).push_back(std::pair<size_t, double>(targetIndex, double(weight)/double(totWeight)));
            }
            std::sort(retGraph[originIndex].begin(), retGraph[originIndex].end(), weightedEdgeCmp);
        }
        return retGraph;
};

/*
    File format
    [offset] [type]         [value]                     [description]
    0000     8 byte int     MetaGraph.size()            number of terminal leaves in the graph
    0008     8 byte int     firstEntry.first            blockNumber of first leaf
    0016     8 byte int     firstEntry.second.size()    number of edges pointing away from leaf
    0024     8 byte int     edge.target                 blockNumber of target node
    0032     8 byte double  edge.weight                 number of neighbors contained in target tree
    0040     8 byte int     edge.target                 blockNumber of target node
    0048     8 byte double  edge.weight                 number of neighbors contained in target tree
    ........ 
    Each graph node is 8*(1 + 2*size) bytes long, or (1 + 2*size) size_t's
    xxxx     8 byte int     secondEntry.first           blockNumber of second leaf
*/

void SerializeMetaGraph(const WeightedGraphEdges& readGraph, const std::string& outputFile){
    std::ofstream outStream(outputFile, std::ios_base::binary);

    SerializeData<size_t, std::endian::big>(outStream, readGraph.size());

    for(const auto& pair : readGraph){
        SerializeData<size_t, std::endian::big>(outStream, pair.first);
        SerializeData<size_t, std::endian::big>(outStream, pair.second.size());
        for (const auto& edge : pair.second){
            SerializeData<size_t, std::endian::big>(outStream, edge.first);
            SerializeData<double, std::endian::big>(outStream, edge.second);
        }
    }

}


}




#endif