#ifndef NND_METAGRAPH_HPP
#define NND_METAGRAPH_HPP

#include <vector>
#include <valarray>

#include "MNISTData.hpp"
#include "GraphStructures.hpp"

namespace nnd{

// Todo might need to use something other than valarray if I need to handle truly arbitrary data types.
struct MetaPoint{
    int weight;
    std::valarray<double> centerOfMass;
};

//template<typename DataType>
MetaPoint CalculateCOM(const DataBlock<std::valarray<unsigned char>>& dataBlock){

    MetaPoint retPoint;
    retPoint.weight = dataBlock.blockData.size();
    retPoint.centerOfMass = std::valarray<double>(dataBlock.blockData[0].size());
    
    for (size_t i = 0; i<dataBlock.blockData.size(); i += 1){
        for(size_t j = 0; j<dataBlock.blockData[i].size(); j += 1){
            retPoint.centerOfMass[j] += static_cast<double>(dataBlock.blockData[i][j]);
        }
    }

    //divide the COM by the weight to put it in the center of the cluster
    retPoint.centerOfMass /= retPoint.weight;

    return retPoint;
}

template<typename DataType, typename FloatType>
void BruteForceGraph(Graph<size_t, FloatType>& uninitGraph, size_t numNeighbors, const std::vector<MetaPoint>& dataVector, SpaceMetric<DataType, FloatType> distanceFunctor){
    
    // I can make this branchless. Check to see if /O2 or /O3 can make this branchless (I really doubt it)
    for (size_t i = 0; i < dataVector.size(); i += 1){
        for (size_t j = i+1; j < dataVector.size(); j += 1){
            FloatType distance = distanceFunctor(dataVector[i].centerOfMass, dataVector[j].centerOfMass);
            if (uninitGraph[i].neighbors.size() < numNeighbors){
                uninitGraph[i].neighbors.push_back(std::pair<size_t, FloatType>(j, distance));
                if (uninitGraph[i].neighbors.size() == numNeighbors){
                    std::make_heap(uninitGraph[i].neighbors.begin(), uninitGraph[i].neighbors.end(), NeighborDistanceComparison<size_t, FloatType>);
                }
            } else if (distance < uninitGraph[i].neighbors[0].second){
                uninitGraph[i].PushNeigbor(std::pair<size_t, FloatType>(j, distance));
            }
            if (uninitGraph[j].neighbors.size() < numNeighbors){
                uninitGraph[j].neighbors.push_back(std::pair<size_t, FloatType>(i, distance));
                if (uninitGraph[j].neighbors.size() == numNeighbors){
                    std::make_heap(uninitGraph[j].neighbors.begin(), uninitGraph[j].neighbors.end(), NeighborDistanceComparison<size_t, FloatType>);
                }
            } else if (distance < uninitGraph[j].neighbors[0].second){
                uninitGraph[j].PushNeigbor(std::pair<size_t, FloatType>(i, distance));
            }
        }
    }
}

struct MetaGraph{
    std::vector<MetaPoint> points;
    Graph<size_t, double> verticies;

    //template<typename DataEntry>
    MetaGraph(const std::vector<DataBlock<std::valarray<unsigned char>>>& dataBlocks): points(0){
        for (const auto& dataBlock: dataBlocks){
            points.push_back(CalculateCOM(dataBlock));
        }
        verticies = ConstructInitialGraph<size_t, double>(points.size(), size_t(5));
        BruteForceGraph<std::valarray<double>, double>(verticies, size_t(20), points, EuclideanNorm<double, double>);
    }
};








}

#endif