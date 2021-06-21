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
#include <bit>

#include "MNISTData.hpp"
#include "NND/SpaceMetrics.hpp"
#include "NND/GraphStructures.hpp"
#include "NND/Algorithm.hpp"

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"



//namespace chrono = std::chrono;
using namespace nnd;



std::vector<size_t> LabelIndecies(const RandomProjectionForest& forest){

    std::vector<size_t> classifications(forest.indexArray.size());

    std::vector<size_t> treePath;
    std::vector<char> pathState;
    treePath.push_back(0);
    pathState.push_back(0);

    size_t highestIndex = 0;

    const RandomProjectionForest::TreeLeaf* currentLeaf = &(forest.treeLeaves[0]);


    while (treePath.size() != 0){

        if(currentLeaf->children.first != nullptr && currentLeaf->children.second != nullptr){
            if (pathState.back() == 0){
                pathState.back() = 1;
                currentLeaf = currentLeaf->children.first;
                treePath.push_back(currentLeaf->childrenSplittingIndex);
                pathState.push_back(0);
                continue;    
            } else if (pathState.back() == 1) {
                pathState.back() = 2;
                currentLeaf = currentLeaf->children.second;
                treePath.push_back(currentLeaf->childrenSplittingIndex);
                pathState.push_back(0);
                continue;
            } else if (pathState.back() == 2) {
                currentLeaf = currentLeaf->parent;
                pathState.pop_back();
                treePath.pop_back();
                continue;
            }
            throw std::logic_error("Invalid Crawl State");
            
        } else if (currentLeaf->children.first == nullptr && currentLeaf->children.second == nullptr){
            currentLeaf = currentLeaf->parent;
            pathState.pop_back();
            treePath.pop_back();
            highestIndex = std::max(highestIndex, currentLeaf->childrenSplittingIndex);
            for (size_t index = currentLeaf->splitRange.first; index < currentLeaf->splitRange.second; index += 1){
                classifications[forest.indexArray[index]] = currentLeaf->childrenSplittingIndex;
            }
            continue;
        }
        throw std::logic_error("Invalid Tree State");
        //size_t currentIndex = treePath.back();

    }

    return classifications;

};

struct MetaPoint{
    int weight;
    std::valarray<double> centerOfMass;
};
template<typename DataType>
std::unordered_map<size_t, MetaPoint> CalculateCOMs(const DataSet<DataType>& data, const std::vector<size_t>& classifications){

    std::unordered_map<size_t, MetaPoint> centerOfMasses;

    for (size_t i = 0; i<classifications.size(); i += 1){

        centerOfMasses[classifications[i]].weight += 1;
        for (size_t j = 0; j<data.sampleLength; j += 1){
            centerOfMasses[classifications[i]].centerOfMass[j] += static_cast<double>(data.samples[i][j]);
        }
    }
        /*
        if (centerOfMasses.find(classifications[i]) == centerOfMasses.end()){
            centerOfMasses[classifications[i]].weight = 1;
            centerOfMasses[classifications[i]].centerOfMass = std::valarray<double>(digits.vectorLength);
            for (size_t j = 0; j<digits.vectorLength; j += 1){
                centerOfMasses[classifications[i]].centerOfMass[j] = static_cast<double>(digits.samples[i][j]);
            }
        } else {
            centerOfMasses[classifications[i]].weight += 1;
            for (size_t j = 0; j<digits.vectorLength; j += 1){
                centerOfMasses[classifications[i]].centerOfMass[j] += static_cast<double>(digits.samples[i][j]);
            }
        }
        */

    return centerOfMasses;
}

//My download speed is getting hammered for some reason.
template<typename DataType>
[[nodiscard]] Graph<DataType> BruteForceGroundTruth(const DataSet<DataType>& dataSource,
                                           size_t numNeighbors,
                                           SpaceMetric<std::valarray<DataType>> distanceFunctor){
    NeighborSearchFunctor searchFunctor;
    Graph<DataType> retGraph(0);
    retGraph.reserve(dataSource.numberOfSamples);

    for (size_t i = 0; i<dataSource.numberOfSamples; i+=1){
        //std::slice vertexSlice(0, dataSource.vectorLength, 1);

        retGraph.push_back(GraphVertex<DataType>(i, dataSource.samples[i], numNeighbors));
        
    };

    for (size_t i = 0; i<dataSource.numberOfSamples; i += 1){
        for (size_t j = i+1; j<dataSource.numberOfSamples; j += 1){
            //if (i == j) continue;
            double distance = distanceFunctor(dataSource.samples[i], dataSource.samples[j]);
            if (distance < retGraph[i].neighbors[0].second) retGraph[i].PushNeigbor(std::pair(j, distance));
            if (distance < retGraph[j].neighbors[0].second) retGraph[j].PushNeigbor(std::pair(i, distance));
        }
    }

    return retGraph;
}


using MetaGraph = std::unordered_map<size_t, std::unordered_map<size_t, size_t>>;

template<typename DataType>
MetaGraph NeighborsOutOfBlock(const Graph<DataType>& groundTruth, const std::vector<size_t>& classifications){
    MetaGraph retGraph;
    for(size_t i = 0; i<classifications.size(); i += 1){
        size_t treeIndex = classifications[i];
        for(const auto& neighbor: groundTruth[i].neighbors){
            retGraph[treeIndex][classifications[neighbor.first]] += 1;
        }
    }

    return retGraph;
};


int main(){

    std::cout << "Test String" << std::endl;
    std::string dataFilePath("./TestData/MNIST-Fashion-Data.bin");
    //MNISTData digits(digitsFilePath);
    static const std::endian dataEndianness = std::endian::big;
    DataSet<float> mnistFashion(dataFilePath, 28*28, 10'000, dataEndianness);

    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), mnistFashion.numberOfSamples - 1);
    StlRngFunctor<std::mt19937_64, std::uniform_int_distribution, size_t> rngFunctor(std::move(rngEngine), std::move(rngDist));

    EuclidianSplittingScheme<double, float> splittingScheme(mnistFashion);

    //StlRngFunctor<> rngFunctor, SplittingScheme<FloatType> getSplitComponents, int splits = 8
    RandomProjectionForest rpTrees(size_t(mnistFashion.numberOfSamples), rngFunctor, SplittingScheme(splittingScheme));

    std::vector<size_t> classifications = LabelIndecies(rpTrees);

    std::unordered_map<size_t, MetaPoint> centerOfMasses = CalculateCOMs(mnistFashion, classifications);

    //MetaGraph treeGraph = NeighborsOutOfBlock()
    //SpaceMetric<std::valarray<unsigned char>> distFunc = &EuclideanNorm<unsigned char>

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