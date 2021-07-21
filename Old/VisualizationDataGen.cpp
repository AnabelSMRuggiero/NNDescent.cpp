//This was used to generate data to use for data visualization

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

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"

#include "Utilities/DataSerialization.hpp"
#include "Utilities/DataDeserialization.hpp"


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
                treePath.push_back(currentLeaf->splittingIndex);
                pathState.push_back(0);
                continue;    
            } else if (pathState.back() == 1) {
                pathState.back() = 2;
                currentLeaf = currentLeaf->children.second;
                treePath.push_back(currentLeaf->splittingIndex);
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
            highestIndex = std::max(highestIndex, currentLeaf->splittingIndex);
            for (size_t index = currentLeaf->splitRange.first; index < currentLeaf->splitRange.second; index += 1){
                classifications[forest.indexArray[index]] = currentLeaf->splittingIndex;
            }
            currentLeaf = currentLeaf->parent;
            pathState.pop_back();
            treePath.pop_back();
            
            
            continue;
        }
        throw std::logic_error("Invalid Tree State");
        //size_t currentIndex = treePath.back();

    }

    return classifications;

};


// START METAGRAPH PROTOTYPING
struct MetaPoint{
    int weight;
    std::valarray<double> centerOfMass;
};
template<typename DataType>
std::unordered_map<size_t, MetaPoint> CalculateCOMs(const DataSet<DataType>& data, const std::vector<size_t>& classifications){

    std::unordered_map<size_t, MetaPoint> centerOfMasses;

    for (size_t i = 0; i<classifications.size(); i += 1){

        if (centerOfMasses.find(classifications[i]) == centerOfMasses.end()){
            centerOfMasses[classifications[i]].weight = 1;
            centerOfMasses[classifications[i]].centerOfMass = std::valarray<double>(data.samples[i].size());
            for (size_t j = 0; j<data.samples[0].size(); j += 1){
                centerOfMasses[classifications[i]].centerOfMass[j] = static_cast<double>(data.samples[i][j]);
            }
        } else {
            centerOfMasses[classifications[i]].weight += 1;
            for (size_t j = 0; j<data.samples[0].size(); j += 1){
                centerOfMasses[classifications[i]].centerOfMass[j] += static_cast<double>(data.samples[i][j]);
            }
        }
        
    }

    //divide the COM by the weight to put it in the center of the cluster
    for(auto& pair : centerOfMasses){
        pair.second.centerOfMass /= pair.second.weight;
    }

    return centerOfMasses;
}

/*
    File format
    [offset] [type]         [value]                         [description]
    0000     8 byte int     COMs.size()                     number of node positions
    0008     8 byte int     firstEntry.centerOfMass.size()  length of COM vectors (all vectors have the same dimensionality)
    0016     8 byte int     firstEntry.key                  splittingIndex of first COM
    0024     8 byte int     firstEntry.point.weight         weight of first COM
    0032     8 byte double  firstEntry.point.COM[0]         0th (first) dimension value
    0040     8 byte double  firstEntry.point.COM[1]         1st dimension value
    ........ 
    xxxx     8 byte double  firstEntry.point.COM[n-1]       (n-1)th (last) dimension value
    xxxx     8 byte int     secondEntry.key                 splittingIndex of second entry 
    xxxx     8 byte int     secondEntry.point.weight        weight of second COM
    xxxx     8 byte double  secondEntry.point.COM[0]        0th (first) dimension value

*/


void SerializeCOMS(const std::unordered_map<size_t, MetaPoint>& COMs, const std::string& outputFile){
    std::ofstream outStream(outputFile, std::ios_base::binary);

    SerializeData<size_t, std::endian::big>(outStream, COMs.size());
    SerializeData<size_t, std::endian::big>(outStream, COMs.begin()->second.centerOfMass.size());
    
    for(const auto& pair : COMs){
        SerializeData<size_t, std::endian::big>(outStream, pair.first);
        SerializeData<size_t, std::endian::big>(outStream, pair.second.weight);
        for(const auto& extent : pair.second.centerOfMass){
            SerializeData<double, std::endian::big>(outStream, extent);
        }

    }

}


//My download speed is getting hammered for some reason.
//Shouldn't need this at this point
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

MetaGraph NeighborsOutOfBlock(const DataSet<std::valarray<int32_t>>& groundTruth, const std::vector<size_t>& trainClassifications, const std::vector<size_t>& testClassifications){
    MetaGraph retGraph;
    for(size_t i = 0; i<groundTruth.samples.size(); i += 1){
        size_t treeIndex = testClassifications[i];
        for(const auto& neighbor: groundTruth.samples[i]){
            (retGraph[treeIndex])[trainClassifications[neighbor]] += 1;
        }
    }

    return retGraph;
};

/*
    File format
    [offset] [type]         [value]                     [description]
    0000     8 byte int     MetaGraph.size()            number of terminal leaves in the graph
    0008     8 byte int     firstEntry.first            splittingIndex of first leaf
    0016     8 byte int     firstEntry.second.size()    number of edges pointing away from leaf
    0024     8 byte int     edge.target                 splittingIndex of target node
    0032     8 byte int     edge.weight                 number of neighbors contained in target tree
    0040     8 byte int     edge.target                 splittingIndex of target node
    0048     8 byte int     edge.weight                 number of neighbors contained in target tree
    ........ 
    Each graph node is 16*(1 + size) bytes long, or 2*(1 + size) size_t's
    xxxx     8 byte int     secondEntry.first           splittingIndex of second leaf
*/

void SerializeMetaGraph(const MetaGraph& readGraph, const std::string& outputFile){
    std::ofstream outStream(outputFile, std::ios_base::binary);

    SerializeData<size_t, std::endian::big>(outStream, readGraph.size());

    for(const auto& pair : readGraph){
        SerializeData<size_t, std::endian::big>(outStream, pair.first);
        SerializeData<size_t, std::endian::big>(outStream, pair.second.size());
        for (const auto& edge : pair.second){
            SerializeData<size_t, std::endian::big>(outStream, edge.first);
            SerializeData<size_t, std::endian::big>(outStream, edge.second);
        }
    }

}



template<TriviallyCopyable DataType>
void SerializeVector(const std::vector<DataType>& readVector, const std::string& outputFile){
    std::ofstream outStream(outputFile, std::ios_base::binary);
    SerializeData<size_t, std::endian::big>(outStream, readVector.size());

    for(const auto& entry : readVector){
        SerializeData<DataType, std::endian::big>(outStream, entry);
    }

}

// END METAGRAPH PROTOTYPING

int main(){

    
    std::string testDataFilePath("./TestData/MNIST-Fashion-Data.bin");
    std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");
    std::string testNeighborsFilePath("./TestData/MNIST-Fashion-Neighbors.bin");
    //MNISTData digits(digitsFilePath);
    static const std::endian dataEndianness = std::endian::big;
    DataSet<std::valarray<float>> mnistFashionTest(testDataFilePath, 28*28, 10'000, &ExtractNumericArray<float,dataEndianness>);
    DataSet<std::valarray<float>> mnistFashionTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<float,dataEndianness>);
    DataSet<std::valarray<int32_t>> mnistFashionTestNeighbors(testNeighborsFilePath, 100, 10'000, &ExtractNumericArray<int32_t,dataEndianness>);

    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), mnistFashionTest.numberOfSamples - 1);
    StlRngFunctor<std::mt19937_64, std::uniform_int_distribution, size_t> rngFunctor(std::move(rngEngine), std::move(rngDist));

    EuclidianSplittingScheme<double, float> splittingScheme(mnistFashionTrain);
    SplittingScheme splitterFunc(splittingScheme);
    //StlRngFunctor<> rngFunctor, SplittingScheme<FloatType> getSplitComponents, int splits = 8
    RandomProjectionForest rpTreesTrain(size_t(mnistFashionTrain.numberOfSamples), rngFunctor, splitterFunc);

    EuclidianSplittingScheme<double, float> testSplitterScheme(mnistFashionTest);
    testSplitterScheme.splittingVectors = splitterFunc.target<EuclidianSplittingScheme<double, float>>()->splittingVectors;

    SplittingScheme testSplitterFunc(testSplitterScheme);

    std::unordered_set<size_t> splittingIndicies;
    for (auto pair: testSplitterScheme.splittingVectors){
        splittingIndicies.insert(pair.first);
    }
    RandomProjectionForest rpTreesTest(size_t(mnistFashionTest.numberOfSamples), rngFunctor, testSplitterFunc, splittingIndicies);


    std::vector<size_t> testClassifications = LabelIndecies(rpTreesTest);
    auto testResult = std::find(testClassifications.begin(), testClassifications.end(), 0);
    std::vector<size_t> trainClassifications = LabelIndecies(rpTreesTrain);
    auto trainResult = std::find(trainClassifications.begin(), trainClassifications.end(), 0);
    
    std::unordered_map<size_t, MetaPoint> centerOfMasses = CalculateCOMs(mnistFashionTrain, trainClassifications);
    SerializeCOMS(centerOfMasses, "./TestData/MNIST-Fashion-Train-COMs.bin");
    MetaGraph neighborsOOB = NeighborsOutOfBlock(mnistFashionTestNeighbors, trainClassifications, testClassifications);
    SerializeMetaGraph(neighborsOOB, "./TestData/MNIST-Fashion-Test-MetaGraphEdges.bin");
    SerializeVector<size_t>(trainClassifications, "./TestData/MNIST-Fashion-Train-SplittingIndicies.bin");
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
