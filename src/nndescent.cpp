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


template<typename Functor>
void CrawlTerminalLeaves(const RandomProjectionForest& forest, Functor& terminalFunctor){

    //std::vector<size_t> classifications(forest.indexArray.size());

    std::vector<size_t> treePath;
    std::vector<char> pathState;
    treePath.push_back(0);
    pathState.push_back(0);

    size_t highestIndex = 0;

    size_t currentIndex = 0;

    auto leafAccesor = [&] (size_t index) -> const std::vector<RandomProjectionForest::TreeLeaf>::const_iterator{
        return forest.treeLeaves.begin()+index;
    };
    //const RandomProjectionForest::TreeLeaf* currentLeaf = &(forest.treeLeaves[0]);


    while (treePath.size() != 0){

        if(leafAccesor(currentIndex)->children.first != 0 && leafAccesor(currentIndex)->children.second != 0){
            if (pathState.back() == 0){
                pathState.back() = 1;
                currentIndex = leafAccesor(currentIndex)->children.first;
                treePath.push_back(leafAccesor(currentIndex)->splittingIndex);
                pathState.push_back(0);
                continue;    
            } else if (pathState.back() == 1) {
                pathState.back() = 2;
                currentIndex = leafAccesor(currentIndex)->children.second;
                treePath.push_back(leafAccesor(currentIndex)->splittingIndex);
                pathState.push_back(0);
                continue;
            } else if (pathState.back() == 2) {
                currentIndex = leafAccesor(currentIndex)->parent;
                pathState.pop_back();
                treePath.pop_back();
                continue;
            }
            throw std::logic_error("Invalid Crawl State");
            
        } else if (leafAccesor(currentIndex)->children.first == 0 && leafAccesor(currentIndex)->children.second == 0){
            highestIndex = std::max(highestIndex, leafAccesor(currentIndex)->splittingIndex);
            
            std::span indexSpan(&(forest.indexArray[leafAccesor(currentIndex)->splitRange.first]),
                              size_t(leafAccesor(currentIndex)->splitRange.second - leafAccesor(currentIndex)->splitRange.first));

            terminalFunctor(leafAccesor(currentIndex)->splittingIndex, indexSpan);

            currentIndex = leafAccesor(currentIndex)->parent;
            pathState.pop_back();
            treePath.pop_back();
            
            
            continue;
        }
        throw std::logic_error("Invalid Tree State");
        //size_t currentIndex = treePath.back();

    }

    return;

};
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

template<typename DataEntry>
struct DataMapper{

    const DataSet<DataEntry>& dataSource;
    size_t blockCounter;
    std::vector<DataBlock<DataEntry>> dataBlocks;
    std::unordered_map<size_t, size_t> splitToBlockNum;
    std::unordered_map<BlockIndex, size_t> blockIndexToSource;
    std::vector<BlockIndex> sourceToBlockIndex;
    std::vector<size_t> sourceToSplitIndex;

    DataMapper(const DataSet<DataEntry>& source):
        dataSource(source), sourceToBlockIndex(dataSource.numberOfSamples), sourceToSplitIndex(dataSource.numberOfSamples) {};

    void operator()(size_t splittingIndex, std::span<const size_t> indicies){
        //[[unlikely]]if (indicies.size() == 0) return;
        splitToBlockNum[splittingIndex] = blockCounter;
        for (size_t i = 0; i<indicies.size(); i += 1){
            size_t index = indicies[i];
            sourceToBlockIndex[index] = BlockIndex(blockCounter, i);
            sourceToSplitIndex[index] = splittingIndex;
            blockIndexToSource[BlockIndex(blockCounter, i)] = index;
        }
        dataBlocks.push_back(DataBlock(dataSource, indicies, blockCounter++));
    };



};

using UnweightedGraphEdges = std::unordered_map<size_t, std::unordered_map<size_t, size_t>>;
using WeightedGraphEdges = std::unordered_map<size_t, std::vector<std::pair<size_t, double>>>;

WeightedGraphEdges NeighborsOutOfBlock(const DataSet<std::valarray<int32_t>>& groundTruth,
    const std::vector<BlockIndex>& trainClassifications,
    const std::vector<size_t>& testClassifications){
        UnweightedGraphEdges unweightedGraph;
        for(size_t i = 0; i<groundTruth.samples.size(); i += 1){
            size_t treeIndex = testClassifications[i];
            for(const auto& neighbor: groundTruth.samples[i]){
            //for (size_t j = 0; j<30; j +=1){
            //    int32_t neighbor = groundTruth.samples[i][j];
                (unweightedGraph[treeIndex])[trainClassifications[neighbor].blockNumber] += 1;
            }
        }

        WeightedGraphEdges retGraph;

        auto weightedEdgeCmp = [](std::pair<size_t, double> lhs, std::pair<size_t, double> rhs){
            return lhs.second > rhs.second;
        };

        for (const auto [originIndex, edges] : unweightedGraph){
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
    0008     8 byte int     firstEntry.first            splittingIndex of first leaf
    0016     8 byte int     firstEntry.second.size()    number of edges pointing away from leaf
    0024     8 byte int     edge.target                 splittingIndex of target node
    0032     8 byte double  edge.weight                 number of neighbors contained in target tree
    0040     8 byte int     edge.target                 splittingIndex of target node
    0048     8 byte double  edge.weight                 number of neighbors contained in target tree
    ........ 
    Each graph node is 16*(1 + size) bytes long, or 2*(1 + size) size_t's
    xxxx     8 byte int     secondEntry.first           splittingIndex of second leaf
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

template<TriviallyCopyable DataType>
void SerializeVector(const std::vector<DataType>& readVector, const std::string& outputFile){
    std::ofstream outStream(outputFile, std::ios_base::binary);
    SerializeData<size_t, std::endian::big>(outStream, readVector.size());

    for(const auto& entry : readVector){
        SerializeData<DataType, std::endian::big>(outStream, entry);
    }

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

void SerializeCOMS(const std::vector<MetaPoint>& COMs, const std::string& outputFile){
    std::ofstream outStream(outputFile, std::ios_base::binary);

    SerializeData<size_t, std::endian::big>(outStream, COMs.size());
    SerializeData<size_t, std::endian::big>(outStream, COMs.begin()->centerOfMass.size());
    
    for(const auto& point : COMs){
        SerializeData<size_t, std::endian::big>(outStream, point.weight);
        for(const auto& extent : point.centerOfMass){
            SerializeData<double, std::endian::big>(outStream, extent);
        }

    }

}

//DataBlock(const DataSet<DataEntry>& dataSource, std::span<size_t> dataPoints, SpaceMetric<DataEntry, FloatType> metric, size_t blockNumber):

int main(){

    static const std::endian dataEndianness = std::endian::big;

    //std::string trainDataFilePath("./TestData/train-images.idx3-ubyte");

    std::string testDataFilePath("./TestData/MNIST-Fashion-Data.bin");
    std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");
    std::string testNeighborsFilePath("./TestData/MNIST-Fashion-Neighbors.bin");
    

    //DataSet<std::valarray<unsigned char>> mnistDigitsTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<unsigned char,dataEndianness>);

    DataSet<std::valarray<float>> mnistFashionTest(testDataFilePath, 28*28, 10'000, &ExtractNumericArray<float,dataEndianness>);
    DataSet<std::valarray<float>> mnistFashionTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<float,dataEndianness>);
    DataSet<std::valarray<int32_t>> mnistFashionTestNeighbors(testNeighborsFilePath, 100, 10'000, &ExtractNumericArray<int32_t,dataEndianness>);
    

    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), mnistFashionTrain.numberOfSamples - 1);
    StlRngFunctor<std::mt19937_64, std::uniform_int_distribution, size_t> rngFunctor(std::move(rngEngine), std::move(rngDist));

    EuclidianTrain<float, double> splittingScheme(mnistFashionTrain);
    TrainingSplittingScheme splitterFunc(splittingScheme);
    //StlRngFunctor<> rngFunctor, SplittingScheme<FloatType> getSplitComponents, int splits = 8
    RandomProjectionForest rpTreesTrain(size_t(mnistFashionTrain.numberOfSamples), rngFunctor, splitterFunc);


    std::vector<size_t> trainClassifications(mnistFashionTrain.numberOfSamples);
    

    //CrawlTerminalLeaves(rpTreesTrain, classificationFunction);
    //auto trainResult = std::find(trainClassifications.begin(), trainClassifications.end(), 0);
    


    DataMapper<std::valarray<float>> trainMapper(mnistFashionTrain);
    CrawlTerminalLeaves(rpTreesTrain, trainMapper);



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

    //DataMapper<std::valarray<float>> testMapper(mnistFashionTest);
    CrawlTerminalLeaves(rpTreesTest, testClassificationFunction);

    std::vector<Graph<BlockIndex, double>> blockGraphs(0);
    blockGraphs.reserve(trainMapper.dataBlocks.size());
    for (const auto& dataBlock : trainMapper.dataBlocks){
        Graph<BlockIndex, double> blockGraph = ConstructInitialGraph<BlockIndex, double>(dataBlock.blockData.size(), size_t(5));
        BruteForceBlock<std::valarray<float>, double>(blockGraph, 5, dataBlock, EuclideanNorm<float, double>);
        blockGraphs.push_back(std::move(blockGraph));
    }
    
    MetaGraph metaGraph(trainMapper.dataBlocks);


    WeightedGraphEdges graphEdges = NeighborsOutOfBlock(mnistFashionTestNeighbors, trainMapper.sourceToBlockIndex, testClassifications);

    for (size_t i = 0; i < trainMapper.sourceToBlockIndex.size(); i += 1){
        trainClassifications[i] = trainMapper.sourceToBlockIndex[i].blockNumber;
    }

    //SerializeCOMS(metaGraph.points, "./TestData/MNIST-Fashion-Train-COMs.bin");
    //SerializeMetaGraph(graphEdges, "./TestData/MNIST-Fashion-Test-MetaGraphEdges.bin");
    //SerializeVector<size_t>(trainClassifications, "./TestData/MNIST-Fashion-Train-SplittingIndicies.bin");
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