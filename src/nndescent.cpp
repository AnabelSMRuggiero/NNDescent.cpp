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

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"

#include "Utilities/DataSerialization.hpp"
#include "Utilities/DataDeserialization.hpp"


//namespace chrono = std::chrono;
using namespace nnd;


template<typename Functor>
void CrawlTerminalLeaves(const RandomProjectionForest& forest, Functor terminalFunctor){

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

template<typename DataEntry, typename FloatType>
auto GenerateDataBlockConstructor(const DataSet<DataEntry>& dataSource, 
                                  SpaceMetric<DataEntry, FloatType> metric,
                                  std::vector<DataBlock<DataEntry, FloatType>>& accumulationVector){

    size_t blockCounter(0);
    auto retLambda = [&, metric](size_t ignored, std::span<const size_t> indicies){
        accumulationVector.push_back(DataBlock(dataSource, indicies, metric, ++blockCounter));
    };
    return retLambda;
}

//DataBlock(const DataSet<DataEntry>& dataSource, std::span<size_t> dataPoints, SpaceMetric<DataEntry, FloatType> metric, size_t blockNumber):

int main(){

    

    std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");

    //MNISTData digits(digitsFilePath);
    static const std::endian dataEndianness = std::endian::big;
    
    DataSet<std::valarray<float>> mnistDigitsTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<float,dataEndianness>);
    

    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), mnistDigitsTrain.numberOfSamples - 1);
    StlRngFunctor<std::mt19937_64, std::uniform_int_distribution, size_t> rngFunctor(std::move(rngEngine), std::move(rngDist));

    EuclidianSplittingScheme<float, double> splittingScheme(mnistDigitsTrain);
    SplittingScheme splitterFunc(splittingScheme);
    //StlRngFunctor<> rngFunctor, SplittingScheme<FloatType> getSplitComponents, int splits = 8
    RandomProjectionForest rpTreesTrain(size_t(mnistDigitsTrain.numberOfSamples), rngFunctor, splitterFunc);

    std::vector<size_t> trainClassifications(mnistDigitsTrain.numberOfSamples);

    auto classificationFunction = [&trainClassifications](size_t splittingIndex, std::span<const size_t> indicies){
        for (const auto& index : indicies){
            trainClassifications[index] = splittingIndex;
        }
    };

    CrawlTerminalLeaves(rpTreesTrain, classificationFunction);
    auto trainResult = std::find(trainClassifications.begin(), trainClassifications.end(), 0);
    std::vector<DataBlock<std::valarray<float>, double>> dataBlocks;

    CrawlTerminalLeaves(rpTreesTrain, GenerateDataBlockConstructor<std::valarray<float>, double>(mnistDigitsTrain, EuclideanNorm<float, double>, dataBlocks));
    

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