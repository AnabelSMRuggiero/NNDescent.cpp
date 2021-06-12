//This is primarily for testing an debugging

#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <valarray>
#include <numeric>
#include <cmath>

#include "MNISTData.hpp"
#include "NND/SpaceMetrics.hpp"
#include "NND/DataStructures.hpp"
#include "NND/Algorithm.hpp"


namespace chrono = std::chrono;
using namespace nnd;
template<typename DataType, typename FloatType>
std::valarray<FloatType> EuclidianSplittingPlaneNormal(const std::valarray<DataType>& pointA, const std::valarray<DataType>& pointB){
    std::valarray<FloatType> splittingLine = pointB - pointA;
    FloatType splittingLineMag(0);
    for (FloatType i : splittingLine){
        splittingLineMag += 1;
    }
    splittingLineMag = std::sqrt(splittingLineMag);
    splittingLine /= splittingLineMag;

    return splittingLine;
}

template<typename Iterator, typename rIterator, typename SplittingFunction>
int Split(Iterator fromBegin, Iterator fromEnd, Iterator toBegin, rIterator toRev, SplittingFunction splitter){
    int numTrue = 0;
    for ( ; fromBegin != fromEnd; fromBegin++){
        if (splitter(*fromBegin)){
            *toRev = *fromBegin;
            toRev++;
            numTrue++;
        } else {
            *toBegin = *fromBegin;
            toBegin++;
        }
    }
    return numTrue;
}

template<typename DataType, typename FloatType>
void EuclidianBuildRPTrees(const MNISTData& digits, std::function<size_t ()> rngFunctor, int numberOfSplits = 8){
    std::vector<size_t> indexVector1(digits.numberOfSamples);
    std::iota(indexVector1.begin(), indexVector1.end(), 0);
    std::vector<size_t> indexVector2(digits.numberOfSamples);
    std::vector<std::valarray<FloatType>> normalVectors((1<<numberOfSplits) - 1);

    auto splitter = [&](size_t index){
        
    };



}

int main(){

    MNISTData digits(std::string("./TestData/train-images.idx3-ubyte"));
    
    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), digits.numberOfSamples - 1);
    StlRngFunctor<std::mt19937_64, std::uniform_int_distribution, size_t> rngFunctor(rngEngine, rngDist);


    //SpaceMetric<std::valarray<unsigned char>> distFunc = &EuclideanNorm<unsigned char>
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
    

    // compQueues(0);

    return 0;
}