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

#include "MNISTData.hpp"
#include "NND/SpaceMetrics.hpp"
#include "NND/GraphStructures.hpp"
#include "NND/Algorithm.hpp"

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"



//namespace chrono = std::chrono;
using namespace nnd;






int main(){

    MNISTData digits(std::string("./TestData/train-images.idx3-ubyte"));
    
    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), digits.numberOfSamples - 1);
    StlRngFunctor<std::mt19937_64, std::uniform_int_distribution, size_t> rngFunctor(rngEngine, rngDist);

    EuclidianSplittingScheme<double, unsigned char> splittingScheme(digits);

    //StlRngFunctor<> rngFunctor, SplittingScheme<FloatType> getSplitComponents, int splits = 8
    RandomProjectionForest rpTrees(size_t(digits.numberOfSamples), rngFunctor, SplittingScheme(splittingScheme));

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