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
#include "NND/DataStructures.hpp"
#include "NND/Algorithm.hpp"


namespace chrono = std::chrono;
using namespace nnd;
template<typename DataType, typename FloatType>
std::valarray<FloatType> EuclidianSplittingPlaneNormal(const std::valarray<DataType>& pointA, const std::valarray<DataType>& pointB){
    std::valarray<FloatType> splittingLine(pointA.size());
    for (size_t i = 0; i < pointA.size(); i += 1){
        splittingLine[i] = FloatType(pointA[i]) - FloatType(pointB[i]);
    }
    FloatType splittingLineMag(0);
    for (FloatType i : splittingLine){
        splittingLineMag += i*i;
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
            *toBegin = *fromBegin;
            toBegin++;
            numTrue++;
        } else {
            *toRev = *fromBegin;
            toRev++;
        }
    }
    return numTrue;
}

/*
template<typename DataType, typename FloatType>
std::pair<std::valarray<FloatType>, FloatType> GetSplittingVector(const MNISTData& digits, std::function<size_t ()>& rngFunctor){
    size_t index1(rngFunctor()), index2(rngFunctor());
    while (index2 == index1){
        index2 = rngFunctor();
    }
    index1 = indexVector1[index1];
    index2 = indexVector1[index2];

    std::valarray<FloatType> splittingVector = EuclidianSplittingPlaneNormal(digits.samples[index1], digits.samples[index2]);
    
    std::valarray<FloatType> offsets = splittingVector * (digits.samples[index1] - digits.samples[index2])/2.0;
    FloatType offset = 0;
    for (FloatType component : offsets){
        offset -= component;
    }

    return std::make_pair(splittingVector, offset);
}
*/

/*
    Oh boy, oh boy; this type allias is a mouthful.
    Short version: A Splitting Scheme is a function/functor.
    It can take
        size_t: the splitting index, first split being 0
        pair<size_t, size_t>: two valid points to contruct a splitting plane if one needs to be built.

    It returns(as a pair)
        A function: bool f(size_t dataIndex)
*/
//Gah, curse the indirection, but I'll use std::function for now here
using SplittingScheme = std::function<std::function<bool(size_t)> (size_t, std::pair<size_t, size_t>)>;


template<typename FloatType, typename DataType>
struct EuclidianSplittingScheme{

    const std::vector<std::valarray<DataType>>& dataSource;
    std::unordered_map<size_t, std::pair<std::valarray<FloatType>, FloatType>> splittingVectors;
    FloatType projectionOffset;



    EuclidianSplittingScheme(const MNISTData& digits) : dataSource(digits.samples), splittingVectors(){};

    std::function<bool(size_t)> operator()(size_t splitIndex, std::pair<size_t, size_t> splittingPoints){
        
        std::valarray<FloatType> splittingVector;

        if (splittingVectors.find(splitIndex) == splittingVectors.end()){
            splittingVector = EuclidianSplittingPlaneNormal<DataType, FloatType>(dataSource[splittingPoints.first], dataSource[splittingPoints.second]);

            //std::valarray<FloatType> pointAverage(splittingVector.size());

            FloatType projectionOffset = 0;
            for (size_t i = 0; i<dataSource[splittingPoints.first].size(); i+=1){
                projectionOffset -= splittingVector[i] * FloatType(dataSource[splittingPoints.first][i] + dataSource[splittingPoints.second][i])/2.0;
            };
            /*
            std::valarray<FloatType> offsets = splittingVector * (dataSource[splittingPoints.first] + dataSource[splittingPoints.second])/2.0;
            projectionOffset = 0;
            for (FloatType component : offsets){
                projectionOffset -= component;
            };
            */
           splittingVectors[splitIndex] = std::pair<std::valarray<FloatType>, FloatType>(splittingVector, projectionOffset);

        };
        
        
        auto comparisonFunction = [=, 
                                   &data = std::as_const(this->dataSource), 
                                   &splitter = splittingVectors[splitIndex].first,
                                   offset = splittingVectors[splitIndex].second]
                                   (size_t comparisonIndex) -> bool{
                // This is some janky type conversion shenanigans here. I need to either
                // remove the assertion of arbitrary (but single) data type and turn everything into a float/double
                // or define these ops in terms of something other than valarrays.
                // I just wanna get prototyping rptrees done.
                // TODO: KILL THIS IMPLEMENTATION (jfc, this is NOT GOOD)
                std::valarray<FloatType> temporaryArr(data[comparisonIndex].size());
                for(size_t i = 0; i < temporaryArr.size(); i += 1){
                    temporaryArr[i] = FloatType(data[comparisonIndex][i]);
                }

                FloatType distanceFromPlane = (Dot(temporaryArr, splitter) + offset);
                //std::cout << distanceFromPlane << std::endl;
                bool result = 0.0 < distanceFromPlane;

                return result;

        };
        return std::function<bool(size_t)> (comparisonFunction);
    };
    /*
    bool operator()(size_t comparisonIndex){
        return 0 < (Dot(dataSource[comparisonIndex], splittingVector) - offset)
    }
    */
    //private:
    //std::valarray<FloatType> 
};




//TODO: add some way to track best splits
struct RandomProjectionForest{

    std::vector<size_t> indexArray;
    int numberOfSplits;
    //std::vector<std::valarray<FloatType>> splittingVectors;
    //The second value is the first element past the range.
    std::vector<std::pair<size_t, size_t>> splitRanges;

    struct TreeLeaf{
    
        //This is for the split that produced the children
        size_t childrenSplittingIndex;
        std::pair<size_t,size_t> splitRange;
        //Pretty sure the Leaves can be refs. Either way, non of these pointers own the underlying objects
        //All of the leaves of the tree are owned by the enclosing Vector.
        //Wait. Using pointers like this means the tree can't easily be copied at all...
        //I don't even know if this needs to be copied
        std::pair<TreeLeaf*, TreeLeaf*> children;
        TreeLeaf* parent;
        std::vector<TreeLeaf>* enclosingVector;
        /*
        This structure probably can be condensed back down to just a tuple of {bool, size_t, size_t}
        (or two arrays, bool[] and std::pair/std::range[]).
        That'd remove 4 pointers (4/7s) of the struct at this point, add a literal bit, and require
        alocating enough memory for the entire theoretical max tree.
        That assumption may not be a good one in the parallel case (although I am already reserving an agressive amount of space.)
        I'll leave as is for now in case this makes some other piece easier to work on
        
        */
        TreeLeaf() : splitRange(0,0), childrenSplittingIndex(-1), children(nullptr, nullptr), parent(nullptr), enclosingVector(nullptr){};

        TreeLeaf(size_t index1, size_t index2, size_t splittingIndex, std::vector<TreeLeaf>* enclose) : splitRange(index1, index2),
                                                                                childrenSplittingIndex(splittingIndex), 
                                                                                children(nullptr, nullptr),
                                                                                parent(nullptr), 
                                                                                enclosingVector(enclose){};

        TreeLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex, std::vector<TreeLeaf>* enclose) : splitRange(indecies),
                                                                                    childrenSplittingIndex(splittingIndex), 
                                                                                    children(nullptr, nullptr),
                                                                                    parent(nullptr), 
                                                                                    enclosingVector(enclose){};

        TreeLeaf& AddLeftLeaf(size_t index1, size_t index2, size_t splittingIndex){
            TreeLeaf& newLeaf = enclosingVector->emplace_back(index1, index2, splittingIndex, enclosingVector);
            newLeaf.parent = this;
            this->children.first = &newLeaf;
            return newLeaf;
        };

        TreeLeaf& AddLeftLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex){
            TreeLeaf& newLeaf = enclosingVector->emplace_back(indecies, splittingIndex, enclosingVector);
            newLeaf.parent = this;
            this->children.first = &newLeaf;
            return newLeaf;
        };

        TreeLeaf& AddRightLeaf(size_t index1, size_t index2, size_t splittingIndex){
            TreeLeaf& newLeaf = enclosingVector->emplace_back(index1, index2, splittingIndex, enclosingVector);
            newLeaf.parent = this;
            this->children.second = &newLeaf;
            return newLeaf;
        };

        TreeLeaf& AddRightLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex){
            TreeLeaf& newLeaf = enclosingVector->emplace_back(indecies, splittingIndex, enclosingVector);
            newLeaf.parent = this;
            this->children.second = &newLeaf;
            return newLeaf;
        };

    };
    
    std::vector<TreeLeaf> treeLeaves;

    //template<typename DataType>
    RandomProjectionForest(size_t numberOfSamples, StlRngFunctor<> rngFunctor, SplittingScheme getSplitComponents, int splits = 8, int splitThreshold = 255) : 
        splitRanges(1, std::pair<size_t, size_t>(0, numberOfSamples)), numberOfSplits(splits), treeLeaves(0){

        
        splitRanges.reserve(1<<(numberOfSplits+1));
        //splittingVectors.reserve((1<<numberOfSplits) - 1);
        treeLeaves.reserve(1<<(numberOfSplits+1));
        treeLeaves.emplace_back(std::pair<size_t, size_t>(0, numberOfSamples), 0, &treeLeaves);

        std::vector<TreeLeaf*> splitQueue1(1, &treeLeaves[0]);
        std::vector<TreeLeaf*> splitQueue2(0);

        std::vector<size_t> indexVector1(numberOfSamples);
        std::iota(indexVector1.begin(), indexVector1.end(), 0);
        std::vector<size_t> indexVector2(numberOfSamples);

        // splittingIndex is for the case of 
        //size_t splittingIndex(0);
        size_t rangeIndexOffset(0);
        std::function<bool(size_t)> splittingFunction;

        //size_t beginIndex(0), endIndex(numberOfSamples - 1);

        for (size_t i = 0; i<numberOfSplits; i+=1){
            
            while(splitQueue1.size() > 0){
                
                TreeLeaf& currentSplit = *(splitQueue1.back());

                //std::pair<size_t, size_t> rangeIndecies = splitRanges[rangeIndexOffset + j];

                //The 1 and/or 0 member split case
                /*
                if (currentSplit.splitRange.second - currentSplit.splitRange.first <= splitThreshold){
                    splitQueue1.pop_back();
                    continue;
                }
                */
                //This is bootleg af, need to refactor how I do rng.
                decltype(rngFunctor.functorDistribution)::param_type newRange(currentSplit.splitRange.first, currentSplit.splitRange.second - 1);
                rngFunctor.functorDistribution.param(newRange);

                size_t index1(rngFunctor());
                size_t index2(rngFunctor());
                while (index2 == index1){
                    index2 = rngFunctor();
                }
                index1 = indexVector1[index1];
                index2 = indexVector1[index2];

                // Get the splitting vector, this can be fed into this function in the parallel/distributed case.
                //std::valarray<FloatType> splittingVector;
                splittingFunction = getSplitComponents(currentSplit.childrenSplittingIndex, std::pair<size_t, size_t>(index1, index2));
                //splittingVectors.push_back(splittingVector);

                auto beginIt = indexVector1.begin();
                std::advance(beginIt, currentSplit.splitRange.first);
                auto endIt = indexVector1.end();
                std::advance(endIt, currentSplit.splitRange.second - numberOfSamples);
                auto toBegin = indexVector2.begin();
                std::advance(toBegin, currentSplit.splitRange.first);
                auto toRev = indexVector2.rbegin();
                std::advance(toRev, numberOfSamples - currentSplit.splitRange.second);

                int numSplit = Split(beginIt, endIt, toBegin, toRev, splittingFunction);
                //splitRanges.push_back(std::pair<size_t, size_t>(currentSplit.splitRange.first, currentSplit.splitRange.first + numSplit));
                //splitRanges.push_back(std::pair<size_t, size_t>(currentSplit.splitRange.first + numSplit, rangeIndecies.second));

                TreeLeaf* leftSplit = &(currentSplit.AddLeftLeaf(std::pair<size_t, size_t>(currentSplit.splitRange.first, currentSplit.splitRange.first + numSplit),
                                                                 currentSplit.childrenSplittingIndex * 2 + 1));
                TreeLeaf* rightSplit = &(currentSplit.AddRightLeaf(std::pair<size_t, size_t>(currentSplit.splitRange.first + numSplit, currentSplit.splitRange.second),
                                                                   currentSplit.childrenSplittingIndex * 2 + 2));

                if (leftSplit->splitRange.second - leftSplit->splitRange.first > splitThreshold) splitQueue2.push_back(leftSplit);
                if (rightSplit->splitRange.second - rightSplit->splitRange.first > splitThreshold) splitQueue2.push_back(rightSplit);
                splitQueue1.pop_back();

                //splittingIndex++;
            }

            //rangeIndexOffset += 1<<i;
            std::swap(indexVector1, indexVector2);
            std::swap(splitQueue1, splitQueue2);
        }

        indexArray = std::move(indexVector1);

    }
    

};




/*
// TODO: Add in some way to track the best splits
template<typename DataType, typename FloatType>
void EuclidianBuildRPTrees(const MNISTData& digits, StlRngFunctor<> rngFunctor, int numberOfSplits = 8){
    std::vector<size_t> indexVector1(digits.numberOfSamples);
    std::iota(indexVector1.begin(), indexVector1.end(), 0);
    std::vector<size_t> indexVector2(digits.numberOfSamples);

    RandomProjectionForest<FloatType> forest;
    
    std::vector<std::valarray<FloatType>> normalVectors(0);
    normalVectors.reserve((1<<numberOfSplits) - 1);

    FloatType offset(0);
    std::valarray<FloatType> splittingVector;
    auto splitter = [&](size_t index){
        return 0 < (Dot(digits.samples[index], splittingVector) - offset)
    };

    //The second value is the first element past the range.
    std::vector<std::pair<size_t, size_t>> splitRanges(0);
    splitRanges.reserve(1<<(numberOfSplits+1));
    splitRanges.push_back(std::pair(0, digits.numberOfSamples));

    size_t rangeIndexOffset(0);

    size_t beginIndex(0), endIndex(digits.numberOfSamples - 1);

    for (size_t i = 0; i<numberOfSplits; i+=1){
        
        for (size_t j = 0; j < (1<<i); j += 1){
            
            std::pair<size_t, size_t> = splitRanges[rangeIndexOffset + j];

            // Get the splitting vector, this can be fed into this function in the parallel/distributed case.
            rngFunctor.functorDistribution.param(splitRanges.first, splitRanges.second - 1);
            std::tie(splittingVector, offset) = GetSplittingVector(digits, rngFunctor);

            int numSplit = Split(indexVector1.begin(), indexVector1.end(), indexVector2.begin(), indexVector2.rbegin(), splitter);
            splitRanges.push_back(std::pair(splitRanges.first, splitRanges.first + numSplit));
            splitRanges.push_back(std::pair(splitRanges.first + numSplit, splitRanges.second));
        }
        rangeIndexOffset += 1<<i;
        std::swap(indexVector1, indexVector2);
    }



}
*/
int main(){

    MNISTData digits(std::string("./TestData/train-images.idx3-ubyte"));
    
    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), digits.numberOfSamples - 1);
    StlRngFunctor<std::mt19937_64, std::uniform_int_distribution, size_t> rngFunctor(rngEngine, rngDist);

    EuclidianSplittingScheme<double, unsigned char> splittingScheme(digits);

    //StlRngFunctor<> rngFunctor, SplittingScheme<FloatType> getSplitComponents, int splits = 8
    RandomProjectionForest rpTrees(size_t(digits.numberOfSamples), rngFunctor, SplittingScheme(splittingScheme));

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