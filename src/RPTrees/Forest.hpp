/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef RPT_FOREST_HPP
#define RPT_FOREST_HPP

#include <utility>
#include <vector>
#include <functional>
#include <numeric>
#include <exception>
#include <unordered_set>
#include <cstddef>
#include <iostream>
#include <span>

#include "../Utilities/Type.hpp"

#include "../NND/RNG.hpp"
#include "SplittingScheme.hpp"

namespace nnd{

/*
This seems out of place here. The utility functions header under NND is mainly for functions
related to NND, and didn't want to tuck this there. This is a super general function though.
*/


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



SplittingHeurisitcs test = {8, 64, 8};

//TODO: add some way to track best splits
struct RandomProjectionForest{

    std::vector<size_t> indexArray;
    int numberOfSplits;


    struct TreeLeaf{
    
        //This is for the split that produced the children
        size_t splittingIndex;
        std::pair<size_t,size_t> splitRange;
        //Pretty sure the Leaves can be refs. Either way, non of these pointers own the underlying objects
        //All of the leaves of the tree are owned by the enclosing Vector.
        //This needs to be copied because std::vector moves when you don't expect it to.
        std::pair<size_t, size_t> children;
        size_t parent;
        size_t thisIndex;
        //std::vector<TreeLeaf>* enclosingVector;
        /*
        This structure probably can be condensed back down to just a tuple of {bool, size_t, size_t}
        (or two arrays, bool[] and std::pair/std::range[]).
        That'd remove 4 pointers (4/7s) of the struct at this point, add a literal bit, and require
        alocating enough memory for the entire theoretical max tree.
        That assumption may not be a good one in the parallel case (although I am already reserving an agressive amount of space.)
        I'll leave as is for now in case this makes some other piece easier to work on
        
        */
        TreeLeaf() : splitRange(0,0), splittingIndex(-1), children(0, 0), parent(0){};

        TreeLeaf(size_t index1, size_t index2, size_t splittingIndex, size_t thisIndex) : splitRange(index1, index2),
                                                                                splittingIndex(splittingIndex), 
                                                                                children(0,0),
                                                                                parent(0), 
                                                                                thisIndex(thisIndex){};

        TreeLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex, size_t thisIndex) : splitRange(indecies),
                                                                                    splittingIndex(splittingIndex), 
                                                                                    children(0,0),
                                                                                    parent(0), 
                                                                                    thisIndex(thisIndex){};

        TreeLeaf AddLeftLeaf(size_t index1, size_t index2, size_t splittingIndex, std::vector<TreeLeaf>& enclosingVector){
            this->children.first = enclosingVector.size();
            size_t indexToWrite = thisIndex;
            TreeLeaf& newLeaf = enclosingVector.emplace_back(index1, index2, splittingIndex, enclosingVector.size());
            newLeaf.parent = indexToWrite;
            
            return newLeaf;
        };

        TreeLeaf AddLeftLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex, std::vector<TreeLeaf>& enclosingVector){
            this->children.first = enclosingVector.size();
            size_t indexToWrite = thisIndex;
            TreeLeaf& newLeaf = enclosingVector.emplace_back(indecies, splittingIndex, enclosingVector.size());
            newLeaf.parent = indexToWrite;

            return newLeaf;
        };

        TreeLeaf AddRightLeaf(size_t index1, size_t index2, size_t splittingIndex, std::ptrdiff_t parentDiff, std::vector<TreeLeaf>& enclosingVector){
            this->children.second = enclosingVector.size();
            size_t indexToWrite = thisIndex;
            TreeLeaf& newLeaf = enclosingVector.emplace_back(index1, index2, splittingIndex, enclosingVector.size());
            newLeaf.parent = indexToWrite;

            return newLeaf;
        };

        TreeLeaf AddRightLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex, std::vector<TreeLeaf>& enclosingVector){
            this->children.second = enclosingVector.size();
            size_t indexToWrite = thisIndex;
            TreeLeaf& newLeaf = enclosingVector.emplace_back(indecies, splittingIndex, enclosingVector.size());
            newLeaf.parent = indexToWrite;
            return newLeaf;
        };

    };
    
    std::vector<TreeLeaf> treeLeaves;

    // Training Constructor
    RandomProjectionForest(size_t numberOfSamples, StlRngFunctor<> rngFunctor, TrainingSplittingScheme& getSplitComponents, const SplittingHeurisitcs heurisitics = SplittingHeurisitcs()) : 
        numberOfSplits(heurisitics.splits), treeLeaves(0){

        //splittingVectors.reserve((1<<numberOfSplits) - 1);
        treeLeaves.reserve(std::min(size_t(1<<(numberOfSplits+1)), 2*numberOfSamples/size_t(heurisitics.splitThreshold)));
        treeLeaves.emplace_back(std::pair<size_t, size_t>(0, numberOfSamples), 0, 0);

        std::vector<size_t> splitQueue1(1, 0);
        std::vector<size_t> splitQueue2(0);

        std::vector<size_t> indexVector1(numberOfSamples);
        std::iota(indexVector1.begin(), indexVector1.end(), 0);

        
        size_t sum = std::accumulate(indexVector1.begin(), indexVector1.end(), 0);
        size_t tmpSum(sum);
        

        std::vector<size_t> indexVector2(numberOfSamples);

        std::function<bool(size_t)> splittingFunction;

        //size_t beginIndex(0), endIndex(numberOfSamples - 1);

        while (splitQueue1.size() > 0){
            
            while(splitQueue1.size() > 0){
                
                size_t currentIndex = splitQueue1.back();

                retry:
                //This is bootleg af, need to refactor how I do rng.
                decltype(rngFunctor.functorDistribution)::param_type newRange(treeLeaves[currentIndex].splitRange.first, treeLeaves[currentIndex].splitRange.second - 1);
                rngFunctor.functorDistribution.param(newRange);

                size_t index1(rngFunctor());
                size_t index2(rngFunctor());
                while (index2 == index1){
                    index2 = rngFunctor();
                }
                
                index1 = indexVector1[index1];
                index2 = indexVector1[index2];

                // Get the splitting vector, this can be fed into this function in the parallel/distributed case.
                splittingFunction = getSplitComponents(treeLeaves[currentIndex].splittingIndex, std::pair<size_t, size_t>(index1, index2));


                auto beginIt = indexVector1.begin();
                std::advance(beginIt, treeLeaves[currentIndex].splitRange.first);
                auto endIt = indexVector1.end();
                std::advance(endIt, treeLeaves[currentIndex].splitRange.second - numberOfSamples);
                auto toBegin = indexVector2.begin();
                std::advance(toBegin, treeLeaves[currentIndex].splitRange.first);
                auto toRev = indexVector2.rbegin();
                std::advance(toRev, numberOfSamples - treeLeaves[currentIndex].splitRange.second);

                int numSplit = Split(beginIt, endIt, toBegin, toRev, splittingFunction);

                if (numSplit>heurisitics.childThreshold &&
                    (treeLeaves[currentIndex].splitRange.second - treeLeaves[currentIndex].splitRange.first - numSplit)>heurisitics.childThreshold){

                    TreeLeaf leftSplit = treeLeaves[currentIndex].AddLeftLeaf(std::pair<size_t, size_t>(treeLeaves[currentIndex].splitRange.first, treeLeaves[currentIndex].splitRange.first + numSplit),
                                                                    treeLeaves[currentIndex].splittingIndex * 2 + 1, treeLeaves);
                    

                    if (leftSplit.splitRange.second - leftSplit.splitRange.first > heurisitics.splitThreshold) splitQueue2.push_back(leftSplit.thisIndex);
                    else{
                        //copy section of vec2 back to vec 1;
                        /*
                        auto fromIt = indexVector2.begin();
                        std::advance(fromIt, leftSplit->splitRange.first);
                        auto endFrom = indexVector2.end();
                        std::advance(endFrom, leftSplit->splitRange.second - numberOfSamples);

                        auto toIt = indexVector1.begin();
                        std::advance(toIt, leftSplit->splitRange.first);
                        */
                        auto fromIt = indexVector2.begin()+leftSplit.splitRange.first;
                        auto endFrom = indexVector2.begin()+leftSplit.splitRange.second;
                        auto toIt = indexVector1.begin()+leftSplit.splitRange.first;

                        std::copy(fromIt, endFrom, toIt);
                    }

                    TreeLeaf rightSplit = treeLeaves[currentIndex].AddRightLeaf(std::pair<size_t, size_t>(treeLeaves[currentIndex].splitRange.first + numSplit, treeLeaves[currentIndex].splitRange.second),
                        treeLeaves[currentIndex].splittingIndex * 2 + 2, treeLeaves);

                    if (rightSplit.splitRange.second - rightSplit.splitRange.first > heurisitics.splitThreshold) splitQueue2.push_back(rightSplit.thisIndex);
                    else{
                        //copy section of vec2 back to vec 1;
                        /*
                        auto fromIt = indexVector2.begin();
                        std::advance(fromIt, rightSplit->splitRange.first);
                        auto endFrom = indexVector2.end();
                        std::advance(endFrom, rightSplit->splitRange.second - numberOfSamples);

                        auto toIt = indexVector1.begin();
                        std::advance(toIt, rightSplit->splitRange.first);
                        */

                        auto fromIt = indexVector2.begin() + rightSplit.splitRange.first;
                        auto endFrom = indexVector2.begin() + rightSplit.splitRange.second;
                        auto toIt = indexVector1.begin() + rightSplit.splitRange.first;


                        std::copy(fromIt, endFrom, toIt);
                    }
                } else if ((treeLeaves[currentIndex].splitRange.second - treeLeaves[currentIndex].splitRange.first) > heurisitics.maxTreeSize){
                    // I know, gross. This is a (hopefully) a place holder for a more robust way to handle rejected splits.
                    goto retry;
                }else{
                    // Undo the attempted split. This may be unneeded, but I want to be safe for now.
                    // TODO: Check to see if omitting this copy violates the invariance of sum
                    auto fromIt = indexVector1.begin() + treeLeaves[currentIndex].splitRange.first;
                    auto endFrom = indexVector1.begin() + treeLeaves[currentIndex].splitRange.second;
                    auto toIt = indexVector2.begin() + treeLeaves[currentIndex].splitRange.first;


                    //std::copy(fromIt, endFrom, toIt);
                }
                splitQueue1.pop_back();

            } //end while
            
            std::swap(indexVector1, indexVector2);
            std::swap(splitQueue1, splitQueue2);
            
            tmpSum = std::accumulate(indexVector1.begin(), indexVector1.end(), 0);
            if (sum != tmpSum){
                throw std::logic_error("Sum of indicies should be invariant.");
            };
            

        } //end for
        indexArray = std::move(indexVector1);
    } //end constructor
    
    RandomProjectionForest(size_t numberOfSamples, TransformingSplittingScheme& getSplitComponents, std::unordered_set<size_t> splitsToDo);

};

//Transforming Constructor
RandomProjectionForest::RandomProjectionForest(size_t numberOfSamples, TransformingSplittingScheme& getSplitComponents, std::unordered_set<size_t> splitIndicies) : 
    numberOfSplits(8), treeLeaves(0){

        //splittingVectors.reserve((1<<numberOfSplits) - 1);
        treeLeaves.reserve(1<<(numberOfSplits+1));
        treeLeaves.emplace_back(std::pair<size_t, size_t>(0, numberOfSamples), 0, 0);

        std::vector<size_t> splitQueue1(1, 0);
        std::vector<size_t> splitQueue2(0);

        std::vector<size_t> indexVector1(numberOfSamples);
        std::iota(indexVector1.begin(), indexVector1.end(), 0);
        size_t sum = std::accumulate(indexVector1.begin(), indexVector1.end(), 0);
        size_t tmpSum(sum);
        std::vector<size_t> indexVector2(numberOfSamples);

        std::function<bool(size_t)> splittingFunction;

        //size_t beginIndex(0), endIndex(numberOfSamples - 1);

        while(splitQueue1.size() > 0){
            
            while(splitQueue1.size() > 0){
                
                size_t currentIndex = splitQueue1.back();

                
                // Get the splitting vector, this can be fed into this function in the parallel/distributed case.
                splittingFunction = getSplitComponents(treeLeaves[currentIndex].splittingIndex);


                auto beginIt = indexVector1.begin();
                std::advance(beginIt, treeLeaves[currentIndex].splitRange.first);
                auto endIt = indexVector1.end();
                std::advance(endIt, treeLeaves[currentIndex].splitRange.second - numberOfSamples);
                auto toBegin = indexVector2.begin();
                std::advance(toBegin, treeLeaves[currentIndex].splitRange.first);
                auto toRev = indexVector2.rbegin();
                std::advance(toRev, numberOfSamples - treeLeaves[currentIndex].splitRange.second);

                int numSplit = Split(beginIt, endIt, toBegin, toRev, splittingFunction);


                TreeLeaf leftSplit = treeLeaves[currentIndex].AddLeftLeaf(std::pair<size_t, size_t>(treeLeaves[currentIndex].splitRange.first, treeLeaves[currentIndex].splitRange.first + numSplit),
                                                                 treeLeaves[currentIndex].splittingIndex * 2 + 1, treeLeaves);
                TreeLeaf rightSplit = treeLeaves[currentIndex].AddRightLeaf(std::pair<size_t, size_t>(treeLeaves[currentIndex].splitRange.first + numSplit, treeLeaves[currentIndex].splitRange.second),
                                                                   treeLeaves[currentIndex].splittingIndex * 2 + 2, treeLeaves);
                auto result = std::find(splitIndicies.begin(),splitIndicies.end(), leftSplit.splittingIndex);
                if ((result != splitIndicies.end()) && (leftSplit.splitRange.second - leftSplit.splitRange.first > 1)) splitQueue2.push_back(leftSplit.thisIndex);
                else {
                    //copy section of vec2 back to vec 1;
                    /*
                    auto fromIt = indexVector2.begin();
                    std::advance(fromIt, leftSplit->splitRange.first);
                    auto endFrom = indexVector2.end();
                    std::advance(endFrom, leftSplit->splitRange.second - numberOfSamples);

                    auto toIt = indexVector1.begin();
                    std::advance(toIt, leftSplit->splitRange.first);
                    */
                    auto fromIt = &(indexVector2[leftSplit.splitRange.first]);
                    auto endFrom = &(indexVector2[leftSplit.splitRange.second]);
                    auto toIt = &(indexVector1[leftSplit.splitRange.first]);

                    std::copy(fromIt, endFrom, toIt);
                }

                result = std::find(splitIndicies.begin(),splitIndicies.end(), rightSplit.splittingIndex);
                if ((result != splitIndicies.end())&&(rightSplit.splitRange.second - rightSplit.splitRange.first > 1)) splitQueue2.push_back(rightSplit.thisIndex);
                else{
                    //copy section of vec2 back to vec 1;
                    /*
                    auto fromIt = indexVector2.begin();
                    std::advance(fromIt, rightSplit->splitRange.first);
                    auto endFrom = indexVector2.end();
                    std::advance(endFrom, rightSplit->splitRange.second - numberOfSamples);

                    auto toIt = indexVector1.begin();
                    std::advance(toIt, rightSplit->splitRange.first);
                    */

                    auto fromIt = &(indexVector2[rightSplit.splitRange.first]);
                    auto endFrom = &(indexVector2[rightSplit.splitRange.second]);
                    auto toIt = &(indexVector1[rightSplit.splitRange.first]);


                    std::copy(fromIt, endFrom, toIt);
                }
                splitQueue1.pop_back();

            } //end while
            
            std::swap(indexVector1, indexVector2);
            
            tmpSum = std::accumulate(indexVector1.begin(), indexVector1.end(), 0);
            if (sum != tmpSum){
                throw std::logic_error("Sum of indicies should be invariant.");
            };
            //if (splitQueue2.size() == 0) break;
            std::swap(splitQueue1, splitQueue2);
        } //end while
        indexArray = std::move(indexVector1);
} //end constructor
    

template<typename Functor>
void CrawlTerminalLeaves(const RandomProjectionForest& forest, Functor& terminalFunctor){

    //std::vector<size_t> classifications(forest.indexArray.size());

    std::vector<size_t> treePath;
    std::vector<char> pathState;
    treePath.push_back(0);
    pathState.push_back(0);

    size_t highestIndex = 0;
    size_t counter = 0;
    size_t currentIndex = 0;

    //Fix this
    auto leafAccesor = [&] (size_t index) -> const RandomProjectionForest::TreeLeaf{
        return *(forest.treeLeaves.begin()+index);
    };
    //const RandomProjectionForest::TreeLeaf* currentLeaf = &(forest.treeLeaves[0]);


    while (treePath.size() != 0){

        if(forest.treeLeaves[currentIndex].children.first != 0 && forest.treeLeaves[currentIndex].children.second != 0){
            if (pathState.back() == 0){
                pathState.back() = 1;
                currentIndex = forest.treeLeaves[currentIndex].children.first;
                treePath.push_back(forest.treeLeaves[currentIndex].splittingIndex);
                pathState.push_back(0);
                continue;    
            } else if (pathState.back() == 1) {
                pathState.back() = 2;
                currentIndex = forest.treeLeaves[currentIndex].children.second;
                treePath.push_back(forest.treeLeaves[currentIndex].splittingIndex);
                pathState.push_back(0);
                continue;
            } else if (pathState.back() == 2) {
                currentIndex = forest.treeLeaves[currentIndex].parent;
                pathState.pop_back();
                treePath.pop_back();
                continue;
            }
            
            throw std::logic_error("Invalid Crawl State");
            
        } else if (forest.treeLeaves[currentIndex].children.first == 0 && forest.treeLeaves[currentIndex].children.second == 0){
            highestIndex = std::max(highestIndex, forest.treeLeaves[currentIndex].splittingIndex);
            counter += 1;
            std::span indexSpan(&(forest.indexArray[forest.treeLeaves[currentIndex].splitRange.first]),
                              size_t(forest.treeLeaves[currentIndex].splitRange.second - forest.treeLeaves[currentIndex].splitRange.first));

            terminalFunctor(forest.treeLeaves[currentIndex].splittingIndex, indexSpan);

            currentIndex = forest.treeLeaves[currentIndex].parent;
            pathState.pop_back();
            treePath.pop_back();
            
            
            continue;
        }
        throw std::logic_error("Invalid Tree State");
        //size_t currentIndex = treePath.back();

    }

    return;

};


}
#endif //RPT_FOREST_HPP