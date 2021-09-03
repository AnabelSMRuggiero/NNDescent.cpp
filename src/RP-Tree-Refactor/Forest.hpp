/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef RPT_FORESTRF_HPP
#define RPT_FORESTRF_HPP

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
#include "./SplittingScheme.hpp"

#include <memory_resource>

namespace nnd{

/*
This seems out of place here. The utility functions header under NND is mainly for functions
related to NND, and didn't want to tuck this there. This is a super general function though.
*/

struct SplittingHeurisitcs{
    int splits = 16;
    int splitThreshold = 80;
    int childThreshold = 32;
    int maxTreeSize = 130;
};

template<typename Iterator, typename rIterator, typename Predicate>
int Split(Iterator fromBegin, Iterator fromEnd, Iterator toBegin, rIterator toRev, Predicate splitter){
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

struct TreeLeaf{
    
    //This is for the split that produced the children
    size_t splittingIndex;
    std::pair<size_t, size_t> splitRange;
    TreeLeaf* parent;
    std::pair<TreeLeaf*, TreeLeaf*> children;
    
    
    TreeLeaf() : splitRange(0,0), splittingIndex(-1), children(nullptr, nullptr), parent(nullptr){};

    TreeLeaf(size_t index1, size_t index2, size_t splittingIndex, TreeLeaf* parent) : splitRange(index1, index2),
                                                                            splittingIndex(splittingIndex), 
                                                                            parent(parent),
                                                                            children(nullptr, nullptr) {};

    TreeLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex, TreeLeaf* parent) : splitRange(indecies),
                                                                                splittingIndex(splittingIndex), 
                                                                                parent(parent),
                                                                                children(nullptr, nullptr){};
    
};

//std::pmr::synchronized_pool_resource;

struct RandomProjectionForest{

    //std::pmr::synchronized_pool_resource treeBuffer;
    std::pmr::polymorphic_allocator<std::byte> alloc;
    TreeLeaf* topNode;

    RandomProjectionForest(std::pmr::memory_resource* upstream, std::pair<size_t, size_t> topIndexRange):
        alloc(upstream), topNode(alloc.new_object<TreeLeaf>(topIndexRange, 0, nullptr)){};

    
};

struct TreeRef{
    std::pmr::polymorphic_allocator<std::byte> alloc;
    TreeLeaf* refNode;


    TreeRef(RandomProjectionForest& forest): alloc(forest.alloc), refNode(forest.topNode){
        GetChunk();
    };
    
    TreeLeaf* AddLeftLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex){
        if (buildMemory == buildEnd) GetChunk();
        alloc.construct<TreeLeaf>(buildMemory, indecies, splittingIndex, refNode);
        refNode->children.first = buildMemory;
        buildMemory++;
        return buildMemory-1;
    }
    
    TreeLeaf* AddRightLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex){
        if (buildMemory == buildEnd) GetChunk();
        alloc.construct<TreeLeaf>(buildMemory, indecies, splittingIndex, refNode);
        refNode->children.second = buildMemory;
        buildMemory++;
        return buildMemory-1;
    }
    
    private:

    void GetChunk(){
        buildMemory = alloc.allocate_object<TreeLeaf>(chunkSize);
        buildEnd = buildMemory + 32;
    }

    TreeLeaf* buildMemory;
    TreeLeaf* buildEnd;
    constexpr static size_t chunkSize = 32;

};

//SplittingHeurisitcs test = {8, 64, 8};
template<typename SplittingScheme>
struct ForestBuilder{

    //std::vector<size_t> indexArray;
    StlRngFunctor<> rngFunctor;
    const SplittingHeurisitcs heurisitics;
    SplittingScheme& getSplitComponents;
    

    //ForestBuilder

    // Training operator()
    RandomProjectionForest operator()(std::span<size_t> samples, std::span<size_t> workSpace, std::pmr::memory_resource* upstream){

        //splittingVectors.reserve((1<<numberOfSplits) - 1);
        
        #ifdef _DEBUG
        size_t sum = std::accumulate(samples.begin(), samples.end(), 0);
        size_t tmpSum(sum);
        #endif

        RandomProjectionForest forest(upstream, {samples.front(), samples.back()});

        TreeRef builder(forest);

        std::vector<TreeLeaf*> splitQueue1 = {forest.topNode};
        std::vector<TreeLeaf*> splitQueue2;
        std::function<bool(size_t)> splittingFunction;

        //size_t beginIndex(0), endIndex(samples.size() - 1);

        while (splitQueue1.size() > 0){
            
            while(splitQueue1.size() > 0){
                
                builder.refNode = splitQueue1.back();

                retry:
                //This is bootleg af, need to refactor how I do rng.
                decltype(rngFunctor.functorDistribution)::param_type newRange(builder.refNode->splitRange.first, builder.refNode->splitRange.second - 1);
                rngFunctor.functorDistribution.param(newRange);

                size_t index1(rngFunctor());
                size_t index2(rngFunctor());
                while (index2 == index1){
                    index2 = rngFunctor();
                }
                
                index1 = samples[index1];
                index2 = samples[index2];

                // Get the splitting vector, this can be fed into this function in the parallel/distributed case.
                splittingFunction = getSplitComponents(builder.refNode->splittingIndex, std::pair<size_t, size_t>(index1, index2));


                size_t* beginIt = samples.begin() + builder.refNode->splitRange.first;
                size_t* endIt = samples.begin() + builder.refNode->splitRange.second;
                size_t* toBegin = workSpace.begin() + builder.refNode->splitRange.first;
                size_t* toRev = workSpace.rbegin() + workSpace.size()-builder.refNode->splitRange.second;
                

                int numSplit = Split(beginIt, endIt, toBegin, toRev, splittingFunction);

                if (numSplit>heurisitics.childThreshold &&
                    (builder.refNode->splitRange.second - builder.refNode->splitRange.first - numSplit)>heurisitics.childThreshold){

                    TreeLeaf* leftSplit = builder.AddLeftLeaf(std::pair<size_t, size_t>(builder.refNode->splitRange.first, builder.refNode->splitRange.first + numSplit),
                                                                    builder.refNode->splittingIndex * 2 + 1);
                    

                    if (leftSplit->splitRange.second - leftSplit->splitRange.first > heurisitics.splitThreshold) splitQueue2.push_back(leftSplit);
                    else{
                        //copy section of vec2 back to vec 1;
                        
                        size_t* fromIt = workSpace.begin()+leftSplit->splitRange.first;
                        size_t* endFrom = workSpace.begin()+leftSplit->splitRange.second;
                        size_t* toIt = samples.begin()+leftSplit->splitRange.first;

                        std::copy(fromIt, endFrom, toIt);
                    }

                    TreeLeaf* rightSplit = builder.refNode->AddRightLeaf(std::pair<size_t, size_t>(builder.refNode->splitRange.first + numSplit, builder.refNode->splitRange.second),
                        builder.refNode->splittingIndex * 2 + 2);

                    if (rightSplit->splitRange.second - rightSplit->splitRange.first > heurisitics.splitThreshold) splitQueue2.push_back(rightSplit->thisIndex);
                    else{
                        //copy section of vec2 back to vec 1;
                        /*
                        auto fromIt = indexVector2.begin();
                        std::advance(fromIt, rightSplit->splitRange.first);
                        auto endFrom = indexVector2.end();
                        std::advance(endFrom, rightSplit->splitRange.second - samples.size());

                        auto toIt = indexVector1.begin();
                        std::advance(toIt, rightSplit->splitRange.first);
                        */

                        size_t* fromIt = workSpace.begin() + rightSplit->splitRange.first;
                        size_t* endFrom = workSpace.begin() + rightSplit->splitRange.second;
                        size_t* toIt = samples.begin() + rightSplit->splitRange.first;


                        std::copy(fromIt, endFrom, toIt);
                    }
                } else if ((builder.refNode->splitRange.second - builder.refNode->splitRange.first) > heurisitics.maxTreeSize){
                    // I know, gross. This is a (hopefully) a place holder for a more robust way to handle rejected splits.
                    goto retry;
                }else{
                    // Undo the attempted split. This may be unneeded, but I want to be safe for now.
                    // TODO: Check to see if omitting this copy violates the invariance of sum
                    auto fromIt = indexVector1.begin() + builder.refNode->splitRange.first;
                    auto endFrom = indexVector1.begin() + builder.refNode->splitRange.second;
                    auto toIt = indexVector2.begin() + builder.refNode->splitRange.first;


                    //std::copy(fromIt, endFrom, toIt);
                }
                splitQueue1.pop_back();

            } //end while
            
            std::swap(samples, workSpace);
            std::swap(splitQueue1, splitQueue2);
            #ifdef _DEBUG
            tmpSum = std::accumulate(indexVector1.begin(), indexVector1.end(), 0);
            if (sum != tmpSum){
                throw std::logic_error("Sum of indicies should be invariant.");
            };
            #endif

        } //end for
    } //end operator()
    
    RandomProjectionForest operator()(std::span<size_t> samples, std::span<size_t> workSpace, std::pmr::memory_resource* upstream, std::unordered_set<size_t> splitsToDo);

};

//Transforming Constructor
template<typename SplittingScheme>
RandomProjectionForest ForestBuilder<SplittingScheme>::operator()(std::span<size_t> samples, std::span<size_t> workSpace, std::pmr::memory_resource* upstream, std::unordered_set<size_t> splitIndicies){

        

        


    RandomProjectionForest forest(upstream, {samples.front(), samples.back()});

    TreeRef builder(forest);

    std::vector<TreeLeaf*> splitQueue1 = {forest.topNode};
    std::vector<TreeLeaf*> splitQueue2;
    std::function<bool(size_t)> splittingFunction;

    //size_t beginIndex(0), endIndex(samples.size() - 1);

    while(splitQueue1.size() > 0){
        
        while(splitQueue1.size() > 0){
            


            builder.refNode = splitQueue1.back();


            // Get the splitting vector, this can be fed into this function in the parallel/distributed case.
            splittingFunction = getSplitComponents(builder.refNode->splittingIndex);


            size_t* beginIt = samples.begin() + builder.refNode->splitRange.first;
            size_t* endIt = samples.begin() + builder.refNode->splitRange.second;
            size_t* toBegin = workSpace.begin() + builder.refNode->splitRange.first;
            size_t* toRev = workSpace.rbegin() + workSpace.size()-builder.refNode->splitRange.second;
            

            int numSplit = Split(beginIt, endIt, toBegin, toRev, splittingFunction);



            TreeLeaf* leftSplit = builder.AddLeftLeaf(std::pair<size_t, size_t>(builder.refNode->splitRange.first, builder.refNode->splitRange.first + numSplit),
                                                                builder.refNode->splittingIndex * 2 + 1);
            TreeLeaf* rightSplit = builder.refNode->AddRightLeaf(std::pair<size_t, size_t>(builder.refNode->splitRange.first + numSplit, builder.refNode->splitRange.second),
                    builder.refNode->splittingIndex * 2 + 2);
            auto result = splitIndicies.find(leftSplit->splittingIndex);
            if ((result != splitIndicies.end()) && (leftSplit->splitRange.second - leftSplit->splitRange.first > 1)) splitQueue2.push_back(leftSplit);
            else {
                //copy section of vec2 back to vec 1;
                /*
                auto fromIt = indexVector2.begin();
                std::advance(fromIt, leftSplit->splitRange.first);
                auto endFrom = indexVector2.end();
                std::advance(endFrom, leftSplit->splitRange.second - samples.size());

                auto toIt = indexVector1.begin();
                std::advance(toIt, leftSplit->splitRange.first);
                */
                auto fromIt = &(workSpace[leftSplit->splitRange.first]);
                auto endFrom = &(workSpace[leftSplit->splitRange.second]);
                auto toIt = &(samples[leftSplit->splitRange.first]);

                std::copy(fromIt, endFrom, toIt);
            }

            result = splitIndicies.find(rightSplit->splittingIndex);
            if ((result != splitIndicies.end())&&(rightSplit->splitRange.second - rightSplit->splitRange.first > 1)) splitQueue2.push_back(rightSplit);
            else{
                //copy section of vec2 back to vec 1;
                /*
                auto fromIt = indexVector2.begin();
                std::advance(fromIt, rightSplit->splitRange.first);
                auto endFrom = indexVector2.end();
                std::advance(endFrom, rightSplit->splitRange.second - samples.size());

                auto toIt = indexVector1.begin();
                std::advance(toIt, rightSplit->splitRange.first);
                */

                auto fromIt = &(workSpace[rightSplit->splitRange.first]);
                auto endFrom = &(workSpace[rightSplit->splitRange.second]);
                auto toIt = &(samples[rightSplit->splitRange.first]);


                std::copy(fromIt, endFrom, toIt);
            }
            splitQueue1.pop_back();

        } //end while
        
        std::swap(samples, workSpace);
        /*
        tmpSum = std::accumulate(indexVector1.begin(), indexVector1.end(), 0);
        if (sum != tmpSum){
            throw std::logic_error("Sum of indicies should be invariant.");
        };
        */
        //if (splitQueue2.size() == 0) break;
        std::swap(splitQueue1, splitQueue2);
    } //end while
    //indexArray = std::move(indexVector1);
} //end operator()
    

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