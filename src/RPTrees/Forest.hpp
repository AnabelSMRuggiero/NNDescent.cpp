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
#include <execution>
#include <bit>
#include <future>
#include <type_traits>
#include <thread>

#include "../Utilities/Type.hpp"
#include "Parallelization/ThreadPool.hpp"
#include "../NND/RNG.hpp"
#include "./SplittingScheme.hpp"

#include <memory_resource>

namespace nnd{





/*
This seems out of place here. The utility functions header under NND is mainly for functions
related to NND, and didn't want to tuck this there. This is a super general function though.
*/

template<typename Iterator, typename rIterator, typename Predicate>
size_t Split(Iterator fromBegin, Iterator fromEnd, Iterator toBegin, rIterator toRev, Predicate splitter){
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

struct SplitState{

    SplitState(size_t* toBegin, size_t* toEnd): atoBegin(toBegin), atoEnd(toEnd) {};

    std::atomic<size_t> numTrue;
    std::atomic<size_t*> atoBegin;
    std::atomic<size_t*> atoEnd;
    std::atomic<size_t> threadsDone;
    std::promise<size_t> result;
};

template<typename Predicate, typename Pool>
std::unique_ptr<SplitState> Split(size_t* fromBegin, const size_t* fromEnd, size_t* toBegin, size_t* toEnd, Predicate splitter, Pool& threadPool){


    std::unique_ptr<SplitState> ptrToState = std::make_unique<SplitState>(toBegin, toEnd);

    auto splitTaskGenerator = [&](size_t* taskBegin, const size_t* taskEnd)->auto{
        auto splitTask = [&state = *ptrToState, splitter, taskBegin, taskEnd, numThreads = threadPool.ThreadCount()](auto&)mutable->void{
            for ( ; taskBegin != taskEnd; taskBegin++){
                if (splitter(*taskBegin)){
                    size_t* toBegin = state.atoBegin.fetch_add(1);
                    *toBegin = *taskBegin;
                    //taskBegin++;
                    state.numTrue++;
                } else {
                    size_t* toRev = state.atoEnd.fetch_sub(1)-1;
                    *toRev = *taskBegin;
                }
            }
            if(state.threadsDone.fetch_add(1)+1 == numThreads) state.result.set_value(state.numTrue);
        };

        return splitTask;
    };
    size_t* taskPtr = fromBegin;
    for (size_t i = 0; i<threadPool.ThreadCount()-1; i+=1){
        threadPool.DelegateTask(splitTaskGenerator(taskPtr, taskPtr + (fromEnd-fromBegin)/threadPool.ThreadCount()));
        taskPtr += (fromEnd-fromBegin)/threadPool.ThreadCount();
    }
    threadPool.DelegateTask(splitTaskGenerator(taskPtr, fromEnd));
    

    
    return ptrToState;
}

struct TreeLeaf{
    
    //This is for the split that produced the children
    size_t splittingIndex;
    std::pair<size_t, size_t> splitRange;
    TreeLeaf* parent;
    std::pair<TreeLeaf*, TreeLeaf*> children;
    
    
    TreeLeaf() : splittingIndex(-1), splitRange(0,0), parent(nullptr), children(nullptr, nullptr){};

    TreeLeaf(size_t index1, size_t index2, size_t splittingIndex, TreeLeaf* parent) : splittingIndex(splittingIndex), 
                                                                            splitRange(index1, index2),
                                                                            parent(parent),
                                                                            children(nullptr, nullptr) {};

    TreeLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex, TreeLeaf* parent) : splittingIndex(splittingIndex), 
                                                                                splitRange(indecies),
                                                                                parent(parent),
                                                                                children(nullptr, nullptr){};
    
};

//std::pmr::synchronized_pool_resource;

struct RandomProjectionForest{

    friend struct TreeRef;

    std::unique_ptr<size_t[]> indecies;
    const size_t numIndecies;
    std::pmr::polymorphic_allocator<std::byte> alloc;
    TreeLeaf topNode;

    RandomProjectionForest(RandomProjectionForest&& other):indecies(std::move(other.indecies)),
        numIndecies(other.numIndecies),
        alloc(std::move(other.alloc)),
        topNode(std::move(other.topNode)),
        memManager(std::move(other.memManager)){
            //memManager->Put(other.memManager.TryTakeAll());
            topNode.children.first->parent = &topNode;
            topNode.children.second->parent = &topNode;
        }
    

    RandomProjectionForest(std::unique_ptr<size_t[]>&& indecies, const size_t numIndecies, std::pmr::memory_resource* upstream = std::pmr::get_default_resource()):
        indecies(std::move(indecies)),
        numIndecies(numIndecies),
        alloc(upstream),
        topNode(0, numIndecies, 0, nullptr),
        memManager(std::make_unique<AsyncQueue<TreeLeaf*>>()){};

    std::span<size_t> GetView() const{
        return std::span{indecies.get(), numIndecies};
    }
    
    ~RandomProjectionForest(){
        if(memManager){
            std::list<TreeLeaf*> mem = memManager->TryTakeAll();
            for(auto ptr: mem){
                if(ptr != nullptr) alloc.deallocate_object(ptr, chunkSize);
            }
        }
    }

    private:
    constexpr static size_t chunkSize = 32;
    std::unique_ptr<AsyncQueue<TreeLeaf*>> memManager;
};

struct TreeRef{
    std::pmr::polymorphic_allocator<std::byte> alloc;
    TreeLeaf* refNode;

    TreeRef() = default;

    TreeRef(TreeRef&& rhs): alloc(rhs.alloc), 
        refNode(rhs.refNode), 
        buildMemory(rhs.buildMemory),
        buildEnd(rhs.buildEnd),
        memManager(rhs.memManager){
            rhs.refNode = nullptr;
            rhs.buildMemory = nullptr;
            rhs.buildEnd = nullptr;
            rhs.memManager = nullptr;
    }

    TreeRef(RandomProjectionForest& forest): alloc(forest.alloc), refNode(&forest.topNode), memManager(forest.memManager.get()){
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
        buildMemory = alloc.allocate_object<TreeLeaf>(RandomProjectionForest::chunkSize);
        buildEnd = buildMemory + 32;
        TreeLeaf* somethingToMove = buildMemory;
        //If I'm not using the new_delete_resource, I should be able to get away with letting the resource
        //manage the memory.
        if(alloc.resource() == std::pmr::new_delete_resource()) memManager->Put(std::move(somethingToMove));
    }

    TreeLeaf* buildMemory;
    TreeLeaf* buildEnd;
    AsyncQueue<TreeLeaf*>* memManager;

};

//SplittingHeurisitcs test = {8, 64, 8};
template<typename SplittingScheme>
struct ForestBuilder{

    constexpr static bool debugRPTrees = false;

    //std::vector<size_t> indexArray;
    RngFunctor rngFunctor;
    const SplittingHeurisitcs heurisitics;
    SplittingScheme& getSplitComponents;
    

    ForestBuilder(RngFunctor&& rngFunctor, const SplittingHeurisitcs heurisitics, SplittingScheme& scheme):
        rngFunctor(std::move(rngFunctor)),
        heurisitics(heurisitics),
        getSplitComponents(scheme) {};

    // Training operator()
    RandomProjectionForest operator()(std::unique_ptr<size_t[]>&& indecies, 
                                      const size_t numIndecies, 
                                      std::pmr::memory_resource* upstream = std::pmr::get_default_resource());

    RandomProjectionForest operator()(std::execution::parallel_unsequenced_policy,
                                      std::unique_ptr<size_t[]>&& indecies, 
                                      const size_t numIndecies,
                                      ThreadPool<TreeRef>& threadPool,
                                      std::pmr::memory_resource* upstream = std::pmr::get_default_resource());
    
    RandomProjectionForest operator()(std::unique_ptr<size_t[]>&& indecies, 
                                      const size_t numIndecies, 
                                      const std::unordered_set<size_t>& splitIndicies, 
                                      std::pmr::memory_resource* upstream = std::pmr::get_default_resource());

    private:
    /*
    void AddLeaves(TreeRef& nodeRef,
                   std::span<size_t> samples,
                   std::span<size_t> workSpace,
                   const size_t numTrue,
                   std::vector<TreeLeaf*>& queue) const;
    */
    //auto SplitTask(std::span<size_t> indecies, LeafNode* splitToDo)

};

//Training version
void AddLeaves(TreeRef& nodeRef, std::span<size_t> samples, std::span<size_t> workSpace, const size_t numTrue, std::vector<TreeLeaf*>& queue, const size_t splitThreshold){
    
    TreeLeaf* leftSplit = nodeRef.AddLeftLeaf(std::pair<size_t, size_t>(nodeRef.refNode->splitRange.first, nodeRef.refNode->splitRange.first + numTrue),
                                                        nodeRef.refNode->splittingIndex * 2 + 1);
            
    if (leftSplit->splitRange.second - leftSplit->splitRange.first > splitThreshold) queue.push_back(leftSplit);
    else{
        //copy section of vec2 back to vec 1;
        
        auto fromIt = workSpace.begin()+leftSplit->splitRange.first;
        auto endFrom = workSpace.begin()+leftSplit->splitRange.second;
        auto toIt = samples.begin()+leftSplit->splitRange.first;

        std::copy(fromIt, endFrom, toIt);
    }

    TreeLeaf* rightSplit = nodeRef.AddRightLeaf(std::pair<size_t, size_t>(nodeRef.refNode->splitRange.first + numTrue, nodeRef.refNode->splitRange.second),
        nodeRef.refNode->splittingIndex * 2 + 2);

    if (rightSplit->splitRange.second - rightSplit->splitRange.first > splitThreshold) queue.push_back(rightSplit);
    else{
        //copy section of vec2 back to vec 1;

        auto fromIt = workSpace.begin() + rightSplit->splitRange.first;
        auto endFrom = workSpace.begin() + rightSplit->splitRange.second;
        auto toIt = samples.begin() + rightSplit->splitRange.first;

        std::copy(fromIt, endFrom, toIt);
    }
}

template<typename RNG>
std::pair<size_t, size_t> UnequalIndecies(RNG& rng){

    size_t index1(rng());
    size_t index2(rng());
    while (index2 == index1){
        index2 = rng();
    }
    return {index1, index2};

}

template<typename SplittingScheme>
RandomProjectionForest ForestBuilder<SplittingScheme>::operator()(std::unique_ptr<size_t[]>&& indecies, const size_t numIndecies, std::pmr::memory_resource* upstream){

    //splittingVectors.reserve((1<<numberOfSplits) - 1);
    RandomProjectionForest forest(std::move(indecies), numIndecies, upstream);

    std::span<size_t> samples = forest.GetView();

    std::unique_ptr<size_t[]> workSpaceArr = std::make_unique<size_t[]>(numIndecies);
    std::span<size_t> workSpace = {workSpaceArr.get(), numIndecies};

    size_t sum(0);
    if constexpr(debugRPTrees) sum = std::accumulate(samples.begin(), samples.end(), 0);
        //size_t tmpSum(sum);
    
    

    TreeRef builder(forest);

    std::vector<TreeLeaf*> splitQueue1 = {&forest.topNode};
    std::vector<TreeLeaf*> splitQueue2;
    //auto splittingFunction;

    //size_t beginIndex(0), endIndex(samples.size() - 1);

    while (splitQueue1.size() > 0){
        
        while(splitQueue1.size() > 0){
            //builder, samples, this
            builder.refNode = splitQueue1.back();

            
            rngFunctor.SetRange(builder.refNode->splitRange.first, builder.refNode->splitRange.second - 1);

            auto [index1, index2] = UnequalIndecies(rngFunctor);
            index1 = samples[index1];
            index2 = samples[index2];

            // Get the splitting vector, this can be fed into this function in the parallel/distributed case.
            auto splittingFunction = getSplitComponents(builder.refNode->splittingIndex, std::pair<size_t, size_t>(index1, index2));


            auto beginIt = samples.begin() + builder.refNode->splitRange.first;
            auto endIt = samples.begin() + builder.refNode->splitRange.second;
            auto toBegin = workSpace.begin() + builder.refNode->splitRange.first;
            auto toRev = workSpace.rbegin() + workSpace.size()-builder.refNode->splitRange.second;
            

            size_t numSplit = Split(beginIt, endIt, toBegin, toRev, splittingFunction);

            if (numSplit>heurisitics.childThreshold &&
                (builder.refNode->splitRange.second - builder.refNode->splitRange.first - numSplit)>heurisitics.childThreshold){

                AddLeaves(builder, samples, workSpace, numSplit, splitQueue2, heurisitics.splitThreshold);
                
            } else if ((builder.refNode->splitRange.second - builder.refNode->splitRange.first) > heurisitics.maxTreeSize){
                splitQueue2.push_back(builder.refNode);
            }
            splitQueue1.pop_back();

        } //end while
        
        std::swap(samples, workSpace);
        std::swap(splitQueue1, splitQueue2);
        if constexpr(debugRPTrees){
            size_t tmpSum = std::accumulate(samples.begin(), samples.end(), 0);
            if (sum != tmpSum){
                throw std::logic_error("Sum of indicies should be invariant.");
            };
        }

    } //end for
    if (samples.data() == workSpaceArr.get()){
        std::swap(forest.indecies, workSpaceArr);
    }
    return forest;
} //end operator()

template<typename SplittingScheme>
RandomProjectionForest ForestBuilder<SplittingScheme>::operator()(std::execution::parallel_unsequenced_policy,
                                      std::unique_ptr<size_t[]>&& indecies, 
                                      const size_t numIndecies,
                                      ThreadPool<TreeRef>& threadPool,
                                      std::pmr::memory_resource* upstream){
    
    RandomProjectionForest forest(std::move(indecies), numIndecies, upstream);
    threadPool.RebuildStates(std::reference_wrapper(forest));
    std::span<size_t> samples = forest.GetView();

    std::unique_ptr<size_t[]> workSpaceArr = std::make_unique<size_t[]>(samples.size());
    std::span<size_t> workSpace = {workSpaceArr.get(), samples.size()};
    
    size_t sum(0);
    if constexpr(debugRPTrees) sum = std::accumulate(samples.begin(), samples.end(), 0);

    TreeRef builder(forest);

    std::vector<TreeLeaf*> splitQueue1 = {&forest.topNode};
    std::vector<TreeLeaf*> splitQueue2;

    size_t topIters = std::bit_width(std::bit_ceil(threadPool.ThreadCount()));

    for (size_t i = 0; i < topIters; i+=1){
        std::vector<std::pair<TreeLeaf*, std::unique_ptr<SplitState>>> pendingResults;
        for (auto ptr: splitQueue1){
            builder.refNode = ptr;

            rngFunctor.SetRange(builder.refNode->splitRange.first, builder.refNode->splitRange.second - 1);

            auto [index1, index2] = UnequalIndecies(rngFunctor);
            index1 = samples[index1];
            index2 = samples[index2];
            

            auto splittingFunction = getSplitComponents(builder.refNode->splittingIndex, std::pair<size_t, size_t>(index1, index2));


            size_t* beginIt = samples.data() + builder.refNode->splitRange.first;
            size_t* endIt = samples.data() + builder.refNode->splitRange.second;
            size_t* toBegin = workSpace.data() + builder.refNode->splitRange.first;
            size_t* toEnd = workSpace.data() + builder.refNode->splitRange.second;

            pendingResults.push_back({ptr, Split(beginIt, endIt, toBegin, toEnd, splittingFunction, threadPool)});
        }
        

        for (auto& [nodePtr, splitState]: pendingResults){
            std::future<size_t> result = splitState->result.get_future();
            size_t numSplit = result.get();
            builder.refNode = nodePtr;
            if (numSplit>heurisitics.childThreshold &&
                (builder.refNode->splitRange.second - builder.refNode->splitRange.first - numSplit)>heurisitics.childThreshold){

                AddLeaves(builder, samples, workSpace, numSplit, splitQueue2, heurisitics.splitThreshold);
                
            } else if ((builder.refNode->splitRange.second - builder.refNode->splitRange.first) > heurisitics.maxTreeSize){
                splitQueue2.push_back(builder.refNode);
            }
            
        }
        std::swap(samples, workSpace);
        splitQueue1.clear();
        std::swap(splitQueue1, splitQueue2);

        if constexpr(debugRPTrees){
            size_t tmpSum = std::accumulate(samples.begin(), samples.end(), 0);
            if (sum != tmpSum){
                throw std::logic_error("Sum of indicies should be invariant.");
            };
        }
        
    }

    AsyncQueue<TreeLeaf*> incomingNodes;
    auto splitTaskGenerator = [&](TreeLeaf* nodeToSplit)->auto{
        auto splitTask = [&getSplitComponents =this->getSplitComponents,
                          &incomingNodes, 
                          nodeToSplit, 
                          samples, 
                          workSpace,
                          heurisitics = this->heurisitics,
                          rngFunctor = this->rngFunctor] (TreeRef& nodeBuilder) mutable ->void{
            std::vector<TreeLeaf*> splitQueue1 = {nodeToSplit};
            std::vector<TreeLeaf*> splitQueue2;
            for(size_t i = 0; i<2; i+=1){
                while(splitQueue1.size() > 0){
                    //builder, samples, this
                    nodeBuilder.refNode = splitQueue1.back();

                    retry:
                    //This is bootleg af, need to refactor how I do rng.
                    
                    rngFunctor.SetRange(nodeBuilder.refNode->splitRange.first, nodeBuilder.refNode->splitRange.second - 1);

                    auto [index1, index2] = UnequalIndecies(rngFunctor);
                    index1 = samples[index1];
                    index2 = samples[index2];

                    // Get the splitting vector, this can be fed into this function in the parallel/distributed case.
                    auto splittingFunction = getSplitComponents(nodeBuilder.refNode->splittingIndex, std::pair<size_t, size_t>(index1, index2));


                    auto beginIt = samples.begin() + nodeBuilder.refNode->splitRange.first;
                    auto endIt = samples.begin() + nodeBuilder.refNode->splitRange.second;
                    auto toBegin = workSpace.begin() + nodeBuilder.refNode->splitRange.first;
                    auto toRev = workSpace.rbegin() + workSpace.size()-nodeBuilder.refNode->splitRange.second;
                    

                    size_t numSplit = Split(beginIt, endIt, toBegin, toRev, splittingFunction);

                    if (numSplit>heurisitics.childThreshold &&
                        (nodeBuilder.refNode->splitRange.second - nodeBuilder.refNode->splitRange.first - numSplit)>heurisitics.childThreshold){

                        AddLeaves(nodeBuilder, samples, workSpace, numSplit, splitQueue2, heurisitics.splitThreshold);
                        
                    } else if ((nodeBuilder.refNode->splitRange.second - nodeBuilder.refNode->splitRange.first) > heurisitics.maxTreeSize){
                        // I know, gross. This is a (hopefully) a place holder for a more robust way to handle rejected splits.
                        goto retry;
                    }
                    splitQueue1.pop_back();
                }
                std::swap(samples, workSpace);
                std::swap(splitQueue1, splitQueue2);
            }
            if(splitQueue1.size() != 0) incomingNodes.Put(std::move(splitQueue1));
        };

        return splitTask;
    };

    for(auto ptr: splitQueue1) threadPool.DelegateTask(splitTaskGenerator(ptr));
    getSplitComponents.ConsumeNewVectors();
    while(true){
        while(threadPool.TaskCount() > 0){
            std::list<TreeLeaf*> newPtrs = incomingNodes.TryTakeAll();
            for (auto ptr: newPtrs) threadPool.DelegateTask(splitTaskGenerator(ptr));
            getSplitComponents.ConsumeNewVectors([](const size_t num){ return num > 10;});
        }

        threadPool.Latch();
        std::list<TreeLeaf*> newPtrs = incomingNodes.TryTakeAll();
        if(newPtrs.size() != 0){
            for (auto ptr: newPtrs) threadPool.DelegateTask(splitTaskGenerator(ptr));
            getSplitComponents.ConsumeNewVectors([](const size_t num){ return num > 10;});
        }else{
            getSplitComponents.ConsumeNewVectors();
            break;
        }
    }

    if constexpr(debugRPTrees){
        size_t tmpSum = std::accumulate(samples.begin(), samples.end(), 0);
        if (sum != tmpSum){
            throw std::logic_error("Sum of indicies should be invariant.");
        };
    }
    return forest;
}




//Transforming Constructor
template<typename SplittingScheme>
RandomProjectionForest ForestBuilder<SplittingScheme>::operator()(std::unique_ptr<size_t[]>&& indecies, const size_t numIndecies, const std::unordered_set<size_t>& splitIndicies, std::pmr::memory_resource* upstream){


    RandomProjectionForest forest(std::move(indecies), numIndecies, upstream);

    std::span<size_t> samples = forest.GetView();


    std::unique_ptr<size_t[]> workSpaceArr = std::make_unique<size_t[]>(numIndecies);
    std::span<size_t> workSpace = {workSpaceArr.get(), numIndecies};
    TreeRef builder(forest);

    std::vector<TreeLeaf*> splitQueue1 = {&forest.topNode};
    std::vector<TreeLeaf*> splitQueue2;
    //auto splittingFunction;

    size_t sum(0);
    if constexpr(debugRPTrees) sum = std::accumulate(samples.begin(), samples.end(), 0);

    //size_t beginIndex(0), endIndex(samples.size() - 1);

    while(splitQueue1.size() > 0){
        
        while(splitQueue1.size() > 0){
            


            builder.refNode = splitQueue1.back();


            // Get the splitting vector, this can be fed into this function in the parallel/distributed case.
            auto splittingFunction = getSplitComponents(builder.refNode->splittingIndex, transformTag);


            auto beginIt = samples.begin() + builder.refNode->splitRange.first;
            auto endIt = samples.begin() + builder.refNode->splitRange.second;
            auto toBegin = workSpace.begin() + builder.refNode->splitRange.first;
            auto toRev = workSpace.rbegin() + workSpace.size()-builder.refNode->splitRange.second;
            

            size_t numSplit = Split(beginIt, endIt, toBegin, toRev, splittingFunction);



            TreeLeaf* leftSplit = builder.AddLeftLeaf(std::pair<size_t, size_t>(builder.refNode->splitRange.first, builder.refNode->splitRange.first + numSplit),
                                                                builder.refNode->splittingIndex * 2 + 1);
            TreeLeaf* rightSplit = builder.AddRightLeaf(std::pair<size_t, size_t>(builder.refNode->splitRange.first + numSplit, builder.refNode->splitRange.second),
                    builder.refNode->splittingIndex * 2 + 2);
            auto result = splitIndicies.find(leftSplit->splittingIndex);
            if ((result != splitIndicies.end()) && (leftSplit->splitRange.second - leftSplit->splitRange.first > 1)) splitQueue2.push_back(leftSplit);
            else {
                //copy section of vec2 back to vec 1;
                
                auto fromIt = &(workSpace[leftSplit->splitRange.first]);
                auto endFrom = &(workSpace[leftSplit->splitRange.second]);
                auto toIt = &(samples[leftSplit->splitRange.first]);

                std::copy(fromIt, endFrom, toIt);
            }

            result = splitIndicies.find(rightSplit->splittingIndex);
            if ((result != splitIndicies.end())&&(rightSplit->splitRange.second - rightSplit->splitRange.first > 1)) splitQueue2.push_back(rightSplit);
            else{
                //copy section of vec2 back to vec 1;
                

                auto fromIt = &(workSpace[rightSplit->splitRange.first]);
                auto endFrom = &(workSpace[rightSplit->splitRange.second]);
                auto toIt = &(samples[rightSplit->splitRange.first]);


                std::copy(fromIt, endFrom, toIt);
            }
            splitQueue1.pop_back();

        } //end while
        
        std::swap(samples, workSpace);
        
        if constexpr(debugRPTrees){
            size_t tmpSum = std::accumulate(samples.begin(), samples.end(), 0);
            if (sum != tmpSum){
                throw std::logic_error("Sum of indicies should be invariant.");
            };
        }
        
        //if (splitQueue2.size() == 0) break;
        std::swap(splitQueue1, splitQueue2);
    } //end while
    if (samples.data() == workSpaceArr.get()){
        std::swap(forest.indecies, workSpaceArr);
    }
    if constexpr(debugRPTrees){
        size_t tmpSum = std::accumulate(samples.begin(), samples.end(), 0);
        if (sum != tmpSum){
            throw std::logic_error("Sum of indicies should be invariant.");
        };
    }
    return forest;
    //indexArray = std::move(indexVector1);
} //end operator()
    

template<typename Functor>
void CrawlTerminalLeaves(const RandomProjectionForest& forest, Functor& terminalFunctor){


    std::vector<char> pathState;
    pathState.push_back(0);

    //size_t highestIndex = 0;
    //size_t counter = 0;
    
    std::span<const size_t> indecies = forest.GetView();
    
    const TreeLeaf* currentNode = &forest.topNode;

    


    while (pathState.size() != 0){

        if(currentNode->children.first != nullptr && currentNode->children.second != nullptr){
            if (pathState.back() == 0){
                pathState.back() = 1;
                currentNode = currentNode->children.first;
                //treePath.push_back(currentNode->splittingIndex);
                pathState.push_back(0);
                continue;    
            } else if (pathState.back() == 1) {
                pathState.back() = 2;
                currentNode = currentNode->children.second;
                //treePath.push_back(currentNode->splittingIndex);
                pathState.push_back(0);
                continue;
            } else if (pathState.back() == 2) {
                currentNode = currentNode->parent;
                pathState.pop_back();
                //treePath.pop_back();
                continue;
            }
            
            throw std::logic_error("Invalid Crawl State");
            
        } else if (currentNode->children.first == nullptr && currentNode->children.second == nullptr){
            //highestIndex = std::max(highestIndex, currentNode->splittingIndex);
            //counter += 1;
            std::span indexSpan(&(indecies[currentNode->splitRange.first]),
                              size_t(currentNode->splitRange.second - currentNode->splitRange.first));

            terminalFunctor(currentNode->splittingIndex, indexSpan);

            currentNode = currentNode->parent;
            pathState.pop_back();
            //treePath.pop_back();
            
            
            continue;
        }
        throw std::logic_error("Invalid Tree State");
        //size_t currentIndex = treePath.back();

    }
    
    return;

};

template<typename Functor>
void CrawlLeaves(const RandomProjectionForest& forest, Functor& nodeFunctor){

    std::vector<char> pathState;
    pathState.push_back(0);

    //size_t highestIndex = 0;
    //size_t counter = 0;
    
    std::span<const size_t> indecies = forest.GetView();
    
    const TreeLeaf* currentNode = &forest.topNode;

    nodeFunctor(*currentNode, indecies);


    while (pathState.size() != 0){
        
        if(currentNode->children.first != nullptr && currentNode->children.second != nullptr){
            if (pathState.back() == 0){
                pathState.back() = 1;
                currentNode = currentNode->children.first;
                std::span indexSpan(&(indecies[currentNode->splitRange.first]),
                              size_t(currentNode->splitRange.second - currentNode->splitRange.first));

                nodeFunctor(*currentNode, indexSpan);
                //treePath.push_back(currentNode->splittingIndex);
                pathState.push_back(0);
                continue;    
            } else if (pathState.back() == 1) {
                pathState.back() = 2;
                currentNode = currentNode->children.second;
                std::span indexSpan(&(indecies[currentNode->splitRange.first]),
                              size_t(currentNode->splitRange.second - currentNode->splitRange.first));
                nodeFunctor(*currentNode, indexSpan);
                //treePath.push_back(currentNode->splittingIndex);
                pathState.push_back(0);
                continue;
            } else if (pathState.back() == 2) {
                currentNode = currentNode->parent;
                pathState.pop_back();
                //treePath.pop_back();
                continue;
            }
            
            throw std::logic_error("Invalid Crawl State");
            
        } else if (currentNode->children.first == nullptr && currentNode->children.second == nullptr){
            //highestIndex = std::max(highestIndex, currentNode->splittingIndex);
            //counter += 1;
            std::span indexSpan(&(indecies[currentNode->splitRange.first]),
                              size_t(currentNode->splitRange.second - currentNode->splitRange.first));

            nodeFunctor(*currentNode, indexSpan);

            currentNode = currentNode->parent;
            pathState.pop_back();
            //treePath.pop_back();
            
            
            continue;
        }
        throw std::logic_error("Invalid Tree State");
        //size_t currentIndex = treePath.back();

    }
    
    return;

};


template<typename SplittingScheme, typename DataEntry>
    requires std::is_same_v<std::true_type, typename SplittingScheme::SerialScheme>
std::pair<RandomProjectionForest, typename SplittingScheme::SplittingVectors> BuildRPForest(std::execution::sequenced_policy, const DataSet<DataEntry>& data, const SplittingHeurisitcs params, std::pmr::memory_resource* upstream = std::pmr::get_default_resource()){
    

    RngFunctor rngFunctor(data.IndexStart(), data.size() - data.IndexStart());

    std::unique_ptr<size_t[]> indecies = std::make_unique<size_t[]>(data.size());
    std::iota(indecies.get(), indecies.get()+data.size(), data.IndexStart());

    SplittingScheme splittingScheme(data);
    ForestBuilder builder{std::move(rngFunctor), params, splittingScheme};
    RandomProjectionForest rpTrees = builder(std::move(indecies), data.size(), upstream);

    return {std::move(rpTrees), std::move(splittingScheme.splittingVectors)};
}

template<typename SplittingScheme, typename DataEntry>
    requires std::is_same_v<std::true_type, typename SplittingScheme::ParallelScheme>
std::pair<RandomProjectionForest, typename SplittingScheme::SplittingVectors> BuildRPForest(std::execution::parallel_unsequenced_policy, const DataSet<DataEntry>& data, const SplittingHeurisitcs params, const size_t numThreads, std::pmr::memory_resource* upstream = std::pmr::get_default_resource()){
    

    RngFunctor rngFunctor(data.IndexStart(), data.size() - data.IndexStart());

    std::unique_ptr<size_t[]> indecies = std::make_unique<size_t[]>(data.size());
    std::iota(indecies.get(), indecies.get()+data.size(), data.IndexStart());

    SplittingScheme splittingScheme(data);
    ForestBuilder builder{std::move(rngFunctor), params, splittingScheme};
    ThreadPool<TreeRef> pool(numThreads);
    pool.StartThreads();
    RandomProjectionForest rpTrees = builder(std::execution::par_unseq,
                                             std::move(indecies), 
                                             data.size(),
                                             pool,
                                             upstream);

    pool.StopThreads();

    return {std::move(rpTrees), std::move(splittingScheme.splittingVectors)};
    
}


}
#endif //RPT_FOREST_HPP