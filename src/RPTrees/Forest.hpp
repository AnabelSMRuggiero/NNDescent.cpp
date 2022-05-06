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

#include <bit>
#include <concepts>
#include <cstddef>
#include <exception>
#include <execution>
#include <functional>
#include <future>
#include <iostream>
#include <memory_resource>
#include <numeric>
#include <ranges>
#include <span>
#include <type_traits>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>


#include "../ann/Type.hpp"
#include "../ann/AlignedMemory/DynamicArray.hpp"
#include "NND/MemoryInternals.hpp"
#include "NND/Type.hpp"
#include "Parallelization/AsyncQueue.hpp"
#include "Parallelization/ThreadPool.hpp"
#include "../NND/RNG.hpp"
#include "./SplittingScheme.hpp"

namespace nnd{





/*
This seems out of place here. The utility functions header under NND is mainly for functions
related to NND, and didn't want to tuck this there. This is a super general function though.
*/

template<typename Iterator, typename rIterator, typename Predicate>
size_t Split(Iterator fromBegin, Iterator fromEnd, Iterator toBegin, rIterator toRev, Predicate&& splitter){
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
    
    template< std::ranges::contiguous_range RangeToView >
    std::span<std::ranges::range_value_t<RangeToView>> to_span(RangeToView& range) const{
        return std::span{range.data() + splitRange.first, splitRange.second - splitRange.first};
    }

    std::size_t begin(){
        return splitRange.first;
    }
    std::size_t end(){
        return splitRange.second;
    }
    std::size_t size(){
        return end() - begin();
    }

};

//std::pmr::synchronized_pool_resource;

struct RandomProjectionForest{

    friend struct TreeRef;

    ann::dynamic_array<size_t> indecies;
    //const size_t numIndecies;
    PolymorphicAllocator<std::byte> alloc;
    TreeLeaf topNode;

    RandomProjectionForest(RandomProjectionForest&& other):indecies(std::move(other.indecies)),
        //indecies(std::move(other.indecies)),
        alloc(std::move(other.alloc)),
        topNode(std::move(other.topNode)),
        memManager(std::move(other.memManager)){
            //memManager->Put(other.memManager.TryTakeAll());
            topNode.children.first->parent = &topNode;
            topNode.children.second->parent = &topNode;
        }
    
    RandomProjectionForest(ann::dynamic_array<size_t>&& index_array): 
        indecies(std::move(index_array)),
        alloc{},
        topNode(0, indecies.size(), 0, nullptr),
        memManager(std::make_unique<AsyncQueue<TreeLeaf*>>()){};

    RandomProjectionForest(const std::unique_ptr<size_t[]>& indecies, const size_t numIndecies, std::pmr::memory_resource* upstream = internal::GetInternalResource()):
        indecies(std::span(indecies.get(), numIndecies)),
        alloc(upstream),
        topNode(0, numIndecies, 0, nullptr),
        memManager(std::make_unique<AsyncQueue<TreeLeaf*>>()){};

    std::span<size_t> GetView(){
        return std::span{indecies.data(), indecies.size()};
    }
    
    std::span<const size_t> GetView() const{
        return std::span{indecies.data(), indecies.size()};
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
    PolymorphicAllocator<std::byte> alloc;
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
        //This is cheesy to drop the memory though...
        memManager->Put(std::move(somethingToMove));
    }

    TreeLeaf* buildMemory;
    TreeLeaf* buildEnd;
    AsyncQueue<TreeLeaf*>* memManager;

};

enum class SplitQuality{
    done = 0,
    accept = 1,
    reject
};

inline SplitQuality EvaluateSplit(const SplittingHeurisitcs& heuristics, const size_t rangeSize, const size_t splitSize){
    size_t smallerSide = std::min(splitSize, rangeSize-splitSize);
    double splitFraction = double(smallerSide)/double(rangeSize);

    if (smallerSide>heuristics.childThreshold && splitFraction>heuristics.maxSplitFraction){

        return SplitQuality::accept;
        //done/accept rightSplit->splitRange.second - rightSplit->splitRange.first > splitThreshold
                
    } else if (rangeSize < heuristics.maxTreeSize){
        return SplitQuality::done;
    } else {
        return SplitQuality::reject;
    }

}
inline constexpr bool debugRPTrees = false;
//SplittingHeurisitcs test = {8, 64, 8};
template<typename SplittingScheme>
struct ForestBuilder{

    

    //std::vector<size_t> indexArray;
    RngFunctor rngFunctor;
    SplittingHeurisitcs heurisitics;
    SplittingScheme& getSplitComponents;
    

    ForestBuilder(RngFunctor&& rngFunctor, const SplittingHeurisitcs heurisitics, SplittingScheme& scheme):
        rngFunctor(std::move(rngFunctor)),
        heurisitics(heurisitics),
        getSplitComponents(scheme) {};

    // Training operator()

    RandomProjectionForest operator()(ann::dynamic_array<size_t>&& indecies);

    RandomProjectionForest operator()(ann::dynamic_array<size_t>&& indecies, 
                                      ThreadPool<TreeRef>& threadPool);
    
    //RandomProjectionForest operator()(ann::dynamic_array<size_t>&& indecies,
    //                                  const std::unordered_set<size_t>& splitIndicies);

    //RandomProjectionForest operator()(ann::dynamic_array<size_t>&& indecies,
    //                                  const std::unordered_set<size_t>& splitIndicies,
    //                                  ThreadPool<TreeRef>& threadPool);  

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
inline void AddLeaves(TreeRef& nodeRef, std::span<size_t> samples, std::span<size_t> workSpace, const size_t numTrue, std::vector<std::pair<std::size_t, TreeLeaf*>>& queue, const size_t splitThreshold){
    

    TreeLeaf* leftSplit = nodeRef.AddLeftLeaf(std::pair<size_t, size_t>(nodeRef.refNode->splitRange.first, nodeRef.refNode->splitRange.first + numTrue),
                                                        nodeRef.refNode->splittingIndex * 2 + 1);
            
    if (leftSplit->size() > splitThreshold) queue.emplace_back(0, leftSplit);
    else{
        //copy section of vec2 back to vec 1;
        
        auto fromIt = workSpace.begin()+leftSplit->splitRange.first;
        auto endFrom = workSpace.begin()+leftSplit->splitRange.second;
        auto toIt = samples.begin()+leftSplit->splitRange.first;

        std::copy(fromIt, endFrom, toIt);
    }

    TreeLeaf* rightSplit = nodeRef.AddRightLeaf(std::pair<size_t, size_t>(nodeRef.refNode->begin() + numTrue, nodeRef.refNode->end()),
        nodeRef.refNode->splittingIndex * 2 + 2);

    if (rightSplit->size() > splitThreshold) queue.emplace_back(0, rightSplit);
    else{
        //copy section of vec2 back to vec 1;

        auto fromIt = workSpace.begin() + rightSplit->splitRange.first;
        auto endFrom = workSpace.begin() + rightSplit->splitRange.second;
        auto toIt = samples.begin() + rightSplit->splitRange.first;

        std::copy(fromIt, endFrom, toIt);
    }
}

//Transforming Version
inline std::pair<TreeLeaf*, TreeLeaf*> AddLeaves(TreeRef& nodeRef,
                                          std::span<size_t> samples, 
                                          std::span<size_t> workSpace,
                                          const size_t numTrue,
                                          std::vector<TreeLeaf*>& queue,
                                          const std::unordered_set<size_t>& splitIndicies){
    
    std::pair<TreeLeaf*, TreeLeaf*> retPtrs = {nullptr, nullptr};
    if (numTrue == 0){
        nodeRef.refNode->splittingIndex = nodeRef.refNode->splittingIndex * 2 + 2;
        auto result = splitIndicies.find(nodeRef.refNode->splittingIndex);
        if (result != splitIndicies.end()) queue.push_back(nodeRef.refNode);
        return retPtrs;
    }
    if(numTrue == nodeRef.refNode->size()){
        nodeRef.refNode->splittingIndex = nodeRef.refNode->splittingIndex * 2 + 1;
        auto result = splitIndicies.find(nodeRef.refNode->splittingIndex);
        if (result != splitIndicies.end()) queue.push_back(nodeRef.refNode);
        return retPtrs;
    }

    TreeLeaf* leftSplit = nodeRef.AddLeftLeaf(std::pair<size_t, size_t>(nodeRef.refNode->begin(), nodeRef.refNode->begin() + numTrue),
                                                        nodeRef.refNode->splittingIndex * 2 + 1);

    auto result = splitIndicies.find(leftSplit->splittingIndex);
    if ((result == splitIndicies.end()) || (leftSplit->size() == 1)){
        
        if (result != splitIndicies.end()) retPtrs.first = leftSplit;
        
        auto fromIt = workSpace.begin()+leftSplit->begin();
        auto endFrom = workSpace.begin()+leftSplit->end();
        auto toIt = samples.begin()+leftSplit->begin();

        std::copy(fromIt, endFrom, toIt);  
    } else{ 
        
        queue.push_back(leftSplit);
        
    }

    TreeLeaf* rightSplit = nodeRef.AddRightLeaf(std::pair<size_t, size_t>(nodeRef.refNode->splitRange.first + numTrue, nodeRef.refNode->splitRange.second),
        nodeRef.refNode->splittingIndex * 2 + 2);
        
    result = splitIndicies.find(rightSplit->splittingIndex);
    if ((result == splitIndicies.end()) || (rightSplit->size() == 1)){
        
        if (result != splitIndicies.end()) retPtrs.second = rightSplit;
        
        auto fromIt = workSpace.begin()+rightSplit->splitRange.first;
        auto endFrom = workSpace.begin()+rightSplit->splitRange.second;
        auto toIt = samples.begin()+rightSplit->splitRange.first;

        std::copy(fromIt, endFrom, toIt);  
    } else{ 
        
        queue.push_back(rightSplit);
        
    }
    /*
    if ((result != splitIndicies.end())&&(rightSplit->splitRange.second - rightSplit->splitRange.first > 1)) queue.push_back(rightSplit);
    else{
        //copy section of vec2 back to vec 1;

        auto fromIt = workSpace.begin() + rightSplit->splitRange.first;
        auto endFrom = workSpace.begin() + rightSplit->splitRange.second;
        auto toIt = samples.begin() + rightSplit->splitRange.first;

        std::copy(fromIt, endFrom, toIt);
    }
    */
   return retPtrs;
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
RandomProjectionForest ForestBuilder<SplittingScheme>::operator()(ann::dynamic_array<size_t>&& indecies){

    //splittingVectors.reserve((1<<numberOfSplits) - 1);
    RandomProjectionForest forest(std::move(indecies));

    std::span<size_t> samples = forest.GetView();

    ann::dynamic_array<size_t> workSpaceArr(forest.indecies.size(), forest.indecies.get_allocator());
    std::span<size_t> workSpace = {workSpaceArr.data(), workSpaceArr.size()};

    size_t sum(0);
    if constexpr(debugRPTrees) sum = std::accumulate(samples.begin(), samples.end(), 0);
        //size_t tmpSum(sum);
    
    

    TreeRef builder(forest);

    std::vector<std::pair<std::size_t, TreeLeaf*>> splitQueue1 = {{0, &forest.topNode}};
    std::vector<std::pair<std::size_t, TreeLeaf*>> splitQueue2;
    //auto splittingFunction;

    //size_t beginIndex(0), endIndex(samples.size() - 1);

    while (splitQueue1.size() > 0){
        
        while(splitQueue1.size() > 0){
            //builder, samples, this
            builder.refNode = splitQueue1.back().second;

            
            rngFunctor.SetRange(builder.refNode->begin(), builder.refNode->end() - 1);

            auto [index1, index2] = UnequalIndecies(rngFunctor);
            index1 = samples[index1];
            index2 = samples[index2];

            auto splittingFunction = getSplitComponents(builder.refNode->splittingIndex, std::pair<size_t, size_t>(index1, index2));
            
            if (splittingFunction != std::nullopt){

                auto beginIt = samples.begin() + builder.refNode->begin();
                auto endIt = samples.begin() + builder.refNode->end();
                auto toBegin = workSpace.begin() + builder.refNode->begin();
                auto toRev = workSpace.rbegin() + workSpace.size()-builder.refNode->end();
                

                size_t numSplit = Split(beginIt, endIt, toBegin, toRev, *splittingFunction);
                size_t splitRange = builder.refNode->splitRange.second - builder.refNode->splitRange.first;
                SplitQuality splitQual = EvaluateSplit(this->heurisitics, splitRange, numSplit);
                switch(splitQual){
                    case SplitQuality::accept:
                        AddLeaves(builder, samples, workSpace, numSplit, splitQueue2, heurisitics.splitThreshold);
                        break;
                    case SplitQuality::reject:
                        if(splitQueue1.back().first < heurisitics.max_retry){
                            splitQueue2.emplace_back(splitQueue1.back().first+1, builder.refNode);
                        }
                        break;
                    default:
                        break;
                }
            } else {
                if(splitQueue1.back().first < heurisitics.max_retry){
                    splitQueue2.emplace_back(splitQueue1.back().first+1, builder.refNode);
                }
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

    }

    if (samples.data() == workSpaceArr.data()){
        std::swap(forest.indecies, workSpaceArr);
        //std::swap(forest.indecies, workSpaceArr);
    }
    return forest;
} //end operator()

template<typename SplittingScheme>
RandomProjectionForest ForestBuilder<SplittingScheme>::operator()(ann::dynamic_array<size_t>&& indecies,
                                      ThreadPool<TreeRef>& threadPool){
    
    RandomProjectionForest forest(std::move(indecies));
    threadPool.RebuildStates(std::reference_wrapper(forest));
    std::span<size_t> samples = forest.GetView();

    ann::dynamic_array<size_t> workSpaceArr(forest.indecies.size(), forest.indecies.get_allocator());
    std::span<size_t> workSpace = {workSpaceArr.data(), workSpaceArr.size()};
    
    size_t sum(0);
    if constexpr(debugRPTrees) sum = std::accumulate(samples.begin(), samples.end(), 0);

    TreeRef builder(forest);

    std::vector<std::pair<std::size_t, TreeLeaf*>> splitQueue1 = {{0, &forest.topNode}};
    std::vector<std::pair<std::size_t, TreeLeaf*>> splitQueue2;

    size_t topIters = std::bit_width(std::bit_ceil(threadPool.ThreadCount()));

    for (size_t i = 0; i < topIters; i+=1){
        std::vector<std::tuple<std::size_t, TreeLeaf*, std::unique_ptr<SplitState>>> pendingResults;
        for (auto [retry_count, ptr]: splitQueue1){
            builder.refNode = ptr;

            rngFunctor.SetRange(builder.refNode->splitRange.first, builder.refNode->splitRange.second - 1);
            
            
            auto [index1, index2] = UnequalIndecies(rngFunctor);
            index1 = samples[index1];
            index2 = samples[index2];
            

            auto predicate_opt = getSplitComponents(builder.refNode->splittingIndex, std::pair<size_t, size_t>(index1, index2));
            
            if (predicate_opt){
                size_t* beginIt = samples.data() + builder.refNode->splitRange.first;
                size_t* endIt = samples.data() + builder.refNode->splitRange.second;
                size_t* toBegin = workSpace.data() + builder.refNode->splitRange.first;
                size_t* toEnd = workSpace.data() + builder.refNode->splitRange.second;

                pendingResults.push_back({retry_count, ptr, Split(beginIt, endIt, toBegin, toEnd, *predicate_opt, threadPool)});
            }else {
                if(splitQueue1.back().first < heurisitics.max_retry){
                    splitQueue2.push_back({retry_count+1, builder.refNode});
                }
            }
        }

        

        for (auto& [retry_count, nodePtr, splitState]: pendingResults){
            std::future<size_t> result = splitState->result.get_future();
            size_t numSplit = result.get();
            builder.refNode = nodePtr;
            size_t splitRange = builder.refNode->splitRange.second - builder.refNode->splitRange.first;
            SplitQuality splitQual = EvaluateSplit(this->heurisitics, splitRange, numSplit);
            switch(splitQual){
                case SplitQuality::accept:
                    AddLeaves(builder, samples, workSpace, numSplit, splitQueue2, heurisitics.splitThreshold);
                    break;
                case SplitQuality::reject:
                    if(splitQueue1.back().first < heurisitics.max_retry){
                        splitQueue2.push_back({retry_count+1, builder.refNode});
                    }
                    break;
                default:
                    break;
            }
            /*
            if (splitQual){

                AddLeaves(builder, samples, workSpace, numSplit, splitQueue2, heurisitics.splitThreshold);
                
            } else if ((builder.refNode->splitRange.second - builder.refNode->splitRange.first) > heurisitics.maxTreeSize){
                splitQueue2.push_back(builder.refNode);
            }
            */
            
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

    AsyncQueue<std::pair<std::size_t, TreeLeaf*>> incomingNodes;
    auto splitTaskGenerator = [&](std::size_t retry_count, TreeLeaf* nodeToSplit)->auto{
        auto splitTask = [&getSplitComponents =this->getSplitComponents,
                          &incomingNodes,
                          retry_count, 
                          nodeToSplit, 
                          samples, 
                          workSpace,
                          heurisitics = this->heurisitics,
                          rngFunctor = this->rngFunctor] (TreeRef& nodeBuilder) mutable ->void{
            std::vector<std::pair<std::size_t, TreeLeaf*>> splitQueue1 = {{retry_count, nodeToSplit}};
            std::vector<std::pair<std::size_t, TreeLeaf*>> splitQueue2;
            for(size_t i = 0; i<2; i+=1){
                while(splitQueue1.size() > 0){
                    //builder, samples, this
                    nodeBuilder.refNode = splitQueue1.back().second;

                    retry:
                    //This is bootleg af, need to refactor how I do rng.
                    
                    rngFunctor.SetRange(nodeBuilder.refNode->splitRange.first, nodeBuilder.refNode->splitRange.second - 1);

                    auto [index1, index2] = UnequalIndecies(rngFunctor);
                    index1 = samples[index1];
                    index2 = samples[index2];

                    // Get the splitting vector, this can be fed into this function in the parallel/distributed case.
                    auto opt_predicate = getSplitComponents(nodeBuilder.refNode->splittingIndex, std::pair<size_t, size_t>(index1, index2));

                    if (opt_predicate != std::nullopt){

                        auto beginIt = samples.begin() + nodeBuilder.refNode->splitRange.first;
                        auto endIt = samples.begin() + nodeBuilder.refNode->splitRange.second;
                        auto toBegin = workSpace.begin() + nodeBuilder.refNode->splitRange.first;
                        auto toRev = workSpace.rbegin() + workSpace.size()-nodeBuilder.refNode->splitRange.second;
                        

                        size_t numSplit = Split(beginIt, endIt, toBegin, toRev, *opt_predicate);

                        size_t splitRange = nodeBuilder.refNode->splitRange.second - nodeBuilder.refNode->splitRange.first;
                        SplitQuality splitQual = EvaluateSplit(heurisitics, splitRange, numSplit);
                        switch(splitQual){
                            case SplitQuality::accept:
                                AddLeaves(nodeBuilder, samples, workSpace, numSplit, splitQueue2, heurisitics.splitThreshold);
                                break;
                            case SplitQuality::reject:
                                if(splitQueue1.back().first < heurisitics.max_retry){
                                    ++splitQueue1.back().first;
                                    goto retry;
                                }
                                break;
                            default:
                                break;
                        }
                    } else {
                        if(splitQueue1.back().first < heurisitics.max_retry){
                            ++splitQueue1.back().first;
                            goto retry;
                        }
                    }
                    /*
                    if (numSplit>heurisitics.childThreshold &&
                        (nodeBuilder.refNode->splitRange.second - nodeBuilder.refNode->splitRange.first - numSplit)>heurisitics.childThreshold){

                        AddLeaves(nodeBuilder, samples, workSpace, numSplit, splitQueue2, heurisitics.splitThreshold);
                        
                    } else if ((nodeBuilder.refNode->splitRange.second - nodeBuilder.refNode->splitRange.first) > heurisitics.maxTreeSize){
                        // I know, gross. This is a (hopefully) a place holder for a more robust way to handle rejected splits.
                        goto retry;
                    }
                    */
                    splitQueue1.pop_back();
                }
                std::swap(samples, workSpace);
                std::swap(splitQueue1, splitQueue2);
            }
            if(splitQueue1.size() != 0) incomingNodes.Put(std::move(splitQueue1));
        };

        return splitTask;
    };

    for(auto& [retry, ptr]: splitQueue1) threadPool.DelegateTask(splitTaskGenerator(retry, ptr));
    getSplitComponents.ConsumeNewVectors();
    while(true){
        while(threadPool.TaskCount() > 0){
            std::list<std::pair<std::size_t, TreeLeaf*>> newPtrs = incomingNodes.TryTakeAll();
            for (auto [retry, ptr]: newPtrs) threadPool.DelegateTask(splitTaskGenerator(retry, ptr));
            getSplitComponents.ConsumeNewVectors([](const size_t num){ return num > 10;});
        }

        threadPool.Latch();
         std::list<std::pair<std::size_t, TreeLeaf*>> newPtrs = incomingNodes.TryTakeAll();
        if(newPtrs.size() != 0){
            for (auto [retry, ptr]: newPtrs) threadPool.DelegateTask(splitTaskGenerator(retry, ptr));
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
RandomProjectionForest transform(ann::dynamic_array<size_t>&& indecies, const std::unordered_set<size_t>& split_indicies, const SplittingScheme& splitting_scheme){


    RandomProjectionForest forest(std::move(indecies));

    std::span<size_t> samples = forest.GetView();


    ann::dynamic_array<size_t> workSpaceArr(forest.indecies.size(), forest.indecies.get_allocator());
    std::span<size_t> workSpace = {workSpaceArr.data(), workSpaceArr.size()};

    TreeRef builder(forest);

    std::vector<TreeLeaf*> splitQueue1 = {&forest.topNode};
    std::vector<TreeLeaf*> splitQueue2;
    //auto splittingFunction;

    auto finishSplit = [&](TreeLeaf* nodeToFinish)->void{
        
        std::unordered_set<size_t>::const_iterator result;
        do{
            auto splittingFunction = splitting_scheme(nodeToFinish->splittingIndex, transformTag);
            bool dropLeft = splittingFunction(*(samples.begin() + nodeToFinish->splitRange.first));
            nodeToFinish->splittingIndex = nodeToFinish->splittingIndex * 2 + 1 + !dropLeft;
            result = split_indicies.find(nodeToFinish->splittingIndex);
        } while (result != split_indicies.end());
    };

    size_t sum(0);
    //if constexpr(debugRPTrees) sum = std::accumulate(samples.begin(), samples.end(), 0);

    //size_t beginIndex(0), endIndex(samples.size() - 1);

    while(splitQueue1.size() > 0){
        
        while(splitQueue1.size() > 0){
            


            builder.refNode = splitQueue1.back();


            // Get the splitting vector, this can be fed into this function in the parallel/distributed case.
            auto splittingFunction = splitting_scheme(builder.refNode->splittingIndex, transformTag);


            auto beginIt = samples.begin() + builder.refNode->splitRange.first;
            auto endIt = samples.begin() + builder.refNode->splitRange.second;
            auto toBegin = workSpace.begin() + builder.refNode->splitRange.first;
            auto toRev = workSpace.rbegin() + workSpace.size()-builder.refNode->splitRange.second;
            

            size_t numSplit = Split(beginIt, endIt, toBegin, toRev, splittingFunction);



            std::pair<TreeLeaf*, TreeLeaf*> ptrsToFinish = AddLeaves(builder, samples, workSpace, numSplit, splitQueue2, split_indicies);

            if (ptrsToFinish.first != nullptr) finishSplit(ptrsToFinish.first);
            if (ptrsToFinish.second != nullptr) finishSplit(ptrsToFinish.second);

            splitQueue1.pop_back();

        } //end while
        
        std::swap(samples, workSpace);
        /*
        if constexpr(debugRPTrees){
            size_t tmpSum = std::accumulate(samples.begin(), samples.end(), 0);
            if (sum != tmpSum){
                throw std::logic_error("Sum of indicies should be invariant.");
            };
        }
        */

        std::swap(splitQueue1, splitQueue2);
    } //end while
    if (samples.data() == workSpaceArr.data()){
        std::swap(forest.indecies, workSpaceArr);
    }
    /*
    if constexpr(debugRPTrees){
        size_t tmpSum = std::accumulate(samples.begin(), samples.end(), 0);
        if (sum != tmpSum){
            throw std::logic_error("Sum of indicies should be invariant.");
        };
    }
    */
    return forest;
} //end operator()

template<typename SplittingScheme>
RandomProjectionForest transform(ann::dynamic_array<size_t>&& indecies,
                                                                  const std::unordered_set<size_t>& splitIndicies,
                                                                  const SplittingScheme& splitting_scheme,
                                                                  ThreadPool<TreeRef>& threadPool){


    RandomProjectionForest forest(std::move(indecies));
    threadPool.RebuildStates(std::reference_wrapper(forest));
    std::span<size_t> samples = forest.GetView();


    ann::dynamic_array<size_t> workSpaceArr(forest.indecies.size(), forest.indecies.get_allocator());
    std::span<size_t> workSpace = {workSpaceArr.data(), workSpaceArr.size()};
    TreeRef builder(forest);

    std::vector<TreeLeaf*> splitQueue1 = {&forest.topNode};
    std::vector<TreeLeaf*> splitQueue2;
    //auto splittingFunction;

    auto finishGenerator = [&](TreeLeaf* nodeToFinish)->auto{;
        auto finishSplit = [&, nodeToFinish](auto&) mutable ->void{
            std::unordered_set<size_t>::const_iterator result;
            do{
                auto splittingFunction = splitting_scheme(nodeToFinish->splittingIndex, transformTag);
                bool dropLeft = splittingFunction(*(samples.begin() + nodeToFinish->splitRange.first));
                nodeToFinish->splittingIndex = nodeToFinish->splittingIndex * 2 + 1 + !dropLeft;
                result = splitIndicies.find(nodeToFinish->splittingIndex);
            } while (result != splitIndicies.end());
        };
        return finishSplit;
    };

    size_t sum(0);
    if constexpr(debugRPTrees) sum = std::accumulate(samples.begin(), samples.end(), 0);

    size_t topIters = std::bit_width(std::bit_ceil(threadPool.ThreadCount()));

    //size_t beginIndex(0), endIndex(samples.size() - 1);

    for (size_t i = 0; i < topIters; i+=1){
        std::vector<std::pair<TreeLeaf*, std::unique_ptr<SplitState>>> pendingResults;
        for (auto ptr: splitQueue1){
            


            builder.refNode = ptr;


            // Get the splitting vector, this can be fed into this function in the parallel/distributed case.
            auto splittingFunction = splitting_scheme(builder.refNode->splittingIndex, transformTag);


            auto beginIt = samples.data() + builder.refNode->splitRange.first;
            auto endIt = samples.data() + builder.refNode->splitRange.second;
            auto toBegin = workSpace.data() + builder.refNode->splitRange.first;
            auto toEnd = workSpace.data() + builder.refNode->splitRange.second;
            

            pendingResults.push_back({ptr, Split(beginIt, endIt, toBegin, toEnd, splittingFunction, threadPool)});

        }

        for (auto& [nodePtr, splitState]: pendingResults){
            std::future<size_t> result = splitState->result.get_future();
            size_t numSplit = result.get();
            builder.refNode = nodePtr;

            std::pair<TreeLeaf*, TreeLeaf*> ptrsToFinish = AddLeaves(builder, samples, workSpace, numSplit, splitQueue2, splitIndicies);

            if (ptrsToFinish.first != nullptr) threadPool.DelegateTask(finishGenerator(ptrsToFinish.first));
            if (ptrsToFinish.second != nullptr) threadPool.DelegateTask(finishGenerator(ptrsToFinish.second));
            splitQueue1.pop_back();

        } //end while
        
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
        auto splitTask = [&splitting_scheme,
                          &incomingNodes, 
                          nodeToSplit, 
                          samples, 
                          workSpace,
                          splitIndicies = splitIndicies] (TreeRef& nodeBuilder) mutable ->void{

            auto finishSplit = [&](TreeLeaf* nodeToFinish)->void{
        
                std::unordered_set<size_t>::const_iterator result;
                do{
                    auto splittingFunction = splitting_scheme(nodeToFinish->splittingIndex, transformTag);
                    bool dropLeft = splittingFunction(*(samples.begin() + nodeToFinish->splitRange.first));
                    nodeToFinish->splittingIndex = nodeToFinish->splittingIndex * 2 + 1 + !dropLeft;
                    result = splitIndicies.find(nodeToFinish->splittingIndex);
                } while (result != splitIndicies.end());
            };

            std::vector<TreeLeaf*> splitQueue1 = {nodeToSplit};
            std::vector<TreeLeaf*> splitQueue2;
            for(size_t i = 0; i<2; i+=1){
                while(splitQueue1.size() > 0){
                    //builder, samples, this
                    nodeBuilder.refNode = splitQueue1.back();


                    auto splittingFunction = splitting_scheme(nodeBuilder.refNode->splittingIndex, transformTag);


                    auto beginIt = samples.begin() + nodeBuilder.refNode->splitRange.first;
                    auto endIt = samples.begin() + nodeBuilder.refNode->splitRange.second;
                    auto toBegin = workSpace.begin() + nodeBuilder.refNode->splitRange.first;
                    auto toRev = workSpace.rbegin() + workSpace.size()-nodeBuilder.refNode->splitRange.second;
                    

                    size_t numSplit = Split(beginIt, endIt, toBegin, toRev, splittingFunction);


                    std::pair<TreeLeaf*, TreeLeaf*> ptrsToFinish = AddLeaves(nodeBuilder, samples, workSpace, numSplit, splitQueue2, splitIndicies);
                        
                    if (ptrsToFinish.first != nullptr) finishSplit(ptrsToFinish.first);
                    if (ptrsToFinish.second != nullptr) finishSplit(ptrsToFinish.second);


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
    while(true){
        while(threadPool.TaskCount() > 0){
            std::list<TreeLeaf*> newPtrs = incomingNodes.TryTakeAll();
            for (auto ptr: newPtrs) threadPool.DelegateTask(splitTaskGenerator(ptr));
        }

        threadPool.Latch();
        std::list<TreeLeaf*> newPtrs = incomingNodes.TryTakeAll();
        if(newPtrs.size() != 0){
            for (auto ptr: newPtrs) threadPool.DelegateTask(splitTaskGenerator(ptr));
        }else{
            break;
        }
    }

    if (samples.data() == workSpaceArr.data()){
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
}
    

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
std::pair<RandomProjectionForest, typename SplittingScheme::SplittingVectors> BuildRPForest(const DataSet<DataEntry>& data, const SplittingHeurisitcs params, std::pmr::memory_resource* upstream = std::pmr::get_default_resource()){
    

    RngFunctor rngFunctor(data.IndexStart(), data.size() - data.IndexStart());

    ann::dynamic_array<size_t> indecies(data.size());
    std::iota(indecies.data(), indecies.data()+data.size(), data.IndexStart());

    SplittingScheme splittingScheme(data);
    ForestBuilder builder{std::move(rngFunctor), params, splittingScheme};
    RandomProjectionForest rpTrees = builder(std::move(indecies));

    return {std::move(rpTrees), std::move(splittingScheme.splittingVectors)};
}

template<typename SplittingScheme, typename DataEntry>
    requires std::is_same_v<std::true_type, typename SplittingScheme::SerialScheme>
std::pair<RandomProjectionForest, typename SplittingScheme::SplittingVectors> BuildRPForest(const DataSet<DataEntry>& data, ann::dynamic_array<size_t>&& indecies, const SplittingHeurisitcs params){
    

    RngFunctor rngFunctor(0, indecies.size());

    SplittingScheme splittingScheme(data);
    ForestBuilder builder{std::move(rngFunctor), params, splittingScheme};
    RandomProjectionForest rpTrees = builder(std::move(indecies));

    return {std::move(rpTrees), std::move(splittingScheme.splittingVectors)};
}

template<typename SplittingScheme, typename DataEntry>
    requires std::is_same_v<std::true_type, typename SplittingScheme::ParallelScheme>
std::pair<RandomProjectionForest, typename SplittingScheme::SplittingVectors> BuildRPForest(const DataSet<DataEntry>& data, const SplittingHeurisitcs params, const size_t numThreads, std::pmr::memory_resource* upstream = std::pmr::get_default_resource()){
    

    RngFunctor rngFunctor(data.IndexStart(), data.size() - data.IndexStart());

    ann::dynamic_array<size_t> indecies(data.size());
    std::iota(indecies.data(), indecies.data()+data.size(), data.IndexStart());

    SplittingScheme splittingScheme(data);
    ForestBuilder builder{std::move(rngFunctor), params, splittingScheme};
    ThreadPool<TreeRef> pool(numThreads);
    pool.StartThreads();
    RandomProjectionForest rpTrees = builder(std::move(indecies),
                                             pool);

    pool.StopThreads();

    return {std::move(rpTrees), std::move(splittingScheme.splittingVectors)};
    
}


inline RandomProjectionForest RPTransformData(const DataSet<float>& testSet,
                     const std::unordered_set<size_t>& splittingIndecies,
                     std::unordered_map<size_t, std::pair<ann::aligned_array<float>, ann::aligned_array<float>::value_type>>&& splitting_vectors){

    euclidean_scheme<float, ann::aligned_array<float>> transforming_scheme(testSet);

    transforming_scheme.splittingVectors = std::move(splitting_vectors);

    ann::dynamic_array<size_t> testIndecies(testSet.size());
    std::iota(testIndecies.data(), testIndecies.data()+testSet.size(), 0);



    return transform(std::move(testIndecies), splittingIndecies, transforming_scheme);

}

template<std::regular_invocable<std::size_t, TransformTag> Scheme> 
inline RandomProjectionForest RPTransformData(const DataSet<float>& testSet,
    const std::unordered_set<size_t>& splittingIndecies,
    const Scheme& transforming_scheme) {


    ann::dynamic_array<size_t> testIndecies(testSet.size());
    std::iota(testIndecies.data(), testIndecies.data() + testSet.size(), 0);

    

    


    return transform(std::move(testIndecies), splittingIndecies, transforming_scheme);

}

inline RandomProjectionForest RPTransformData(const DataSet<float>& testSet,
                     const std::unordered_set<size_t>& splittingIndecies,
                     std::unordered_map<size_t, std::pair<ann::aligned_array<float>, ann::aligned_array<float>::value_type>>&& splittingVectors,
                     const size_t numThreads){

    euclidean_scheme<float, ann::aligned_array<float>> transforming_scheme(testSet);

    transforming_scheme.splittingVectors = std::move(splittingVectors);

    ann::dynamic_array<size_t> testIndecies(testSet.size());
    std::iota(testIndecies.data(), testIndecies.data()+testSet.size(), 0);

    RngFunctor testFunctor(size_t(0), testSet.size() - 1);

    ForestBuilder testBuilder{std::move(testFunctor), SplittingHeurisitcs{}, transforming_scheme};

    ThreadPool<TreeRef> pool(numThreads);
    pool.StartThreads();
    RandomProjectionForest retForest = transform(std::move(testIndecies), splittingIndecies, transforming_scheme, pool);
    pool.StopThreads();

    return retForest;

}

template<std::regular_invocable<std::size_t, TransformTag> Scheme> 
inline RandomProjectionForest RPTransformData(
    std::size_t dataset_size,
    const std::unordered_set<size_t>& splittingIndecies,
    const Scheme& transformingScheme,
    size_t numThreads) {


    ann::dynamic_array<size_t> testIndecies(std::views::iota(size_t{0}, dataset_size));
    //std::iota(testIndecies.data(), testIndecies.data() + dataset_size, 0);

    ThreadPool<TreeRef> pool{numThreads};
    pool.StartThreads();
    RandomProjectionForest retForest = transform(std::move(testIndecies), splittingIndecies, transformingScheme, pool);
    pool.StopThreads();

    return retForest;

}


}
#endif //RPT_FOREST_HPP