/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_PARALLELIZATIONOBJECTS_HPP
#define NND_PARALLELIZATIONOBJECTS_HPP

#include <memory>
#include <atomic>

#include "../FunctorErasure.hpp"
#include "../GraphStructures.hpp"
#include "../BlockwiseAlgorithm.hpp"
#include "ann/AlignedMemory/DynamicArray.hpp"

namespace nnd{

template<typename DistType, typename COMExtent>
struct old_thread_functors{
    erased_binary_binder<DistType> dispatchFunctor;
    cache_state<DistType> cache;
    erased_unary_binder<COMExtent> comDistFunctor;

    old_thread_functors() = default;

    old_thread_functors(const old_thread_functors&) = default;

    old_thread_functors& operator=(const old_thread_functors&) = default;

    template<typename DistanceFunctor, typename COMFunctor>
    old_thread_functors(DistanceFunctor distanceFunctor, COMFunctor comFunctor, size_t maxBlockSize, size_t numNeighbors):
        dispatchFunctor(distanceFunctor),
        cache(maxBlockSize, numNeighbors),
        comDistFunctor(comFunctor) {};

};

template<typename DistType>
struct thread_functors{
    erased_binary_binder<DistType> dispatchFunctor;
    cache_state<DistType> cache;

    thread_functors() = default;

    thread_functors(const thread_functors&) = default;

    thread_functors& operator=(const thread_functors&) = default;

    template<typename DistanceFunctor>
    thread_functors(DistanceFunctor distanceFunctor, size_t maxBlockSize, size_t numNeighbors):
        dispatchFunctor(distanceFunctor),
        cache(maxBlockSize, numNeighbors){};

};

template<typename DistType>
struct BlocksAndState{
    ann::dynamic_array<BlockUpdateContext<DistType>> blocks;
    std::unique_ptr<std::atomic<bool>[]> isReady;
};


}

#endif