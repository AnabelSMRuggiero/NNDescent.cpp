/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_SUBGRAPHQUERY_HPP
#define NND_SUBGRAPHQUERY_HPP

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory_resource>
#include <optional>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ann/DelayConstruct.hpp"
#include "NND/GraphStructures/UndirectedGraph.hpp"
#include "ann/Data.hpp"
#include "ann/DataDeserialization.hpp"
#include "ann/DataSerialization.hpp"
#include "ann/AlignedMemory/DynamicArray.hpp"

#include "NND/GraphStructures/GraphVertex.hpp"
#include "RNG.hpp"
#include "FunctorErasure.hpp"
#include "GraphStructures.hpp"
#include "MemoryInternals.hpp"
#include "NND/Type.hpp"

namespace nnd {

template<TriviallyCopyable DataIndexType, typename DataEntry, typename DistType>
struct SubProblemData {

    const Graph<DataIndexType, DistType>& subGraph;
    const DataBlock<DataEntry>& dataBlock;

    SubProblemData(const Graph<DataIndexType, DistType>& subGraph, const DataBlock<DataEntry>& dataBlock)
        : subGraph(subGraph), dataBlock(dataBlock){};
};

template<TriviallyCopyable IndexType, typename DataEntry, std::floating_point FloatType>
std::tuple<size_t, size_t, FloatType> BruteNearestNodes(
    const Graph<IndexType, FloatType>& subGraphA, const DataBlock<DataEntry>& dataBlockA, const Graph<IndexType, FloatType>& subGraphB,
    const DataBlock<DataEntry>& dataBlockB, erased_metric<FloatType> distanceFunctor) {

    std::pair<size_t, size_t> bestPair;
    FloatType bestDistance(std::numeric_limits<FloatType>::max());

    for (size_t i = 0; i < subGraphA.dataBlock.size(); i += 1) {
        for (size_t j = 0; j < subGraphB.dataBlock.size(); j += 1) {
            FloatType distance = distanceFunctor(subGraphA.dataBlock[i], subGraphB.dataBlock[j]);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestPair = std::pair<size_t, size_t>(i, j);
            }
        }
    }

    return { bestPair.first, bestPair.second, bestDistance };
}

template<std::floating_point COMExtent, typename DistType, typename IndexType, typename Functor>
GraphVertex<IndexType, COMExtent> QueryCOMNeighbors(
    const size_t pointIndex, const Graph<IndexType, DistType>& subProb, const size_t numCandidates, Functor& distanceFunctor) {

    GraphVertex<IndexType, COMExtent> COMneighbors(numCandidates);
    // auto problemView = subProb.GetOffsetView(indexOffset);
    // Just gonna dummy it and select the first few nodes. Since the splitting process is randomized, this is a totally random selection,
    // right? /s
    NodeTracker nodesVisited(subProb.size());
    for (size_t i = 0; i < numCandidates; i += 1) {
        COMneighbors.push_back(std::pair<IndexType, COMExtent>(i, distanceFunctor(pointIndex, i)));
        nodesVisited[i] = true;
    }
    COMneighbors.JoinPrep();
    // std::make_heap(COMneighbors.begin(), COMneighbors.neighbors.end(), NeighborDistanceComparison<size_t, COMExtent>);

    bool breakVar = false;
    GraphVertex<IndexType, COMExtent> newState(COMneighbors);
    while (!breakVar) {
        breakVar = true;
        for (const auto& curCandidate : COMneighbors) {
            for (const auto& joinTarget : subProb[curCandidate.first]) {
                if (nodesVisited[joinTarget.first]) continue;
                nodesVisited[joinTarget.first] = true;
                COMExtent distance = distanceFunctor(pointIndex, joinTarget.first);
                if (distance < newState.PushThreshold()) {
                    newState.PushNeighbor({ joinTarget.first, distance });
                    breakVar = false;
                }
            }
        }

        COMneighbors = newState;
    }

    return COMneighbors;
}

/*
template<typename DistType>
struct QueryPoint{
    const GraphVertex<size_t, DistType>& queryHint;
    const size_t dataIndex;
    QueryPoint(const GraphVertex<size_t, DistType>& hint, const size_t index): queryHint(hint), dataIndex(index){}
};
*/

template<typename DistType, typename COMExtent, typename IndexType>
GraphVertex<IndexType, DistType> QueryHintFromCOM(
    const size_t metaPointIndex, const Graph<IndexType, DistType>& subProb, const size_t numCandidates,
    erased_metric<COMExtent> distanceFunctor) {
    GraphVertex<IndexType, COMExtent> comNeighbors = QueryCOMNeighbors<COMExtent>(metaPointIndex, subProb, numCandidates, distanceFunctor);
    if constexpr (std::is_same<DistType, COMExtent>()) {
        for (auto& neighbor : comNeighbors) {
            neighbor.second = std::numeric_limits<DistType>::max();
        }
        return comNeighbors;
    } else {
        GraphVertex<IndexType, DistType> retHint;
        for (auto& hint : comNeighbors) {
            // This should be an emplace_back
            retHint.push_back({ hint.first, std::numeric_limits<DistType>::max() });
        }
        return retHint;
    }
}


/*
template<typename DistType>
GraphVertex<DataIndex_t, DistType> DispersedQueryHint(erased_metric<DistType> distance_functor, std::size_t block_size, std::size_t num_candidates) {
    
    stack_fed_buffer<4096> stack_buffer;
    std::pmr::polymorphic_allocator<> allocator{stack_buffer};

    using distance_cache = std::pmr::unordered_map<comparison_key<DataIndex_t>, DistType>;

    distance_cache& cache = *allocator.new_object<distance_cache>();

    auto distance = [&](DataIndex_t lhs, DataIndex_t rhs)->DistType{
        auto cache_itr = cache.find({lhs, rhs});
        if(cache_itr != cache.end()){
            return *cache_itr;
        } else{
            return cache[{lhs, rhs}] = distance_functor(lhs, rhs);
        }
    };

    ann::pmr::dynamic_array<DataIndex_t>& indecies = *allocator.new_object<ann::pmr::dynamic_array<DataIndex_t>>(num_candidates);
    
}
*/
template<typename IndexType, typename DistType>
GraphVertex<IndexType, DistType> RandomQueryHint(std::size_t block_size, std::size_t num_candidates) {
    
    stack_fed_buffer<4096> stack_buffer;
    std::pmr::polymorphic_allocator<> allocator{stack_buffer};
    
    std::pmr::unordered_set<IndexType>& selected_indecies = *allocator.new_object<std::pmr::unordered_set<IndexType>>();

    RngFunctor rng{0, block_size-1};

    while (selected_indecies.size()<num_candidates){
        selected_indecies.insert(rng());
    }

    GraphVertex<IndexType, DistType> result{num_candidates};
    
    for (const auto& index : selected_indecies){
        result.push_back({index, std::numeric_limits<DistType>::max()});
    }

    return result;
}

/*
struct get_arg {
    std::size_t& number_to_generate;
    constexpr bool await_ready() const noexcept {return false;} //Call await_suspend

    template<typename PromiseType>
    bool await_suspend(std::coroutine_handle<PromiseType> h) {
        return true; //Don't suspend; call await_resume()
    }
    std::size_t await_resume() const noexcept {return number_to_generate;}
};

struct query_generator{
    struct promise_type{
        std::vector<int> return_value;
        std::size_t number_to_generate;
        ~promise_type(){
            //std::cout << "promise_type destroyed" << std::endl;
        }
        query_generator get_return_object(){ return {std::coroutine_handle<promise_type>::from_promise(*this)};}
        std::suspend_never initial_suspend() {return {};}
        std::suspend_always final_suspend() noexcept {return {};}
        void unhandled_exception() {}

        
        get_arg yield_value(std::vector<int>&& geneterated_vector){
            return_value = std::move(geneterated_vector);
            return get_arg{number_to_generate};
        }
        template<typename T>
        get_arg await_transform(T&&) { return get_arg{number_to_generate};}

    };

    std::vector<int> operator()(std::size_t number_of_elements){
        
        if (h.done()) throw "Something";
        
        h.promise().number_to_generate = number_of_elements;
        h();
        return std::move(h.promise().return_value);
    }

    std::coroutine_handle<promise_type> h;
};

query_generator make_vectors(){
    std::size_t vector_size = co_await 1;
    std::mt19937 rng_eng{};
    std::uniform_int_distribution<int> dist;
    while(true){
        std::vector<int> coret_vec(vector_size, 0);
        for (auto& element : coret_vec){
            element = dist(rng_eng);
        }
        vector_size = co_yield std::move(coret_vec);
    }
}


some_promise generate_queries( NodeTrackerImpl<TrackerAlloc>& nodesVisited) {

        constexpr size_t bufferSize = sizeof(size_t) * (internal::maxBatch + 5);
        char stackBuffer[bufferSize];
        std::pmr::monotonic_buffer_resource stackResource(stackBuffer, bufferSize);

        std::pmr::vector<size_t> joinQueue(&stackResource);
        constexpr size_t maxBatch = internal::maxBatch; // Compile time constant
        joinQueue.reserve(maxBatch);

        NodeTrackerImpl<std::pmr::polymorphic_allocator<bool>> nodesCompared(blockSize, internal::GetThreadResource());

        auto notVisited = [&](const auto index) { return !nodesVisited[index]; };

        auto notCompared = [&](const auto index) -> bool {
            return (nodesCompared[index]) ? false
                                          : !(nodesCompared[index] = std::none_of(
                                                  std::make_reverse_iterator(subGraph[index].end()),
                                                  std::make_reverse_iterator(subGraph[index].begin()),
                                                  notVisited));
        };
        auto toNeighborView = [&](const auto index) { return subGraph[index]; }; // returns a view into a data block

        auto toNeighbor = [&](const auto edge) { return edge.first; };

        auto nodes_to_compare = std::span{ initVertex.cbegin(), querySearchDepth } | std::views::reverse
                                                                                   | std::views::filter([&](const auto& index){ return !nodes_compared[index]});

        std::unordered_map<std::size_t, std::span<IndexType>> stored_views;
        auto itr = std::ranges::find_if(nodes_to_compare, [&](const auto& index){ return !nodes_compared[index]});
        if (itr != nodes_to_compare.end())
        bool breakVar = true;
        while (breakVar) {
            

            auto view = (stored_views.contains(*itr)) ? stored_views.at(*itr)
                                                      : stored_views[*itr] = subGraph[*itr];
            
            span

            joinQueue.resize(maxBatch); 
            auto [ignore, outItr] = std::ranges::transform(
                nodesToCompare | std::views::transform(toNeighbor) | std::views::filter(notCompared) | std::views::transform(toNeighborView)
                    | std::views::join | std::views::filter(notVisited) | std::views::take(maxBatch),
                joinQueue.begin(),
                [&](const auto joinTarget) {
                    nodesVisited[joinTarget] = true;
                    return joinTarget;
                });
            joinQueue.resize(outItr - joinQueue.begin());
    
*/

struct query_params{
    size_t size;
    size_t search_depth;
};

template<std::unsigned_integral IndexType, std::totally_ordered DistType>
struct QueryContext {

    UndirectedGraph<IndexType> subGraph;
    GraphVertex<IndexType, DistType> queryHint;
    GraphFragment_t graphFragment{ GraphFragment_t(-1) };
    BlockNumber_t blockNumber{ BlockNumber_t(-1) };
    size_t blockSize{ size_t(-1) };

    QueryContext() = default;

    QueryContext(
        const Graph<IndexType, DistType>& subGraph, GraphVertex<IndexType, DistType>&& queryHint,
        const size_t graphFragment, const size_t blockNumber, const size_t blockSize)
        :

          subGraph(BuildUndirectedGraph(subGraph)), queryHint(std::move(queryHint)),
          graphFragment(graphFragment), blockNumber(blockNumber), blockSize(blockSize) {
        // defaultQueryFunctor = DefaultQueryFunctor<IndexType, DataEntry, DistType>(distanceFunctor, dataBlock);
    };

    QueryContext(std::ifstream& inFile)
        : subGraph(Extract<UndirectedGraph<IndexType>>(inFile)),
          // subGraph(inFile),
          queryHint(Extract<GraphVertex<IndexType, DistType>>(inFile)), 
          graphFragment(Extract<GraphFragment_t>(inFile)),
          blockNumber(Extract<BlockNumber_t>(inFile)), blockSize(Extract<size_t>(inFile)) {}

    //QueryContext(QueryContext&&) = default;

    template<typename QueryFunctor>
    GraphVertex<IndexType, DistType, PolymorphicAllocator> Query(
        std::span<const IndexType> initHints,
        const size_t queryIndex, // Can realistically be any parameter passed through to the Functor
        query_params params,
        QueryFunctor&& queryFunctor) const {

        auto [vertex, nodeTracker] = ForwardQueryInit(initHints, queryIndex, params, queryFunctor);

        return std::move(QueryLoop(vertex, queryIndex, params, queryFunctor, nodeTracker));
    }


    template<typename QueryFunctor>
    GraphVertex<IndexType, DistType>& Query(
        GraphVertex<IndexType, DistType>& initVertex,
        size_t queryIndex, // Can realistically be any parameter passed through to the Functor
        query_params params,
        QueryFunctor& queryFunctor, NodeTracker& nodesVisited) const {

        if (initVertex.size() < params.size) {
            ReverseQueryInit(initVertex, queryIndex, params, queryFunctor, nodesVisited);
        }

        return QueryLoop(initVertex, queryIndex, params, queryFunctor, nodesVisited);
    }

    template<VertexLike<IndexType, DistType> Vertex, typename QueryFunctor, typename TrackerAlloc>
    Vertex& QueryLoop(
        Vertex& initVertex,
        const size_t queryIndex, // Can realistically be any parameter passed through to the Functor
        query_params params,
        QueryFunctor& queryFunctor, NodeTrackerImpl<TrackerAlloc>& nodesVisited) const {

        constexpr size_t bufferSize = sizeof(size_t) * (internal::maxBatch + 5);
        char stackBuffer[bufferSize];
        std::pmr::monotonic_buffer_resource stackResource(stackBuffer, bufferSize);

        std::pmr::vector<size_t> joinQueue(&stackResource);
        constexpr size_t maxBatch = internal::maxBatch; // Compile time constant
        joinQueue.reserve(maxBatch);

        NodeTrackerImpl<std::pmr::polymorphic_allocator<bool>> nodesCompared(blockSize, internal::GetThreadResource());

        auto notVisited = [&](const auto index) { return !nodesVisited[index]; };
        /*
        auto notCompared = [&](const auto index) -> bool {
            return (nodesCompared[index]) ? false
                                          : !(nodesCompared[index] = std::none_of(
                                                  std::make_reverse_iterator(subGraph[index].end()),
                                                  std::make_reverse_iterator(subGraph[index].begin()),
                                                  notVisited));
        };
        */
        auto toNeighborView = [&](const auto index) { return subGraph[index]; }; // returns a view into a data block

        auto toNeighbor = [&](const auto edge) { return edge.first; };

        /*
        auto visited = [&](const auto index) {
            bool alreadyVisited = nodesVisited[index];
            nodesVisited[index] = true;
            return alreadyVisited;
        };

        auto notCompared = [&](const auto index) -> bool {
            bool alreadyCompared = nodesCompared[index];
            nodesCompared[index] = true;
            return !alreadyCompared;
        };
        auto toNeighborView = [&](const auto index) { return subGraph[index]; }; // returns a view into a data block

        auto toNeighbor = [&](const auto edge) { return edge.first; };
        */
        std::span<const IndexType> nodesToCompare = initVertex.view_first().subspan(0, params.search_depth);
        
        //constexpr size_t bufferSize = sizeof(size_t) * (internal::maxBatch + 5);
        std::byte mapBuffer[5'000];
        std::pmr::monotonic_buffer_resource mapResource(mapBuffer, 5'000);
        std::pmr::polymorphic_allocator alloc{&mapResource};
        using SubGraphView = typename UndirectedGraph<IndexType>::const_reference;

        auto& stored_views = *alloc.new_object<std::pmr::unordered_map<IndexType, SubGraphView>>();
        auto notCompared = [&](const auto& edge) -> bool {
            if (stored_views.contains(edge)){
                return stored_views[edge].size() > 0;
            } else {
                stored_views[edge] = subGraph[edge];
                return true;
            }
        };
        
        auto getView = [&](const auto& edge) -> SubGraphView& {
            auto [iter, inserted] = stored_views.insert(
                std::pair{ edge, subGraph[edge]}
            );
            return iter->second;
        };

        bool breakVar = true;
        while (breakVar) {
            
            [&]{
            joinQueue.resize(maxBatch);
            auto beginItr = joinQueue.begin();
            auto endItr = joinQueue.end();
//            for(auto& candidates : nodesToCompare | std::views::filter(notCompared)
//                                                  | std::views::transform([&](const auto& edge)-> auto&{return stored_views[edge.first];})){
            for(auto& candidates : nodesToCompare | std::views::transform(getView)
                                                  | std::views::filter([](auto& span){return span.size() > 0;})){

                auto canBegin = candidates.begin();
                auto canEnd = candidates.end();
                canBegin = std::find_if(canBegin, canEnd, notVisited);
                while(canBegin != canEnd && beginItr != endItr){
                    *beginItr = *canBegin;
                    nodesVisited[*canBegin] = true;
                    ++beginItr;
                    canBegin = std::find_if(++canBegin, canEnd, notVisited);
                }
                candidates = {canBegin, canEnd};
                if (beginItr == endItr) break;
            }
            joinQueue.resize(beginItr - joinQueue.begin());
            }();
            
            //joinQueue.resize(outItr - joinQueue.begin());
            /*
            [&]{
            joinQueue.resize(maxBatch);
            auto [ignore, outItr] = std::ranges::transform(
                nodesToCompare | std::views::transform(toNeighbor) | std::views::filter(notCompared) | std::views::transform(toNeighborView)
                    | std::views::join | std::views::filter(notVisited) | std::views::take(maxBatch),
                joinQueue.begin(),
                [&](const auto joinTarget) {
                    nodesVisited[joinTarget] = true;
                    return joinTarget;
                });
            joinQueue.resize(outItr - joinQueue.begin());
            }();
            */
            /*
            std::span<const typename Vertex::EdgeType> nodesToPickFrom{ initVertex.begin(), params.search_depth };
            std::vector<IndexType> nodesToCompare(params.search_depth);
            // clang-format off
            auto [ignore, outItr] = std::ranges::copy(nodesToPickFrom | std::views::transform(toNeighbor)
                                                                      | std::views::filter(notCompared),
                                                      nodesToCompare.begin());
            // clang-format on
            nodesToCompare.resize(outItr - nodesToCompare.begin());
            size_t maxJoinListSize = 0;
            for (const auto &index : nodesToCompare) {
                maxJoinListSize += subGraph[index].size();
            }
            joinQueue.resize(maxJoinListSize);

            // clang-format off
            auto [ignore2, joinQueueItr] = std::ranges::copy(nodesToCompare | std::views::transform(toNeighborView)
                                                                            | std::views::join,
                                                             joinQueue.begin());
            // clang-format on
            joinQueue.resize(joinQueueItr - joinQueue.begin());
            std::erase_if(joinQueue, visited);

            //for (const auto &joinee : joinQueue) {
            //    nodesVisited[joinee] = true;
            //}
            */
            std::ranges::contiguous_range auto distances = queryFunctor(queryIndex, joinQueue);

            breakVar = std::transform_reduce(
                joinQueue.begin(),
                joinQueue.end(),
                distances.begin(),
                joinQueue.size() == maxBatch,
                std::logical_or<bool>{},
                [&](const size_t index, const DistType distance) {
                    return initVertex.PushNeighbor({ static_cast<IndexType>(index), distance });
                });
        }

        return initVertex;
    }

    template<typename QueryFunctor>
    auto ForwardQueryInit(
        std::span<const IndexType> startHint,
        const size_t queryIndex, // Can realistically be any parameter passed through to the Functor
        query_params params,
        QueryFunctor& queryFunctor) const
        -> std::pair<GraphVertex<IndexType, DistType, PolymorphicAllocator>, NodeTrackerImpl<std::pmr::polymorphic_allocator<bool>>> {

        NodeTrackerImpl<std::pmr::polymorphic_allocator<bool>> nodesJoined(blockSize, internal::GetThreadResource());
        for (const auto& hint : startHint){
            nodesJoined[hint] = true;
        }

        constexpr size_t bufferSize = sizeof(size_t) * (internal::maxBatch * 3);
        char stackBuffer[bufferSize];
        std::pmr::monotonic_buffer_resource stackResource(stackBuffer, bufferSize);

        std::pmr::vector<size_t> initDestinations(&stackResource);
        initDestinations.reserve(params.size);
        initDestinations.resize(startHint.size());

        if constexpr (std::is_same_v<IndexType, size_t>) {
            std::ranges::copy(startHint, initDestinations.begin());
        } else {
            std::ranges::transform(startHint, initDestinations.begin(), std::identity{});
        }

        auto notJoined = [&](const auto& index) { return !nodesJoined[index]; };

        auto padIndecies = [&](const auto& range) {
            for (const auto& index : range | std::views::filter(notJoined) | std::views::take(params.size - initDestinations.size())) {
                initDestinations.push_back(index);
                nodesJoined[index] = true;
            }
        };

        if (initDestinations.size() < params.size) {
            padIndecies(queryHint | std::views::transform([&](const auto& pair) { return pair.first; }));
            [[unlikely]] if (initDestinations.size() < params.size) padIndecies(std::views::iota(size_t{ 0 }, blockSize));
        }

        std::ranges::contiguous_range auto initDistances = queryFunctor(queryIndex, initDestinations);

        GraphVertex<IndexType, DistType, PolymorphicAllocator> retVertex(initDistances.size()); // This constructor merely reserves
        retVertex.resize(initDistances.size());

        std::ranges::transform(initDestinations, initDistances, retVertex.begin(), [&](const auto& index, const auto& distance) {
            return typename GraphVertex<IndexType, DistType, PolymorphicAllocator>::EdgeType{ index, distance };
        });
        retVertex.JoinPrep();

        return std::pair{ std::move(retVertex), nodesJoined };
    }

    template<typename QueryFunctor>
    void ReverseQueryInit(
        GraphVertex<IndexType, DistType>& initVertex,
        const size_t queryIndex, // Can realistically be any parameter passed through to the Functor
        query_params params,
        QueryFunctor& queryFunctor, NodeTracker& previousVisits) const {

        int sizeDif = params.size - initVertex.size();

        int indexOffset(0);
        std::vector<size_t> initComputations;
        for (int i = 0; i < sizeDif; i += 1) {
            while (previousVisits[queryHint[i + indexOffset].first]) {
                indexOffset += 1;
            }
            initComputations.push_back(queryHint[i + indexOffset].first);
            previousVisits[queryHint[i + indexOffset].first] = true;
        }

        std::ranges::contiguous_range auto initDistances = queryFunctor(queryIndex, initComputations);
        for (size_t i = 0; i < initDistances.size(); i += 1) {
            initVertex.push_back({ static_cast<IndexType>(initComputations[i]), initDistances[i] });
        }
        initVertex.JoinPrep();
    }

    template<typename QueryFunctor>
    std::tuple<IndexType, IndexType, DistType> NearestNodes(const QueryContext& rhs, QueryFunctor&& distanceFunctor) const {


        std::pair<IndexType, IndexType> bestPair;
        DistType bestDistance(std::numeric_limits<DistType>::max());
        // NodeTracker nodesVisitedA(subGraphA.dataBlock.size());
        // NodeTracker nodesVisitedB(subGraphB.dataBlock.size());

        for (const auto& starterA : this->queryHint) {
            // nodesVisitedA[starterA.first] = true;
            for (const auto& starterB : rhs.queryHint) {
                // nodesVisitedB[starterB.first] = true;
                DistType distance = distanceFunctor(starterA.first, starterB.first);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestPair = std::pair<IndexType, IndexType>(starterA.first, starterB.first);
                }
            }
        }

        bool breakVar = false;
        while (!breakVar) {
            breakVar = true;
            std::pair<IndexType, IndexType> tmpPair = bestPair;
            for (const auto& neighborA : this->subGraph[bestPair.first]) {
                // if (!nodesVisitedA[neighborA.first]){
                DistType distance = distanceFunctor(neighborA, tmpPair.second);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    tmpPair.first = neighborA;
                    breakVar = false;
                }
                // nodesVisitedA[neighborA.first] = true;
                //}

                for (const auto& neighborOfNeighborA : this->subGraph[neighborA]) {
                    // if (nodesVisitedA[neighborOfNeighborA.first]) continue;
                    // nodesVisitedA[neighborOfNeighborA.first] = true;
                    DistType distance = distanceFunctor(neighborOfNeighborA, tmpPair.second);
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        tmpPair.first = neighborOfNeighborA;
                        breakVar = false;
                    }
                }
            }
            for (const auto& neighborB : rhs.subGraph[bestPair.second]) {
                // if (!nodesVisitedB[neighborB.first]){
                DistType distance = distanceFunctor(tmpPair.first, neighborB);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    tmpPair.second = neighborB;
                    breakVar = false;
                }
                //  nodesVisitedB[neighborB.first] = true;
                //}
                for (const auto& neighborOfNeighborB : rhs.subGraph[neighborB]) {
                    // nodesVisitedB[neighborOfNeighborB.first] = true;
                    DistType distance = distanceFunctor(tmpPair.first, neighborOfNeighborB);
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        tmpPair.second = neighborOfNeighborB;
                        breakVar = false;
                    }
                }
            }
            bestPair = tmpPair;
        }

        return { bestPair.first, bestPair.second, bestDistance };
    }

    void serialize(std::ofstream& outFile) const {

        auto outputter = BindSerializer(outFile);

        outputter(this->subGraph.graphBlock);

        outputter(this->queryHint);
        

        outputter(this->graphFragment);
        outputter(this->blockNumber);
        outputter(this->blockSize);
    }
};

} // namespace nnd

#endif