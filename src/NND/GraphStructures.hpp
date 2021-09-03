/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_GRAPHSTRUCTURES_HPP
#define NND_GRAPHSTRUCTURES_HPP
#include <vector>
#include <valarray>
#include <algorithm>
#include <functional>
#include <ranges>
#include <limits>
#include <numeric>
#include <execution>
#include <array>
#include <utility>

#include "UtilityFunctions.hpp"

#include "RNG.hpp"
#include "Type.hpp"

#include "../Utilities/Type.hpp"
#include "../Utilities/Data.hpp"
#include "../Utilities/Metrics/FunctorErasure.hpp"

#include "Utilities/DataDeserialization.hpp"

namespace nnd{

template<TriviallyCopyable IndexType, typename FloatType>
struct GraphVertex{

    using iterator = typename std::vector<std::pair<IndexType, FloatType>>::iterator;
    using const_iterator = typename std::vector<std::pair<IndexType, FloatType>>::const_iterator;
    std::vector<std::pair<IndexType, FloatType>> neighbors;
    //std::vector<size_t> reverseNeighbor;

    GraphVertex(): neighbors(0){};

    GraphVertex(size_t numNeighbors): neighbors(0) {
        this->neighbors.reserve(numNeighbors + 1);
    };

    //GraphVertex(GraphVertex&& rval): neighbors(std::forward<std::vector<std::pair<IndexType, FloatType>>>(rval.neighbors)){};
    //Incorporate size checking in here?
    bool PushNeighbor(std::pair<IndexType, FloatType> newNeighbor){
        if (newNeighbor.second > neighbors[0].second) return false;
        neighbors.push_back(newNeighbor);
        std::push_heap(neighbors.begin(), neighbors.end(), NeighborDistanceComparison<IndexType, FloatType>);
        std::pop_heap(neighbors.begin(), neighbors.end(), NeighborDistanceComparison<IndexType, FloatType>);
        neighbors.pop_back();
        return true;
    };

    void JoinPrep(){
        std::make_heap(neighbors.begin(), neighbors.end(), NeighborDistanceComparison<IndexType, FloatType>);
    }

    void UnPrep(){
        std::sort_heap(neighbors.begin(), neighbors.end(), NeighborDistanceComparison<IndexType, FloatType>);
    }
    
    //Object Composition stuff below here

    constexpr void pop_back(){
        neighbors.pop_back();
    }

    constexpr void push_back(const std::pair<IndexType, FloatType>& value){
        neighbors.push_back(value);
    }

    //template<typename PairReferenceType>
    constexpr void push_back(std::pair<IndexType, FloatType>&& value){
        neighbors.push_back(std::forward<std::pair<IndexType, FloatType>>(value));
    }

    constexpr std::pair<IndexType, FloatType>& operator[](size_t i){
        return neighbors[i];
    }

    constexpr const std::pair<IndexType, FloatType>& operator[](size_t i) const{
        return neighbors[i];
    }

    constexpr std::pair<IndexType, FloatType>& operator[](BlockIndecies i){
        // I'm assuming the block number is correct
        return neighbors[i.dataIndex];
    }

    constexpr const std::pair<IndexType, FloatType>& operator[](BlockIndecies i) const{
        // I'm assuming the block number is correct
        return neighbors[i.dataIndex];
    }

    size_t size() const noexcept{
        return neighbors.size();
    }
    
    constexpr iterator begin() noexcept{
        return neighbors.begin();
    }

    constexpr const_iterator begin() const noexcept{
        return neighbors.begin();
    }

    constexpr const_iterator cbegin() const noexcept{
        return neighbors.cbegin();
    }

    constexpr iterator end() noexcept{
        return neighbors.end();
    }

    constexpr const_iterator end() const noexcept{
        return neighbors.end();
    }

    constexpr const_iterator cend() const noexcept{
        return neighbors.cend();
    }

    constexpr void resize(size_t count){
        neighbors.resize(count);
    }

    constexpr iterator erase(const_iterator first, const_iterator last) {
        return neighbors.erase(first, last);
    }

    //private:
    
};

//Rewrite as stream operator?
template<typename DistType>
unsigned int ConsumeVertex(GraphVertex<BlockIndecies, DistType>& consumer, GraphVertex<BlockIndecies, DistType>& consumee){
    std::sort(consumee.begin(), consumee.end(), NeighborDistanceComparison<BlockIndecies, DistType>);
    unsigned int neighborsAdded(0);
    for (auto& pair: consumee){
        if (pair.second >= consumer.neighbors[0].second) return neighborsAdded;
        consumer.PushNeighbor(pair);
        neighborsAdded++;
    }
    return neighborsAdded;
}



template<TriviallyCopyable OtherIndex, typename OtherDist, typename ConsumerDist>
unsigned int ConsumeVertex(GraphVertex<BlockIndecies, ConsumerDist>& consumer, GraphVertex<OtherIndex, OtherDist>& consumee, size_t consumeeBlockNum){
    std::sort(consumee.begin(), consumee.end(), NeighborDistanceComparison<OtherIndex, OtherDist>);
    unsigned int neighborsAdded(0);
    for (auto& pair: consumee){
        if (pair.second >= consumer.neighbors[0].second) return neighborsAdded;
        consumer.PushNeighbor({{consumeeBlockNum, pair.first}, static_cast<ConsumerDist>(pair.second)});
        neighborsAdded++;
    }
    return neighborsAdded;
}


//Prototype

struct CacheLineVertex{

    using iterator = std::pair<uint32_t, float>*;
    using const_iterator = const std::pair<uint32_t, float>*;
    // The 8th spot is a sink value, only holds 7 neighbors
    private:
    std::pair<uint32_t, float> neighbors[8];
    //std::vector<size_t> reverseNeighbor;

    public:
    constexpr CacheLineVertex(std::array<std::pair<uint32_t, float>, 7> initNeighbors): neighbors{
        initNeighbors[0],
        initNeighbors[1],
        initNeighbors[2],
        initNeighbors[3],
        initNeighbors[4],
        initNeighbors[5],
        initNeighbors[6],
        {0, std::numeric_limits<float>::max()}
    }{};
    
    //Branchless insertion
    void PushNeighbor(std::pair<uint32_t, float> newNeighbor){
        auto transformFunc = [=](std::pair<uint32_t, float> operand)->uint32_t{
            return uint32_t(newNeighbor.second > operand.second);
        };
        int pos = std::transform_reduce(std::execution::unseq,
                                        &neighbors[0],
                                        &neighbors[7],
                                        uint32_t(0),
                                        std::plus<uint32_t>(),
                                        transformFunc);
        
        std::copy_backward(&neighbors[pos], &neighbors[7], &neighbors[8]);
        neighbors[pos] = newNeighbor;
//<std::execution::unsequenced_policy, iterator, uint32_t, std::plus<uint32_t>, transformFunc>
    };

    std::pair<uint32_t, float>& operator[](size_t i){
        return neighbors[i];
    }

    std::pair<uint32_t, float>& operator[](BlockIndecies i){
        // I'm assuming the block number is correct
        return neighbors[i.dataIndex];
    }

    constexpr size_t size(){
        return 7;
    }
    
    constexpr iterator begin() noexcept{
        return &neighbors[0];
    }

    constexpr const_iterator begin() const noexcept{
        return &neighbors[0];
    }

    constexpr const_iterator cbegin() const noexcept{
        return &neighbors[0];
    }

    constexpr iterator end() noexcept{
        return &neighbors[7];
    }

    constexpr const_iterator end() const noexcept{
        return &neighbors[7];
    }

    constexpr const_iterator cend() const noexcept{
        return &neighbors[7];
    }

    //private:
    
};


//Thin layer over std::vector<GraphVertex<IndexType, FloatType>> to help deal with
//Static polymorphism via templates
template<TriviallyCopyable IndexType, typename FloatType>
struct Graph{

    using iterator = typename std::vector<GraphVertex<IndexType, FloatType>>::iterator;
    using const_iterator = typename std::vector<GraphVertex<IndexType, FloatType>>::const_iterator;

    std::vector<GraphVertex<IndexType, FloatType>> verticies;

    Graph(): verticies(){};

    Graph(size_t numVerticies, size_t numNeighbors): 
        verticies(numVerticies){
            for (auto& vertex: verticies){
                vertex.neighbors.reserve(numNeighbors+1);
            }
        };

    //Graph(Graph&&) = default;

    //Graph& operator=(Graph&&) = default; 	

    Graph(const Graph& otherGraph): verticies(otherGraph.size(), GraphVertex<IndexType, FloatType>(otherGraph[0].size())){
        for (size_t i = 0; const auto& vertex: otherGraph){
            for (const auto& neighbor: vertex){
                verticies[i].push_back(neighbor);
            }
            i++;
        }
    }

    Graph& operator=(Graph other){
        verticies = std::move(other.verticies);
        return *this;
    }

    GraphVertex<IndexType, FloatType>& operator[](size_t i){
        return verticies[i];
    }

    GraphVertex<IndexType, FloatType>& operator[](BlockIndecies i){
        // I'm assuming the block number is correct
        return verticies[i.dataIndex];
    }

    constexpr const GraphVertex<IndexType, FloatType>& operator[](size_t i) const{
        return this->verticies[i];
    }

    constexpr const GraphVertex<IndexType, FloatType>& operator[](BlockIndecies i) const{
        return this->verticies[i.dataIndex];
    }

    constexpr void push_back(const GraphVertex<IndexType, FloatType>& value){
        verticies.push_back(value);
    }

    template<typename VertexReferenceType>
    constexpr void push_back(GraphVertex<IndexType, FloatType>&& value){
        verticies.push_back(std::forward<VertexReferenceType>(value));
    }

    size_t size() const noexcept{
        return verticies.size();
    }
    
    constexpr iterator begin() noexcept{
        return verticies.begin();
    }

    constexpr const_iterator begin() const noexcept{
        return verticies.begin();
    }

    constexpr const_iterator cbegin() const noexcept{
        return verticies.cbegin();
    }

    constexpr iterator end() noexcept{
        return verticies.end();
    }

    constexpr const_iterator end() const noexcept{
        return verticies.end();
    }

    constexpr const_iterator cend() const noexcept{
        return verticies.cend();
    }
};

template<typename BlockNumberType, typename DataIndexType, typename DistType>
Graph<BlockIndecies, DistType> ToBlockIndecies(const Graph<DataIndexType, DistType>& blockGraph, const BlockNumberType blockNum){
    Graph<BlockIndecies, DistType> newGraph(blockGraph.size(), blockGraph[0].size());
    for (size_t j = 0; const auto& vertex: blockGraph){
        newGraph[j].resize(blockGraph[j].size());
        for(size_t k = 0; const auto& neighbor: vertex){
            newGraph[j][k] = {{blockNum, neighbor.first}, neighbor.second};
            k++;
        }   
        j++;
    }
    return newGraph;
}


template<TriviallyCopyable IndexType>
struct UndirectedGraph{

    using iterator = typename std::vector<std::vector<IndexType>>::iterator;
    using const_iterator = typename std::vector<std::vector<IndexType>>::const_iterator;

    std::vector<std::vector<IndexType>> verticies;

    UndirectedGraph(): verticies(){};

    UndirectedGraph(size_t numVerticies, size_t numNeighbors): 
        verticies(numVerticies, std::vector<IndexType>(numNeighbors)){};

    template<typename DistType>
    UndirectedGraph(Graph<IndexType, DistType> directedGraph): verticies(directedGraph.size()){
        
        for (size_t i = 0; auto& vertex: directedGraph){
            vertex.neighbors.reserve(vertex.size()*2);
            for (const auto& neighbor: vertex){
                if(std::find_if(directedGraph[neighbor.first].begin(), directedGraph[neighbor.first].end(), NeighborSearchFunctor<IndexType, DistType>(i)) == directedGraph[neighbor.first].end()) 
                    directedGraph[neighbor.first].push_back({i, neighbor.second});
            }
            i++;
        }
        
        for (size_t i = 0; auto& vertex: directedGraph){
            std::ranges::sort(vertex, NeighborDistanceComparison<IndexType, DistType>);
            verticies[i].reserve(vertex.size());
            for (const auto& neighbor:vertex){
                verticies[i].push_back(neighbor.first);
            }

            i++;
        }
        /*
        for (size_t i = 0; const auto& vertex: directedGraph){
            for (const auto& neighbor: vertex){
                if(std::ranges::find(verticies[i], neighbor.first) == verticies[i].end()) verticies[i].push_back(neighbor.first);
                if(std::ranges::find(verticies[neighbor.first], i) == verticies[neighbor.first].end()) verticies[neighbor.first].push_back(i);
            }
            i++;
        }
        */
    }

    std::vector<IndexType>& operator[](size_t i){
        return verticies[i];
    }

    std::vector<IndexType>& operator[](BlockIndecies i){
        // I'm assuming the block number is correct
        return verticies[i.dataIndex];
    }

    constexpr const std::vector<IndexType>& operator[](size_t i) const{
        return this->verticies[i];
    }

    constexpr const std::vector<IndexType>& operator[](BlockIndecies i) const{
        return this->verticies[i.dataIndex];
    }

    constexpr void push_back(const std::vector<IndexType>& value){
        verticies.push_back(value);
    }

    template<typename VertexReferenceType>
    constexpr void push_back(std::vector<IndexType>&& value){
        verticies.push_back(std::forward<VertexReferenceType>(value));
    }

    size_t size() const noexcept{
        return verticies.size();
    }
    
    constexpr iterator begin() noexcept{
        return verticies.begin();
    }

    constexpr const_iterator begin() const noexcept{
        return verticies.begin();
    }

    constexpr const_iterator cbegin() const noexcept{
        return verticies.cbegin();
    }

    constexpr iterator end() noexcept{
        return verticies.end();
    }

    constexpr const_iterator end() const noexcept{
        return verticies.end();
    }

    constexpr const_iterator cend() const noexcept{
        return verticies.cend();
    }
};






template<typename DistType>
//numNeighbors, blockSize, distanceFunctor
Graph<size_t, DistType> BruteForceBlock(const size_t numNeighbors, const size_t blockSize, DispatchFunctor<DistType>& distanceFunctor){
    Graph<size_t, DistType> retGraph(blockSize, numNeighbors);
    // I can make this branchless. Check to see if /O2 or /O3 can make this branchless (I really doubt it)
    for (size_t i = 0; i < blockSize; i += 1){
        for (size_t j = i+1; j < blockSize; j += 1){
            DistType distance = distanceFunctor(i, j);
            if (retGraph[i].size() < numNeighbors){
                retGraph[i].push_back(std::pair<size_t, DistType>(static_cast<size_t>(j), distance));
                if (retGraph[i].size() == numNeighbors){
                    retGraph[i].JoinPrep();
                }
            } else if (distance < retGraph[i][0].second){
                retGraph[i].PushNeighbor(std::pair<size_t, DistType>(static_cast<size_t>(j), distance));
            }
            if (retGraph[j].size() < numNeighbors){
                retGraph[j].push_back(std::pair<size_t, DistType>(static_cast<size_t>(i), distance));
                if (retGraph[j].size() == numNeighbors){
                    retGraph[j].JoinPrep();
                }
            } else if (distance < retGraph[j].neighbors[0].second){
                retGraph[j].PushNeighbor(std::pair<size_t, DistType>(static_cast<size_t>(i), distance));
            }
        }
    }

    return retGraph;
}

template<typename DistType>
struct CachingFunctor{

    DispatchFunctor<DistType> metricFunctor;
    //DistanceCache<DistType> cache;
    Graph<size_t, DistType> reverseGraph;
    std::vector<NodeTracker> nodesJoined;
    //std::vector<DistType> minDists;
    size_t numNeighbors;
    size_t maxBlockSize;

    CachingFunctor(DispatchFunctor<DistType>& metricFunctor, size_t maxBlockSize, size_t numNeighbors):
        metricFunctor(metricFunctor), 
        reverseGraph(maxBlockSize, numNeighbors),
        numNeighbors(numNeighbors),
        nodesJoined(maxBlockSize, NodeTracker(maxBlockSize)),
        maxBlockSize(maxBlockSize){
        //cache.reserve(cacheSize);
    }

    CachingFunctor() = default;

    CachingFunctor(const CachingFunctor&) = default;

    CachingFunctor& operator= (const CachingFunctor&) = default;

    DistType operator()(const size_t queryIndex, const size_t targetIndex){
        DistType distance = this->metricFunctor(queryIndex, targetIndex);
        cachedGraphSize = std::max(targetIndex, cachedGraphSize);
        //minDists[targetIndex] = std::min(minDists[targetIndex], distance);
        if(reverseGraph[targetIndex].size() == numNeighbors){
            reverseGraph[targetIndex].PushNeighbor({queryIndex, distance});
        } else if(reverseGraph[targetIndex].size() == numNeighbors-1){
            reverseGraph[targetIndex].push_back({queryIndex, distance});
            reverseGraph[targetIndex].JoinPrep();
        } else{
            reverseGraph[targetIndex].push_back({queryIndex, distance});
        }
        
        nodesJoined[targetIndex][queryIndex] = true;
        
        return distance;
    };

    std::vector<DistType> operator()(const size_t queryIndex, const std::vector<size_t>& targetIndecies){
        std::vector<DistType> distances = this->metricFunctor(queryIndex, targetIndecies);
        for (size_t i = 0; i<targetIndecies.size(); i+=1){
            cachedGraphSize = std::max(targetIndecies[i], cachedGraphSize);
            if(reverseGraph[targetIndecies[i]].size() == numNeighbors){
                reverseGraph[targetIndecies[i]].PushNeighbor({queryIndex, distances[i]});
            } else if(reverseGraph[targetIndecies[i]].size() == numNeighbors-1){
                reverseGraph[targetIndecies[i]].push_back({queryIndex, distances[i]});
                reverseGraph[targetIndecies[i]].JoinPrep();
            } else{
                reverseGraph[targetIndecies[i]].push_back({queryIndex, distances[i]});
            }
            nodesJoined[targetIndecies[i]][queryIndex] = true;
        }
        return distances;
    };

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum){
        cachedGraphSize = 0;
        for (auto& vertex: reverseGraph){
            vertex.resize(0);
        }
        for (auto& tracker: nodesJoined){
            tracker = NodeTracker(maxBlockSize);
        }
        //for (auto& minDist: minDists){
        //    minDist = std::numeric_limits<DistType>::max();
        //}
        this->metricFunctor.SetBlocks(lhsBlockNum, rhsBlockNum);
    }

    using iterator = typename std::vector<GraphVertex<size_t, DistType>>::iterator;
    using const_iterator = typename std::vector<GraphVertex<size_t, DistType>>::const_iterator;

    size_t size() const noexcept{
        return cachedGraphSize + 1;
    }
    
    constexpr iterator begin() noexcept{
        return reverseGraph.begin();
    }

    constexpr const_iterator begin() const noexcept{
        return reverseGraph.begin();
    }

    constexpr const_iterator cbegin() const noexcept{
        return reverseGraph.cbegin();
    }

    constexpr iterator end() noexcept{
        return reverseGraph.begin() + cachedGraphSize + 1;
    }

    constexpr const_iterator end() const noexcept{
        return reverseGraph.begin() + cachedGraphSize + 1;
    }

    constexpr const_iterator cend() const noexcept{
        return reverseGraph.begin() + cachedGraphSize + 1;
    }

    private:
    size_t cachedGraphSize;
};



/*
Queue that accepts up to a maxmimum number of elements
Once the max is reached, new elements are added based on the number of total elements
That have been attempted to be added since the last time the queue was flushed
Currently, storing the index of the pointA neighbor is redundant, but the plan is to
associate a queue for blocks of data, not individual vertecies.
*/
template<typename IndexType>
struct ComparisonQueue{

    std::vector<IndexType> queue;
    size_t queueMaxLength;
    size_t queueWeight;
    typename std::vector<IndexType>::iterator ringIterator;
    StlRngFunctor<std::mt19937_64, std::uniform_real_distribution, float> rngFunctor;

    ComparisonQueue():
        queue(0),
        queueMaxLength(0),
        queueWeight(0),
        rngFunctor(std::mt19937_64(0), std::uniform_real_distribution<float>()){
            ringIterator = queue.begin();
        }

    ComparisonQueue(size_t maxLength):
        queue(0),
        queueMaxLength(maxLength),
        queueWeight(0),
        rngFunctor(std::mt19937_64(0), std::uniform_real_distribution<float>()){
            queue.reserve(maxLength);
            ringIterator = queue.begin();
        }
    
    ComparisonQueue(ComparisonQueue&& rhsQueue):
        queue(std::move(rhsQueue.queue)),
        queueMaxLength(rhsQueue.queueMaxLength),
        queueWeight(rhsQueue.queueWeight),
        rngFunctor(std::move(rhsQueue.rngFunctor)),
        ringIterator(std::move(rhsQueue.ringIterator)){};

    //ComparisonQueue& operator=(const ComparisonQueue& rhs)

    void PushQueue(const IndexType& indecies){
        queueWeight+=1;
        if (queue.size() < queueMaxLength){
            queue.push_back(indecies);
            return;
        } else {
            /*
            size_t rand = size_t(rngFunctor()*queueWeight);
            if (rand<queueMaxLength) queue[rand] = indecies;
            return;
            */
            *ringIterator = indecies;
            ringIterator++;
            if (ringIterator == queue.end()) ringIterator = queue.begin();
            return;
        }
    }

    void FlushQueue(){
        queue.clear();
        queueWeight = 0;
        ringIterator = queue.begin();
    }

    size_t size(){
        return queue.size();
    }
};

template<typename IndexType>
std::vector<ComparisonQueue<IndexType>> ConstructQueues(size_t numQueues, size_t queueMax){

    std::vector<ComparisonQueue<IndexType>> retQueues(0);
    retQueues.reserve(numQueues);
    //ComparisonQueue copyQueue(queueMax);
    //retQueues.insert(retQueues.begin(), {queueMax});
    //retQueues.push_back(std::move(copyQueue));
    for (size_t i = 0; i < numQueues; i +=1){
        retQueues.emplace_back(queueMax);
    }
    return retQueues;
}

//Mainly For Debugging to make sure I didn't screw up my graph state.
template<typename FloatType>
void VerifyGraphState(const Graph<size_t, FloatType>& currentGraph){
    for (const auto& vertex : currentGraph){
        for (const auto& neighbor : vertex.neighbors){
            if (neighbor.first == vertex.dataIndex) throw("Vertex is own neighbor");
            for (const auto& neighbor1 : vertex.neighbors){
                if (&neighbor == &neighbor1) continue;
                if (neighbor.first == neighbor.second) throw("Duplicate neighbor in heap");
            }
        }
    }
}

template<typename FloatType>
void VerifySubGraphState(const Graph<BlockIndecies, FloatType>& currentGraph, size_t blockNum){
    for (size_t i = 0; i<currentGraph.size(); i+=1){
        const GraphVertex<BlockIndecies, FloatType>& vertex = currentGraph[i];
        for (const auto& neighbor : vertex.neighbors){
            if (neighbor.first == BlockIndecies{blockNum, i}) throw("Vertex is own neighbor");
            for (const auto& neighbor1 : vertex.neighbors){
                if (&neighbor == &neighbor1) continue;
                if (neighbor.first == neighbor1.first) throw("Duplicate neighbor in heap");
            }
        }
    }
}


}
#endif //DATASTRUCTURES