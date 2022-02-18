/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_GRAPHVERTEX_HPP
#define NND_GRAPHVERTEX_HPP

#include <vector>
#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>

#include "../UtilityFunctions.hpp"
#include "../../ann/Type.hpp"
#include "../../ann/DataSerialization.hpp"
#include "../Type.hpp"

namespace nnd{

struct ReturnRemoved {};

static const ReturnRemoved returnRemovedTag;

//This might need to be more precisely constrained, but works for now 
template<typename Vertex, typename IndexType, typename DistType>
concept VertexLike = std::same_as<typename Vertex::Index_t, IndexType> && std::same_as<typename Vertex::Distance_t, DistType>;


template<TriviallyCopyable IndexType, TriviallyCopyable FloatType, template<typename> typename Alloc = std::allocator>
struct GraphVertex{
    using EdgeType = std::pair<IndexType, FloatType>;
    using Index_t = IndexType;
    using Distance_t = FloatType;
    using iterator = typename std::vector<std::pair<IndexType, FloatType>, Alloc<EdgeType>>::iterator;
    using const_iterator = typename std::vector<std::pair<IndexType, FloatType>, Alloc<EdgeType>>::const_iterator;

    using allocator_type = Alloc<EdgeType>;

    std::vector<std::pair<IndexType, FloatType>, Alloc<EdgeType>> neighbors;
    //std::vector<size_t> reverseNeighbor;

    GraphVertex(allocator_type allocator = allocator_type{}): neighbors(0, allocator){};

    GraphVertex(size_t numNeighbors, allocator_type allocator = allocator_type{}): neighbors(0, allocator) {
        this->neighbors.reserve(numNeighbors + 1);
    };
    
    GraphVertex(const GraphVertex&) = default;
    GraphVertex& operator=(const GraphVertex&) = default;
    
    //Explicit copy constructor for when allocators don't match
    template<VertexLike<IndexType, FloatType> Other>
        requires (is_not<GraphVertex, Other>)
    GraphVertex(const Other& other, allocator_type allocator = allocator_type{}): neighbors(other.size()+1, allocator) {
        neighbors.resize(other.size());
        std::ranges::copy(other, this->begin());
    }

    GraphVertex(const GraphVertex& other, allocator_type allocator): neighbors(other.size()+1, allocator) {
        neighbors.resize(other.size());
        std::ranges::copy(other, this->begin());
    }

    GraphVertex(GraphVertex&& other, allocator_type allocator): neighbors(std::move(other.neighbors), allocator) {
    }

    GraphVertex(GraphVertex&&) = default;
    GraphVertex& operator=(GraphVertex&&) = default;

    //GraphVertex()

    //GraphVertex(GraphVertex&& rval): neighbors(std::forward<std::vector<std::pair<IndexType, FloatType>>>(rval.neighbors)){};
    //Incorporate size checking in here?
    bool PushNeighbor(std::pair<IndexType, FloatType> newNeighbor){
        if (newNeighbor.second > neighbors.back().second) return false;
        
        size_t index = neighbors.size();
        for ( ; index>0; index -= 1){
            if (edge_ops::lessThan(neighbors[index-1], newNeighbor)) break;
        }
        
        neighbors.push_back(newNeighbor);
        std::memmove(&neighbors[index+1], &neighbors[index], sizeof(std::pair<IndexType, FloatType>)*(neighbors.size()-1 - index));
        neighbors[index] = newNeighbor;
        
        neighbors.pop_back();
        return true;
    };

    std::pair<IndexType, FloatType> PushNeighbor(std::pair<IndexType, FloatType> newNeighbor, ReturnRemoved){
        if (newNeighbor.second > neighbors.back().second) return newNeighbor;
        
        size_t index = neighbors.size();
        for ( ; index>0; index -= 1){
            if (edge_ops::lessThan(neighbors[index-1], newNeighbor)) break;
        }

        neighbors.push_back(newNeighbor);
        std::memmove(&neighbors[index+1], &neighbors[index], sizeof(std::pair<IndexType, FloatType>)*(neighbors.size()-1 - index));
        neighbors[index] = newNeighbor;

        std::pair<IndexType, FloatType> retValue = neighbors.back();
        neighbors.pop_back();

        return retValue;
    };

    void JoinPrep(){
        std::sort(neighbors.begin(), neighbors.end(), edge_ops::lessThan);
    }

    void UnPrep(){
        //std::sort_heap(neighbors.begin(), neighbors.end(), NeighborDistanceComparison<IndexType, FloatType>);
        return; //noop
    }
    
    FloatType PushThreshold() const noexcept{
        return neighbors.back().second;
    }
    
    size_t FindIndexBack(const FloatType dist) const noexcept{
        size_t index = neighbors.size();
        for ( ; index>0; index -= 1){
            if (neighbors[index-1].second < dist) break;
        }
        return index;
    }

    template<typename OtherIndex>
    size_t FindIndexBack(const std::pair<OtherIndex, FloatType>& neighbor) const noexcept{
        return FindIndexBack(neighbor.second);
    }

    size_t FindIndexFront(const FloatType dist) const noexcept{
        size_t index = 0;
        for ( ; index<neighbors.size(); index += 1){
            if (neighbors[index].second > dist) break;
        }
        return index;
    }

    template<typename OtherIndex>
    size_t FindIndexFront(const std::pair<OtherIndex, FloatType>& neighbor) const noexcept{
        return FindIndexFront(neighbor.second);
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


    //std::pair of trivially copyable types is not trivially copyable,
    //and swaping out for a separate type is enough work at this point to make it worth defering for now

    void serialize(std::ofstream& outFile) const requires (!TriviallyCopyable<std::pair<IndexType, FloatType>>){
        Serialize(this->size(), outFile);
        outFile.write(reinterpret_cast<const char*>(this->neighbors.data()), this->size()*sizeof(std::pair<IndexType, FloatType>));
    }

    GraphVertex(std::ifstream& inFile): neighbors(Extract<size_t>(inFile)+1) {
        this->neighbors.resize(neighbors.size() - 1);
        inFile.read(reinterpret_cast<char *>(neighbors.data()), sizeof(std::pair<IndexType, FloatType>)*neighbors.size());
    }


    //private:
    
};
template<typename Type>
static constexpr bool isGraphVertex = false;

template<TriviallyCopyable IndexType, TriviallyCopyable DistType, template<typename> typename Alloc>
static constexpr bool isGraphVertex<GraphVertex<IndexType, DistType, Alloc>> = true;

template<typename Vertex>
    requires isGraphVertex<Vertex>
Vertex& EraseRemove(Vertex& vertex, typename Vertex::Distance_t minValue){

    size_t index = vertex.FindIndexFront(minValue);
    vertex.erase(vertex.begin()+index, vertex.end());
    
    return vertex;
}

//Rewrite as stream operator?
template<typename DistType>
unsigned int ConsumeVertex(VertexLike<BlockIndecies, DistType> auto& consumer, VertexLike<BlockIndecies, DistType> auto&& consumee){
    //std::sort(consumee.begin(), consumee.end(), NeighborDistanceComparison<BlockIndecies, DistType>);
    consumee.UnPrep();
    unsigned int neighborsAdded(0);
    for (auto& pair: consumee){
        if (pair.second >= consumer.PushThreshold()) return neighborsAdded;
        consumer.PushNeighbor(pair);
        neighborsAdded++;
    }
    return neighborsAdded;
}
/*
template<typename DistType>
unsigned int ConsumeVertex(GraphVertex<BlockIndecies, DistType>& consumer, GraphVertex<BlockIndecies, DistType>&& consumee){
    //std::sort(consumee.begin(), consumee.end(), NeighborDistanceComparison<BlockIndecies, DistType>);
    consumee.UnPrep();
    unsigned int neighborsAdded(0);
    for (auto& pair: consumee){
        if (pair.second >= consumer.PushThreshold()) return neighborsAdded;
        consumer.PushNeighbor(pair);
        neighborsAdded++;
    }
    return neighborsAdded;
}
*/

template<typename Consumer, typename OtherVertex>
    requires isGraphVertex<Consumer> && isGraphVertex<OtherVertex>
unsigned int ConsumeVertex(Consumer& consumer, OtherVertex& consumee, size_t consumeeFragment, size_t consumeeBlock){
    //std::sort(consumee.begin(), consumee.end(), NeighborDistanceComparison<OtherIndex, OtherDist>);
    using ConsumerDist = typename Consumer::Distance_t;
    consumee.UnPrep();
    unsigned int neighborsAdded(0);
    for (auto& pair: consumee){
        if (pair.second >= consumer.PushThreshold()) return neighborsAdded;
        consumer.PushNeighbor({{consumeeFragment, consumeeBlock, pair.first}, static_cast<ConsumerDist>(pair.second)});
        neighborsAdded++;
    }
    return neighborsAdded;
}



template<TriviallyCopyable OtherIndex, typename OtherDist, typename ConsumerDist>
unsigned int ConsumeVertex(GraphVertex<std::pair<BlockIndecies, bool>, ConsumerDist>& consumer, GraphVertex<OtherIndex, OtherDist>& consumee, size_t consumeeBlockNum){
    //std::sort(consumee.begin(), consumee.end(), NeighborDistanceComparison<OtherIndex, OtherDist>);
    consumee.UnPrep();
    unsigned int neighborsAdded(0);
    for (auto& pair: consumee){
        if (pair.second >= consumer.PushThreshold()) return neighborsAdded;
        consumer.PushNeighbor({{{consumeeBlockNum, pair.first}, false}, static_cast<ConsumerDist>(pair.second)});
        neighborsAdded++;
    }
    return neighborsAdded;
}

template<is_not<BlockIndecies> DataIndexType, typename DistType>
GraphVertex<BlockIndecies, DistType> ToBlockIndecies(const GraphVertex<DataIndexType, DistType>& vertexToConvert, const size_t fragmentNum, const size_t blockNum){
    GraphVertex<BlockIndecies, DistType> newVertex(vertexToConvert.size());
    for(const auto& neighbor: vertexToConvert){
        newVertex.push_back({{fragmentNum, blockNum, neighbor.first}, neighbor.second});
    }   
    return newVertex;
}


}

#endif

