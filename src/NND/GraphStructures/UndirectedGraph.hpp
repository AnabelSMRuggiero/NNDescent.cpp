/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_UNDIRECTEDGRAPH_HPP
#define NND_UNDIRECTEDGRAPH_HPP

#include <span>
#include <memory_resource>

#include "../GraphStructures.hpp"
#include "../../Utilities/Type.hpp"

namespace nnd{

template<TriviallyCopyable IndexType>
struct UndirectedGraphIterator{
    using value_type = std::span<IndexType>;
    using difference_type = std::ptrdiff_t;
    using reference = std::span<IndexType>;

    private:
    size_t* vertexStart;
    IndexType* vertexNeighbors;

    public:
    UndirectedGraphIterator& operator++(){
        vertexNeighbors += *(vertexStart+1) - *vertexStart;
        ++vertexStart;
        return *this;
    }

    UndirectedGraphIterator operator++(int){
        UndirectedGraphIterator copy = *this;
        vertexNeighbors += *(vertexStart+1) - *vertexStart;
        ++vertexStart;
        return copy;
    }

    UndirectedGraphIterator& operator--(){
        vertexNeighbors -= *vertexStart - *(vertexStart-1);
        --vertexStart;
        return *this;
    }

    UndirectedGraphIterator operator--(int){
        UndirectedGraphIterator copy = *this;
        vertexNeighbors -= *vertexStart - *(vertexStart-1);
        --vertexStart;
        return copy;
    }

    UndirectedGraphIterator operator+(std::ptrdiff_t inc){
        UndirectedGraphIterator copy{vertexStart+inc, vertexNeighbors + (*(vertexStart+inc) - *vertexStart)};
        return copy;
    }

    UndirectedGraphIterator operator-(std::ptrdiff_t inc){
        UndirectedGraphIterator copy{vertexStart-inc, vertexNeighbors - (*vertexStart - *(vertexStart-inc))};
        return copy;
    }

    std::ptrdiff_t operator-(UndirectedGraphIterator other){
        return vertexStart - other.vertexStart;
    }

    bool operator==(UndirectedGraphIterator other){
        return vertexStart == other.vertexStart;
    }

    reference operator*(){
        return reference{vertexNeighbors, *(vertexStart+1) - *vertexStart};
    }

    reference operator[](size_t i){
        return *(*this + i);
    }
};

//Placeholder name while I work on the constructor for the new version
template<TriviallyCopyable IndexType>
struct NewUndirectedGraph{

    using iterator = typename std::vector<std::vector<IndexType>>::iterator;
    using const_iterator = typename std::vector<std::vector<IndexType>>::const_iterator;
    private:
    DynamicArray<std::byte, std::max(alignof(size_t), alignof(IndexType))> dataStorage;
    size_t numVerticies;
    IndexType* firstIndex;
    public:

    NewUndirectedGraph(): verticies(){};

    //NewUndirectedGraph(size_t numVerticies, size_t numNeighbors): 
    //    verticies(numVerticies, std::vector<IndexType>(numNeighbors)){};

    template<typename DistType>
    NewUndirectedGraph(const size_t numBytes, const size_t numVerticies, const size_t headerPadding, const size_t numIndecies, std::pmr::memory_resource* resource): dataStorage(numBytes, resource), numVerticies(numVerticies), firstIndex(nullptr){
        //std::pmr::polymorphic_allocator<std::byte> alloc(resource);
        //alloc.construct
        size_t* vertexStart = new (dataStorage.begin()) size_t[numVerticies+1];
        //*vertexStart = 0;
        firstIndex = new (dataStorage.begin() + sizeof(size_t)*(numVerticies+1) + padding) IndexType[numIndecies];

        
        
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

template<typename IndexType, typename DistType>
void BuildUndirectedGraph(Graph<IndexType, DistType> directedGraph, std::pmr::memory_resource* resource = std::pmr::get_default_resource()){
    for (size_t i = 0; auto& vertex: directedGraph){
        vertex.neighbors.reserve(vertex.size()*2);
        for (const auto& neighbor: vertex){
            if(std::find_if(directedGraph[neighbor.first].begin(), directedGraph[neighbor.first].end(), NeighborSearchFunctor<IndexType, DistType>(i)) == directedGraph[neighbor.first].end()) 
                directedGraph[neighbor.first].push_back({static_cast<IndexType>(i), neighbor.second});
        }
        i++;
    }

    //size_t totalSize = directedGraph.size();
    size_t numberOfVerticies = directedGraph.size();
    size_t totalIndecies = 0;

    for(const auto& vertex : directedGraph){
        totalIndecies += vertex.size();
    }

    size_t numberOfBytes = 0;
    size_t headerBytes = sizeof(size_t)*(numberOfVerticies+1);
    size_t headerPadding = 0;

    if constexpr(alignof(IndexType)>alignof(size_t)){
        headerPadding = alignof(IndexType) - headerBytes%alignof(IndexType);
        numberOfBytes = headerBytes + headerPadding + sizeof(IndexType)*totalIndecies;
    } else {
        numberOfBytes = headerBytes + sizeof(IndexType)*totalIndecies;
    }
    //Okay, now I have enough information to call the constructor.

    NewUndirectedGraph<IndexType> retGraph(numberOfBytes, numberOfVerticies, headerPadding, totalIndecies, resource);

    std::transform_inclusive_scan(directedGraph.begin(), directedGraph.end(), retGraph.dataStorage.get(), 0, std::plus<size_t>{},[](const auto& vertex){
        return vertex.size();
    });

    IndexType* indexStart = retGraph.firstIndex;
    for (auto& vertex: directedGraph){
        std::ranges::sort(vertex, NeighborDistanceComparison<IndexType, DistType>);
        //This orders everything from closest to furtherest. Opposite of what it used to be
        std::transform(vertex.begin(), vertex.end(), indexStart, [](const auto& neighbor){ return neighbor.first });
        indexStart += vertex.size();
    }


    return retGraph;

}


}

#endif