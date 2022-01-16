/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_DIRECTEDGRAPH_HPP
#define NND_DIRECTEDGRAPH_HPP

#include <vector>

#include "ann/Type.hpp"
#include "../FunctorErasure.hpp"
#include "ann/Data.hpp"
#include "../Type.hpp"

namespace nnd {

/*
template<TriviallyCopyable IndexType>
struct DirectedGraph{

    using iterator = typename std::vector<std::vector<IndexType>>::iterator;
    using const_iterator = typename std::vector<std::vector<IndexType>>::const_iterator;

    std::vector<std::vector<IndexType>> verticies;

    DirectedGraph(): verticies(){};

    DirectedGraph(size_t numVerticies, size_t numNeighbors): 
        verticies(numVerticies, std::vector<IndexType>(numNeighbors)){};

    template<typename DistType>
    DirectedGraph(Graph<IndexType, DistType> directedGraph): verticies(directedGraph.size()){
        
        
        for (size_t i = 0; auto& vertex: directedGraph){
            verticies[i].reserve(vertex.size());
            for (const auto& neighbor:vertex){
                verticies[i].push_back(neighbor.first);
            }

            i++;
        }
        
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
*/

template<TriviallyCopyable IndexType>
struct DirectedGraph{

    using iterator = typename DataBlock<IndexType, alignof(IndexType)>::iterator;
    using const_iterator = typename DataBlock<IndexType, alignof(IndexType)>::const_iterator;
    using reference = typename DataBlock<IndexType, alignof(IndexType)>::reference;
    using const_reference = typename DataBlock<IndexType, alignof(IndexType)>::const_reference;

    DataBlock<IndexType, alignof(IndexType)> verticies;

    DirectedGraph(): verticies(){};

    DirectedGraph(size_t numVerticies, size_t numNeighbors): 
        verticies(numVerticies, numNeighbors, 0){};

    template<typename DistType>
    DirectedGraph(Graph<IndexType, DistType> directedGraph): verticies(directedGraph.size(), directedGraph[0].size(), 0){
        
        
        for (size_t i = 0; auto& vertex: directedGraph){
            std::transform(vertex.begin(), vertex.end(), verticies[i].begin(), [](const auto neighbor){
                return neighbor.first;
            });
            i++;
        }
        
    }

    reference operator[](size_t i){
        return verticies[i];
    }

    reference operator[](BlockIndecies i){
        // I'm assuming the block number is correct
        return verticies[i.dataIndex];
    }

    const_reference operator[](size_t i) const{
        return this->verticies[i];
    }

    const_reference operator[](BlockIndecies i) const{
        return this->verticies[i.dataIndex];
    }

    size_t size() const noexcept{
        return verticies.size();
    }
    
    iterator begin() noexcept{
        return verticies.begin();
    }

    const_iterator begin() const noexcept{
        return verticies.begin();
    }

    const_iterator cbegin() const noexcept{
        return verticies.cbegin();
    }

    iterator end() noexcept{
        return verticies.end();
    }

    const_iterator end() const noexcept{
        return verticies.end();
    }

    const_iterator cend() const noexcept{
        return verticies.cend();
    }
};



}

#endif