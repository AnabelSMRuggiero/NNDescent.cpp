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

#include <new>
#include <vector>

#include "ann/AlignedMemory/DynamicArray.hpp"
#include "ann/Type.hpp"
#include "../FunctorErasure.hpp"
#include "ann/Data.hpp"
#include "../Type.hpp"
#include "Graph.hpp"

namespace nnd {

template<TriviallyCopyable IndexType>
struct DirectedGraph{

    using iterator = typename DataBlock<IndexType, ann::align_val_of<IndexType>>::iterator;
    using const_iterator = typename DataBlock<IndexType, ann::align_val_of<IndexType>>::const_iterator;
    using reference = typename DataBlock<IndexType, ann::align_val_of<IndexType>>::reference;
    using const_reference = typename DataBlock<IndexType, ann::align_val_of<IndexType>>::const_reference;

    DataBlock<IndexType, ann::align_val_of<IndexType>> verticies;

    DirectedGraph(): verticies(){};

    DirectedGraph(size_t numVerticies, size_t numNeighbors): 
        verticies(numVerticies, numNeighbors, 0){};

    template<typename DistType>
    DirectedGraph(Graph<IndexType, DistType> graph): verticies(graph.size(), graph[0].size(), 0){
        
        for (std::size_t i = 0; auto& vertex: graph){
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