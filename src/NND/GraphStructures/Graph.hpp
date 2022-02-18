/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_GRAPH_HPP
#define NND_GRAPH_HPP

#include <vector>
#include <concepts>

#include "ann/Type.hpp"
#include "../FunctorErasure.hpp"
#include "../Type.hpp"
#include "../MemoryInternals.hpp"

#include "GraphVertex.hpp"

namespace nnd{


template<TriviallyCopyable IndexType, typename FloatType>
struct Graph{

    using iterator = typename std::vector<GraphVertex<IndexType, FloatType>>::iterator;
    using const_iterator = typename std::vector<GraphVertex<IndexType, FloatType>>::const_iterator;
    using vertex_type = GraphVertex<IndexType, FloatType>;
    using edge_type = typename GraphVertex<IndexType, FloatType>::EdgeType;

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

template<is_not<BlockIndecies> DataIndexType, typename DistType>
Graph<BlockIndecies, DistType> ToBlockIndecies(const Graph<DataIndexType, DistType>& blockGraph, const size_t fragmentNum, const size_t blockNum){
    Graph<BlockIndecies, DistType> newGraph(blockGraph.size(), blockGraph[0].size());
    for (size_t j = 0; const auto& vertex: blockGraph){
        newGraph[j].resize(blockGraph[j].size());
        for(size_t k = 0; const auto& neighbor: vertex){
            newGraph[j][k] = {{fragmentNum, blockNum, neighbor.first}, neighbor.second};
            k++;
        }   
        j++;
    }
    return newGraph;
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
void VerifySubGraphState(const Graph<BlockIndecies, FloatType>& currentGraph, size_t fragmentNum, size_t blockNum){
    for (size_t i = 0; i<currentGraph.size(); i+=1){
        const GraphVertex<BlockIndecies, FloatType>& vertex = currentGraph[i];
        for (const auto& neighbor : vertex.neighbors){
            if (neighbor.first == BlockIndecies{fragmentNum, blockNum, i}) throw("Vertex is own neighbor");
            for (const auto& neighbor1 : vertex.neighbors){
                if (&neighbor == &neighbor1) continue;
                if (neighbor.first == neighbor1.first) throw("Duplicate neighbor in heap");
            }
        }
    }
}

template<typename DistType>
//numNeighbors, blockSize, distanceFunctor
Graph<DataIndex_t, DistType> BruteForceBlock(const size_t numNeighbors, const size_t blockSize, erased_metric<DistType> distanceFunctor){
    
    using edge_type = typename Graph<DataIndex_t, DistType>::edge_type;

    Graph<DataIndex_t, DistType> retGraph(blockSize, numNeighbors);

    auto addResult = [&](auto from, auto to, auto distance)->void{
        if (retGraph[from].size() < numNeighbors){
            retGraph[from].push_back(edge_type{static_cast<size_t>(to), distance});
            if (retGraph[from].size() == numNeighbors){
                retGraph[from].JoinPrep();
            }
        } else {
            retGraph[from].PushNeighbor(edge_type{static_cast<size_t>(to), distance});
        }
    };

    std::pmr::vector<size_t> indecies{blockSize-1, internal::GetThreadResource()};
    std::iota(indecies.rbegin(), indecies.rend(), size_t{1});
    
    for (size_t i = 0; i < blockSize-1; i += 1){
        std::ranges::contiguous_range auto distances = distanceFunctor(i, indecies);

        for (auto indeciesItr = indecies.begin(); const auto& distance : distances){
            size_t j = *indeciesItr;

            addResult(i, j, distance);
            addResult(j, i, distance);
            
            indeciesItr++;
        }
        indecies.pop_back();
    }
    /*
    for (size_t i = 0; i < blockSize; i += 1){
        for (size_t j = i+1; j < blockSize; j += 1){
            DistType distance = distanceFunctor(i, j);
            if (retGraph[i].size() < numNeighbors){
                retGraph[i].push_back(std::pair<DataIndex_t, DistType>(static_cast<size_t>(j), distance));
                if (retGraph[i].size() == numNeighbors){
                    retGraph[i].JoinPrep();
                }
            } else {
                retGraph[i].PushNeighbor(std::pair<DataIndex_t, DistType>(static_cast<size_t>(j), distance));
            }
            if (retGraph[j].size() < numNeighbors){
                retGraph[j].push_back(std::pair<DataIndex_t, DistType>(static_cast<size_t>(i), distance));
                if (retGraph[j].size() == numNeighbors){
                    retGraph[j].JoinPrep();
                }
            } else {
                retGraph[j].PushNeighbor(std::pair<DataIndex_t, DistType>(static_cast<size_t>(i), distance));
            }
        }
    }
    */
    return retGraph;
}

}

#endif