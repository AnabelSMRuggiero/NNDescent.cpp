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

#include <algorithm>


#include "UtilityFunctions.hpp"


#include "Type.hpp"

#include "GraphStructures/GraphVertex.hpp"
#include "GraphStructures/Graph.hpp"
#include "GraphStructures/DirectedGraph.hpp"
#include "GraphStructures/UndirectedGraph.hpp"
#include "GraphStructures/CachingFunctor.hpp"


#include "Utilities/DataDeserialization.hpp"

namespace nnd{




/*
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
                    directedGraph[neighbor.first].push_back({static_cast<IndexType>(i), neighbor.second});
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
        */
        /*
        for (size_t i = 0; const auto& vertex: directedGraph){
            for (const auto& neighbor: vertex){
                if(std::ranges::find(verticies[i], neighbor.first) == verticies[i].end()) verticies[i].push_back(neighbor.first);
                if(std::ranges::find(verticies[neighbor.first], i) == verticies[neighbor.first].end()) verticies[neighbor.first].push_back(i);
            }
            i++;
        }
        */
       /*
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














}
#endif //DATASTRUCTURES