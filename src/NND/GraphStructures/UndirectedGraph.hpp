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
#include <fstream>

#include "Graph.hpp"
#include "ann/Type.hpp"
#include "ann/Data.hpp"
#include "ann/DataSerialization.hpp"
#include "ann/DataDeserialization.hpp"

namespace nnd{

/*
template<TriviallyCopyable IndexType>
    requires std::is_trivially_constructible_v<IndexType> && std::is_trivially_destructible_v<IndexType>
struct UndirectedGraph;
*/


template<TriviallyCopyable IndexType>
    requires std::is_trivially_constructible_v<IndexType> && std::is_trivially_destructible_v<IndexType>
struct UndirectedGraph{

    using iterator = typename UnevenBlock<IndexType>::iterator;
    using const_iterator = typename UnevenBlock<IndexType>::const_iterator;
    using reference = typename UnevenBlock<IndexType>::reference;
    using const_reference = typename UnevenBlock<const IndexType>::reference;


    //private:
    UnevenBlock<IndexType> graphBlock;
    //public:

    UndirectedGraph(): graphBlock() {};

    UndirectedGraph(std::ifstream& inFile): graphBlock(Extract<UnevenBlock<IndexType>>(inFile)) {}
    //graphBlock(UnevenBlock<IndexType>::deserialize(inFile)) {}
    

    //NewUndirectedGraph(size_t numVerticies, size_t numNeighbors): 
    //    verticies(numVerticies, std::vector<IndexType>(numNeighbors)){};

    UndirectedGraph(const size_t numVerticies, const size_t numIndecies): graphBlock(UninitUnevenBlock<IndexType>(numVerticies, numIndecies)){
    }

    reference operator[](size_t i){
        return graphBlock[i];
    }

    reference operator[](BlockIndecies i){
        // I'm assuming the block number is correct
        return graphBlock[i.dataIndex];
    }

    constexpr const const_reference operator[](size_t i) const{
        return graphBlock[i];
    }

    constexpr const const_reference operator[](BlockIndecies i) const{
        return graphBlock[i.dataIndex];
    }


    size_t size() const noexcept{
        return graphBlock.size();
    }
    
    constexpr iterator begin() noexcept{
        return graphBlock.begin();
    }

    constexpr const_iterator begin() const noexcept{
        return graphBlock.begin();
    }

    constexpr const_iterator cbegin() const noexcept{
        return graphBlock.cbegin();
    }

    constexpr iterator end() noexcept{
        return graphBlock.end();
    }

    constexpr const_iterator end() const noexcept{
        return graphBlock.end();
    }

    constexpr const_iterator cend() const noexcept{
        return graphBlock.cend();
    }
};

template<typename IndexType, typename DistType>
UndirectedGraph<IndexType> BuildUndirectedGraph(Graph<IndexType, DistType> directedGraph){
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

    //Okay, now I have enough information to call the constructor.

    UndirectedGraph<IndexType> retGraph(numberOfVerticies, totalIndecies);
    size_t* headerStart = std::launder(static_cast<size_t*>(static_cast<void*>(retGraph.graphBlock.data())));
    std::transform_inclusive_scan(directedGraph.begin(), directedGraph.end(), headerStart+1, std::plus<size_t>{},[](const auto& vertex){
        return vertex.size();
    }, 0);

    IndexType* indexStart = retGraph.graphBlock.firstIndex;
    for (auto& vertex: directedGraph){
        std::sort(vertex.begin(), vertex.end(), edge_ops::lessThan);
        std::transform(vertex.begin(), vertex.end(), indexStart, [](const auto& neighbor){ return neighbor.first; });
        indexStart += vertex.size();
    }


    return retGraph;

}


}

#endif