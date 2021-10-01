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
#include "../../Utilities/Data.hpp"

namespace nnd{


//Placeholder name while I work on the constructor for the new version
template<TriviallyCopyable IndexType>
    requires std::is_trivially_constructible_v<IndexType> && std::is_trivially_destructible_v<IndexType>
struct NewUndirectedGraph{

    using iterator = typename UnevenBlock<IndexType>::iterator;
    using const_iterator = typename UnevenBlock<IndexType>::const_iterator;
    private:
    UnevenBlock<IndexType> graphBlock;
    public:

    NewUndirectedGraph(): graphBlock(){};

    //NewUndirectedGraph(size_t numVerticies, size_t numNeighbors): 
    //    verticies(numVerticies, std::vector<IndexType>(numNeighbors)){};

    template<typename DistType>
    NewUndirectedGraph(const size_t numVerticies, const size_t numIndecies, std::pmr::memory_resource* resource): graphBlock(UninitUnevenBlock(numVerticies, numVerticies, resource)){
    }

    std::vector<IndexType>& operator[](size_t i){
        return graphBlock[i];
    }

    std::vector<IndexType>& operator[](BlockIndecies i){
        // I'm assuming the block number is correct
        return graphBlock[i.dataIndex];
    }

    constexpr const std::vector<IndexType>& operator[](size_t i) const{
        return graphBlock[i];
    }

    constexpr const std::vector<IndexType>& operator[](BlockIndecies i) const{
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

    //Okay, now I have enough information to call the constructor.

    NewUndirectedGraph<IndexType> retGraph(numberOfVerticies, totalIndecies, resource);

    std::transform_inclusive_scan(directedGraph.begin(), directedGraph.end(), retGraph.dataStorage.get(), 0, std::plus<size_t>{},[](const auto& vertex){
        return vertex.size();
    });

    IndexType* indexStart = retGraph.firstIndex;
    for (auto& vertex: directedGraph){
        std::ranges::sort(vertex, NeighborDistanceComparison<IndexType, DistType>);
        std::transform(vertex.begin(), vertex.end(), indexStart, [](const auto& neighbor){ return neighbor.first });
        indexStart += vertex.size();
    }


    return retGraph;

}


}

#endif