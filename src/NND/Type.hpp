#ifndef NND_NNDTYPE_HPP
#define NND_NNDTYPE_HPP

#include <vector>
#include <concepts>
#include "../ann/Type.hpp"
#include "../ann/Data.hpp"
#include <cstdint>
//#include <memory>

namespace nnd{

struct SplittingHeurisitcs{
    uint32_t splitThreshold = 80;
    uint32_t childThreshold = 32;
    uint32_t maxTreeSize = 130;
    float maxSplitFraction = 0.0f;
};

using DataIndex_t = uint32_t;
using BlockNumber_t = uint32_t;
using GraphFragment_t = uint32_t;

struct Override {};
static inline constexpr Override overrideTag{};

namespace internal{
    static constexpr size_t maxBatch = 14;
}

struct BlockIndecies{
    GraphFragment_t graphFragment;
    BlockNumber_t blockNumber;
    DataIndex_t dataIndex;

    BlockIndecies() = default;

    BlockIndecies(std::unsigned_integral auto graphFragment, std::unsigned_integral auto blockNumber, std::unsigned_integral auto dataIndex):
        graphFragment(graphFragment),
        blockNumber(blockNumber),
        dataIndex(dataIndex) {};

};

inline bool operator==(const BlockIndecies lhs, const BlockIndecies& rhs){
    return (lhs.graphFragment == rhs.graphFragment) && (lhs.blockNumber == rhs.blockNumber) && (lhs.dataIndex == rhs.dataIndex);
}

inline constexpr bool debugNND = false;
//Maybe a block specific one that reads i.blockNumber from a BlockIndecies
template<typename Alloc = std::allocator<bool>>
struct NodeTrackerImpl{

    using reference = typename std::vector<bool, Alloc>::reference;
    using const_reference = typename std::vector<bool, Alloc>::const_reference;
    using size_type = typename std::vector<bool, Alloc>::size_type;

    std::vector<bool, Alloc> flags;

    NodeTrackerImpl() = default;

    explicit NodeTrackerImpl(size_t graphSize): flags(graphSize, false){};

    explicit NodeTrackerImpl(size_t graphSize, Alloc allocator): flags(graphSize, false, allocator){};

    reference operator[](size_type i){
        return flags[i];
    }

    constexpr const_reference operator[](size_type i) const {
        return flags[i];
    }

    reference operator[](BlockIndecies i){
        //Assuming block index lines up here;
        return flags[i.dataIndex];
    }

    constexpr const_reference operator[](BlockIndecies i) const{
        //Assuming block index lines up here;
        return flags[i.dataIndex];
    }

    size_t size() const{
        return flags.size();
    }

    void clear(){
        flags.clear();
    }

    void resize(size_t newSize){
        flags.resize(newSize);
    }
};

using NodeTracker = NodeTrackerImpl<>;
//using SearchQueue = std::vector<std::vector<std::pair<BlockIndecies, size_t>>>;
// Two member struct with the following properties. hash({x,y}) == hash({y,x}) and {x,y} == {y,x}
// This way a set can be used to queue up an operation between two blocks without worrying which is first or second.
template<typename IndexType>
struct ComparisonKey{
    IndexType first;
    IndexType second;
};

template<typename IndexType>
bool operator==(ComparisonKey<IndexType> lhs, ComparisonKey<IndexType> rhs){
    return (lhs.first == rhs.first && lhs.second == rhs.second) ||
           (lhs.first == rhs.second && lhs.second == rhs.first);
}


struct IndexParameters{
    size_t blockGraphNeighbors;
    size_t COMNeighbors;
    size_t nearestNodeNeighbors;
    size_t queryDepth;
};

struct SearchParameters{
    size_t searchNeighbors;
    size_t searchDepth;
    size_t maxSearchesQueued;
};




struct HyperParameterValues{
    SplittingHeurisitcs splitParams;
    IndexParameters indexParams;
    SearchParameters searchParams;
};

//using IndexBlock = std::vector<std::vector<BlockIndecies>>;

using IndexBlock = UnevenBlock<BlockIndecies>;

}

template<>
struct std::hash<nnd::BlockIndecies>{

    size_t operator()(const nnd::BlockIndecies& index) const noexcept{
        return std::hash<size_t>()(index.blockNumber ^ std::hash<size_t>()(index.dataIndex));
    };

};

// If first = second, I screwed up before calling this
template<typename IndexType>
struct std::hash<nnd::ComparisonKey<IndexType>>{

    size_t operator()(const nnd::ComparisonKey<IndexType>& key) const noexcept{
        return std::hash<size_t>()(key.first) ^ std::hash<size_t>()(key.second);
    };

};

#endif