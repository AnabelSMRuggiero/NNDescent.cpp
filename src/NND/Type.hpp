#ifndef NND_NNDTYPE_HPP
#define NND_NNDTYPE_HPP

#include <iterator>
#include <memory_resource>
#include <vector>
#include <concepts>
#include "ann/Type.hpp"
#include "ann/Data.hpp"
#include <cstdint>


namespace nnd{

struct splitting_heurisitcs{
    std::uint32_t split_threshold = 80;
    std::uint32_t child_threshold = 32;
    std::uint32_t max_tree_size = 130;
    std::size_t max_retry = std::size_t(-1);
    float max_split_fraction = 0.0f;
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
struct comparison_key{
    IndexType first;
    IndexType second;
};

template<typename IndexType>
bool operator==(comparison_key<IndexType> lhs, comparison_key<IndexType> rhs){
    return (lhs.first == rhs.first && lhs.second == rhs.second) ||
           (lhs.first == rhs.second && lhs.second == rhs.first);
}


struct index_parameters{
    std::size_t block_graph_neighbors;
    std::size_t COM_neighbors;
    std::size_t nearest_node_neighbors;
    std::size_t query_depth;
};

struct search_parameters{
    std::size_t search_neighbors;
    std::size_t search_depth;
    std::size_t max_searches_queued;
};


struct hyper_parameters{
    splitting_heurisitcs split_params;
    index_parameters index_params;
    search_parameters search_params;
};

//using IndexBlock = std::vector<std::vector<BlockIndecies>>;

using IndexBlock = UnevenBlock<BlockIndecies>;

template<std::ranges::contiguous_range Range>
auto as_const_span(Range&& range){
    using value_type = std::ranges::range_value_t<Range>;
    return std::span<const value_type>{range};
}

using candidate_set = ann::dynamic_array<std::vector<BlockIndecies>>;

template<std::size_t BufferSize>
struct stack_fed_buffer{
    std::byte buffer[BufferSize];
    std::pmr::monotonic_buffer_resource memoryResource{buffer, BufferSize};

    template<typename ValueType>
    operator std::pmr::polymorphic_allocator<ValueType>(){
        return &memoryResource;
    }
};

}

template<>
struct std::hash<nnd::BlockIndecies>{

    size_t operator()(const nnd::BlockIndecies& index) const noexcept{
        return std::hash<size_t>()(index.blockNumber ^ std::hash<size_t>()(index.dataIndex));
    };

};

// If first = second, I screwed up before calling this
template<typename IndexType>
struct std::hash<nnd::comparison_key<IndexType>>{

    size_t operator()(const nnd::comparison_key<IndexType>& key) const noexcept{
        return std::hash<IndexType>()(key.first) ^ std::hash<IndexType>()(key.second);
    };

};

#endif