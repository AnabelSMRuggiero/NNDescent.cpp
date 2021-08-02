#ifndef NND_NNDTYPE_HPP
#define NND_NNDTYPE_HPP

#include <vector>
#include "../Utilities/Type.hpp"

namespace nnd{

//Maybe a block specific one that reads i.blockNumber from a BlockIndecies
struct NodeTracker{

    using reference = std::vector<bool>::reference;
    using const_reference = std::vector<bool>::const_reference;
    using size_type = std::vector<bool>::size_type;

    std::vector<bool> flags;

    NodeTracker() = default;

    NodeTracker(size_t graphSize): flags(graphSize, false){};

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

};

    using SearchQueue = std::vector<std::vector<std::pair<BlockIndecies, size_t>>>;

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



}

// If first = second, I screwed up before calling this
template<typename IndexType>
struct std::hash<nnd::ComparisonKey<IndexType>>{

    size_t operator()(const nnd::ComparisonKey<IndexType>& key) const noexcept{
        return std::hash<size_t>()(key.first) ^ std::hash<size_t>()(key.second);
    };

};

#endif