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

}

#endif