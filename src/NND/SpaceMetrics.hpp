#ifndef NND_SPACEMETRICS_HPP
#define NND_SPACEMETRICS_HPP

#include <valarray>
#include <cmath>
//#include <functional>

namespace nnd{

// I might want to not use valarrays or use template specifications for valarrays. There's some nifty numerical algorithms in the STL
// that use container semantics, but apparently .begin() and .end() on valarray doesn't support it.
// /shrug

template<typename DataType, typename RetType=double>
using SpaceMetric = RetType (*)(const DataType&, const DataType&);
//using SpaceMetric = std::function<RetType(const std::valarray<DataType>&, const std::valarray<DataType>&)>;

template<typename DataType, typename RetType=double>
RetType EuclideanNorm(const std::valarray<DataType>& pointA, const std::valarray<DataType>& pointB){
    std::valarray<DataType> diffs = pointB-pointA;
    RetType accum(0);
    for(DataType i : diffs){
        accum += i*i;
    }
    return std::sqrt(accum);
};

template<typename DataType, typename RetType=double>
RetType Dot(const std::valarray<DataType>& pointA, const std::valarray<DataType>& pointB){
    std::valarray<DataType> components = pointB*pointA;
    RetType accum(0);
    for(DataType i : components){
        accum += i;
    }
    return accum;
};


}

#endif 