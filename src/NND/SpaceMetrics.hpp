#ifndef NND_SPACEMETRICS_HPP
#define NND_SPACEMETRICS_HPP

#include <valarray>
#include <cmath>
//#include <functional>

namespace nnd{

template<typename DataType, typename RetType=double>
using SpaceMetric = RetType (*)(const DataType&, const DataType&);
//using SpaceMetric = std::function<RetType(const std::valarray<DataType>&, const std::valarray<DataType>&)>;

template<typename DataType, typename RetType=double>
RetType EuclideanNorm(const std::valarray<DataType>& pointA, const std::valarray<DataType>& pointB){
    std::valarray<DataType> diffs = pointB-pointA;
    double accum(0);
    for(DataType i : diffs){
        accum += i*i;
    }
    return std::sqrt(accum);
};


}

#endif 