/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_SPACEMETRICS_HPP
#define NND_SPACEMETRICS_HPP

#include <valarray>
#include <cmath>
//#include <functional>

namespace nnd{

// I might want to not use valarrays or use template specifications for valarrays. There's some nifty numerical algorithms in the STL
// that use container semantics, but apparently .begin() and .end() on valarray doesn't support that.
// /shrug
// I can worry about optimizing these later. Get the core stuff done first.

template<typename DataType, typename RetType=double>
using SpaceMetric = RetType (*)(const DataType&, const DataType&);

template<typename DataType, typename COMType, typename RetType=double>
using COMMetric = RetType (*)(const COMType&, const DataType&);
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
RetType EuclideanNorm(const std::valarray<RetType>& pointA, const std::valarray<DataType>& pointB){
    std::valarray<RetType> diffs(pointA);
    for (size_t i = 0; i<diffs.size(); i+=1){
        diffs[i] -= RetType(pointB[i]);
    }
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