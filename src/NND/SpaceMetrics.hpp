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
#include <execution>
//#include <functional>

namespace nnd{

// I might want to not use valarrays or use template specifications for valarrays. There's some nifty numerical algorithms in the STL
// that use container semantics, but apparently .begin() and .end() on valarray doesn't support that.
// /shrug
// I can worry about optimizing these later. Get the core stuff done first.

template<typename DataTypeA, typename DataTypeB, typename RetType=double>
using SpaceMetric = RetType (*)(const DataTypeA&, const DataTypeB&);

//using SpaceMetric = std::function<RetType(const std::valarray<DataType>&, const std::valarray<DataType>&)>;
/*
template<typename DataTypeA, typename DataTypeB, typename RetType=double>
RetType EuclideanNorm(const std::valarray<DataTypeA>& pointA, const std::valarray<DataTypeB>& pointB){
    RetType accum(0);
    for (size_t i = 0; i<diffs.size(); i+=1){
        RetType diff = RetType(pointB[i]) - RetType(pointA[i]);
        accum += diff*diff;
    }
    return std::sqrt(accum);
};
*/
template<typename DataTypeA, typename DataTypeB, typename RetType=double>
RetType EuclideanNorm(const std::valarray<DataTypeA>& pointA, const std::valarray<DataTypeB>& pointB){
    auto transformFunc = [](DataTypeA operandA, DataTypeB operandB){
        RetType diff = static_cast<RetType>(operandA) - static_cast<RetType>(operandB);
        return diff*diff;
    };
    RetType accum = std::transform_reduce(std::execution::unseq,
                                    std::begin(pointA),
                                    std::end(pointA),
                                    std::begin(pointB),
                                    RetType(0),
                                    std::plus<RetType>(),
                                    transformFunc);
    return std::sqrt(accum);
};

/*
template<typename DataTypeA, typename DataTypeB, typename RetType=double>
RetType EuclideanNorm(const std::valarray<DataTypeA>& pointA, const std::valarray<DataTypeB>& pointB){
    std::valarray<RetType> diffs(pointA.size());
    for (size_t i = 0; i<diffs.size(); i+=1){
        diffs[i] = RetType(pointB[i]) - RetType(pointA[i]);
    }
    RetType accum(0);
    for(RetType i : diffs){
        accum += i*i;
    }
    return std::sqrt(accum);
};
*/

template<typename DataTypeA, typename DataTypeB, typename RetType=double>
RetType Dot(const std::valarray<DataTypeA>& pointA, const std::valarray<DataTypeB>& pointB){
    auto transformFunc = [](DataTypeA operandA, DataTypeB operandB){
        return static_cast<RetType>(operandA) * static_cast<RetType>(operandB);
    };

    RetType accum = std::transform_reduce(std::execution::unseq,
                                    std::begin(pointA),
                                    std::end(pointA),
                                    std::begin(pointB),
                                    RetType(0),
                                    std::plus<RetType>(),
                                    transformFunc);

    return accum;
};


}

#endif 