#include "NND/SpaceMetrics.hpp"
#include <numeric>
#include <span>
#include <algorithm>
#include <execution>

/*

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


*/

template<typename DataTypeA, typename DataTypeB, typename RetType=double>
std::vector<RetType> BatchEuclideanNorm(const std::vector<std::span<const DataTypeA>>& pointsTo, const std::valarray<DataTypeB>& pointB){
    std::vector<RetType> results(pointsTo.size());

    auto transformFunc = [&](DataTypeA operandA, size_t index){
        std::vector<RetType> transformedElements(pointsTo.size());
        auto elementWiseTransform = [&](std::span<const DataTypeA> targetArray){
            RetType diff = targetArray[index] - operandA;
            return diff*diff;
        };
        std::transform(std::execution::unseq, pointsTo.begin(), pointsTo.end(), transformedElements.begin(), elementWiseTransform);
        //RetType diff = static_cast<RetType>(operandA) - static_cast<RetType>(operandB);
        return transformedElements;
    };

    auto accumulator = [] (std::vector<RetType> operandA, std::vector<RetType> operandB){
        for (size_t i = 0; i<operandA.size(); i+=1){
          operandA[i] += operandB[i];
        }
        return operandA;
    };

    std::vector<RetType> accum = std::transform_reduce(
                                    std::begin(pointB),
                                    std::end(pointB),
                                    std::views::iota(0).begin(),
                                    results,
                                    accumulator,
                                    transformFunc);
    
    return accum;

}