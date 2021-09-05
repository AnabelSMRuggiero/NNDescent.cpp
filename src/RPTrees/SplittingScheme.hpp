/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef RPT_SPLITTINGSCHEMERF_HPP
#define RPT_SPLITTINGSCHEMERF_HPP

#include <valarray>
#include <functional>
#include "Utilities/Metrics/SpaceMetrics.hpp"
#include "Utilities/Data.hpp"

namespace nnd{

template<typename DataEntry, typename DistType>
AlignedArray<DistType> EuclidianSplittingPlaneNormal(const DataEntry& pointA, const DataEntry& pointB){
    AlignedArray<DistType> splittingLine(pointA.size());
    for (size_t i = 0; i < pointA.size(); i += 1){
        splittingLine[i] = DistType(pointA[i]) - DistType(pointB[i]);
    }
    DistType splittingLineMag(0);
    for (DistType i : splittingLine){
        splittingLineMag += i*i;
    }
    splittingLineMag = std::sqrt(splittingLineMag);

    for(auto& extent: splittingLine) extent /= splittingLineMag;

    return splittingLine;
}



struct TransformTag {};
 static const TransformTag transformTag;

//The serial case
template<typename DataEntry, typename SplittingVector>
struct EuclidianScheme{
    using OffSetType = typename SplittingVector::value_type;
    using SplittingView = typename DefaultDataView<SplittingVector>::ViewType;
    
    const DataSet<DataEntry>& dataSource;
    std::unordered_map<size_t, std::pair<SplittingVector, OffSetType>> splittingVectors;
    //DistType projectionOffset;

    EuclidianScheme(const DataSet<DataEntry>& data) : dataSource(data), splittingVectors(){};

    auto operator()(size_t splitIndex, std::pair<size_t, size_t> splittingPoints){
        
        
        // For right now at least, in the serial case I want to be able to get a new splitting vector
        //if (splittingVectors.find(splitIndex) == splittingVectors.end()){
        SplittingVector splittingVector = EuclidianSplittingPlaneNormal<DataEntry, OffSetType>(dataSource[splittingPoints.first], dataSource[splittingPoints.second]);


        OffSetType projectionOffset = 0;
        for (size_t i = 0; i<dataSource[splittingPoints.first].size(); i+=1){
            projectionOffset -= splittingVector[i] * OffSetType(dataSource[splittingPoints.first][i] + dataSource[splittingPoints.second][i])/2.0;
        };

        splittingVectors[splitIndex] = std::pair<SplittingVector, OffSetType>(std::move(splittingVector), projectionOffset);

        //};
        if constexpr(isAlignedArray_v<SplittingVector>){
            auto comparisonFunction = [=, 
                                    &data = std::as_const(this->dataSource), 
                                    splitter = SplittingView(splittingVectors[splitIndex].first),
                                    offset = splittingVectors[splitIndex].second]
                                    (size_t comparisonIndex) -> bool{
                    return 0.0 < (Dot(data[comparisonIndex], splitter) + offset);
            };
            return comparisonFunction;
        } else {
            auto comparisonFunction = [=, 
                                    &data = std::as_const(this->dataSource), 
                                    splitter = SplittingView(splittingVectors[splitIndex].first.begin(), splittingVectors[splitIndex].first.end()),
                                    offset = splittingVectors[splitIndex].second]
                                    (size_t comparisonIndex) -> bool{
                    return 0.0 < (Dot(data[comparisonIndex], splitter) + offset);
            };
            return comparisonFunction;
        }

        
    };

    auto operator()(size_t splitIndex, TransformTag){
        std::pair<SplittingVector, OffSetType>& splitPair = splittingVectors.at(splitIndex);
        if constexpr(isAlignedArray_v<SplittingVector>){
            auto comparisonFunction = [=, 
                                    &data = std::as_const(this->dataSource), 
                                    splitter = SplittingView(splitPair.first),
                                    offset = splitPair.second]
                                    (size_t comparisonIndex) -> bool{
                    return 0.0 < (Dot(data[comparisonIndex], splitter) + offset);
            };
            return comparisonFunction;
        } else {
            auto comparisonFunction = [=, 
                                    &data = std::as_const(this->dataSource), 
                                    splitter = SplittingView(splitPair.first.begin(), splitPair.first.size()),
                                    offset = splitPair.second]
                                    (size_t comparisonIndex) -> bool{
                    return 0.0 < (Dot(data[comparisonIndex], splitter) + offset);
            };
            return comparisonFunction;
        }

        
    };
    
};

/*
template<typename DataEntry, typename SplittingVector>
struct EuclidianTransform{

    using OffSetType = typename SplittingVector::value_type;
    using SplittingView = DefaultDataView<SplittingVector>;

    const DataSet<DataEntry>& dataSource;
    std::unordered_map<size_t, std::pair<SplittingVector, OffSetType>>& splittingVectors;
    

    EuclidianTransform(const DataSet<DataEntry>& data,
                       std::unordered_map<size_t, std::pair<SplittingVector, typename SplittingVector::value_type>>& splits):
                       splittingVectors(splits),
                       dataSource(data){};

    
    

};
*/

}
#endif //RPT_SPLITTINGSCHEME_HPP