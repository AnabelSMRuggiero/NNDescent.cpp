/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef RPT_SPLITTINGSCHEME_HPP
#define RPT_SPLITTINGSCHEME_HPP

#include <valarray>
#include <functional>
#include "Utilities/Metrics/SpaceMetrics.hpp"

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




/*
    Oh boy, oh boy; this type allias is a mouthful.
    Short version: A Splitting Scheme is a function/functor.
    It can take
        size_t: the splitting index, first split being 0
        pair<size_t, size_t>: two valid points to contruct a splitting plane if one needs to be built.

    It returns(as a pair)
        A function: bool f(size_t dataIndex)
*/
//Gah, curse the indirection, but I'll use std::function for now here
using TrainingSplittingScheme = std::function<std::function<bool(size_t)> (size_t, std::pair<size_t, size_t>)>;
using TransformingSplittingScheme = std::function<std::function<bool(size_t)> (size_t)>;

//The serial case
template<typename DataEntry, typename SplittingVector>
struct EuclidianTrain{

    using DistType = typename SplittingVector::value_type;
    const DataSet<DataEntry>& dataSource;
    std::unordered_map<size_t, std::pair<SplittingVector, DistType>> splittingVectors;
    DistType projectionOffset;

    EuclidianTrain(const DataSet<DataEntry>& data) : dataSource(data), splittingVectors(){};

    std::function<bool(size_t)> operator()(size_t splitIndex, std::pair<size_t, size_t> splittingPoints){
        
        SplittingVector splittingVector;

        // For right now at least, in the serial case I want to be able to get a new splitting vector
        //if (splittingVectors.find(splitIndex) == splittingVectors.end()){
        splittingVector = EuclidianSplittingPlaneNormal<DataEntry, DistType>(dataSource[splittingPoints.first], dataSource[splittingPoints.second]);


        DistType projectionOffset = 0;
        for (size_t i = 0; i<dataSource[splittingPoints.first].size(); i+=1){
            projectionOffset -= splittingVector[i] * DistType(dataSource[splittingPoints.first][i] + dataSource[splittingPoints.second][i])/2.0;
        };

        splittingVectors[splitIndex] = std::pair<SplittingVector, DistType>(std::move(splittingVector), projectionOffset);

        //};
              
        auto comparisonFunction = [=, 
                                   &data = std::as_const(this->dataSource), 
                                   &splitter = splittingVectors[splitIndex].first,
                                   offset = splittingVectors[splitIndex].second]
                                   (size_t comparisonIndex) -> bool{
                /*
                FloatType distanceFromPlane = (Dot(data[comparisonIndex], splitter) + offset);
                //std::cout << distanceFromPlane << std::endl;
                bool result = 0.0 < distanceFromPlane;
                */
                return 0.0 < (Dot(data[comparisonIndex], splitter) + offset);

        };
        return std::function<bool(size_t)> (comparisonFunction);
    };
    /*
    bool operator()(size_t comparisonIndex){
        return 0 < (Dot(dataSource[comparisonIndex], splittingVector) - offset)
    }
    */

};


template<typename DataEntry, typename SplittingVector>
struct EuclidianTransform{

    const DataSet<DataEntry>& dataSource;
    // Long term I need to make sure this is owned elsewhere
    std::unordered_map<size_t, std::pair<SplittingVector, typename SplittingVector::value_type>>& splittingVectors;
    typename SplittingVector::value_type projectionOffset;

    EuclidianTransform(const DataSet<DataEntry>& data,
                       std::unordered_map<size_t, std::pair<SplittingVector, typename SplittingVector::value_type>>& splits):
                       splittingVectors(splits),
                       dataSource(data){};

    std::function<bool(size_t)> operator()(size_t splitIndex){
        
        auto comparisonFunction = [=, 
                                   &data = std::as_const(this->dataSource), 
                                   &splitter = splittingVectors[splitIndex].first,
                                   offset = splittingVectors[splitIndex].second]
                                   (size_t comparisonIndex) -> bool{

                /*
                FloatType distanceFromPlane = (Dot(data[comparisonIndex], splitter) + offset);
                //std::cout << distanceFromPlane << std::endl;
                bool result = 0.0 < distanceFromPlane;
                */
                return 0.0 < (Dot(data[comparisonIndex], splitter) + offset);

        };
        return std::function<bool(size_t)>(comparisonFunction);
    };
    /*
    bool operator()(size_t comparisonIndex){
        return 0 < (Dot(dataSource[comparisonIndex], splittingVector) - offset)
    }
    */

};


}
#endif //RPT_SPLITTINGSCHEME_HPP