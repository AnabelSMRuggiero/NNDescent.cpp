#ifndef RPT_SPLITTINGSCHEME_HPP
#define RPT_SPLITTINGSCHEME_HPP

#include <valarray>
#include <functional>

namespace nnd{

template<typename DataType, typename FloatType>
std::valarray<FloatType> EuclidianSplittingPlaneNormal(const std::valarray<DataType>& pointA, const std::valarray<DataType>& pointB){
    std::valarray<FloatType> splittingLine(pointA.size());
    for (size_t i = 0; i < pointA.size(); i += 1){
        splittingLine[i] = FloatType(pointA[i]) - FloatType(pointB[i]);
    }
    FloatType splittingLineMag(0);
    for (FloatType i : splittingLine){
        splittingLineMag += i*i;
    }
    splittingLineMag = std::sqrt(splittingLineMag);
    splittingLine /= splittingLineMag;

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
using SplittingScheme = std::function<std::function<bool(size_t)> (size_t, std::pair<size_t, size_t>)>;


template<typename FloatType, typename DataType>
struct EuclidianSplittingScheme{

    const std::vector<std::valarray<DataType>>& dataSource;
    std::unordered_map<size_t, std::pair<std::valarray<FloatType>, FloatType>> splittingVectors;
    FloatType projectionOffset;



    EuclidianSplittingScheme(const DataSet<DataType>& data) : dataSource(data.samples), splittingVectors(){};

    std::function<bool(size_t)> operator()(size_t splitIndex, std::pair<size_t, size_t> splittingPoints){
        
        std::valarray<FloatType> splittingVector;

        if (splittingVectors.find(splitIndex) == splittingVectors.end()){
            splittingVector = EuclidianSplittingPlaneNormal<DataType, FloatType>(dataSource[splittingPoints.first], dataSource[splittingPoints.second]);


            FloatType projectionOffset = 0;
            for (size_t i = 0; i<dataSource[splittingPoints.first].size(); i+=1){
                projectionOffset -= splittingVector[i] * FloatType(dataSource[splittingPoints.first][i] + dataSource[splittingPoints.second][i])/2.0;
            };

           splittingVectors[splitIndex] = std::pair<std::valarray<FloatType>, FloatType>(splittingVector, projectionOffset);

        };
        
        
        auto comparisonFunction = [=, 
                                   &data = std::as_const(this->dataSource), 
                                   &splitter = splittingVectors[splitIndex].first,
                                   offset = splittingVectors[splitIndex].second]
                                   (size_t comparisonIndex) -> bool{
                // This is some janky type conversion shenanigans here. I need to either
                // remove the assertion of arbitrary (but single) data type and turn everything into a float/double
                // or define these ops in terms of something other than valarrays.
                // I just wanna get prototyping rptrees done.
                // TODO: KILL THIS IMPLEMENTATION (jfc, this is NOT GOOD)
                std::valarray<FloatType> temporaryArr(data[comparisonIndex].size());
                for(size_t i = 0; i < temporaryArr.size(); i += 1){
                    temporaryArr[i] = FloatType(data[comparisonIndex][i]);
                }

                FloatType distanceFromPlane = (Dot(temporaryArr, splitter) + offset);
                //std::cout << distanceFromPlane << std::endl;
                bool result = 0.0 < distanceFromPlane;

                return result;

        };
        return std::function<bool(size_t)> (comparisonFunction);
    };
    /*
    bool operator()(size_t comparisonIndex){
        return 0 < (Dot(dataSource[comparisonIndex], splittingVector) - offset)
    }
    */

};


}
#endif //RPT_SPLITTINGSCHEME_HPP