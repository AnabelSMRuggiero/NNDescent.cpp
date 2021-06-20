#ifndef MNISTDATA_HPP
#define MNISTDATA_HPP
#include <valarray>
#include <string>
#include <vector>

namespace nnd{

//TODO: Abstract this into the base for a data block
struct MNISTData{

    //std::valarray<unsigned char> rawData;

    std::vector<std::valarray<unsigned char>> samples;

    unsigned long numberOfSamples;
    size_t vectorLength;
    unsigned long imageWidth;
    unsigned long imageHeight;

    MNISTData(std::string& dataLocation, int targetMagic = 2051);
};


//Conceptual layout

//struct DataSet

//Presumably, each project would only need to instantiate for a single FloatType
template<typename DataEntry, typename FloatType>
struct DataBlock{

    size_t blockIndex;
    std::pair<size_t, size_t> dataIndexRange;

    
    std::vector<DataEntry> blockData;

    SpaceMetric<DataEntry, FloatType> distanceMetric;

    std::vector<FloatType> BulkDistances(std::vector<std::pair<size_t, size_t>> indicies){
        std::vector<FloatType> retVector;
        retVector.reserve(indicies.size());
        for (const auto& pair : indicies){
            retVector.pushBack(distanceMetric(blockData[indicies.first], blockData[indicies.second]));
        };
        return std::move(retVector);
    };


};



}

#endif //MNISTDATA_HPP