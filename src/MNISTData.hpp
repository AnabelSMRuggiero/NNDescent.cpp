#ifndef MNISTDATA_HPP
#define MNISTDATA_HPP
#include <valarray>
#include <string>
#include <vector>
#include <bit>
#include <ranges>
#include <iostream>
#include <fstream>
#include <type_traits>

#include "NND/SpaceMetrics.hpp"
#include "Utilities/DataSerialization.hpp"

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

template<typename DataEntry>
struct DataSet{

    //std::valarray<unsigned char> rawData;

    std::vector<DataEntry> samples;
    size_t sampleLength;
    size_t numberOfSamples;

    DataSet(std::string& dataLocation, size_t entryLength, size_t numSamples, DataExtractor<DataEntry> extractionFunction):
        samples(0),
        sampleLength(entryLength),
        numberOfSamples(numSamples){
            std::ifstream dataStream;
            dataStream.open(dataLocation, std::ios_base::binary);        
            samples.reserve(numberOfSamples);
            for (size_t i = 0; i < numberOfSamples; i+=1){
                samples.emplace_back(extractionFunction(dataStream, entryLength));
                //dataStream.read(reinterpret_cast<char *>(&(samples[i][0])), vectorLength);
            };
    }

};

//Conceptual layout

//struct DataSet

struct BlockIndex{
    // The block a data point exists in
    size_t blockNumber;
    // The index within that block
    size_t blockIndex;

};

//Presumably, each project would only need to instantiate for a single FloatType
template<typename DataEntry, typename FloatType>
struct DataBlock{

    size_t blockNumber;
    std::vector<DataEntry> blockData;
    SpaceMetric<DataEntry, FloatType> distanceMetric;

    DataBlock(): blockNumber(0), blockData(0), distanceMetric(nullptr){};

    DataBlock(const DataSet<DataEntry>& dataSource, std::span<const size_t> dataPoints, SpaceMetric<DataEntry, FloatType> metric, size_t blockNumber):
    blockNumber(blockNumber), blockData(), distanceMetric(metric){
        blockData.reserve(dataPoints.size());
        for (const size_t& index : dataPoints){
            blockData.push_back(dataSource.samples[index]);
        }
    }

    std::vector<FloatType>& BulkDistances(std::vector<BlockIndex> indicies){
        std::vector<FloatType> retVector;
        retVector.reserve(indicies.size());
        for (const auto& pair : indicies){
            retVector.pushBack(distanceMetric<std::valarray<DataEntry>, FloatType>(blockData[pair.first], blockData[pair.second]));
        };
        return retVector;
    };


};



}

#endif //MNISTDATA_HPP