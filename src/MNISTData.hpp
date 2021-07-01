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

inline bool operator==(const BlockIndex lhs, const BlockIndex& rhs){
    return (lhs.blockNumber == rhs.blockNumber) && (lhs.blockIndex == rhs.blockIndex);
}


//Presumably, each project would only need to instantiate for a single FloatType
template<typename DataEntry>
struct DataBlock{

    size_t blockNumber;
    std::vector<DataEntry> blockData;

    DataBlock(): blockNumber(0), blockData(0){};

    DataBlock(const DataSet<DataEntry>& dataSource, std::span<const size_t> dataPoints, size_t blockNumber):
    blockNumber(blockNumber), blockData(){
        blockData.reserve(dataPoints.size());
        for (const size_t& index : dataPoints){
            blockData.push_back(dataSource.samples[index]);
        }
    }
    /*
    DataBlock(const DataSet<DataEntry>& dataSource, std::span<const size_t> dataPoints, size_t blockNumber, std::vector<BlockIndex>& indexRemapping):
    blockNumber(blockNumber), blockData(){
        blockData.reserve(dataPoints.size());
        for (size_t i = 0; i<dataPoints.size(); i+=1){
            size_t index = dataPoints[i];
            indexRemapping[index] = BlockIndex(blockNumber, i);
            blockData.push_back(dataSource.samples[index]);
        }
    }
    */
};



}

template<>
struct std::hash<nnd::BlockIndex>{

    size_t operator()(const nnd::BlockIndex& index) const noexcept{
        return std::hash<size_t>()(index.blockNumber ^ std::hash<size_t>()(index.blockIndex));
    };

};


#endif //MNISTDATA_HPP