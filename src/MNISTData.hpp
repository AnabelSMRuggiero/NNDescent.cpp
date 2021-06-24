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

template<typename DataType>
concept TriviallyCopyable = std::is_trivially_copyable<DataType>::value == true;

template<TriviallyCopyable DataType, std::endian DataEndianness>
void SerializeData(std::ofstream& dataStream, const DataType& dataEntry){
    //Don't wanna deal with mixed endianess. Unlikely to be an issue I need to deal with
    static_assert((DataEndianness == std::endian::big) || (DataEndianness == std::endian::little));
    static_assert((std::endian::native == std::endian::big) || (std::endian::native == std::endian::little));

    //If the endianness of the data coming in matches the system data, we can stream it right in.
    //Otherwise, we gotta reverse the order of bytes.
    constexpr int endianMod = (DataEndianness == std::endian::native) ? 0 : 1;
    constexpr int numBytes = sizeof(DataType);

    const unsigned char* dataRep = reinterpret_cast<const unsigned char*>(&dataEntry);
    //array indexing jank to avoid branches
    const unsigned char* start = &dataRep[(numBytes) * endianMod];
    const unsigned char* end = &dataRep[(numBytes) * (1 - endianMod)];

    //1 if native = data, -1 otherwise
    std::ptrdiff_t pointerIncrement = (1 + -2*endianMod);

    for (; start != end; start += pointerIncrement){
        dataStream.put(*(start - endianMod));
    }

    return;
};

template<TriviallyCopyable DataType, std::endian DataEndianness>
DataType ExtractData(std::ifstream &dataStream){
    //Don't wanna deal with mixed endianess. Unlikely to be an issue I need to deal with
    static_assert((DataEndianness == std::endian::big) || (DataEndianness == std::endian::little));
    static_assert((std::endian::native == std::endian::big) || (std::endian::native == std::endian::little));

    //If the endianness of the data coming in matches the system data, we can stream it right in.
    //Otherwise, we gotta reverse the order of bytes.
    constexpr int endianMod = (DataEndianness == std::endian::native) ? 0 : 1;
    constexpr int numBytes = sizeof(DataType);

    char retVal[numBytes];
    //array indexing jank to avoid branches
    char* start = &retVal[(numBytes) * endianMod];
    char* end = &retVal[(numBytes) * (1 - endianMod)];

    //1 if native = data, -1 otherwise
    std::ptrdiff_t pointerIncrement = (1 + -2*endianMod);

    for (; start != end; start += pointerIncrement){
        *(start - endianMod) = dataStream.get();
    }

    return *(DataType*)retVal;
};

template<typename DataEntry>
using DataExtractor = DataEntry (*)(std::ifstream, size_t);

template<typename NumericType, std::endian DataEndianness>
std::valarray<NumericType> ExtractNumericArray(std::ifstream dataStream, size_t entryLength){
    std::valarray<NumericType> sample(entryLength);
    for(size_t i = 0; i <entryLength; i+=1){
        sample[i] = ExtractData<DataType, DataEndianness>(dataStream);
    }
}

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
                samples[i] = extractionFunction(dataStream, entryLength);
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

    DataBlock(const DataSet<DataEntry>& dataSource, std::span<size_t> dataPoints, SpaceMetric<DataEntry, FloatType> metric, size_t blockNumber):
    blockNumber(blockNumber), blockData(), distanceMetric(metric){
        blockData.reserve(dataPoints.size());
        for (const size_t& index : dataPoints){
            blockData.push_back(dataSource.samples[index]);
        }
    }

    std::vector<FloatType> BulkDistances(std::vector<BlockIndex> indicies){
        std::vector<FloatType> retVector;
        retVector.reserve(indicies.size());
        for (const auto& pair : indicies){
            retVector.pushBack(distanceMetric<std::valarray<DataEntry>, FloatType>(blockData[pair.first], blockData[pair.second]));
        };
        return std::move(retVector);
    };


};



}

#endif //MNISTDATA_HPP