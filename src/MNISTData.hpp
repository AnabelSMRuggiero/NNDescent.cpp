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


template<typename DataType>
struct DataSet{

    //std::valarray<unsigned char> rawData;

    std::vector<std::valarray<DataType>> samples;
    size_t sampleLength;
    size_t numberOfSamples;

    DataSet(std::string& dataLocation, size_t entryLength, size_t numSamples, const std::endian endianness):
        samples(0),
        sampleLength(entryLength),
        numberOfSamples(numSamples){
            std::ifstream dataStream;
            dataStream.open(dataLocation, std::ios_base::binary);        
            samples.reserve(numberOfSamples);
            if (endianness == std::endian::big){
                for (size_t i = 0; i < numberOfSamples; i+=1){
                    std::valarray<DataType> tempArr(sampleLength);
                    samples.emplace_back(std::move(tempArr));
                    for(size_t j = 0; j <sampleLength; j+=1){
                        samples[i][j] = ExtractData<DataType, std::endian::big>(dataStream);
                    }
                    //dataStream.read(reinterpret_cast<char *>(&(samples[i][0])), vectorLength);
                }
            } else if (endianness == std::endian::little){
                for (size_t i = 0; i < numberOfSamples; i+=1){
                    std::valarray<DataType> tempArr(sampleLength);
                    samples.emplace_back(std::move(tempArr));
                    for(size_t j = 0; j <sampleLength; j+=1){
                        samples[i][j] = ExtractData<DataType, std::endian::little>(dataStream);
                    }
                    //dataStream.read(reinterpret_cast<char *>(&(samples[i][0])), vectorLength);
                }
            }

    }
};

//Conceptual layout

//struct DataSet

//Presumably, each project would only need to instantiate for a single FloatType
template<typename DataEntry, typename FloatType>
struct DataBlock{

    size_t blockIndex;
    std::pair<size_t, size_t> dataIndexRange;

    
    std::vector<DataEntry> blockData;

    SpaceMetric<std::valarray<DataEntry>, FloatType> distanceMetric;

    std::vector<FloatType> BulkDistances(std::vector<std::pair<size_t, size_t>> indicies){
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