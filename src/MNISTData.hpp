/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

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
#include <span>

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
    size_t dataIndex;

};

inline bool operator==(const BlockIndex lhs, const BlockIndex& rhs){
    return (lhs.blockNumber == rhs.blockNumber) && (lhs.dataIndex == rhs.dataIndex);
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

    DataEntry& operator[](size_t i){
        return blockData[i];
    }

    DataEntry& operator[](BlockIndex i){
        static_assert(i.blockNumber == blockNumber);
        return blockData[i.dataIndex];
    }

    const DataEntry& operator[](size_t i) const{
        return blockData[i];
    }

    const DataEntry& operator[](BlockIndex i) const{
        //static_assert(i.blockNumber == blockNumber);
        return blockData[i.dataIndex];
    }

    size_t size() const{
        return blockData.size();
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
        return std::hash<size_t>()(index.blockNumber ^ std::hash<size_t>()(index.dataIndex));
    };

};


#endif //MNISTDATA_HPP