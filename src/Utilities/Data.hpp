/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_DATA_HPP
#define NND_DATA_HPP
#include <valarray>
#include <string>
#include <vector>
#include <bit>
#include <ranges>
#include <iostream>
#include <fstream>
#include <type_traits>
#include <span>


#include "Utilities/DataSerialization.hpp"
#include "Type.hpp"

namespace nnd{


template<typename DataEntry>
struct DataSet{
    using ElementType = typename DataEntry::value_type;
    using DataView = typename DefaultDataView<DataEntry>::ViewType;
    
    //std::valarray<unsigned char> rawData;

    std::vector<DataEntry> samples;
    size_t sampleLength;
    size_t numberOfSamples;
    size_t indexStart;

    DataSet(std::string& dataLocation, size_t entryLength, size_t numSamples, DataExtractor<DataEntry> extractionFunction):
        samples(),
        sampleLength(entryLength),
        numberOfSamples(numSamples),
        indexStart(0){
            std::ifstream dataStream;
            dataStream.open(dataLocation, std::ios_base::binary);        
            samples.reserve(numberOfSamples);
            for (size_t i = 0; i < numberOfSamples; i+=1){
                samples.push_back(extractionFunction(dataStream, entryLength));
                //dataStream.read(reinterpret_cast<char *>(&(samples[i][0])), vectorLength);
            };
    }

    size_t IndexStart() const{
        return indexStart;
    }

    DataEntry& operator[](size_t i){
        return samples[i];
    }

    const DataEntry& operator[](size_t i) const{
        return samples[i];
    }

    size_t size() const{
        return samples.size();
    }

};

//Conceptual layout

//struct DataSet
/*
template<typename BlockNumberType = size_t, typename DataIndexType = size_t>
struct BlockIndecies{
    // The block a data point exists in
    BlockNumberType blockNumber;
    // The index within that block
    DataIndexType dataIndex;

};
*/




inline bool operator==(const BlockIndecies lhs, const BlockIndecies& rhs){
    return (lhs.blockNumber == rhs.blockNumber) && (lhs.dataIndex == rhs.dataIndex);
}


//Presumably, each project would only need to instantiate for a single FloatType
template<typename DataEntry>
struct DataBlock{
    using ElementType = typename DataEntry::value_type;
    using DataView = typename DefaultDataView<DataEntry>::ViewType;

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

    //DataBlock()

    DataEntry& operator[](size_t i){
        return blockData[i];
    }

    DataEntry& operator[](BlockIndecies i){
        static_assert(i.blockNumber == blockNumber);
        return blockData[i.dataIndex];
    }

    const DataEntry& operator[](size_t i) const{
        return blockData[i];
    }

    const DataEntry& operator[](BlockIndecies i) const{
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

template<typename DataEntry, typename TargetDataType, std::endian endianness>
void SerializeDataSet(const DataSet<DataEntry>& dataSet, const std::string filePath){

    std::ofstream outStream(filePath, std::ios_base::binary);

    for(const auto& sample : dataSet.samples){
        for (const auto value : sample){
            TargetDataType valueToSerialize = static_cast<TargetDataType>(value);
            SerializeData<TargetDataType, endianness>(outStream, value);
        }
    };

}



}

template<>
struct std::hash<nnd::BlockIndecies>{

    size_t operator()(const nnd::BlockIndecies& index) const noexcept{
        return std::hash<size_t>()(index.blockNumber ^ std::hash<size_t>()(index.dataIndex));
    };

};





#endif //MNISTDATA_HPP