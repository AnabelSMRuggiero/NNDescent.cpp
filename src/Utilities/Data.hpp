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
#include <cassert>
#include <algorithm>


#include "Utilities/DataSerialization.hpp"
#include "Utilities/Metrics/SpaceMetrics.hpp"
#include "Type.hpp"

namespace nnd{


template<typename DataEntry>
struct DataSet{
    using ElementType = typename DataEntry::value_type;
    using DataView = typename DefaultDataView<DataEntry>::ViewType;
    
    using iterator = typename std::vector<DataEntry>::iterator;
    using const_iterator = typename std::vector<DataEntry>::const_iterator;
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
            //std::cout << dataStream.is_open();      
            samples.reserve(numberOfSamples);
            for (size_t i = 0; i < numberOfSamples; i+=1){
                samples.push_back(extractionFunction(dataStream, entryLength));
                //for (auto& entry: samples.back()) std::cout << entry << std::endl;
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

    constexpr iterator begin() noexcept{
        return samples.begin();
    }

    constexpr const_iterator begin() const noexcept{
        return samples.begin();
    }

    constexpr const_iterator cbegin() const noexcept{
        return samples.cbegin();
    }

    constexpr iterator end() noexcept{
        return samples.end();
    }

    constexpr const_iterator end() const noexcept{
        return samples.end();
    }

    constexpr const_iterator cend() const noexcept{
        return samples.cend();
    }

};

template<typename DataEntry>
void NormalizeDataSet(DataSet<DataEntry>& dataSet){
    for (auto& entry: dataSet){
        Normalize(entry);
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






template<typename DataEntry, size_t alignment>
size_t EntryPadding(const size_t entryLength){
    size_t entryBytes = sizeof(DataEntry)*entryLength;
    size_t excessBytes = entryBytes%alignment;
    if (excessBytes == 0) return 0;
    size_t paddingBytes = alignment - excessBytes;
    size_t paddingEntires = paddingBytes/sizeof(DataEntry);
    assert(paddingEntires*sizeof(DataEntry) == paddingBytes);
    return paddingBytes/sizeof(DataEntry);
    //((sizeof(DataType)*entryLength)%alignment > 0) ? alignment - entryLength%alignment : 0
}


//Presumably, each project would only need to instantiate for a single FloatType
template<typename DataType, size_t align = 32>
    requires (alignof(DataType) <= sizeof(DataType))
struct DataBlock{
    using value_type = DataType;
    using DataView = AlignedSpan<DataType, align>;
    using ConstDataView = AlignedSpan<const DataType, align>;

    static constexpr size_t alignment = align;

    size_t blockNumber;
    size_t numEntries;
    size_t entryLength;
    size_t lengthWithPadding;
    AlignedArray<DataType, alignment> blockData;
    //std::vector<DataEntry> blockData;


    DataBlock() = default;

    DataBlock(const size_t numEntries, const size_t entryLength, size_t blockNumber):
        blockNumber(blockNumber), 
        numEntries(numEntries),
        entryLength(entryLength),
        lengthWithPadding(entryLength + EntryPadding<DataType, alignment>(entryLength)),
        blockData(lengthWithPadding*numEntries){};

    template<typename DataEntry>
        requires std::same_as<typename DataEntry::value_type, DataType>
    DataBlock(const DataSet<DataEntry>& dataSource, std::span<const size_t> dataPoints, const size_t entryLength, size_t blockNumber):
        blockNumber(blockNumber), 
        numEntries(dataPoints.size()),
        entryLength(entryLength),
        lengthWithPadding(entryLength + EntryPadding<DataType, alignment>(entryLength)),
        blockData(lengthWithPadding*dataPoints.size()){
        //blockData = AlignedArray<DataType, alignment>((entryLength + entryPadding));
        //blockData.reserve(dataPoints.size());
        for (DataType* ptrIntoBlock = blockData.begin(); const size_t& index : dataPoints){
            //blockData.push_back(dataSource.samples[index]);
            const DataEntry& pointToCopy = dataSource.samples[index];
            std::copy(pointToCopy.begin(), pointToCopy.end(), ptrIntoBlock);
            ptrIntoBlock += lengthWithPadding;
        }
    }

    DataBlock(DataBlock&& rhs) = default;

    DataBlock& operator=(DataBlock&& rhs) = default;

    //DataBlock()

    DataView operator[](size_t i){
        AlignedPtr<value_type, alignment> ptr = blockData.GetAlignedPtr(lengthWithPadding);
        ptr += i;
        return DataView(ptr, entryLength);
        //return blockData[i];
    }
    /*
    DataView operator[](BlockIndecies i){
        //static_assert(i.blockNumber == blockNumber);
        return operator[](i.dataIndex);
    }
    */
    ConstDataView operator[](size_t i) const{
        AlignedPtr<const value_type, alignment> ptr = blockData.GetAlignedPtr(lengthWithPadding);
        ptr += i;
        return ConstDataView(ptr, entryLength);
    }
    /*
    ConstDataView operator[](BlockIndecies i) const{
        //static_assert(i.blockNumber == blockNumber);
        return operator[](i.dataIndex);
    }
    */
    size_t size() const{
        return numEntries;
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







#endif //MNISTDATA_HPP