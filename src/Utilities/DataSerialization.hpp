/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_DATASERIALIZATION_HPP
#define NND_DATASERIALIZATION_HPP

#include <filesystem>
#include <ranges>
#include "Data.hpp"

#include "DataDeserialization.hpp"

namespace nnd{

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



template<TriviallyCopyable DataType, typename Alloc = std::allocator<DataType>>
void SerializeVector(const std::vector<DataType, Alloc>& readVector, std::ofstream& outStream){
    //std::ofstream outStream(outputFile, std::ios_base::binary);
    SerializeData<size_t, std::endian::big>(outStream, readVector.size());

    outStream.write(reinterpret_cast<const char*>(&readVector.data()), sizeof(DataType)*readVector.size());

};

template<typename Serializee>
void Serialize(const Serializee& objToSerialize, std::ofstream& outFile) = delete;

template<typename Serializee>
concept SerializableClass = requires(Serializee objToSerialize, std::ofstream& outFile){
    objToSerialize.serialize(outFile);
};

template<SerializableClass Serializee>
void Serialize(const Serializee& objToSerialize, std::ofstream& outFile){
    objToSerialize.serialize(outFile);
}

template<typename Serializee>
    requires std::is_trivially_copyable_v<Serializee>
void Serialize(const Serializee& objToSerialize, std::ofstream& outFile){
    outFile.write(reinterpret_cast<const char*>(&objToSerialize), sizeof(Serializee));
}

template<std::ranges::contiguous_range ContRange>
    requires std::is_trivially_copyable_v<std::ranges::range_value_t<ContRange>>
void Serialize(const ContRange& rangeToSerialize, std::ofstream& outFile){
    Serialize(std::ranges::size(rangeToSerialize), outFile);
    outFile.write(reinterpret_cast<const char*>(std::ranges::data(rangeToSerialize)), sizeof(std::ranges::range_value_t<ContRange>) * std::ranges::size(rangeToSerialize));
}


auto BindSerializer(std::ofstream& outFile){

    return [&](const auto& dataToWrite){
        Serialize(dataToWrite, outFile);
    };
    
}

template<typename BlockDataType, size_t blockAlign>
void Serialize(const DataBlock<BlockDataType, blockAlign>& block, std::ofstream& outputFile){
    //std::ofstream outputFile(outputPath, std::ios_base::binary);
    //outputFile << block.size() << block.entryLength << block.lengthWithPadding;
    //outputFile.write(reinterpret_cast<char*>())

    auto outputFunc = BindSerializer(outputFile);
    outputFunc(block.size());
    outputFunc(block.entryLength);
    outputFunc(block.lengthWithPadding);

    outputFile.write(reinterpret_cast<const char*>(block.blockData.begin()), block.blockData.size()*sizeof(float));
}

template<typename BlockDataType>
void Serialize(const UnevenBlock<BlockDataType>& block, std::ofstream& outputFile){
    //std::ofstream outputFile(outputPath, std::ios_base::binary);
    //outputFile << block.size() << block.entryLength << block.lengthWithPadding;
    //outputFile.write(reinterpret_cast<char*>())

    auto outputFunc = BindSerializer(outputFile);
    outputFunc(block.size());
    outputFunc(block.IndexOffset());
    outputFunc(block.dataStorage.size());
                                                                             //this is already the size in bytes
    outputFile.write(reinterpret_cast<const char*>(block.dataStorage.get()), block.dataStorage.size());
}

}

#endif