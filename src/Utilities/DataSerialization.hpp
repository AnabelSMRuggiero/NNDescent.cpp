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
#include <utility>

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


template<typename Range>
concept ContiguousCopyableRange = std::ranges::contiguous_range<Range> && std::is_trivially_copyable_v<std::ranges::range_value_t<Range>>;

template<typename Serializee>
void Serialize(const Serializee& objToSerialize, std::ofstream& outFile) = delete;

//template<std::endian DataEndianness = std::endian::native>
struct Serializer{

    template<typename Serializee, typename StreamType>
    void operator()(Serializee&& objToSerialize, StreamType&& dataStream) const;
    /* {
        Serialize(std::forward<Serializee>(objToSerialize), std::forward<StreamType>(dataStream));
    }*/
};

//template<std::endian DataEndianness = std::endian::native>
inline constexpr Serializer serialize{};

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

template<ContiguousCopyableRange ContRange>
    //requires std::is_trivially_copyable_v<std::ranges::range_value_t<ContRange>>
void Serialize(const ContRange& rangeToSerialize, std::ofstream& outFile){
    serialize(std::ranges::size(rangeToSerialize), outFile);
    outFile.write(reinterpret_cast<const char*>(std::ranges::data(rangeToSerialize)), sizeof(std::ranges::range_value_t<ContRange>) * std::ranges::size(rangeToSerialize));
}

template<std::ranges::input_range InputRange>
    requires (!SerializableClass<InputRange> && !ContiguousCopyableRange<InputRange>)
void Serialize(const InputRange& rangeToSerialize, std::ofstream& outFile){
    serialize(std::ranges::size(rangeToSerialize), outFile);
    std::ranges::for_each(rangeToSerialize, [&](const auto& rangeElement){ serialize(rangeElement, outFile);});
    //outFile.write(reinterpret_cast<const char*>(std::ranges::data(rangeToSerialize)), sizeof(std::ranges::range_value_t<ContRange>) * std::ranges::size(rangeToSerialize));
}

template<typename SerializeeA, typename SerializeeB>
void Serialize(const std::pair<SerializeeA, SerializeeB>& pairToSerialize, std::ofstream& outStream){
    serialize(pairToSerialize.first, outStream);
    serialize(pairToSerialize.second, outStream);
}

//constexpr bool isInput = std::input_iterator<int*>;



auto BindSerializer(std::ofstream& outFile){

    return [&](const auto& dataToWrite){
        serialize(dataToWrite, outFile);
    };
    
}

template<typename Serializee, typename StreamType>
void Serializer::operator()(Serializee&& objToSerialize, StreamType&& dataStream) const{
    Serialize(std::forward<Serializee>(objToSerialize), std::forward<StreamType>(dataStream));
}



}

#endif