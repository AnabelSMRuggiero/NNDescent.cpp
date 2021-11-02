/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_DATADESERIALIZATION_HPP
#define NND_DATADESERIALIZATION_HPP

#include <type_traits>
#include <bit>
#include <fstream>
#include <iterator>
#include <concepts>

#include "./Type.hpp"
namespace nnd{


template<typename DataType, std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream, typename... Ts>
DataType Extract(StreamType&& dataStream, Ts&&... ts) = delete;

template<typename Extractee, typename StreamType, typename... Ts>
constexpr bool hasStaticDeserialize = requires(StreamType&& inFile, Ts&&... ts){
    {Extractee::deserialize(std::forward<StreamType>(inFile), std::forward<Ts>(ts)...)} -> std::same_as<Extractee>;
};

template<typename Extractee, typename StreamType, typename... Ts>
concept ExtractableClass = hasStaticDeserialize<Extractee, StreamType, Ts...> || std::is_constructible_v<Extractee, StreamType&&, Ts...>;

template<typename DataType, std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream, typename... Ts>
    requires ExtractableClass<DataType, StreamType>
DataType Extract(StreamType&& dataStream, Ts&&... ts){
    if constexpr (hasStaticDeserialize<DataType, StreamType>){
        return DataType::deserialize(std::forward<StreamType>(dataStream), std::forward<Ts>(ts)...);
    } else {
        return DataType{std::forward<StreamType>(dataStream), std::forward<Ts>(ts)...};
    }
};



template<TriviallyCopyable DataType, std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream>
DataType Extract(StreamType&& dataStream){
    static_assert((DataEndianness == std::endian::big) || (DataEndianness == std::endian::little));
    static_assert((std::endian::native == std::endian::big) || (std::endian::native == std::endian::little));

    

    if constexpr (DataEndianness == std::endian::native){
        DataType extract;

        dataStream.read(reinterpret_cast<char*>(&extract), sizeof(DataType));

        return extract;
    } else {
        static_assert(DataEndianness == std::endian::native, "Not yet implemented, curse you Linux Mint for having a repo that doesn't have the cutting edge c++ compilers.");
        DataType extract;

        dataStream.read(reinterpret_cast<char*>(&extract), sizeof(DataType));

        return extract;
    }

    
}

template<TriviallyCopyable DataType, std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream>
void Extract(StreamType&& dataStream, DataType* start, size_t count){
    static_assert((DataEndianness == std::endian::big) || (DataEndianness == std::endian::little));
    static_assert((std::endian::native == std::endian::big) || (std::endian::native == std::endian::little));

    

    if constexpr (DataEndianness == std::endian::native){
        

        dataStream.read(reinterpret_cast<char*>(start), sizeof(DataType)*count);

        
    } else {
        static_assert(DataEndianness == std::endian::native, "Not yet implemented, curse you Linux Mint for having a repo that doesn't have the cutting edge c++ compilers.");
    }

}

template<TriviallyCopyable DataType, std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream>
void Extract(StreamType&& dataStream, DataType* start, DataType* end){
    static_assert((DataEndianness == std::endian::big) || (DataEndianness == std::endian::little));
    static_assert((std::endian::native == std::endian::big) || (std::endian::native == std::endian::little));

    

    if constexpr (DataEndianness == std::endian::native){
        

        dataStream.read(reinterpret_cast<char*>(start), sizeof(DataType)*std::distance(start, end));

        
    } else {
        static_assert(DataEndianness == std::endian::native, "Not yet implemented, curse you Linux Mint for having a repo that doesn't have the cutting edge c++ compilers.");
    }

}


template<typename Extractee>
struct ExtractTag{
    using type = Extractee;
};

//template<typename DataType, std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream, typename... Ts>
//DataType Extract(StreamType&& dataStream, Ts&&... ts) = delete;

template<typename Extractee, std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream, typename... Ts>
constexpr static bool extractDispatchable = requires(StreamType&& inFile, Ts&&... ts){
    {Extract<DataEndianness, StreamType>(std::forward<StreamType>(inFile), ts..., ExtractTag<Extractee>{})} -> std::same_as<Extractee>;
};

template<typename Extractee, std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream, typename... Ts>
    requires extractDispatchable<Extractee, DataEndianness, StreamType, Ts...>
Extractee Extract(StreamType&& dataStream, Ts&&... ts){
    return Extract<DataEndianness, StreamType>(std::forward<StreamType>(dataStream), std::forward<Ts>(ts)..., ExtractTag<Extractee>{});
}

template<std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream, typename ExtracteeA, typename ExtracteeB>
std::pair<ExtracteeA, ExtracteeB> Extract(StreamType&& dataStream, ExtractTag<std::pair<ExtracteeA, ExtracteeB>>){
    return {Extract<ExtracteeA, DataEndianness>(std::forward<StreamType>(dataStream)),
            Extract<ExtracteeB, DataEndianness>(std::forward<StreamType>(dataStream))};
}

template<typename Extractee, std::endian DataEndianness = std::endian::native>
struct Extractor{

    template<typename StreamType, typename... Ts>
    Extractee operator()(StreamType&& dataStream, Ts&&... ts) const {
        return Extract<Extractee, DataEndianness>(std::forward<StreamType>(dataStream), std::forward<Ts>(ts)...);
    }
};

template<typename Extractee, std::endian DataEndianness = std::endian::native>
inline constexpr Extractor<Extractee, DataEndianness> extract{};


/*
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
using DataExtractor = DataEntry (*)(std::ifstream&, size_t);

template<typename Container, std::endian DataEndianness>
Container ExtractNumericArray(std::ifstream& dataStream, size_t entryLength){
    Container sample(entryLength);
    for(size_t i = 0; i <entryLength; i+=1){
        sample[i] = ExtractData<typename Container::value_type, DataEndianness>(dataStream);
    }
    return sample;
};
*/



}

#endif