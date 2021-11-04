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
#include <functional>
#include <memory_resource>

#include "./Type.hpp"
#include "./DelayConstruct.hpp"
namespace nnd{



template<typename DataType, std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream, typename... Ts>
DataType Extract(StreamType&& dataStream, Ts&&... ts) = delete;


template<typename Extractee, std::endian DataEndianness, typename StreamType, typename... Ts>
constexpr bool hasStaticDeserialize = requires(StreamType&& inFile, Ts&&... ts){
    {Extractee::template deserialize<DataEndianness>(std::forward<StreamType>(inFile), std::forward<Ts>(ts)...)} -> std::same_as<Extractee>;
    //Extractee::deserialize(...) used to work before I templated deserialize
};


struct ReverseEndiannessTag {};

template<typename Extractee, std::endian DataEndianness, typename StreamType, typename... Ts>
constexpr bool hasExtractConstructor = (DataEndianness == std::endian::native) ? 
                                           std::is_constructible_v<Extractee, StreamType&&, Ts...> :
                                           std::is_constructible_v<Extractee, StreamType&&, Ts..., ReverseEndiannessTag>;


template<typename Extractee>
struct ExtractTag{
    using type = Extractee;
};

template<typename Extractee, std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream, typename... Ts>
constexpr static bool extractDispatchable = requires(StreamType&& inFile, Ts&&... ts){
    {Extract<DataEndianness, StreamType>(std::forward<StreamType>(inFile), std::forward<Ts>(ts)..., ExtractTag<Extractee>{})} -> std::same_as<Extractee>;
};
/*
template<typename Extractee, std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream, typename... Ts>
    requires extractDispatchable<Extractee, DataEndianness, StreamType, Ts...>
Extractee Extract(StreamType&& dataStream, Ts&&... ts){
    return Extract<DataEndianness, StreamType>(std::forward<StreamType>(dataStream), std::forward<Ts>(ts)..., ExtractTag<Extractee>{});
}
*/
template<typename Extractee, std::endian DataEndianness, typename StreamType, typename... Ts>
concept CustomExtract = hasStaticDeserialize<Extractee, DataEndianness, StreamType, Ts...> || 
                        hasExtractConstructor<Extractee, DataEndianness, StreamType, Ts...> ||
                        extractDispatchable<Extractee, DataEndianness, StreamType, Ts...>;

template<typename Extractee, std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream, typename... Ts>
    requires CustomExtract<Extractee, DataEndianness, StreamType, Ts...>
Extractee Extract(StreamType&& dataStream, Ts&&... ts){
    if constexpr (hasStaticDeserialize<Extractee, DataEndianness, StreamType, Ts...>){
        return Extractee::template deserialize<DataEndianness>(std::forward<StreamType>(dataStream), std::forward<Ts>(ts)...);
    } else if constexpr(hasExtractConstructor<Extractee, DataEndianness, StreamType, Ts...>){
        if constexpr(DataEndianness == std::endian::native){
            return Extractee{std::forward<StreamType>(dataStream), std::forward<Ts>(ts)...};
        } else {
            return Extractee{std::forward<StreamType>(dataStream), std::forward<Ts>(ts)..., ReverseEndiannessTag{}};
        }
    } else {
        return Extract<DataEndianness, StreamType>(std::forward<StreamType>(dataStream), std::forward<Ts>(ts)..., ExtractTag<Extractee>{});
    }
    
    //Extract<DataEndianness, StreamType>(std::forward<StreamType>(dataStream), std::forward<Ts>(ts)..., ExtractTag<Extractee>{});
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




template<std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream, typename ExtracteeA, typename ExtracteeB>
std::pair<ExtracteeA, ExtracteeB> Extract(StreamType&& dataStream, ExtractTag<std::pair<ExtracteeA, ExtracteeB>>){
    return {Extract<ExtracteeA, DataEndianness>(std::forward<StreamType>(dataStream)),
            Extract<ExtracteeB, DataEndianness>(std::forward<StreamType>(dataStream))};
}

template<std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream, typename ExtracteeA, typename ExtracteeB>
std::unordered_map<ExtracteeA, ExtracteeB> Extract(StreamType&& dataStream, ExtractTag<std::unordered_map<ExtracteeA, ExtracteeB>>){
    const size_t mapSize = Extract<size_t, DataEndianness>(std::forward<StreamType>(dataStream));
    
    std::unordered_map<ExtracteeA, ExtracteeB> retMap(mapSize);
    for(size_t i = 0; i<mapSize; i+=1) retMap.emplace(DelayConstruct<std::pair<const ExtracteeA, ExtracteeB>>([&](){
        return Extract<std::pair<ExtracteeA, ExtracteeB>>(dataStream);
    }));
    return retMap;
}

template<std::endian dataEndianess = std::endian::native, typename StreamType, typename ValueType, size_t alignment>
DynamicArray<ValueType, alignment> Extract(StreamType&& inFile, std::pmr::memory_resource* resource, ExtractTag<DynamicArray<ValueType, alignment>>){

    size_t arraySize = Extract<size_t, dataEndianess>(inFile);

    if constexpr (std::is_trivially_copyable_v<ValueType>){

        DynamicArray<ValueType, alignment> retArray(uninitTag, arraySize, resource);
        Extract<ValueType, dataEndianess>(inFile, retArray.begin(), retArray.end());

        return retArray;

    } else {

        DynamicArray<ValueType, alignment> retArray(arraySize, resource);
        for(auto&& element: retArray) element = Extract<ValueType, dataEndianess>(inFile);

        return retArray;
    }
}

template<std::endian dataEndianess = std::endian::native, typename StreamType, typename ValueType, size_t alignment>
DynamicArray<ValueType, alignment> Extract(StreamType&& inFile, ExtractTag<DynamicArray<ValueType, alignment>>){
    return Extract<dataEndianess>(inFile, std::pmr::get_default_resource(), ExtractTag<DynamicArray<ValueType, alignment>>{});
}

template<std::ranges::contiguous_range Extractee, std::endian DataEndianness = std::endian::native, typename StreamType = std::ifstream, typename... Ts>
    requires (!CustomExtract<Extractee, DataEndianness, StreamType> && TriviallyCopyable<std::ranges::range_value_t<Extractee>>)
Extractee Extract(StreamType&& dataStream){
    const size_t rangeSize = Extract<size_t, DataEndianness>(std::forward<StreamType>(dataStream));
    
    std::remove_cv_t<Extractee> range(rangeSize);
    Extract<std::ranges::range_value_t<Extractee>, DataEndianness>(std::forward<StreamType>(dataStream), range.data(), rangeSize);

        /*
        std::remove_cv_t<Extractee> range;
        
        std::ranges::for_each(range, [&](const auto&){
            DelayConstruct<std::ranges::range_value_t<Extractee>>([&](){
                return Extract<std::ranges::range_value_t<Extractee>>(std::forward<StreamType>(dataStream));
            })
        });
        */
    

    return range;
    //Extract<DataEndianness, StreamType>(std::forward<StreamType>(dataStream), std::forward<Ts>(ts)..., ExtractTag<Extractee>{});
};


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