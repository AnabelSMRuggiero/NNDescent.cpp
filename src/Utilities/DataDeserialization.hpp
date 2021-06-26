#ifndef NND_DATADESERIALIZATION_HPP
#define NND_DATADESERIALIZATION_HPP

#include <type_traits>
#include <bit>
#include <valarray>
#include <fstream>

namespace nnd{

template<typename DataType>
concept TriviallyCopyable = std::is_trivially_copyable<DataType>::value == true;

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

template<typename NumericType, std::endian DataEndianness>
std::valarray<NumericType> ExtractNumericArray(std::ifstream& dataStream, size_t entryLength){
    std::valarray<NumericType> sample(entryLength);
    for(size_t i = 0; i <entryLength; i+=1){
        sample[i] = ExtractData<NumericType, DataEndianness>(dataStream);
    }
    return sample;
};




}

#endif