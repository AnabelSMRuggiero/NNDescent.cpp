#ifndef NND_DATASERIALIZATION_HPP
#define NND_DATASERIALIZATION_HPP

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

}


#endif