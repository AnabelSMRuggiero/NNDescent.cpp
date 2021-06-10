#ifndef MNISTDATA_HPP
#define MNISTDATA_HPP
#include <valarray>
#include <string>

namespace nnd{

//TODO: Abstract this into the base for a data block
struct MNISTData{

    std::valarray<unsigned char> rawData;

    unsigned long numberOfSamples;
    size_t vectorLength;
    unsigned long imageWidth;
    unsigned long imageHeight;

    MNISTData(std::string& dataLocation, int targetMagic = 2051);
};

}

#endif //MNISTDATA_HPP