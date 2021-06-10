//Basic procedure to pull in the data of the MNIST digits training set

#include <iostream>
#include <fstream>
#include <string>
#include <valarray>
#include <utility>
#include <new>
#include "MNISTData.hpp"

using namespace nnd;

unsigned long ExtractInt(std::ifstream &dataStream){
    unsigned long retVal = 0;
    for (size_t i = 0; i < 4; i += 1){
        retVal <<= 8;
        unsigned char numByte = dataStream.get();
        retVal |= numByte;
    }

    return retVal;
};



MNISTData::MNISTData(std::string& dataLocation, int targetMagic) {

    std::ifstream dataStream;
    dataStream.open(dataLocation, std::ios_base::binary);
    dataStream.fail();

    /*
    File format
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    */

    unsigned long magicnumber = ExtractInt(dataStream);
    
    if (magicnumber != targetMagic) throw ("Magic number in file does not match target number");

    numberOfSamples = ExtractInt(dataStream);
    imageWidth = ExtractInt(dataStream);
    imageHeight = ExtractInt(dataStream);
    vectorLength = imageHeight * imageWidth;

    //Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic> rawImages(numImages, imgRows*imgCols);

    //Todo: make a factory function so I can feed the array length to the constructor directly
    try {
        std::valarray<unsigned char> tempArr(size_t(numberOfSamples * imageWidth * imageHeight));
        rawData = std::move(tempArr);
    } catch (const std::bad_array_new_length &e){
        std::cout << e.what() << std::endl;
        throw e;
    }
    //this->rawData = std::move(tempArr);
    
    //.read() takes specifically a char *, and can't implicitly cast the unsigned char *
    dataStream.read(reinterpret_cast<char *>(&(rawData[0])), rawData.size());


};