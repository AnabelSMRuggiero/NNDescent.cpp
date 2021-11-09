/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_DATASET_HPP
#define NND_DATASET_HPP

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <exception>

#include "../Type.hpp"

namespace nnd{

void OpenData(std::filesystem::path dataPath, const size_t vectorLength, const size_t endEntry, const size_t startEntry = 0){
    using DataType = float;

    std::ifstream dataStream(dataPath, std::ios_base::binary);
    if (!dataStream.is_open()) throw std::filesystem::filesystem_error("File could not be opened.", dataPath, std::make_error_code(std::io_errc::stream));

    const size_t numElements = vectorLength * (endEntry-startEntry);

    DynamicArray<float> dataArr(uninitTag, vectorLength * (endEntry-startEntry));

    dataStream.read(reinterpret_cast<char*>(dataArr.begin()), numElements*sizeof(DataType));

    
}

template<typename DataEntry, size_t alignment>
size_t EntryPadding(const size_t entryLength){
    size_t entryBytes = sizeof(DataEntry)*entryLength;
    size_t excessBytes = entryBytes%alignment;
    if (excessBytes == 0) return 0;
    size_t paddingBytes = alignment - excessBytes;
    size_t paddingEntries = paddingBytes/sizeof(DataEntry);
    assert(paddingEntries*sizeof(DataEntry) == paddingBytes);
    return paddingBytes/sizeof(DataEntry);
    //((sizeof(DataType)*entryLength)%alignment > 0) ? alignment - entryLength%alignment : 0
}

template<typename DataType, size_t align=32>
struct DataSet{
    using value_type = DataType;
    using DataView = typename DefaultDataView<DynamicArray<DataType, align>>::ViewType;
    using ConstDataView = typename DefaultDataView<DynamicArray<DataType, align>>::ViewType;
    //using iterator = typename std::vector<DataEntry>::iterator;
    //using const_iterator = typename std::vector<DataEntry>::const_iterator;
    //std::valarray<unsigned char> rawData;
    static constexpr size_t alignment = align;
    

    private:
    //DynamicArray<DataType> samples;
    size_t sampleLength;
    size_t numberOfSamples;
    size_t indexStart;
    DynamicArray<DataType, align> samples;

    public:
    DataSet(std::filesystem::path dataPath, const size_t entryLength, const size_t endEntry, const size_t startEntry = 0, const size_t fileHeader = 0):
        sampleLength(entryLength),
        numberOfSamples(endEntry-startEntry),
        indexStart(startEntry),
        samples(){
            size_t padding = EntryPadding<DataType, alignment>(entryLength);

            assert(padding==0); 
            std::ifstream dataStream(dataPath, std::ios_base::binary);

            if (!dataStream.is_open()) throw std::filesystem::filesystem_error("File could not be opened.", dataPath, std::make_error_code(std::io_errc::stream));

            const size_t numElements = sampleLength * numberOfSamples;

            DynamicArray<DataType, align> dataArr(uninitTag, numElements);

            dataStream.seekg(fileHeader + entryLength*startEntry);
            dataStream.read(reinterpret_cast<char*>(dataArr.begin()), numElements*sizeof(DataType));

            this->samples = std::move(dataArr);
            //for (const auto& entry: this->samples) std::cout << entry << std::endl;
            
    }


    size_t IndexStart() const{
        return indexStart;
    }
    /*
    DataView operator[](size_t i){
        return {samples.begin() + i*sampleLength, sampleLength};
    }

    ConstDataView operator[](size_t i) const{
        return {samples.begin() + i*sampleLength, sampleLength};
    }
    */
    DataView operator[](size_t i){
        value_type* dataPtr = samples.get();
        dataPtr += i * sampleLength;
        return DataView(MakeAlignedPtr(dataPtr, *this), sampleLength);
        /*
        AlignedPtr<value_type, alignment> ptr = samples.GetAlignedPtr(sampleLength);
        ptr += i;
        return DataView(ptr, sampleLength);
        */
        //return blockData[i];
    }
    
    ConstDataView operator[](size_t i) const{
        const value_type* dataPtr = samples.get();
        dataPtr += i * sampleLength;
        return ConstDataView(MakeAlignedPtr(dataPtr, *this), sampleLength);
        /*
        AlignedPtr<const value_type, alignment> ptr = samples.GetAlignedPtr(sampleLength);
        ptr += i;
        return ConstDataView(ptr, sampleLength);
        */
    }

    size_t size() const{
        return numberOfSamples;
    }

    size_t SampleLength() const{
        return sampleLength;
    }
    /*
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
    */

};

/*
struct DataIterator{
    const size_t sampleLength;
    DataType* pointedSample;

    DataIterator& operator++(){
        pointedSample + sampleLength;
        return *this;
    }

    DataIterator operator++(int){
        DataIterator copy = *this;
        pointedSample + sampleLength;
        return copy;
    }

    DataIterator& operator--(){
        pointedSample - sampleLength;
        return *this;
    }

    DataIterator operator--(int){
        DataIterator copy = *this;
        pointedSample - sampleLength;
        return copy;
    }

    DataIterator operator+(std::ptrdiff_t inc){
        DataIterator copy{sampleLength, pointedSample+inc*sampleLength};
        return copy;
    }

    DataIterator operator-(std::ptrdiff_t inc){
        DataIterator copy{sampleLength, pointedSample-inc*sampleLength};
        return copy;
    }

    std::ptrdiff_t operator-(DataIterator other){
        return (pointedSample - other.pointedSample)/sampleLength;
    }

    bool operator==(DataIterator other){
        return pointedSample == other.pointedSample;
    }

    DataView operator*(){
        return DataView{pointedSample, sampleLength};
    }
};
*/

template<typename DataEntry>
void NormalizeDataSet(DataSet<DataEntry>& dataSet){
    for (auto& entry: dataSet){
        Normalize(entry);
    }
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

#endif