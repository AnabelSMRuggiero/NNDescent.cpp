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
#include <filesystem>
#include <exception>
#include <memory_resource>

#include "Utilities/DataSerialization.hpp"
#include "Utilities/Metrics/SpaceMetrics.hpp"
#include "Type.hpp"

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
            /*
            std::ifstream dataStream;
            dataStream.open(dataLocation, std::ios_base::binary);
            //std::cout << dataStream.is_open();      
            samples.reserve(numberOfSamples);
            for (size_t i = 0; i < numberOfSamples; i+=1){
                samples.push_back(extractionFunction(dataStream, entryLength));
                //for (auto& entry: samples.back()) std::cout << entry << std::endl;
                //dataStream.read(reinterpret_cast<char *>(&(samples[i][0])), vectorLength);
            };
            */
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
        AlignedPtr<value_type, alignment> ptr = samples.GetAlignedPtr(sampleLength);
        ptr += i;
        return DataView(ptr, sampleLength);
        //return blockData[i];
    }
    
    ConstDataView operator[](size_t i) const{
        AlignedPtr<const value_type, alignment> ptr = samples.GetAlignedPtr(sampleLength);
        ptr += i;
        return ConstDataView(ptr, sampleLength);
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
    DynamicArray<DataType, alignment> blockData;
    //std::vector<DataEntry> blockData;


    DataBlock() = default;

    DataBlock(const size_t numEntries, const size_t entryLength, size_t blockNumber):
        blockNumber(blockNumber), 
        numEntries(numEntries),
        entryLength(entryLength),
        lengthWithPadding(entryLength + EntryPadding<DataType, alignment>(entryLength)),
        blockData(lengthWithPadding*numEntries){};

    //template<typename DataEntry>
    //    requires std::same_as<typename DataEntry::value_type, DataType>
    DataBlock(const DataSet<DataType>& dataSource, std::span<const size_t> dataPoints, const size_t entryLength, size_t blockNumber):
        blockNumber(blockNumber), 
        numEntries(dataPoints.size()),
        entryLength(entryLength),
        lengthWithPadding(entryLength + EntryPadding<DataType, alignment>(entryLength)),
        blockData(lengthWithPadding*dataPoints.size()){
        //blockData = AlignedArray<DataType, alignment>((entryLength + entryPadding));
        //blockData.reserve(dataPoints.size());
        for (DataType* ptrIntoBlock = blockData.begin(); const size_t& index : dataPoints){
            //blockData.push_back(dataSource.samples[index]);
            typename DataSet<DataType>::ConstDataView pointToCopy = dataSource[index];
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

template<typename ElementType>
struct UnevenBlockIterator{
    using value_type = std::span<ElementType>;
    using difference_type = std::ptrdiff_t;
    using reference = std::span<ElementType>;

    UnevenBlockIterator(const size_t* vertexStart, ElementType* vertexNeighbors): vertexStart(vertexStart), vertexNeighbors(vertexNeighbors) {}

    private:
    const size_t* vertexStart;
    ElementType* vertexNeighbors;

    public:
    UnevenBlockIterator& operator++(){
        vertexNeighbors += *(vertexStart+1) - *vertexStart;
        ++vertexStart;
        return *this;
    }

    UnevenBlockIterator operator++(int){
        UnevenBlockIterator copy = *this;
        vertexNeighbors += *(vertexStart+1) - *vertexStart;
        ++vertexStart;
        return copy;
    }

    UnevenBlockIterator& operator--(){
        vertexNeighbors -= *vertexStart - *(vertexStart-1);
        --vertexStart;
        return *this;
    }

    UnevenBlockIterator operator--(int){
        UnevenBlockIterator copy = *this;
        vertexNeighbors -= *vertexStart - *(vertexStart-1);
        --vertexStart;
        return copy;
    }

    UnevenBlockIterator operator+(std::ptrdiff_t inc){
        UnevenBlockIterator copy{vertexStart+inc, vertexNeighbors + (*(vertexStart+inc) - *vertexStart)};
        return copy;
    }

    UnevenBlockIterator operator-(std::ptrdiff_t inc){
        UnevenBlockIterator copy{vertexStart-inc, vertexNeighbors - (*vertexStart - *(vertexStart-inc))};
        return copy;
    }

    std::ptrdiff_t operator-(UnevenBlockIterator other){
        return vertexStart - other.vertexStart;
    }
    
    bool operator==(UnevenBlockIterator other){
        return vertexStart == other.vertexStart;
    }
    
    reference operator*(){
        return reference{vertexNeighbors, *(vertexStart+1) - *vertexStart};
    }

    reference operator[](size_t i){
        return *(*this + i);
    }

    UnevenBlockIterator& operator+=(std::ptrdiff_t inc){
        *this = *this + inc;
        return *this;
    }

    UnevenBlockIterator& operator-=(std::ptrdiff_t inc){
        *this = *this - inc;
        return *this;
    }

    auto operator<=>(UnevenBlockIterator& rhs){
        return vertexStart<=> rhs.vertexStart;
    }
};

template<typename ElementType>
    requires std::is_trivially_constructible_v<ElementType> && std::is_trivially_destructible_v<ElementType>
struct UnevenBlock{

    using iterator = UnevenBlockIterator<ElementType>;
    using const_iterator = UnevenBlockIterator<const ElementType>;
    using reference = std::span<ElementType>;
    using const_reference = std::span<const ElementType>;


    DynamicArray<std::byte, std::max(alignof(size_t), alignof(ElementType))> dataStorage;
    size_t numArrays;
    ElementType* firstIndex;


    UnevenBlock() = default;

    //Default Copy Constructor is buggy
    UnevenBlock(const UnevenBlock& other): dataStorage(other.dataStorage), numArrays(other.numArrays), firstIndex(nullptr){
        this->firstIndex =  static_cast<ElementType*>(static_cast<void*>(this->dataStorage.get()))   + (other.firstIndex - static_cast<const ElementType*>(static_cast<const void*>(other.dataStorage.get())));
    }
    //NewUndirectedGraph(size_t numVerticies, size_t numNeighbors): 
    //    verticies(numVerticies, std::vector<IndexType>(numNeighbors)){};

    //template<typename DistType>
    UnevenBlock(const size_t numBytes, const size_t numArrays, const size_t headerPadding, const size_t numIndecies, std::pmr::memory_resource* resource): dataStorage(numBytes, resource), numArrays(numArrays), firstIndex(nullptr){
        //std::pmr::polymorphic_allocator<std::byte> alloc(resource);
        //alloc.construct
        size_t* vertexStart = new (dataStorage.begin()) size_t[numArrays+1];
        //*vertexStart = 0;
        firstIndex = new (dataStorage.begin() + sizeof(size_t)*(numArrays+1) + headerPadding) ElementType[numIndecies];

        
        
    }

    size_t size() const noexcept{
        return numArrays;
    }
    
    constexpr iterator begin() noexcept{
        return iterator{static_cast<size_t*>(static_cast<void*>(dataStorage.begin())), firstIndex};
    }

    constexpr const_iterator begin() const noexcept{
        return const_iterator{static_cast<size_t*>(static_cast<void*>(dataStorage.begin())), firstIndex};
    }

    constexpr const_iterator cbegin() const noexcept{
        return const_iterator{static_cast<const size_t*>(static_cast<const void*>(dataStorage.begin())), firstIndex};
    }

    constexpr iterator end() noexcept{
        return iterator{static_cast<size_t*>(static_cast<void*>(dataStorage.begin()))+numArrays, static_cast<ElementType*>(static_cast<void*>(dataStorage.end()))};
    }

    constexpr const_iterator end() const noexcept{
        return const_iterator{static_cast<size_t*>(static_cast<void*>(dataStorage.begin()))+numArrays, static_cast<ElementType*>(static_cast<void*>(dataStorage.end()))};
    }

    constexpr const_iterator cend() const noexcept{
        return const_iterator{static_cast<const size_t*>(static_cast<const void*>(dataStorage.begin()))+numArrays, static_cast<const ElementType*>(static_cast<const void*>(dataStorage.end()))};
    }

    reference operator[](size_t i){
        return this->begin()[i];
    }

    constexpr const_reference operator[](size_t i) const{
        return this->cbegin()[i];
    }



    std::byte* get(){
        return dataStorage.get();
    }

};

template<typename ElementType>
    requires std::is_trivially_constructible_v<ElementType> && std::is_trivially_destructible_v<ElementType>
UnevenBlock<ElementType> UninitUnevenBlock(const size_t numArrays, const size_t numElements, std::pmr::memory_resource* resource = std::pmr::get_default_resource()){
    size_t numberOfBytes = 0;
    size_t headerBytes = sizeof(size_t)*(numArrays+1);
    size_t headerPadding = 0;

    if constexpr(alignof(ElementType)>alignof(size_t)){
        size_t headerExcess = headerBytes%alignof(ElementType);
        headerPadding = (headerExcess == 0) ? 0 : alignof(ElementType) - headerBytes%alignof(ElementType);
        numberOfBytes = headerBytes + headerPadding + sizeof(ElementType)*numElements;
    } else {
        numberOfBytes = headerBytes + sizeof(ElementType)*numElements;
    }

    return UnevenBlock<ElementType>(numberOfBytes, numArrays, headerPadding, numElements, resource);
}



}







#endif //MNISTDATA_HPP