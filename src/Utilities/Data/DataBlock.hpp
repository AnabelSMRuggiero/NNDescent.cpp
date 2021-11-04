/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_DATABLOCK_HPP
#define NND_DATABLOCK_HPP

#include <cstddef>

#include "../Type.hpp"
#include "../DataSerialization.hpp"
#include "../DataDeserialization.hpp"

namespace nnd{


template<typename ElementType, size_t align>
struct DataBlockIterator{
    using value_type = AlignedSpan<ElementType, align>;
    using difference_type = std::ptrdiff_t;
    using reference = AlignedSpan<ElementType, align>;

    static constexpr size_t alignment = align;

    DataBlockIterator(const size_t arraySize, const size_t viewSize, ElementType* arrayStart): arraySize(arraySize), viewSize(viewSize), arrayStart(arrayStart) {}

    private:
    const size_t arraySize;
    const size_t viewSize;
    ElementType* arrayStart;

    public:
    DataBlockIterator& operator++(){
        arrayStart += arraySize;
        return *this;
    }

    DataBlockIterator operator++(int){
        DataBlockIterator copy = *this;
        arrayStart += arraySize;
        return copy;
    }

    DataBlockIterator& operator--(){
        arrayStart -= arraySize;
        return *this;
    }

    DataBlockIterator operator--(int){
        DataBlockIterator copy = *this;
        arrayStart -= arraySize;
        return copy;
    }

    DataBlockIterator operator+(std::ptrdiff_t inc){
        DataBlockIterator copy{arraySize, viewSize, arrayStart + (arraySize * inc)};
        return copy;
    }

    DataBlockIterator operator-(std::ptrdiff_t inc){
        DataBlockIterator copy{arraySize, viewSize, arrayStart - (arraySize * inc)};
        return copy;
    }

    std::ptrdiff_t operator-(DataBlockIterator other){
        return (arrayStart - other.arrayStart)/arraySize;
    }
    
    bool operator==(DataBlockIterator other){
        return arrayStart == other.arrayStart;
    }
    
    reference operator*(){
        return reference{MakeAlignedPtr(arrayStart, *this), viewSize};
    }

    reference operator[](size_t i){
        return *(*this + i);
    }

    DataBlockIterator& operator+=(std::ptrdiff_t inc){
        *this = *this + inc;
        return *this;
    }

    DataBlockIterator& operator-=(std::ptrdiff_t inc){
        *this = *this - inc;
        return *this;
    }

    auto operator<=>(DataBlockIterator& rhs){
        return arrayStart<=> rhs.arrayStart;
    }
};


//Presumably, each project would only need to instantiate for a single FloatType
template<typename ElementType, size_t align = 32>
    requires (alignof(ElementType) <= sizeof(ElementType))
struct DataBlock{
    using value_type = ElementType;
    using DataView = AlignedSpan<ElementType, align>;
    using ConstDataView = AlignedSpan<const ElementType, align>;

    
    using iterator = DataBlockIterator<ElementType, align>;
    using const_iterator = DataBlockIterator<const ElementType, align>;
    using reference = AlignedSpan<ElementType, align>;
    using const_reference = AlignedSpan<const ElementType, align>;
    

    static constexpr size_t alignment = align;

    size_t blockNumber;
    size_t numEntries;
    size_t entryLength;
    size_t lengthWithPadding;
    DynamicArray<ElementType, alignment> blockData;
    //std::vector<DataEntry> blockData;
    
    template<std::endian dataEndianess = std::endian::native>
    static DataBlock deserialize(std::ifstream& inFile, size_t blockNumber = 0){//, std::pmr::memory_resource* resource = std::pmr::get_default_resource()){

        static_assert(dataEndianess == std::endian::native, "reverseEndianess not implemented yet for this class");

        struct DeserializationArgs{
            size_t size;
            size_t entryLength;
            size_t lengthWithPadding;
        };
        /*
            outputFunc(block.size());
            outputFunc(block.entryLength);
            outputFunc(block.lengthWithPadding);

            outputFile.write(reinterpret_cast<const char*>(block.blockData.begin()), block.blockData.size()*sizeof(float));
        */
        DeserializationArgs args = Extract<DeserializationArgs, dataEndianess>(inFile);

        DataBlock retBlock{args.size, args.entryLength, args.lengthWithPadding, blockNumber};
        Extract<ElementType, dataEndianess>(inFile, retBlock.blockData.begin(), retBlock.blockData.end());

        return retBlock;
    }

    DataBlock() = default;

    DataBlock(const size_t numEntries, const size_t entryLength, size_t blockNumber):
        blockNumber(blockNumber), 
        numEntries(numEntries),
        entryLength(entryLength),
        lengthWithPadding(entryLength + EntryPadding<ElementType, alignment>(entryLength)),
        blockData(lengthWithPadding*numEntries){};

    private:

    DataBlock(const size_t numEntries, const size_t entryLength, const size_t lengthWithPadding, size_t blockNumber):
        blockNumber(blockNumber), 
        numEntries(numEntries),
        entryLength(entryLength),
        lengthWithPadding(lengthWithPadding),
        blockData(lengthWithPadding*numEntries){};

    public:

    //template<typename DataEntry>
    //    requires std::same_as<typename DataEntry::value_type, ElementType>
    DataBlock(const DataSet<ElementType>& dataSource, std::span<const size_t> dataPoints, const size_t entryLength, size_t blockNumber):
        blockNumber(blockNumber), 
        numEntries(dataPoints.size()),
        entryLength(entryLength),
        lengthWithPadding(entryLength + EntryPadding<ElementType, alignment>(entryLength)),
        blockData(lengthWithPadding*dataPoints.size()){
        //blockData = AlignedArray<ElementType, alignment>((entryLength + entryPadding));
        //blockData.reserve(dataPoints.size());
        for (ElementType* ptrIntoBlock = blockData.begin(); const size_t& index : dataPoints){
            //blockData.push_back(dataSource.samples[index]);
            typename DataSet<ElementType>::ConstDataView pointToCopy = dataSource[index];
            std::copy(pointToCopy.begin(), pointToCopy.end(), ptrIntoBlock);
            ptrIntoBlock += lengthWithPadding;
        }
    }

    DataBlock(DataBlock&& rhs) = default;

    DataBlock& operator=(DataBlock&& rhs) = default;

   

    iterator begin(){
        return iterator{lengthWithPadding, entryLength, blockData.get()};
    }

    const_iterator begin() const{
        return const_iterator{lengthWithPadding, entryLength, blockData.get()};
    }

    const_iterator cbegin() const{
        return const_iterator{lengthWithPadding, entryLength, blockData.get()};
    }

    iterator end(){
        return iterator{lengthWithPadding, entryLength, blockData.get() + lengthWithPadding*(numEntries-1)};
    }

    const_iterator end() const{
        return const_iterator{lengthWithPadding, entryLength, blockData.get() + lengthWithPadding*(numEntries-1)};
    }

    const_iterator cend() const{
        return const_iterator{lengthWithPadding, entryLength, blockData.get() + lengthWithPadding*(numEntries-1)};
    }

    DataView operator[](size_t i){
        
        //ptr += i;
        return begin()[i];
        //DataView(ptr, entryLength);
        //return blockData[i];
    }
    
    ConstDataView operator[](size_t i) const{
        //const_iterator ptr = begin()[i];
        //ptr += i;
        return cbegin()[i];
        /*
        AlignedPtr<const value_type, alignment> ptr = blockData.GetAlignedPtr(lengthWithPadding);
        ptr += i;
        return ConstDataView(ptr, entryLength);
        */
    }
    
    size_t size() const{
        return numEntries;
    }


};

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

}

#endif