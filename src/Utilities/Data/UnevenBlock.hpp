/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_UNEVENBLOCK_HPP
#define NND_UNEVENBLOCK_HPP

#include <cstddef>
#include <memory_resource>
#include <type_traits>
#include <new>

#include "../Type.hpp"
#include "../DataSerialization.hpp"
#include "../DataDeserialization.hpp"



namespace nnd{

template <typename RetType, typename ArgType>
std::remove_pointer_t<RetType>* PtrCast(ArgType* ptrToCast){
    if constexpr (std::is_const_v<RetType> && std::is_volatile_v<RetType>){
        return static_cast<std::remove_pointer_t<RetType>*>(static_cast<const volatile void*>(ptrToCast));
    } else if (std::is_const_v<RetType>){
        return static_cast<std::remove_pointer_t<RetType>*>(static_cast<const void*>(ptrToCast));
    } else if (std::is_volatile_v<RetType>){
        return static_cast<std::remove_pointer_t<RetType>*>(static_cast<volatile void*>(ptrToCast));
    } else{
        return static_cast<std::remove_pointer_t<RetType>*>(static_cast<volatile void*>(ptrToCast));
    }
}


template<typename ElementType>
struct UnevenBlockIterator{
    using value_type = std::span<ElementType>;
    using difference_type = std::ptrdiff_t;
    using reference = std::span<ElementType>;
    using const_reference = const std::span<const ElementType>;

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

    UnevenBlockIterator operator+(std::ptrdiff_t inc) const {
        UnevenBlockIterator copy{vertexStart+inc, vertexNeighbors + (*(vertexStart+inc) - *vertexStart)};
        return copy;
    }

    UnevenBlockIterator operator-(std::ptrdiff_t inc) const {
        UnevenBlockIterator copy{vertexStart-inc, vertexNeighbors - (*vertexStart - *(vertexStart-inc))};
        return copy;
    }

    std::ptrdiff_t operator-(UnevenBlockIterator other) const {
        return vertexStart - other.vertexStart;
    }
    
    bool operator==(UnevenBlockIterator other) const {
        return vertexStart == other.vertexStart;
    }
    
    reference operator*(){
        return reference{vertexNeighbors, *(vertexStart+1) - *vertexStart};
    }

    const_reference operator*()const{
        return reference{vertexNeighbors, *(vertexStart+1) - *vertexStart};
    }

    reference operator[](size_t i) {
        return *(*this + i);
    }

    const_reference operator[](size_t i)const {
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

    auto operator<=>(UnevenBlockIterator& rhs) const {
        return vertexStart<=> rhs.vertexStart;
    }
};

template<typename ElementType>
    requires std::is_trivially_constructible_v<ElementType> && std::is_trivially_destructible_v<ElementType>
struct UnevenBlock{

    using iterator = UnevenBlockIterator<ElementType>;
    using const_iterator = UnevenBlockIterator<const ElementType>;
    using reference = std::span<ElementType>;
    using const_reference = const std::span<const ElementType>;


    DynamicArray<std::byte, std::max(alignof(size_t), alignof(ElementType))> dataStorage;
    size_t numArrays;
    ElementType* firstIndex;

    template<std::endian dataEndianess = std::endian::native>
    static UnevenBlock deserialize(std::ifstream& inFile, std::pmr::memory_resource* resource = std::pmr::get_default_resource()){

        static_assert(dataEndianess == std::endian::native, "reverseEndianess not implemented yet for this class");
        
        struct DeserializationArgs{
            size_t numBytes;
            size_t numArrays;
            std::ptrdiff_t indexOffset;
        };

        DeserializationArgs args = Extract<DeserializationArgs>(inFile);

        UnevenBlock retBlock{args.numBytes, args.numArrays, args.indexOffset, resource};
        Extract<std::byte>(inFile, retBlock.dataStorage.begin(), retBlock.dataStorage.end());

        return retBlock;
    }

    UnevenBlock() = default;

    //Default Copy Constructor is buggy
    UnevenBlock(const UnevenBlock& other): dataStorage(other.dataStorage), numArrays(other.numArrays), firstIndex(nullptr){
        this->firstIndex =  static_cast<ElementType*>(static_cast<void*>(this->dataStorage.get()
           + other.IndexOffset()));
    }

    UnevenBlock& operator=(const UnevenBlock& other){
        dataStorage = DynamicArray<std::byte, std::max(alignof(size_t), alignof(ElementType))>(other.dataStorage.size());
        numArrays = other.numArrays;


        size_t* vertexStart = new (dataStorage.begin()) size_t[numArrays+1];
        std::ptrdiff_t indexOffset = static_cast<const std::byte*>(static_cast<const void*>(other.firstIndex)) - other.dataStorage.begin();

        size_t numElements = static_cast<const ElementType*>(static_cast<const void*>(other.dataStorage.end())) - other.firstIndex;
        firstIndex = new (dataStorage.begin() + indexOffset) ElementType[numElements];
        std::copy(other.dataStorage.begin(), other.dataStorage.end(), dataStorage.begin());
        
        return *this;
    }
    //NewUndirectedGraph(size_t numVerticies, size_t numNeighbors): 
    //    verticies(numVerticies, std::vector<IndexType>(numNeighbors)){};

    //template<typename DistType>
    UnevenBlock(const size_t numBytes, const size_t numArrays, const size_t headerPadding, const size_t numIndecies, std::pmr::memory_resource* resource): dataStorage(numBytes, resource), numArrays(numArrays), firstIndex(nullptr){
        size_t* vertexStart = new (dataStorage.begin()) size_t[numArrays+1];
        firstIndex = new (dataStorage.begin() + sizeof(size_t)*(numArrays+1) + headerPadding) ElementType[numIndecies];
    }

    UnevenBlock(const size_t numBytes, const size_t numArrays, const std::ptrdiff_t indexOffset, std::pmr::memory_resource* resource): dataStorage(numBytes, resource), numArrays(numArrays), firstIndex(nullptr){
        size_t* vertexStart = new (dataStorage.begin()) size_t[numArrays+1];
        size_t numIndecies = (dataStorage.end() - (dataStorage.begin() + indexOffset))/sizeof(ElementType);
        firstIndex = new (dataStorage.begin() + indexOffset) ElementType[numIndecies];
    }

    size_t size() const noexcept{
        return numArrays;
    }
    
    constexpr iterator begin() noexcept{
        return iterator{std::launder(static_cast<size_t*>(static_cast<void*>(dataStorage.begin()))), firstIndex};
    }

    constexpr const_iterator begin() const noexcept{
        return const_iterator{std::launder(static_cast<const size_t*>(static_cast<const void*>(dataStorage.begin()))), firstIndex};
    }

    constexpr const_iterator cbegin() const noexcept{
        return const_iterator{std::launder(static_cast<const size_t*>(static_cast<const void*>(dataStorage.begin()))), firstIndex};
    }

    constexpr iterator end() noexcept{
        return iterator{std::launder(static_cast<size_t*>(static_cast<void*>(dataStorage.begin()))+numArrays), std::launder(static_cast<ElementType*>(static_cast<void*>(dataStorage.end())))};
    }

    constexpr const_iterator end() const noexcept{
        return const_iterator{std::launder(static_cast<const size_t*>(static_cast<const void*>(dataStorage.begin()))+numArrays), std::launder(static_cast<const ElementType*>(static_cast<const void*>(dataStorage.end())))};
    }

    constexpr const_iterator cend() const noexcept{
        return const_iterator{std::launder(static_cast<const size_t*>(static_cast<const void*>(dataStorage.begin()))+numArrays), std::launder(static_cast<const ElementType*>(static_cast<const void*>(dataStorage.end())))};
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

    std::ptrdiff_t IndexOffset() const{
        return static_cast<std::byte*>(static_cast<void*>(firstIndex)) - dataStorage.get();
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


template<typename BlockDataType>
void Serialize(const UnevenBlock<BlockDataType>& block, std::ofstream& outputFile){

    auto outputFunc = BindSerializer(outputFile);
    outputFunc(block.dataStorage.size());
    outputFunc(block.size());
    outputFunc(block.IndexOffset());
    
                                                                                 //this is already the size in bytes
    outputFile.write(reinterpret_cast<const char*>(block.dataStorage.get()), block.dataStorage.size());
}

}

#endif