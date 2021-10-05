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

#include "../Type.hpp"

namespace nnd{


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

#endif