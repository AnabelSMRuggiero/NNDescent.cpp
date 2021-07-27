/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_GENERALTYPE_HPP
#define NND_GENERALTYPE_HPP

#include <vector>
#include <concepts>
#include <memory>
#include <ranges>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <utility>
#include <span>

namespace nnd{



template<std::integral IndexType>
struct IntegralPairHasher{

    size_t operator()(const std::pair<IndexType, IndexType>& pair) const noexcept{
        return std::hash<IndexType>()(size_t(pair.first)*634018663193ul ^ std::hash<IndexType>()(pair.second)*354019652443ul);
    }

};

template<typename DistType>
using DistanceCache = std::unordered_map<std::pair<size_t, size_t>, DistType, IntegralPairHasher<size_t>>;


template<typename ValueType, size_t alignment=32>
struct AlignedArray{
    using value_type = ValueType;
    private:

    struct AlignedDeleter{

        void operator()(ValueType* arrayToDelete) {operator delete[](arrayToDelete, std::align_val_t(alignment)); };

    };

    std::unique_ptr<ValueType[], AlignedDeleter> data;
    size_t capacity;

    public:

    AlignedArray() = default;

    AlignedArray(size_t size): data(new (std::align_val_t(alignment)) ValueType[size]), capacity(size) {};

    AlignedArray(const AlignedArray& other): data(new (std::align_val_t(alignment)) ValueType[other.capacity]), capacity(other.capacity) {
        std::copy(other.begin(), other.end(), this->begin());
    };

    AlignedArray& operator=(AlignedArray&& other) = default;

    AlignedArray& operator=(const AlignedArray& other) = default;

    size_t size() const { return capacity; }

    ValueType* begin() { return std::assume_aligned<alignment>(data.get()); }

    ValueType* end() { return data.get() + capacity; }

    ValueType& operator[](size_t index) { return data[index]; }

    const ValueType* begin() const { return data.get(); }

    const ValueType* end() const { return data.get() + capacity; }

    const ValueType& operator[](size_t index) const{ return data[index]; }


};

template<typename ElementType, size_t alignment=32>
struct AlignedSpan{

    using value_type = std::remove_cv_t<ElementType>;
    private:
    ElementType* data;
    size_t extent;

    public:

    template<typename ConvertableToElement>
    AlignedSpan(const AlignedArray<ConvertableToElement, alignment>& dataToView): data(dataToView.begin()), extent(dataToView.size()){};

    template<typename ConvertableToElement>
    AlignedSpan(const AlignedSpan<ConvertableToElement>& spanToCopy): data(spanToCopy.data), extent(spanToCopy.extent){};

    ElementType* begin() const { return std::assume_aligned<alignment>(data); }

    ElementType* end() const { return data + extent; }

    ElementType& operator[](size_t index) const { return data[index]; };

    size_t size() const { return extent; };

};


template<std::ranges::contiguous_range Container>
struct DefaultDataView{ using ViewType = std::span<const typename Container::value_type>; };

template<typename ElementType>
struct DefaultDataView<AlignedArray<ElementType>>{ using ViewType = AlignedSpan<const ElementType>; };

template<typename DataTypeA, typename DataTypeB, typename RetType=double>
using SpaceMetric = RetType (*)(const DataTypeA&, const DataTypeB&);


template<typename DataTypeA, typename DataTypeB, typename RetType=std::vector<double>>
using BatchMetric = RetType (*)(const std::vector<DataTypeA>&, const DataTypeB&);

struct BlockIndecies{
    // The block a data point exists in
    size_t blockNumber;
    // The index within that block
    size_t dataIndex;

};


}


#endif