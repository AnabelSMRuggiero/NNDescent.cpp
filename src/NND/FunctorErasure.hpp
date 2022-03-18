/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_FUNCTORERASURE_HPP
#define NND_FUNCTORERASURE_HPP

#include <concepts>
#include <cstddef>
#include <memory>
#include <memory_resource>
#include <new>
#include <ranges>
#include <type_traits>
#include <vector>

#include "../ann/Data.hpp"
#include "../ann/Metrics/SpaceMetrics.hpp"
#include "../ann/NestedRanges.hpp"
#include "../ann/TemplateManipulation.hpp"
#include "../ann/Type.hpp"
#include "Type.hpp"

namespace nnd {

template<typename Object>
concept empty_object = std::is_empty_v<Object>;

template<typename FirstType, typename SecondType>
struct compressed_pair {

    constexpr compressed_pair() = default;

    constexpr compressed_pair(FirstType first_member, SecondType second_member)
        : first_member{ std::move(first_member) }, second_member{ std::move(second_member) } {}

    FirstType& first() { return first_member; }

    const FirstType& first() const { return first_member; }

    SecondType& second() { return second_member; }

    const SecondType& second() const { return second_member; }

  private:
    FirstType first_member;
    SecondType second_member;
};

template<empty_object FirstType, typename SecondType>
struct compressed_pair<FirstType, SecondType> : private FirstType {

    constexpr compressed_pair() = default;

    constexpr compressed_pair(FirstType first, SecondType second_member)
        : FirstType{ std::move(first) }, second_member{ std::move(second_member) } {}

    FirstType& first() { return static_cast<FirstType&>(*this); }

    const FirstType& first() const { return static_cast<const FirstType&>(*this); }

    SecondType& second() { return second_member; }

    const SecondType& second() const { return second_member; }

  private:
    SecondType second_member;
};

// It might be worth shelving this. It's a lot of work to implement for something I don't expect to effectively use yet.
template<typename AbstractObject, typename Allocator, size_t BufferSize>
struct alignas(64) abstract_storage {

    using alloc_traits = std::allocator_traits<Allocator>;
    using allocator = Allocator;
    static constexpr size_t storage_size = 64 - sizeof(compressed_pair<Allocator, AbstractObject*>);

    constexpr abstract_storage() = default;

    constexpr abstract_storage(std::allocator_arg_t, Allocator alloc) : alloc_ptr_pair{ std::move(alloc), nullptr } {}

    template<typename ConcreteType, typename... StoredArgs>
        requires(sizeof...(StoredArgs) > 0)
    abstract_storage(std::allocator_arg_t, Allocator alloc, ann::type_obj<ConcreteType>, StoredArgs&&... storedArgs)
        : alloc_ptr_pair{ std::move(alloc), nullptr } {
        construct<ConcreteType>(std::forward<StoredArgs>(storedArgs)...);
    }

    template<typename ConcreteType, typename... StoredArgs>
        requires(sizeof...(StoredArgs) > 0)
    abstract_storage(ann::type_obj<ConcreteType>, StoredArgs&&... storedArgs) : alloc_ptr_pair{ Allocator{}, nullptr } {
        construct<ConcreteType>(std::forward<StoredArgs>(storedArgs)...);
    }

    // Copy construct
    // Copy assign
    // Move construct
    // Move Assign
    // Destructor

    operator bool() const { return alloc_ptr_pair.second() != nullptr; }

  private:
    template<typename ConcreteType, typename... StoredArgs>
    void construct(StoredArgs&&... storedArgs) {
        if constexpr (sizeof(ConcreteType) > storage_size) {
            alloc_ptr_pair.second() = static_cast<ConcreteType*>(alloc_traits::allocate(alloc_ptr_pair.first(), sizeof(ConcreteType)));
            auto construct = [&]<typename... Args>(Args && ... args) {
                alloc_traits::construct(
                    static_cast<ConcreteType*>(alloc_ptr_pair.first()), alloc_ptr_pair.second(), std::forward<Args>(args)...);
            };
            std::apply(construct, std::uses_allocator_construction_args(alloc_ptr_pair.first(), std::forward<StoredArgs>(storedArgs)...));
        } else {
            alloc_ptr_pair.second() = std::uninitialized_construct_using_allocator(
                static_cast<ConcreteType*>(storage), alloc_ptr_pair.first(), std::forward<StoredArgs>(storedArgs)...);
        }
    }
    std::byte storage[storage_size];
    compressed_pair<Allocator, AbstractObject*> alloc_ptr_pair;
};

template<typename AbstractObject, typename Allocator, size_t BufferSize>
bool operator==(const abstract_storage<AbstractObject, Allocator, BufferSize>& storage, std::nullptr_t) {
    return !storage;
}

struct swap_binds {};

inline constexpr swap_binds swap_binds_tag{};

template<typename Metric, typename LHSData, typename RHSData>
struct metric_pair {
    using distance_type = typename Metric::DistType;
    using const_data_view = typename RHSData::ConstDataView;
    using const_vector_view = typename RHSData::const_vector_view;
    using lhs_const_iterator = typename LHSData::const_iterator;
    using rhs_const_iterator = typename RHSData::const_iterator;

    [[no_unique_address]] Metric metric;
    lhs_const_iterator lhsBlock;
    rhs_const_iterator rhsBlock;
    // size_t lhsBlockNum, rhsBlockNum;

    metric_pair(const LHSData& lhsData, const RHSData& rhsData)
        : metric{ Metric{} }, lhsBlock{ lhsData.begin() }, rhsBlock{ rhsData.begin() } {};

    metric_pair(Metric metric, const LHSData& lhsData, const RHSData& rhsData)
        : metric(metric), lhsBlock(std::ranges::begin(lhsData)), rhsBlock(std::ranges::begin(rhsData)){};

    metric_pair(Metric metric, lhs_const_iterator lhsItr, rhs_const_iterator rhsItr)
        : metric{ std::move(metric) }, lhsBlock{ std::move(lhsItr) }, rhsBlock{ std::move(rhsItr) } {}

    metric_pair<Metric, RHSData, LHSData> operator()(swap_binds) const {
        return metric_pair<Metric, RHSData, LHSData>{ metric, rhsBlock, lhsBlock };
    }

    distance_type operator()(size_t LHSIndex, size_t RHSIndex) const { return metric(lhsBlock[LHSIndex], rhsBlock[RHSIndex]); };

    std::pmr::vector<distance_type> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const {
        
        return ComputeBatch(
            lhsBlock[lhsIndex],
            rhsIndecies | std::views::transform([rhsBlock = this->rhsBlock](const auto index)->const_vector_view { return rhsBlock[index];
        }), metric);
        
    };
};

template<typename DistanceType>
struct erased_metric {

    erased_metric() = default;

    erased_metric(const erased_metric& other) : ptrToFunc(other.ptrToFunc->clone()){};

    erased_metric& operator=(const erased_metric&) = default;

    erased_metric(erased_metric&&) = default;

    erased_metric& operator=(erased_metric&&) = default;

    template<is_not<erased_metric> DistanceFunctor>
        requires std::is_copy_assignable_v<DistanceFunctor> erased_metric(DistanceFunctor&& distanceFunctor)
            : ptrToFunc(
                std::make_unique<concrete_functor<std::remove_cvref_t<DistanceFunctor>>>(std::forward<DistanceFunctor>(distanceFunctor))){};

        erased_metric operator()(swap_binds) const { return std::invoke(*ptrToFunc, swap_binds_tag); }

        DistanceType operator()(const size_t LHSIndex, const size_t RHSIndex) const { return std::invoke(*ptrToFunc, LHSIndex, RHSIndex); };

        std::pmr::vector<DistanceType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const {
            return std::invoke(*ptrToFunc, lhsIndex, rhsIndecies);
        };

      private:
        struct abstract_functor {
            virtual ~abstract_functor(){};
            virtual std::unique_ptr<abstract_functor> clone() const = 0;
            virtual erased_metric operator()(swap_binds) const = 0;
            virtual DistanceType operator()(size_t LHSIndex, size_t RHSIndex) const = 0;
            virtual std::pmr::vector<DistanceType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const = 0;
        };

        template<typename DistanceFunctor>
        struct concrete_functor final : abstract_functor {

            DistanceFunctor underlyingFunctor;

            concrete_functor(DistanceFunctor underlyingFunctor) : underlyingFunctor(underlyingFunctor){};
            //~ConcreteFunctor() final = default;

            std::unique_ptr<abstract_functor> clone() const final { return std::make_unique<concrete_functor>(underlyingFunctor); }

            erased_metric operator()(swap_binds) const { return erased_metric{ this->underlyingFunctor(swap_binds_tag) }; }

            DistanceType operator()(size_t LHSIndex, size_t RHSIndex) const final { return this->underlyingFunctor(LHSIndex, RHSIndex); };

            std::pmr::vector<DistanceType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const final {
                return this->underlyingFunctor(lhsIndex, rhsIndecies);
            };
        };

      private:
        std::unique_ptr<abstract_functor> ptrToFunc;
};

template<typename Metric, typename LHSContainer, typename RHSContainer>
struct block_binder {
    using distance_type = typename Metric::DistType;

    [[no_unique_address]] Metric metric;
    std::span<const LHSContainer> lhsBlocks;
    std::span<const RHSContainer> rhsBlocks;
    // size_t lhsBlockNum, rhsBlockNum;

    // MetricFunctorRefactor(std::span<const DataBlock<DataType>> blocks, MetricPair metricPair = {}) : metricPair{std::move(metricPair)},
    // lhsBlocks{blocks}, rhsBlocks{blocks}{};

    block_binder(Metric metric, std::span<const LHSContainer> lhsBlocks, std::span<const RHSContainer> rhsBlocks)
        : metric{ std::move(metric) }, lhsBlocks{ lhsBlocks }, rhsBlocks{ rhsBlocks } {};

    block_binder(std::span<const LHSContainer> lhsBlocks, std::span<const RHSContainer> rhsBlocks)
        : metric{ Metric{} }, lhsBlocks{ lhsBlocks }, rhsBlocks{ rhsBlocks } {};

    block_binder<Metric, RHSContainer, LHSContainer> operator()(swap_binds) const {
        return block_binder<Metric, RHSContainer, LHSContainer>{ metric, rhsBlocks, lhsBlocks };
    }

    metric_pair<Metric, LHSContainer, RHSContainer> operator()(size_t lhsIndex, size_t rhsIndex) const {
        return metric_pair<Metric, LHSContainer, RHSContainer>{ metric, lhsBlocks[lhsIndex], rhsBlocks[rhsIndex] };
    }
};

template<typename Metric, typename FixedContainer, typename RHSContainer>
struct fixed_block_binder {
    using distance_type = typename Metric::DistType;

    [[no_unique_address]] Metric metric;
    typename FixedContainer::const_iterator fixedBlock;
    std::span<const RHSContainer> rhsBlocks;
    // size_t lhsBlockNum, rhsBlockNum;

    // MetricFunctorRefactor(std::span<const DataBlock<DataType>> blocks, MetricPair metricPair = {}) : metricPair{std::move(metricPair)},
    // lhsBlocks{blocks}, rhsBlocks{blocks}{};

    fixed_block_binder(Metric metric, const FixedContainer& fixedBlock, std::span<const RHSContainer> rhsBlocks)
        : metric{ std::move(metric) }, fixedBlock{ fixedBlock.begin() }, rhsBlocks{ rhsBlocks } {};

    fixed_block_binder(const FixedContainer& fixedBlock, std::span<const RHSContainer> rhsBlocks)
        : metric{ Metric{} }, fixedBlock{ fixedBlock.begin() }, rhsBlocks{ rhsBlocks } {};

    metric_pair<Metric, FixedContainer, RHSContainer> operator()(size_t targetBlock) const {
        return metric_pair<Metric, FixedContainer, RHSContainer>{ metric, fixedBlock, rhsBlocks[targetBlock].begin() };
    }
};

template<typename DistanceType>
struct erased_unary_binder {

    erased_unary_binder() = default;

    erased_unary_binder(const erased_unary_binder& other) : ptrToFunc(other.ptrToFunc->clone()){};

    erased_unary_binder& operator=(const erased_unary_binder& other) { ptrToFunc = other.ptrToFunc.clone(); }

    erased_unary_binder(erased_unary_binder&&) = default;

    erased_unary_binder& operator=(erased_unary_binder&&) = default;

    operator bool() const{
        return bool(ptrToFunc);
    }

    template<is_not<erased_unary_binder> DistanceFunctor>
        requires std::is_copy_assignable_v<DistanceFunctor> erased_unary_binder(DistanceFunctor&& distanceFunctor)
            : ptrToFunc(
                std::make_unique<concrete_functor<std::remove_cvref_t<DistanceFunctor>>>(std::forward<DistanceFunctor>(distanceFunctor))){};

        erased_metric<DistanceType> operator()(size_t bindIndex) const { return std::invoke(*ptrToFunc, bindIndex); };

      private:
        struct abstract_functor {
            virtual ~abstract_functor(){};
            virtual std::unique_ptr<abstract_functor> clone() const = 0;
            virtual erased_metric<DistanceType> operator()(size_t bindIndex) const = 0;
        };

        template<typename DistanceFunctor>
        struct concrete_functor final : abstract_functor {

            DistanceFunctor underlyingFunctor;

            concrete_functor(DistanceFunctor underlyingFunctor) : underlyingFunctor(underlyingFunctor){};
            //~ConcreteFunctor() final = default;

            std::unique_ptr<abstract_functor> clone() const final { return std::make_unique<concrete_functor>(underlyingFunctor); }

            erased_metric<DistanceType> operator()(size_t bindIndex) const final { return this->underlyingFunctor(bindIndex); };
        };

      private:
        std::unique_ptr<abstract_functor> ptrToFunc;
};

template<typename DistanceType>
struct erased_binary_binder {

    erased_binary_binder() = default;

    erased_binary_binder(const erased_binary_binder& other) : ptrToFunc(other.ptrToFunc->clone()){};

    erased_binary_binder& operator=(const erased_binary_binder& other) { ptrToFunc = other.ptrToFunc.clone(); }

    erased_binary_binder(erased_binary_binder&&) = default;

    erased_binary_binder& operator=(erased_binary_binder&&) = default;

    template<is_not<erased_binary_binder> DistanceFunctor>
        requires std::is_copy_assignable_v<DistanceFunctor> erased_binary_binder(DistanceFunctor&& distanceFunctor)
            : ptrToFunc(
                std::make_unique<concrete_functor<std::remove_cvref_t<DistanceFunctor>>>(std::forward<DistanceFunctor>(distanceFunctor))){};

        erased_metric<DistanceType> operator()(size_t lhsBind, size_t rhsBind) const { return std::invoke(*ptrToFunc, lhsBind, rhsBind); };

      private:
        struct abstract_functor {
            virtual ~abstract_functor(){};
            virtual std::unique_ptr<abstract_functor> clone() const = 0;
            virtual erased_metric<DistanceType> operator()(size_t lhsBind, size_t rhsBind) const = 0;
        };

        template<typename DistanceFunctor>
        struct concrete_functor final : abstract_functor {

            DistanceFunctor underlyingFunctor;

            concrete_functor(DistanceFunctor underlyingFunctor) : underlyingFunctor(underlyingFunctor){};
            //~ConcreteFunctor() final = default;

            std::unique_ptr<abstract_functor> clone() const final { return std::make_unique<concrete_functor>(underlyingFunctor); }

            erased_metric<DistanceType> operator()(size_t lhsBind, size_t rhsBind) const final {
                return this->underlyingFunctor(lhsBind, rhsBind);
            };
        };

      private:
        std::unique_ptr<abstract_functor> ptrToFunc;
};

} // namespace nnd

#endif