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
#include <type_traits>
#include <vector>


#include "../ann/Data.hpp"
#include "../ann/Metrics/SpaceMetrics.hpp"
#include "../ann/NestedRanges.hpp"
#include "../ann/Type.hpp"
#include "../ann/TemplateManipulation.hpp"
#include "Type.hpp"


namespace nnd {

/*
    Lets rethink this nonsense:
    I want f(span, span)         -> g(blockIdx, blockIdx) -> h(dataIdx, dataIdx)
           f(span, blockIdx)     -> k(blockIdx)           -> h(dataIdx, dataIdx)
           f(blockIdx, span)     -> l(blockIdx)           -> h(dataIdx, dataIdx)
           f(blockIdx, blockIdx) ->                       -> h(dataIdx, dataIdx)

*/

/*
template<typename Metric>
struct MetricFunctorRedo {
    using center_type = typename Metric::center_type;
    using data_type = typename Metric::data_type;
    using distance_type = typename Metric::distance_type;
    [[no_unique_address]] Metric metric;

    // template<std::ranges::random_access_range LHSData, std::ranges::random_access_range RHSData>
    // operator(const LHSData& lhsData, const RHSData& rhsData){}

    // ?? operator()(std::span<DataBlock<distance_type>> lhsSpan, std::span<DataBlock<distance_type>> rhsSpan)
    // ?? operator()(DataBlock<distance_type> lhsBlock, std::span<DataBlock<distance_type>> rhsSpan)
    // ?? operator()(DataBlock<distance_type> lhsBlock, DataBlock<distance_type> rhsBlock)
};
*/

template<typename MetricPair, typename LHSData, typename RHSData>
struct BasicFunctor {
    using DistType = typename MetricPair::DistType;
    using ConstDataView = typename RHSData::ConstDataView;
    using const_vector_view = typename RHSData::const_vector_view;

    [[no_unique_address]] MetricPair metricPair;
    const LHSData* lhsBlock;
    const RHSData* rhsBlock;
    // size_t lhsBlockNum, rhsBlockNum;

    BasicFunctor(const LHSData& lhsData, const RHSData& rhsData) : metricPair(MetricPair()), lhsBlock(&lhsData), rhsBlock(&rhsData){};

    BasicFunctor(MetricPair metricPair, const LHSData& lhsData, const RHSData& rhsData)
        : metricPair(metricPair), lhsBlock(&lhsData), rhsBlock(&rhsData){};

    DistType operator()(size_t LHSIndex, size_t RHSIndex) const { return metricPair((*lhsBlock)[LHSIndex], (*rhsBlock)[RHSIndex]); };

    std::pmr::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const {
        constexpr size_t numberOfViews = internal::maxBatch + 5;
        constexpr size_t bufferSize = sizeof(const_vector_view) * numberOfViews + sizeof(std::pmr::vector<const_vector_view>);

        char stackBuffer[bufferSize];
        std::pmr::monotonic_buffer_resource stackResource(stackBuffer, bufferSize);

        void* vectorStorage = stackResource.allocate(sizeof(std::pmr::vector<const_vector_view>));
        std::pmr::vector<const_vector_view>& rhsData = *(new (vectorStorage) std::pmr::vector<const_vector_view>(&stackResource));

        rhsData.resize(rhsIndecies.size());
        std::ranges::transform(rhsIndecies, rhsData.begin(), [&](const auto index) { return (*rhsBlock)[index]; });

        return ComputeBatch((*lhsBlock)[lhsIndex], rhsData, metricPair);
    };
};

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

    constexpr compressed_pair(FirstType first, SecondType second_member) : FirstType{ std::move(first) }, second_member{ std::move(second_member) } {}

    FirstType& first() { return static_cast<FirstType&>(*this); }

    const FirstType& first() const { return static_cast<const FirstType&>(*this); }

    SecondType& second() { return second_member; }

    const SecondType& second() const { return second_member; }

  private:
    SecondType second_member;
};

// It might be worth shelving this. It's a lot of work to implement for something I don't expect to effectively use yet.
template<typename AbstractObject, typename Allocator, size_t BufferSize>
struct alignas(std::hardware_destructive_interference_size) abstract_storage {

    using alloc_traits = std::allocator_traits<Allocator>;
    using allocator = Allocator;
    static constexpr size_t storage_size = std::hardware_destructive_interference_size - sizeof(compressed_pair<Allocator, AbstractObject*>);

    constexpr abstract_storage() = default;

    constexpr abstract_storage(std::allocator_arg_t, Allocator alloc) : alloc_ptr_pair{ std::move(alloc), nullptr } {}

    template<typename ConcreteType, typename... StoredArgs>
        requires(sizeof...(StoredArgs) > 0)
    abstract_storage(std::allocator_arg_t, Allocator alloc, ann::type_obj<ConcreteType>, StoredArgs&&... storedArgs) : alloc_ptr_pair{ std::move(alloc), nullptr } {
        construct<ConcreteType>(std::forward<StoredArgs>(storedArgs)...);
    }

    template<typename ConcreteType, typename... StoredArgs>
        requires(sizeof...(StoredArgs) > 0)
    abstract_storage(ann::type_obj<ConcreteType>, StoredArgs&&... storedArgs) : alloc_ptr_pair{ Allocator{}, nullptr } {
        construct<ConcreteType>(std::forward<StoredArgs>(storedArgs)...);
    }

    //Copy construct
    //Copy assign
    //Move construct
    //Move Assign
    //Destructor

    operator bool() const { return alloc_ptr_pair.second() != nullptr; }

  private:
    template<typename ConcreteType, typename... StoredArgs>
    void construct(StoredArgs&&... storedArgs){
        if constexpr (sizeof(ConcreteType) > storage_size) {
            alloc_ptr_pair.second() = static_cast<ConcreteType*>(alloc_traits::allocate(alloc_ptr_pair.first(), sizeof(ConcreteType)));
            auto construct = [&]<typename... Args>(Args && ... args) {
                alloc_traits::construct(static_cast<ConcreteType*>(alloc_ptr_pair.first()), alloc_ptr_pair.second(), std::forward<Args>(args)...);
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

struct swap_binds{};

inline constexpr swap_binds swap_binds_tag{};

template<typename DistType>
struct erased_metric {

    erased_metric() = default;

    erased_metric(const erased_metric& other) : ptrToFunc(other.ptrToFunc->clone()){};

    erased_metric& operator=(const erased_metric&) = default;

    erased_metric(erased_metric&&) = default;

    erased_metric& operator=(erased_metric&&) = default;

    template<IsNot<erased_metric> DistanceFunctor>
        requires std::is_copy_assignable_v<DistanceFunctor>
    erased_metric(DistanceFunctor&& distanceFunctor)
        : ptrToFunc(std::make_unique<concrete_functor<std::remove_cvref_t<DistanceFunctor>>>(std::forward<DistanceFunctor>(distanceFunctor))){};

    DistType operator()(const size_t LHSIndex, const size_t RHSIndex) const { return this->ptrToFunc->operator()(LHSIndex, RHSIndex); };

    std::pmr::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const {
        return this->ptrToFunc->operator()(lhsIndex, rhsIndecies);
    };

  private:
    struct abstract_functor {
        virtual ~abstract_functor(){};
        virtual std::unique_ptr<abstract_functor> clone() const = 0;
        virtual DistType operator()(size_t LHSIndex, size_t RHSIndex) const = 0;
        virtual std::pmr::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const = 0;
    };

    template<typename DistanceFunctor>
    struct concrete_functor final : abstract_functor {

        DistanceFunctor underlyingFunctor;

        concrete_functor(DistanceFunctor underlyingFunctor) : underlyingFunctor(underlyingFunctor){};
        //~ConcreteFunctor() final = default;

        
        virtual std::unique_ptr<abstract_functor> clone() const {
            return std::make_unique<concrete_functor>(underlyingFunctor);
        }

        DistType operator()(size_t LHSIndex, size_t RHSIndex) const final { return this->underlyingFunctor(LHSIndex, RHSIndex); };

        std::pmr::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const final {
            return this->underlyingFunctor(lhsIndex, rhsIndecies);
        };
    };

  private:
    std::unique_ptr<abstract_functor> ptrToFunc;
};

template<typename DataType, typename MetricPair>
struct MetricFunctorRefactor {
    using DistType = typename MetricPair::DistType;
    using ConstDataView = typename DataBlock<DataType>::ConstDataView;
    using const_vector_view = typename DataBlock<DataType>::const_vector_view;

    [[no_unique_address]] MetricPair metricPair;
    std::span<const DataBlock<DataType>> lhsBlocks;
    std::span<const DataBlock<DataType>> rhsBlocks;
    // size_t lhsBlockNum, rhsBlockNum;

    MetricFunctorRefactor(std::span<const DataBlock<DataType>> blocks, MetricPair metricPair = {}) : metricPair{std::move(metricPair)}, lhsBlocks{blocks}, rhsBlocks{blocks}{};

    MetricFunctorRefactor(std::span<const DataBlock<DataType>> lhsBlocks, std::span<const DataBlock<DataType>> rhsBlocks, MetricPair metricPair = {}) : metricPair{std::move(metricPair)}, lhsBlocks{lhsBlocks}, rhsBlocks{rhsBlocks}{};

    /*

    DistType operator()(size_t LHSIndex, size_t RHSIndex) const { return metricPair((*lhsBlock)[LHSIndex], (*rhsBlock)[RHSIndex]); };

    std::pmr::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const noexcept {
        constexpr size_t numberOfViews = internal::maxBatch + 5;
        constexpr size_t bufferSize = sizeof(const_vector_view) * numberOfViews + sizeof(std::pmr::vector<const_vector_view>);

        char stackBuffer[bufferSize];
        std::pmr::monotonic_buffer_resource stackResource(stackBuffer, bufferSize);

        void* vectorStorage = stackResource.allocate(sizeof(std::pmr::vector<const_vector_view>));
        std::pmr::vector<const_vector_view>& rhsData = *(new (vectorStorage) std::pmr::vector<const_vector_view>(&stackResource));

        rhsData.resize(rhsIndecies.size());
        std::ranges::transform(rhsIndecies, rhsData.begin(), [&](const auto index) { return (*rhsBlock)[index]; });

        return ComputeBatch((*lhsBlock)[lhsIndex], rhsData, metricPair);
    };

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum) {
        this->lhsBlock = &(blocks[lhsBlockNum]);
        this->rhsBlock = &(blocks[rhsBlockNum]);
    }
    */
};


template<typename DataType, typename MetricPair>
struct MetricFunctor {
    using DistType = typename MetricPair::DistType;
    using ConstDataView = typename DataBlock<DataType>::ConstDataView;
    using const_vector_view = typename DataBlock<DataType>::const_vector_view;

    [[no_unique_address]] MetricPair metricPair;
    const DataBlock<DataType>* lhsBlock;
    const DataBlock<DataType>* rhsBlock;
    std::span<const DataBlock<DataType>> blocks;
    // size_t lhsBlockNum, rhsBlockNum;

    MetricFunctor(const std::vector<DataBlock<DataType>>& blocks) : metricPair(MetricPair()), blocks(blocks.data(), blocks.size()){};

    MetricFunctor(MetricPair metricPair, const std::vector<DataBlock<DataType>>& blocks)
        : metricPair(metricPair), blocks(blocks.data(), blocks.size()){};

    DistType operator()(size_t LHSIndex, size_t RHSIndex) const { return metricPair((*lhsBlock)[LHSIndex], (*rhsBlock)[RHSIndex]); };

    std::pmr::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const noexcept {
        constexpr size_t numberOfViews = internal::maxBatch + 5;
        constexpr size_t bufferSize = sizeof(const_vector_view) * numberOfViews + sizeof(std::pmr::vector<const_vector_view>);

        char stackBuffer[bufferSize];
        std::pmr::monotonic_buffer_resource stackResource(stackBuffer, bufferSize);

        void* vectorStorage = stackResource.allocate(sizeof(std::pmr::vector<const_vector_view>));
        std::pmr::vector<const_vector_view>& rhsData = *(new (vectorStorage) std::pmr::vector<const_vector_view>(&stackResource));

        rhsData.resize(rhsIndecies.size());
        std::ranges::transform(rhsIndecies, rhsData.begin(), [&](const auto index) { return (*rhsBlock)[index]; });

        return ComputeBatch((*lhsBlock)[lhsIndex], rhsData, metricPair);
    };

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum) {
        this->lhsBlock = &(blocks[lhsBlockNum]);
        this->rhsBlock = &(blocks[rhsBlockNum]);
    }
};

template<typename DataType, typename MetricPair>
struct CrossFragmentFunctor {
    using DistType = typename MetricPair::DistType;
    using ConstDataView = typename DataBlock<DataType>::ConstDataView;
    using const_vector_view = typename DataBlock<DataType>::const_vector_view;

    [[no_unique_address]] MetricPair metricPair;
    const DataBlock<DataType>* lhsBlock;
    const DataBlock<DataType>* rhsBlock;
    std::span<const DataBlock<DataType>> lhsBlocks;
    std::span<const DataBlock<DataType>> rhsBlocks;
    // size_t lhsBlockNum, rhsBlockNum;

    CrossFragmentFunctor(const std::vector<DataBlock<DataType>>& lhsBlocks, const std::vector<DataBlock<DataType>>& rhsBlocks)
        : metricPair(MetricPair()), lhsBlocks(lhsBlocks.data(), lhsBlocks.size()), rhsBlocks(rhsBlocks.data(), rhsBlocks.size()){};

    CrossFragmentFunctor(
        MetricPair metricPair, const std::vector<DataBlock<DataType>>& lhsBlocks, const std::vector<DataBlock<DataType>>& rhsBlocks)
        : metricPair(metricPair), lhsBlocks(lhsBlocks.data(), lhsBlocks.size()), rhsBlocks(rhsBlocks.data(), rhsBlocks.size()){};

    DistType operator()(size_t LHSIndex, size_t RHSIndex) const { return metricPair((*lhsBlock)[LHSIndex], (*rhsBlock)[RHSIndex]); };

    std::pmr::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const {
        constexpr size_t numberOfViews = internal::maxBatch + 5;
        constexpr size_t bufferSize = sizeof(const_vector_view) * numberOfViews + sizeof(std::pmr::vector<const_vector_view>);

        char stackBuffer[bufferSize];
        std::pmr::monotonic_buffer_resource stackResource(stackBuffer, bufferSize);

        void* vectorStorage = stackResource.allocate(sizeof(std::pmr::vector<const_vector_view>));
        std::pmr::vector<const_vector_view>& rhsData = *(new (vectorStorage) std::pmr::vector<const_vector_view>(&stackResource));

        rhsData.resize(rhsIndecies.size());
        std::ranges::transform(rhsIndecies, rhsData.begin(), [&](const auto index) { return (*rhsBlock)[index]; });

        return ComputeBatch((*lhsBlock)[lhsIndex], rhsData, metricPair);
    };

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum) {
        this->lhsBlock = &(lhsBlocks[lhsBlockNum]);
        this->rhsBlock = &(rhsBlocks[rhsBlockNum]);
    }
};

template<typename DistType>
struct DispatchFunctor {

    DispatchFunctor() = default;

    DispatchFunctor(const DispatchFunctor& other) : ptrToFunc(other.ptrToFunc){};

    DispatchFunctor& operator=(const DispatchFunctor&) = default;

    template<IsNot<DispatchFunctor> DistanceFunctor>
    DispatchFunctor(DistanceFunctor& distanceFunctor) : ptrToFunc(std::make_shared<ConcreteFunctor<DistanceFunctor>>(distanceFunctor)){};

    DistType operator()(const size_t LHSIndex, const size_t RHSIndex) const { return this->ptrToFunc->operator()(LHSIndex, RHSIndex); };

    std::pmr::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const {
        return this->ptrToFunc->operator()(lhsIndex, rhsIndecies);
    };

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum) { this->ptrToFunc->SetBlocks(lhsBlockNum, rhsBlockNum); }

  private:
    struct AbstractFunctor {
        virtual ~AbstractFunctor(){};
        virtual DistType operator()(size_t LHSIndex, size_t RHSIndex) const = 0;
        virtual std::pmr::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const = 0;
        virtual void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum) = 0;
    };

    template<typename DistanceFunctor>
    struct ConcreteFunctor final : AbstractFunctor {

        DistanceFunctor underlyingFunctor;

        ConcreteFunctor(DistanceFunctor underlyingFunctor) : underlyingFunctor(underlyingFunctor){};
        //~ConcreteFunctor() final = default;

        DistType operator()(size_t LHSIndex, size_t RHSIndex) const final { return this->underlyingFunctor(LHSIndex, RHSIndex); };

        std::pmr::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const final {
            return this->underlyingFunctor(lhsIndex, rhsIndecies);
        };

        void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum) final { this->underlyingFunctor.SetBlocks(lhsBlockNum, rhsBlockNum); }
    };

  private:
    std::shared_ptr<AbstractFunctor> ptrToFunc;
};

template<typename COMExtent>
struct MetaGraph;

template<typename DataType, typename COMExtent, typename MetricPair>
struct DataComDistance {
    using DistType = typename MetricPair::DistType;
    using ConstDataView = typename DataBlock<DataType>::ConstDataView;
    using const_vector_view = typename DataBlock<DataType>::const_vector_view;
    // Reference to Com?
    const MetaGraph<COMExtent>& centersOfMass;
    const DataBlock<DataType>* targetBlock;
    std::span<const DataBlock<DataType>> blocks;
    [[no_unique_address]] MetricPair functor;

    DataComDistance(const MetaGraph<COMExtent>& centersOfMass, const std::vector<DataBlock<DataType>>& blocks)
        : centersOfMass(centersOfMass), blocks(blocks.data(), blocks.size()), functor(){};

    DataComDistance(const MetaGraph<COMExtent>& centersOfMass, const std::vector<DataBlock<DataType>>& blocks, MetricPair functor)
        : centersOfMass(centersOfMass), blocks(blocks.data(), blocks.size(), blocks[0].blockNumber), functor(functor){};

    float operator()(const size_t metagraphIndex, const size_t dataIndex) const {
        return functor(centersOfMass.points[metagraphIndex], (*targetBlock)[dataIndex]);
    };

    std::pmr::vector<float> operator()(const size_t metagraphIndex, std::span<const size_t> rhsIndecies) const {
        std::vector<const_vector_view> rhsData;
        for (const auto& index : rhsIndecies) {
            rhsData.push_back((*targetBlock)[index]);
        }
        return ComputeBatch(centersOfMass.points[metagraphIndex], rhsData, functor);
    };

    void SetBlock(size_t targetBlockNum) { this->targetBlock = &(blocks[targetBlockNum]); }
};

template<typename DataType, typename DataSet, typename MetricPair>
struct SearchFunctor {
    using DistType = typename MetricPair::DistType;
    using ConstDataView = typename DataBlock<DataType>::ConstDataView;
    using const_vector_view = typename DataBlock<DataType>::const_vector_view;

    const DataBlock<DataType>* targetBlock;
    std::span<const DataBlock<DataType>> blocks;
    const DataSet& points;
    [[no_unique_address]] MetricPair functor;

    SearchFunctor(const std::vector<DataBlock<DataType>>& blocks, const DataSet& points)
        : blocks(blocks.data(), blocks.size()), points(points), functor(){};

    SearchFunctor(const std::vector<DataBlock<DataType>>& blocks, const DataSet& points, MetricPair functor)
        : blocks(blocks.data(), blocks.size()), points(points), functor(functor){};

    float operator()(const size_t searchIndex, const size_t targetIndex) const {
        return functor(points[searchIndex], (*targetBlock)[targetIndex]);
    };

    std::pmr::vector<typename MetricPair::DistType> operator()(const size_t searchIndex, std::span<const size_t> targetIndecies) const {

        constexpr size_t numberOfViews = internal::maxBatch + 5;
        constexpr size_t bufferSize = sizeof(const_vector_view) * numberOfViews + sizeof(std::pmr::vector<const_vector_view>);

        char stackBuffer[bufferSize];
        std::pmr::monotonic_buffer_resource stackResource(stackBuffer, bufferSize);

        void* vectorStorage = stackResource.allocate(sizeof(std::pmr::vector<const_vector_view>));
        std::pmr::vector<const_vector_view>& targetData = *(new (vectorStorage) std::pmr::vector<const_vector_view>(&stackResource));

        targetData.resize(targetIndecies.size());
        std::ranges::transform(targetIndecies, targetData.begin(), [&](const auto index) { return (*targetBlock)[index]; });

        return ComputeBatch(points[searchIndex], targetData, functor);
    };

    void SetBlock(size_t targetBlockNum) { this->targetBlock = &(blocks[targetBlockNum]); }
};

template<typename DistType>
struct SinglePointFunctor {

    SinglePointFunctor() = default;

    SinglePointFunctor(const SinglePointFunctor& other) : ptrToFunc(other.ptrToFunc){};

    SinglePointFunctor& operator=(const SinglePointFunctor&) = default;

    template<IsNot<SinglePointFunctor> DistanceFunctor>
    SinglePointFunctor(DistanceFunctor& distanceFunctor) : ptrToFunc(std::make_shared<ConcreteFunctor<DistanceFunctor>>(distanceFunctor)){};

    DistType operator()(const size_t functorParam, const size_t targetIndex) const {
        return this->ptrToFunc->operator()(functorParam, targetIndex);
    };

    std::pmr::vector<DistType> operator()(const size_t functorParam, std::span<const size_t> targetIndecies) const {
        return this->ptrToFunc->operator()(functorParam, targetIndecies);
    };

    void SetBlock(size_t targetBlockNum) { this->ptrToFunc->SetBlock(targetBlockNum); };

  private:
    struct AbstractFunctor {
        virtual ~AbstractFunctor(){};
        virtual DistType operator()(const size_t functorParam, const size_t targetIndex) const = 0;
        virtual std::pmr::vector<DistType> operator()(const size_t functorParam, std::span<const size_t> targetIndecies) const = 0;
        virtual void SetBlock(size_t targetBlockNum) = 0;
    };

    template<typename DistanceFunctor>
    struct ConcreteFunctor final : AbstractFunctor {

        DistanceFunctor underlyingFunctor;

        ConcreteFunctor(DistanceFunctor underlyingFunctor) : underlyingFunctor(underlyingFunctor){};
        //~ConcreteFunctor() final = default;

        DistType operator()(const size_t functorParam, size_t targetIndex) const final {
            return this->underlyingFunctor(functorParam, targetIndex);
        };

        std::pmr::vector<DistType> operator()(const size_t functorParam, std::span<const size_t> targetIndecies) const final {
            return this->underlyingFunctor(functorParam, targetIndecies);
        };

        void SetBlock(size_t targetBlockNum) final { this->underlyingFunctor.SetBlock(targetBlockNum); };
    };

  private:
    std::shared_ptr<AbstractFunctor> ptrToFunc;
};

} // namespace nnd

#endif
