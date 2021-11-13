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

#include <vector>
#include <concepts>
#include <memory>
#include <type_traits>
#include <concepts>
#include <memory_resource>

#include "Utilities/Type.hpp"
#include "Utilities/Data.hpp"
#include "Utilities/Metrics/SpaceMetrics.hpp"


namespace nnd{

template<typename MetricPair, typename LHSData, typename RHSData>
struct BasicFunctor{
    using DistType = typename MetricPair::DistType;
    using ConstDataView = typename RHSData::ConstDataView;

    [[no_unique_address]] MetricPair metricPair;
    const LHSData* lhsBlock;
    const RHSData* rhsBlock;
    //size_t lhsBlockNum, rhsBlockNum;
    
    BasicFunctor(const LHSData& lhsData, const RHSData& rhsData): metricPair(MetricPair()), lhsBlock(&lhsData), rhsBlock(&rhsData) {};

    BasicFunctor(MetricPair metricPair, const LHSData& lhsData, const RHSData& rhsData): metricPair(metricPair), lhsBlock(&lhsData), rhsBlock(&rhsData) {};

    DistType operator()(size_t LHSIndex, size_t RHSIndex) const {
        return metricPair((*lhsBlock)[LHSIndex], (*rhsBlock)[RHSIndex]);
    };
    
    std::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const {
        char stackBuffer[sizeof(ConstDataView)*20];
        std::pmr::monotonic_buffer_resource stackResource(stackBuffer, sizeof(ConstDataView)*20);
        std::pmr::vector<ConstDataView> rhsData(&stackResource);
        rhsData.reserve(20);
        for(const auto& index: rhsIndecies){
            rhsData.push_back((*rhsBlock)[index]);
        }
        return metricPair((*lhsBlock)[lhsIndex], rhsData);
    };

};

template<typename DistType>
struct ErasedMetricPair{

    ErasedMetricPair() = default;

    ErasedMetricPair(const ErasedMetricPair& other):
        ptrToFunc(other.ptrToFunc){};

    ErasedMetricPair& operator=(const ErasedMetricPair&) = default;

    template<IsNot<ErasedMetricPair> DistanceFunctor>
    ErasedMetricPair(DistanceFunctor& distanceFunctor):
        ptrToFunc(std::make_shared<ConcreteFunctor<DistanceFunctor>>(distanceFunctor)){};
    
    DistType operator()(const size_t LHSIndex, const size_t RHSIndex) const{
        return this->ptrToFunc->operator()(LHSIndex, RHSIndex);
    };

    std::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const{
        return this->ptrToFunc->operator()(lhsIndex, rhsIndecies);
    };


    private:
    struct AbstractFunctor{
        virtual ~AbstractFunctor(){};
        virtual DistType operator()(size_t LHSIndex, size_t RHSIndex) const = 0;
        virtual std::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const = 0;
    };

    template<typename DistanceFunctor>
    struct ConcreteFunctor final : AbstractFunctor{

        DistanceFunctor underlyingFunctor;

        ConcreteFunctor(DistanceFunctor underlyingFunctor): underlyingFunctor(underlyingFunctor){};
        //~ConcreteFunctor() final = default;

        DistType operator()(size_t LHSIndex, size_t RHSIndex) const final {
            return this->underlyingFunctor(LHSIndex, RHSIndex);
        };

        std::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const final{
            return this->underlyingFunctor(lhsIndex, rhsIndecies);
        };


    };

    private:
    std::shared_ptr<AbstractFunctor> ptrToFunc;
    
};



template<typename DataType, typename MetricPair>
struct MetricFunctor{
    using DistType = typename MetricPair::DistType;
    using ConstDataView = typename DataBlock<DataType>::ConstDataView;

    [[no_unique_address]] MetricPair metricPair;
    const DataBlock<DataType>* lhsBlock;
    const DataBlock<DataType>* rhsBlock;
    std::span<const DataBlock<DataType>> blocks;
    //size_t lhsBlockNum, rhsBlockNum;
    
    MetricFunctor(const std::vector<DataBlock<DataType>>& blocks): metricPair(MetricPair()), blocks(blocks.data(), blocks.size()) {};

    MetricFunctor(MetricPair metricPair, const std::vector<DataBlock<DataType>>& blocks):metricPair(metricPair), blocks(blocks.data(), blocks.size()) {};

    


    DistType operator()(size_t LHSIndex, size_t RHSIndex) const {
        return metricPair((*lhsBlock)[LHSIndex], (*rhsBlock)[RHSIndex]);
    };
    
    std::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const noexcept {
        char stackBuffer[sizeof(ConstDataView)*20];
        std::pmr::monotonic_buffer_resource stackResource(stackBuffer, sizeof(ConstDataView)*20);
        std::pmr::vector<ConstDataView> rhsData(&stackResource);
        rhsData.reserve(20);
        for(const auto& index: rhsIndecies){
            rhsData.push_back((*rhsBlock)[index]);
        }
        return metricPair((*lhsBlock)[lhsIndex], rhsData);
    };

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum){
        this->lhsBlock = &(blocks[lhsBlockNum]);
        this->rhsBlock = &(blocks[rhsBlockNum]);
    }
};

template<typename DataType, typename MetricPair>
struct CrossFragmentFunctor{
    using DistType = typename MetricPair::DistType;
    using ConstDataView = typename DataBlock<DataType>::ConstDataView;

    [[no_unique_address]] MetricPair metricPair;
    const DataBlock<DataType>* lhsBlock;
    const DataBlock<DataType>* rhsBlock;
    std::span<const DataBlock<DataType>> lhsBlocks;
    std::span<const DataBlock<DataType>> rhsBlocks;
    //size_t lhsBlockNum, rhsBlockNum;
    
    CrossFragmentFunctor(const std::vector<DataBlock<DataType>>& lhsBlocks, const std::vector<DataBlock<DataType>>& rhsBlocks): metricPair(MetricPair()),
        lhsBlocks(lhsBlocks.data(), lhsBlocks.size()),
        rhsBlocks(rhsBlocks.data(), rhsBlocks.size()) {};

    CrossFragmentFunctor(MetricPair metricPair, const std::vector<DataBlock<DataType>>& lhsBlocks, const std::vector<DataBlock<DataType>>& rhsBlocks):
        metricPair(metricPair), 
        lhsBlocks(lhsBlocks.data(), lhsBlocks.size()),
        rhsBlocks(rhsBlocks.data(), rhsBlocks.size()) {};

    


    DistType operator()(size_t LHSIndex, size_t RHSIndex) const {
        return metricPair((*lhsBlock)[LHSIndex], (*rhsBlock)[RHSIndex]);
    };
    
    std::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const {
        constexpr size_t bufferSize = sizeof(ConstDataView)*23 + sizeof(std::pmr::vector<ConstDataView>);
        char stackBuffer[sizeof(ConstDataView)*23 + sizeof(std::pmr::vector<ConstDataView>)];
        std::pmr::monotonic_buffer_resource stackResource(stackBuffer, bufferSize);
        std::pmr::vector<ConstDataView>& rhsData = *(new (stackResource.allocate(sizeof(std::pmr::vector<ConstDataView>))) std::pmr::vector<ConstDataView>(&stackResource));
        //
        rhsData.reserve(20);
        for(const auto& index: rhsIndecies){
            rhsData.push_back((*rhsBlock)[index]);
        }
        return metricPair((*lhsBlock)[lhsIndex], rhsData);
    };

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum){
        this->lhsBlock = &(lhsBlocks[lhsBlockNum]);
        this->rhsBlock = &(rhsBlocks[rhsBlockNum]);
    }
};


template<typename DistType>
struct DispatchFunctor{

    DispatchFunctor() = default;

    DispatchFunctor(const DispatchFunctor& other):
        ptrToFunc(other.ptrToFunc){};

    DispatchFunctor& operator=(const DispatchFunctor&) = default;

    template<IsNot<DispatchFunctor> DistanceFunctor>
    DispatchFunctor(DistanceFunctor&& distanceFunctor):
        ptrToFunc(std::make_shared<ConcreteFunctor<DistanceFunctor>>(
                  std::forward<DistanceFunctor>(distanceFunctor))){};
    
    DistType operator()(const size_t LHSIndex, const size_t RHSIndex) const{
        return this->ptrToFunc->operator()(LHSIndex, RHSIndex);
    };

    std::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const{
        return this->ptrToFunc->operator()(lhsIndex, rhsIndecies);
    };

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum){
        this->ptrToFunc->SetBlocks(lhsBlockNum, rhsBlockNum);
    }


    private:
    struct AbstractFunctor{
        virtual ~AbstractFunctor(){};
        virtual DistType operator()(size_t LHSIndex, size_t RHSIndex) const = 0;
        virtual std::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const = 0;
        virtual void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum) = 0;
    };

    template<typename DistanceFunctor>
    struct ConcreteFunctor final : AbstractFunctor{

        DistanceFunctor underlyingFunctor;

        ConcreteFunctor(DistanceFunctor underlyingFunctor): underlyingFunctor(underlyingFunctor){};
        //~ConcreteFunctor() final = default;

        DistType operator()(size_t LHSIndex, size_t RHSIndex) const final {
            return this->underlyingFunctor(LHSIndex, RHSIndex);
        };

        std::vector<DistType> operator()(const size_t lhsIndex, std::span<const size_t> rhsIndecies) const final{
            return this->underlyingFunctor(lhsIndex, rhsIndecies);
        };

        void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum) final{
            this->underlyingFunctor.SetBlocks(lhsBlockNum, rhsBlockNum);
        }

    };

    private:
    std::shared_ptr<AbstractFunctor> ptrToFunc;
    
};




template<typename COMExtent>
struct MetaGraph;

template<typename DataType, typename COMExtent, typename MetricPair>
struct DataComDistance{
    using DistType = typename MetricPair::DistType;
    using ConstDataView = typename DataBlock<DataType>::ConstDataView;
    //Reference to Com?
    const MetaGraph<COMExtent>& centersOfMass;
    const DataBlock<DataType>* targetBlock;
    std::span<const DataBlock<DataType>> blocks;
    [[no_unique_address]] MetricPair functor;

    DataComDistance(const MetaGraph<COMExtent>& centersOfMass, const std::vector<DataBlock<DataType>>& blocks): centersOfMass(centersOfMass), blocks(blocks.data(), blocks.size()), functor(){};

    DataComDistance(const MetaGraph<COMExtent>& centersOfMass, const std::vector<DataBlock<DataType>>& blocks, MetricPair functor):
                        centersOfMass(centersOfMass), blocks(blocks.data(), blocks.size(), blocks[0].blockNumber), functor(functor){};

    float operator()(const size_t metagraphIndex, const size_t dataIndex) const{
        return functor(centersOfMass.points[metagraphIndex], (*targetBlock)[dataIndex]);
    };
    
    std::vector<float> operator()(const size_t metagraphIndex, std::span<const size_t> rhsIndecies) const{
        std::vector<ConstDataView> rhsData;
        for(const auto& index: rhsIndecies){
            rhsData.push_back((*targetBlock)[index]);
        }
        return functor(centersOfMass.points[metagraphIndex], rhsData);
    };

    void SetBlock(size_t targetBlockNum){
        this->targetBlock = &(blocks[targetBlockNum]);
    }
    
};

template<typename DataType, typename DataSet, typename MetricPair>
struct SearchFunctor{
    using DistType = typename MetricPair::DistType;
    using ConstDataView = typename DataBlock<DataType>::ConstDataView;
    const DataBlock<DataType>* targetBlock;
    std::span<const DataBlock<DataType>> blocks;
    const DataSet& points;
    [[no_unique_address]] MetricPair functor;

    SearchFunctor(const std::vector<DataBlock<DataType>>& blocks, const DataSet& points):
        blocks(blocks.data(), blocks.size()), points(points), functor(){};

    SearchFunctor(const std::vector<DataBlock<DataType>>& blocks, const DataSet& points, MetricPair functor):
                        blocks(blocks.data(), blocks.size()), points(points), functor(functor){};

    float operator()(const size_t searchIndex, const size_t targetIndex) const{
        return functor(points[searchIndex], (*targetBlock)[targetIndex]);
    };
    
    std::vector<typename MetricPair::DistType> operator()(const size_t searchIndex, std::span<const size_t> targetIndecies) const{
        constexpr size_t bufferSize = sizeof(ConstDataView)*23 + sizeof(std::pmr::vector<ConstDataView>);
        char stackBuffer[sizeof(ConstDataView)*23 + sizeof(std::pmr::vector<ConstDataView>)];
        std::pmr::monotonic_buffer_resource stackResource(stackBuffer, bufferSize);
        std::pmr::vector<ConstDataView>& targetData = *(new (stackResource.allocate(sizeof(std::pmr::vector<ConstDataView>))) std::pmr::vector<ConstDataView>(&stackResource));
        //std::pmr::vector<ConstDataView> targetData(&stackResource);
        targetData.reserve(20);
        for(const auto& index: targetIndecies){
            targetData.push_back((*targetBlock)[index]);
        }
        return functor(points[searchIndex], targetData);
    };

    void SetBlock(size_t targetBlockNum){
        this->targetBlock = &(blocks[targetBlockNum]);
    }
    
};

template<typename DistType>
struct SinglePointFunctor{

    SinglePointFunctor() = default;

    SinglePointFunctor(const SinglePointFunctor& other): ptrToFunc(other.ptrToFunc){};

    SinglePointFunctor& operator=(const SinglePointFunctor&) = default;

    template<IsNot<SinglePointFunctor> DistanceFunctor>
    SinglePointFunctor(DistanceFunctor& distanceFunctor):
        ptrToFunc(std::make_shared<ConcreteFunctor<DistanceFunctor>>(distanceFunctor)){};
    
    DistType operator()(const size_t functorParam, const size_t targetIndex) const{
        return this->ptrToFunc->operator()(functorParam, targetIndex);
    };

    std::vector<DistType> operator()(const size_t functorParam, std::span<const size_t> targetIndecies) const{
        return this->ptrToFunc->operator()(functorParam, targetIndecies);
    };

    void SetBlock(size_t targetBlockNum){
        this->ptrToFunc->SetBlock(targetBlockNum);
    };

    private:
    struct AbstractFunctor{
        virtual ~AbstractFunctor(){};
        virtual DistType operator()(const size_t functorParam, const size_t targetIndex) const = 0;
        virtual std::vector<DistType> operator()(const size_t functorParam, std::span<const size_t> targetIndecies) const = 0;
        virtual void SetBlock(size_t targetBlockNum) = 0;
    };

    template<typename DistanceFunctor>
    struct ConcreteFunctor final : AbstractFunctor{

        DistanceFunctor underlyingFunctor;

        ConcreteFunctor(DistanceFunctor underlyingFunctor): underlyingFunctor(underlyingFunctor){};
        //~ConcreteFunctor() final = default;

        DistType operator()(const size_t functorParam, size_t targetIndex) const final {
            return this->underlyingFunctor(functorParam, targetIndex);
        };

        std::vector<DistType> operator()(const size_t functorParam, std::span<const size_t> targetIndecies) const final{
            return this->underlyingFunctor(functorParam, targetIndecies);
        };

        void SetBlock(size_t targetBlockNum) final{
            this->underlyingFunctor.SetBlock(targetBlockNum);
        };

    };

    private:
    std::shared_ptr<AbstractFunctor> ptrToFunc;
    
};





}

#endif
