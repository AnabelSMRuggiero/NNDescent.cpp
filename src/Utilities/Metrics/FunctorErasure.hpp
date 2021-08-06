#ifndef NND_FUNCTORERASURE_HPP
#define NND_FUNCTORERASURE_HPP

#include <vector>
#include <concepts>
#include <memory>
#include <type_traits>
#include <concepts>

#include "Utilities/Type.hpp"
#include "Utilities/Data.hpp"
#include "Utilities/Metrics/SpaceMetrics.hpp"


namespace nnd{



struct EuclideanMetricPair{
    using DistType = float;
    float operator()(const AlignedSpan<const float> lhsVector, const AlignedSpan<const float> rhsVector) const{
        return EuclideanNorm<AlignedSpan<const float>, AlignedSpan<const float>, float>(lhsVector, rhsVector);
    };
    
    std::vector<float> operator()(AlignedSpan<const float> lhsVector, const std::vector<AlignedSpan<const float>>& rhsVectors) const{
        return EuclideanBatcher(lhsVector, rhsVectors);
    };
};

template<typename DataEntry, typename MetricPair>
struct MetricFunctor{
    using DistType = typename MetricPair::DistType;
    using DataView = typename DataBlock<DataEntry>::DataView;

    [[no_unique_address]] MetricPair metricPair;
    const DataBlock<DataEntry>* lhsBlock;
    const DataBlock<DataEntry>* rhsBlock;
    const std::vector<DataBlock<DataEntry>>& blocks;
    //size_t lhsBlockNum, rhsBlockNum;
    
    MetricFunctor(const std::vector<DataBlock<DataEntry>>& blocks): metricPair(MetricPair()), blocks(blocks) {};

    MetricFunctor(MetricPair metricPair, const std::vector<DataBlock<DataEntry>>& blocks):metricPair(metricPair), blocks(blocks) {};

    


    DistType operator()(size_t LHSIndex, size_t RHSIndex) const {
        return metricPair((*lhsBlock)[LHSIndex], (*rhsBlock)[RHSIndex]);
    };
    
    std::vector<DistType> operator()(const size_t lhsIndex, const std::vector<size_t>& rhsIndecies) const {
        std::vector<DataView> rhsData;
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

template<typename Type, typename OtherType>
concept IsNot = !std::same_as<Type, OtherType>;

template<typename DistType>
struct DispatchFunctor{

    DispatchFunctor() = default;

    DispatchFunctor(const DispatchFunctor& other):
        ptrToFunc(other.ptrToFunc){};

    DispatchFunctor& operator=(const DispatchFunctor&) = default;

    template<IsNot<DispatchFunctor> DistanceFunctor>
    DispatchFunctor(DistanceFunctor& distanceFunctor):
        ptrToFunc(std::make_shared<ConcreteFunctor<DistanceFunctor>>(distanceFunctor)){};
    
    DistType operator()(const size_t LHSIndex, const size_t RHSIndex) const{
        return this->ptrToFunc->operator()(LHSIndex, RHSIndex);
    };

    std::vector<DistType> operator()(const size_t lhsIndex, const std::vector<size_t>& rhsIndecies) const{
        return this->ptrToFunc->operator()(lhsIndex, rhsIndecies);
    };

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum){
        this->ptrToFunc->SetBlocks(lhsBlockNum, rhsBlockNum);
    }


    private:
    struct AbstractFunctor{
        virtual ~AbstractFunctor(){};
        virtual DistType operator()(size_t LHSIndex, size_t RHSIndex) const = 0;
        virtual std::vector<DistType> operator()(const size_t lhsIndex, const std::vector<size_t>& rhsIndecies) const = 0;
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

        std::vector<DistType> operator()(const size_t lhsIndex, const std::vector<size_t>& rhsIndecies) const final{
            return this->underlyingFunctor(lhsIndex, rhsIndecies);
        };

        void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum) final{
            this->underlyingFunctor.SetBlocks(lhsBlockNum, rhsBlockNum);
        }

    };

    private:
    std::shared_ptr<AbstractFunctor> ptrToFunc;
    
};


struct EuclideanComDistance{
    using DistType = float;
    float operator()(const AlignedSpan<const float> dataVector, const AlignedSpan<const float> comVector) const{
        return EuclideanNorm<AlignedSpan<const float>, AlignedSpan<const float>, float>(comVector, dataVector);
    };
    
    std::vector<float> operator()(const AlignedSpan<const float> comVector, const std::vector<AlignedSpan<const float>>& rhsVectors) const{
        return EuclideanBatcher(comVector, rhsVectors);
    };
};

template<typename COMExtent>
struct MetaGraph;

template<typename DataEntry, typename COMExtent, typename MetricPair>
struct DataComDistance{
    using DistType = typename MetricPair::DistType;
    using DataView = typename DataBlock<DataEntry>::DataView;
    //Reference to Com?
    const MetaGraph<COMExtent>& centersOfMass;
    const DataBlock<DataEntry>* targetBlock;
    const std::vector<DataBlock<DataEntry>>& blocks;
    [[no_unique_address]] MetricPair functor;

    DataComDistance(const MetaGraph<COMExtent>& centersOfMass, const std::vector<DataBlock<DataEntry>>& blocks): centersOfMass(centersOfMass), blocks(blocks), functor(){};

    DataComDistance(const MetaGraph<COMExtent>& centersOfMass, const std::vector<DataBlock<DataEntry>>& blocks, MetricPair functor):
                        centersOfMass(centersOfMass), blocks(blocks), functor(functor){};

    float operator()(const size_t metagraphIndex, const size_t dataIndex) const{
        return functor(centersOfMass.points[metagraphIndex].centerOfMass, (*targetBlock)[dataIndex]);
    };
    
    std::vector<float> operator()(const size_t metagraphIndex, const std::vector<size_t>& rhsIndecies) const{
        std::vector<DataView> rhsData;
        for(const auto& index: rhsIndecies){
            rhsData.push_back((*targetBlock)[index]);
        }
        return functor(centersOfMass.points[metagraphIndex].centerOfMass, rhsData);
    };

    void SetBlock(size_t targetBlockNum){
        this->targetBlock = &(blocks[targetBlockNum]);
    }
    
};

template<typename DataEntry, typename MetricPair>
struct SearchFunctor{
    using DistType = typename MetricPair::DistType;
    using DataView = typename DataBlock<DataEntry>::DataView;
    const DataBlock<DataEntry>* targetBlock;
    const std::vector<DataBlock<DataEntry>>& blocks;
    const DataSet<DataEntry>& points;
    [[no_unique_address]] MetricPair functor;

    SearchFunctor(const std::vector<DataBlock<DataEntry>>& blocks, const DataSet<DataEntry>& points):
        blocks(blocks), points(points), functor(){};

    SearchFunctor(const std::vector<DataBlock<DataEntry>>& blocks, const DataSet<DataEntry>& points, MetricPair functor):
                        blocks(blocks), points(points), functor(functor){};

    float operator()(const size_t searchIndex, const size_t targetIndex) const{
        return functor(points[searchIndex], (*targetBlock)[targetIndex]);
    };
    
    std::vector<float> operator()(const size_t searchIndex, const std::vector<size_t>& targetIndecies) const{
        std::vector<DataView> targetData;
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

    SinglePointFunctor(SinglePointFunctor& other): ptrToFunc(other.ptrToFunc){};

    template<IsNot<SinglePointFunctor> DistanceFunctor>
    SinglePointFunctor(DistanceFunctor& distanceFunctor):
        ptrToFunc(std::make_shared<ConcreteFunctor<DistanceFunctor>>(distanceFunctor)){};
    
    DistType operator()(const size_t functorParam, const size_t targetIndex) const{
        return this->ptrToFunc->operator()(functorParam, targetIndex);
    };

    std::vector<DistType> operator()(const size_t functorParam, const std::vector<size_t>& targetIndecies) const{
        return this->ptrToFunc->operator()(functorParam, targetIndecies);
    };

    void SetBlock(size_t targetBlockNum){
        this->ptrToFunc->SetBlock(targetBlockNum);
    };

    private:
    struct AbstractFunctor{
        virtual ~AbstractFunctor(){};
        virtual DistType operator()(const size_t functorParam, const size_t targetIndex) const = 0;
        virtual std::vector<DistType> operator()(const size_t functorParam, const std::vector<size_t>& targetIndecies) const = 0;
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

        std::vector<DistType> operator()(const size_t functorParam, const std::vector<size_t>& targetIndecies) const final{
            return this->underlyingFunctor(functorParam, targetIndecies);
        };

        void SetBlock(size_t targetBlockNum) final{
            this->underlyingFunctor.SetBlock(targetBlockNum);
        };

    };

    const std::shared_ptr<AbstractFunctor> ptrToFunc;
    
};





}

#endif
