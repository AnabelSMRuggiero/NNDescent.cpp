#ifndef NND_FUNCTORERASURE_HPP
#define NND_FUNCTORERASURE_HPP

#include <vector>
#include <concepts>
#include <memory>

#include "Utilities/Type.hpp"
#include "Utilities/Data.hpp"
#include "Utilities/Metrics/SpaceMetrics.hpp"


namespace nnd{



struct EuclideanMetricPair{
    float operator()(const AlignedSpan<const float> lhsVector, const AlignedSpan<const float> rhsVector) const{
        return EuclideanNorm<AlignedSpan<const float>, AlignedSpan<const float>, float>(lhsVector, rhsVector);
    };
    
    std::vector<float> operator()(const std::vector<AlignedSpan<const float>>& lhsVectors, const AlignedSpan<const float> rhsVector) const{
        return EuclideanBatcher(lhsVectors, rhsVector);
    };
};

template<typename DataEntry, typename MetricPair, typename DistType>
struct MetricFunctor{

    using DataView = typename DataBlock<DataEntry>::DataView;

    MetricPair metricPair;
    const DataBlock<DataEntry>* lhsBlock;
    const DataBlock<DataEntry>* rhsBlock;

    size_t lhsBlockNum, rhsBlockNum;
    
    MetricFunctor(): metricPair(MetricPair()) {};

    MetricFunctor(MetricPair metricPair):metricPair(metricPair), lhsBlock(nullptr), rhsBlock(nullptr) {};

    MetricFunctor(MetricPair metricPair, const DataBlock<DataEntry>* lhsBlock, const DataBlock<DataEntry>* rhsBlock):
        metricPair(metricPair), lhsBlock(lhsBlock), rhsBlock(rhsBlock) {};


    DistType operator()(size_t LHSIndex, size_t RHSIndex) const {
        return metricPair((*lhsBlock)[LHSIndex], (*rhsBlock)[RHSIndex]);
    };
    
    std::vector<DistType> operator()(const std::vector<size_t>& lhsIndecies, size_t rhsIndex) const {
        std::vector<DataView> lhsData;
        for(const auto& index: lhsIndecies){
            lhsData.push_back((*lhsBlock)[index]);
        }
        return metricPair(lhsData, (*rhsBlock)[rhsIndex]);
    };

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum){
        this->lhsBlockNum = lhsBlockNum;
        this->rhsBlockNum = rhsBlockNum;
    }
};



template<typename DistType>
struct DispatchFunctor{

    template<typename DistanceFunctor>
    DispatchFunctor(DistanceFunctor& distanceFunctor):
        ptrToFunc(std::make_shared<ConcreteFunctor<DistanceFunctor>>(distanceFunctor)){};
    
    DistType operator()(const size_t LHSIndex, const size_t RHSIndex) const{
        return this->*ptrToFunc(LHSIndex, RHSIndex);
    };

    std::vector<DistType> operator()(const std::vector<size_t>& LHSIndecies, const size_t RHSIndex) const{
        return this->*ptrToFunc(LHSIndecies, RHSIndex);
    };

    void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum){
        this->ptrToFunc->SetBlocks(lhsBlockNum, rhsBlockNum);
    }


    private:
    struct AbstractFunctor{
        virtual ~AbstractFunctor(){};
        virtual DistType operator()(size_t LHSIndex, size_t RHSIndex) const = 0;
        virtual std::vector<DistType> operator()(const std::vector<size_t>& LHSIndecies, size_t RHSIndex) const = 0;
        virtual void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum) = 0;
    };

    template<typename DistanceFunctor>
    struct ConcreteFunctor final : AbstractFunctor{

        DistanceFunctor& underlyingFunctor;

        ConcreteFunctor(DistanceFunctor& underlyingFunctor): underlyingFunctor(underlyingFunctor){};
        //~ConcreteFunctor() final = default;

        DistType operator()(size_t LHSIndex, size_t RHSIndex) const final {
            return this->underlyingFunctor(LHSIndex, RHSIndex);
        };

        std::vector<DistType> operator()(const std::vector<size_t>& LHSIndecies, size_t RHSIndex) const final{
            return this->underlyingFunctor(LHSIndecies, RHSIndex);
        };

        void SetBlocks(size_t lhsBlockNum, size_t rhsBlockNum) final{
            this->underlyingFunctor.SetBlocks(lhsBlockNum, rhsBlockNum);
        }

    };

    const std::shared_ptr<AbstractFunctor> ptrToFunc;
    
};


template<typename DistType>
struct CachingFunctor{

    DispatchFunctor<DistType> metricFunctor;
    DistanceCache<DistType> cache;

    CachingFunctor(DispatchFunctor<DistType> metricFunctor, size_t cacheSize): metricFunctor(metricFunctor), cache(){
        cache.reserve(cacheSize);
    }

    DistType operator()(size_t LHSIndex, size_t RHSIndex){
        return this->underlyingFunctor(LHSIndex, RHSIndex);
    };

    std::vector<DistType> operator()(const std::vector<size_t>& LHSIndecies, size_t RHSIndex){
        return this->underlyingFunctor(LHSIndecies, RHSIndex);
    };

};

}

#endif
