/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef RPT_SPLITTINGSCHEMERF_HPP
#define RPT_SPLITTINGSCHEMERF_HPP

#include <algorithm>
#include <type_traits>
#include <functional>
#include <thread>
#include <numeric>
#include <iterator>
#include <variant>

#include "Parallelization/AsyncQueue.hpp"
#include "ann/Metrics/SpaceMetrics.hpp"
#include "ann/AlignedMemory/DynamicArray.hpp"
#include "ann/Data.hpp"

namespace nnd{

/*
template<splitting_scheme SchemeType>
struct serial_splitting_scheme;

template<splitting_scheme SchemeType>
struct parallel_splitting_scheme;

template<splitting_scheme SchemeType>
struct borrowed_splitting_scheme;
*/

enum struct splitting_scheme{
    euclidean,
    angular
};

template<typename Metric>
constexpr splitting_scheme choose_scheme = splitting_scheme::euclidean;


using euclidean_splitting_vectors = std::unordered_map<size_t, std::pair<ann::aligned_array<float>, float>>;

using angular_splitting_vectors = std::unordered_map<size_t, ann::aligned_array<float>>;

using splitting_vectors = std::variant<euclidean_splitting_vectors, angular_splitting_vectors>;

template<typename DataEntry, typename DistType>
ann::aligned_array<DistType> EuclidianSplittingPlaneNormal(const DataEntry& pointA, const DataEntry& pointB){
    ann::aligned_array<DistType> splittingLine(pointA.size());
    for (size_t i = 0; i < pointA.size(); i += 1){
        splittingLine[i] = DistType(pointA[i]) - DistType(pointB[i]);
    }
    DistType splittingLineMag(0);
    for (DistType i : splittingLine){
        splittingLineMag += i*i;
    }
    splittingLineMag = std::sqrt(splittingLineMag);

    for(auto& extent: splittingLine) extent /= splittingLineMag;

    return splittingLine;
}



auto bind_euclidean_predicate(std::random_access_iterator auto data, auto splitting_vector, auto offset){
    return [=](std::size_t index){
        return 0.0 < (ann::Dot(data[index], splitting_vector) + offset);
    };
}

//std::random_access_iterator<ann::data_iterator<float, 32_a>>;
struct TransformTag {};
static const TransformTag transformTag;

//The serial case
template<typename DataType, typename SplittingVector>
struct euclidean_scheme{
    using OffSetType = typename SplittingVector::value_type;
    using SplittingView = typename DefaultDataView<SplittingVector>::ViewType;
    using SplittingVectors = std::unordered_map<size_t, std::pair<SplittingVector, OffSetType>>;

    //using DataView = typename DefaultDataView<DataEntry>::ViewType;

    using ParallelScheme = std::false_type;
    using SerialScheme = std::true_type;
    
    using DataView = typename DataSet<DataType>::ConstDataView;

    const DataSet<DataType>& dataSource;
    std::unordered_map<size_t, std::pair<SplittingVector, OffSetType>> splittingVectors;
    //DistType projectionOffset;

    euclidean_scheme(const DataSet<DataType>& data) : dataSource(data), splittingVectors(){};

    auto operator()(size_t splitIndex, std::pair<size_t, size_t> splittingPoints){
        
        
        // For right now at least, in the serial case I want to be able to get a new splitting vector
        //if (splittingVectors.find(splitIndex) == splittingVectors.end()){
        SplittingVector splittingVector = EuclidianSplittingPlaneNormal<DataView, OffSetType>(dataSource[splittingPoints.first], dataSource[splittingPoints.second]);


        OffSetType projectionOffset = 0;
        for (size_t i = 0; i<dataSource[splittingPoints.first].size(); i+=1){
            projectionOffset -= splittingVector[i] * OffSetType(dataSource[splittingPoints.first][i] + dataSource[splittingPoints.second][i])/2.0;
        };

        splittingVectors[splitIndex] = std::pair<SplittingVector, OffSetType>(std::move(splittingVector), projectionOffset);

        //};
        if constexpr(is_aligned_contiguous_range_v<SplittingVector>){
            return bind_euclidean_predicate(dataSource.begin(), SplittingView(splittingVectors[splitIndex].first), splittingVectors[splitIndex].second);

        } else {
            SplittingView splitter_view{splittingVectors[splitIndex].first.begin(), splittingVectors[splitIndex].first.end()};
            return bind_euclidean_predicate(dataSource.begin(), splitter_view, splittingVectors[splitIndex].second);
        }

        
    };

    auto operator()(size_t splitIndex, TransformTag) const {
        const std::pair<SplittingVector, OffSetType>& splitPair = splittingVectors.at(splitIndex);
        if constexpr(is_aligned_contiguous_range_v<SplittingVector>){
            return bind_euclidean_predicate(dataSource.begin(), SplittingView(splitPair.first), splitPair.second);
            /*
            auto comparisonFunction = [=, 
                                    &data = std::as_const(this->dataSource), 
                                    splitter = SplittingView(splitPair.first),
                                    offset = splitPair.second]
                                    (size_t comparisonIndex) -> bool{
                    return 0.0 < (ann::Dot(data[comparisonIndex], splitter) + offset);
            };
            
            return comparisonFunction;
            */
        } else {
            auto comparisonFunction = [=, 
                                    &data = std::as_const(this->dataSource), 
                                    splitter = SplittingView(splitPair.first.begin(), splitPair.first.size()),
                                    offset = splitPair.second]
                                    (size_t comparisonIndex) -> bool{
                    return 0.0 < (ann::Dot(data[comparisonIndex], splitter) + offset);
            };
            return comparisonFunction;
        }

        
    };
    
};

template<typename DataType, typename SplittingVector>
struct parallel_euclidean_scheme{
    using OffSetType = typename SplittingVector::value_type;
    using SplittingView = typename DefaultDataView<SplittingVector>::ViewType;
    using SplittingVectors = std::unordered_map<size_t, std::pair<SplittingVector, OffSetType>>;

    //using DataView = typename DefaultDataView<DataEntry>::ViewType;
    //Using EntryView

    using ParallelScheme = std::true_type;
    using SerialScheme = std::false_type;
    using DataView = typename DataSet<DataType>::DataView;
    
    const DataSet<DataType>& dataSource;
    std::unordered_map<size_t, std::pair<SplittingVector, OffSetType>> splittingVectors;

    AsyncQueue<std::tuple<size_t, SplittingVector, OffSetType>> generatedVectors;
    //DistType projectionOffset;

    parallel_euclidean_scheme(const DataSet<DataType>& data) : dataSource(data), splittingVectors(){};

    auto operator()(size_t splitIndex, std::pair<size_t, size_t> splittingPoints){
        
        
        // For right now at least, in the serial case I want to be able to get a new splitting vector
        //if (splittingVectors.find(splitIndex) == splittingVectors.end()){
        SplittingVector splittingVector = EuclidianSplittingPlaneNormal<DataView, OffSetType>(dataSource[splittingPoints.first], dataSource[splittingPoints.second]);


        OffSetType projectionOffset = 0;
        for (size_t i = 0; i<dataSource[splittingPoints.first].size(); i+=1){
            projectionOffset -= splittingVector[i] * OffSetType(dataSource[splittingPoints.first][i] + dataSource[splittingPoints.second][i])/2.0;
        };

        //splittingVectors[splitIndex] = std::pair<SplittingVector, OffSetType>(std::move(splittingVector), projectionOffset);

        //};
        if constexpr(is_aligned_contiguous_range_v<SplittingVector>){
            
            
            auto comparisonFunction =
                bind_euclidean_predicate(dataSource.begin(), SplittingView(splittingVector), projectionOffset);
            
            generatedVectors.Put({splitIndex, std::move(splittingVector), projectionOffset});
            return comparisonFunction;
        } else {
            auto comparisonFunction =
                bind_euclidean_predicate(dataSource.begin(), SplittingView(splittingVector.begin(), splittingVector.end()), projectionOffset);

            generatedVectors.Put({splitIndex, std::move(splittingVector), projectionOffset});
            return comparisonFunction;
        }

        
    };

    auto operator()(size_t splitIndex, TransformTag) const{
        const std::pair<SplittingVector, OffSetType>& splitPair = splittingVectors.at(splitIndex);
        if constexpr(is_aligned_contiguous_range_v<SplittingVector>){
            return bind_euclidean_predicate(dataSource.begin(), SplittingView(splittingVectors[splitIndex].first), splittingVectors[splitIndex].second);
            /*
            auto comparisonFunction = [=, 
                                    &data = std::as_const(this->dataSource), 
                                    splitter = SplittingView(splitPair.first),
                                    offset = splitPair.second]
                                    (size_t comparisonIndex) -> bool{
                    return 0.0 < (ann::Dot(data[comparisonIndex], splitter) + offset);
            };
            return comparisonFunction;
            */
        } else {
            auto comparisonFunction = [=, 
                                    &data = std::as_const(this->dataSource), 
                                    splitter = SplittingView(splitPair.first.begin(), splitPair.first.size()),
                                    offset = splitPair.second]
                                    (size_t comparisonIndex) -> bool{
                    return 0.0 < (ann::Dot(data[comparisonIndex], splitter) + offset);
            };
            return comparisonFunction;
        }

        
    };
    
    void ConsumeNewVectors(){
        std::list<std::tuple<size_t, SplittingVector, OffSetType>> newVecs = generatedVectors.TryTakeAll();
        for (auto& newVec: newVecs){
            splittingVectors[std::get<0>(newVec)] = {std::move(std::get<1>(newVec)), std::get<2>(newVec)};
        }
    }

    template<typename Predicate>
    void ConsumeNewVectors(Predicate pred){
        if (!pred(generatedVectors.GetCount())) return;
        std::list<std::tuple<size_t, SplittingVector, OffSetType>> newVecs = generatedVectors.TryTakeAll();
        for (auto& newVec: newVecs){
            splittingVectors[std::get<0>(newVec)] = {std::move(std::get<1>(newVec)), std::get<2>(newVec)};
        }
    }
};


template<typename SplittingVector>
auto make_splitting_view(const SplittingVector& vector){
    return typename DefaultDataView<SplittingVector>::ViewType{vector};
}

template<typename SplittingVector>
using euclidean_vector_map = std::unordered_map<size_t, std::pair<SplittingVector, typename SplittingVector::value_type>>;

template<typename DataType, typename SplittingVector>
auto borrowed_euclidean(const DataSet<DataType>& data, const euclidean_vector_map<SplittingVector>& splitting_vectors){

    using offset_type = typename SplittingVector::value_type;
    return [&](size_t splitIndex, TransformTag){
        const std::pair<SplittingVector, offset_type>& splitPair = splitting_vectors.at(splitIndex);
        
        return bind_euclidean_predicate(data.begin(), make_splitting_view(splitPair.first), splitPair.second);
    
    };
}

template<typename DataType, typename SplittingVector>
using borrowed_euclidean_scheme = decltype(borrowed_euclidean(std::declval<DataSet<DataType>>(), std::declval<euclidean_vector_map<SplittingVector>>()));

template<typename DataEntry, typename DistType>
ann::aligned_array<DistType> angular_splitting_plane(const DataEntry& pointA, const DataEntry& pointB){
    ann::aligned_array<DistType> splittingLine(pointA.size());


    //double normA = ann::PNorm<2>(pointA);
    //double normB = ann::PNorm<2>(pointB);

    if (std::lexicographical_compare_three_way(pointA.begin(), pointA.end(), pointB.begin(), pointB.end()) == 0){
        throw "equal vectors";
    }

    for (size_t i = 0; i < pointA.size(); i += 1){
        //splittingLine[i] = DistType(pointA[i])/normA - DistType(pointB[i])/normB;
        splittingLine[i] = DistType(pointA[i]) - DistType(pointB[i]);
    }

    double splitterNorm = ann::PNorm<2>(splittingLine);

    for(auto& extent: splittingLine) extent /= splitterNorm;

    return splittingLine;
}

auto bind_angular_predicate(std::random_access_iterator auto data, auto splitting_vector){
    return [=](std::size_t index){
        return 0.0 < (ann::Dot(data[index], splitting_vector));
    };
}

template<typename DataType, typename SplittingVector>
struct angular_scheme{
    using OffSetType = typename SplittingVector::value_type;
    using SplittingView = typename DefaultDataView<SplittingVector>::ViewType;
    using SplittingVectors = std::unordered_map<size_t, SplittingVector>;

    using ParallelScheme = std::false_type;
    using SerialScheme = std::true_type;
    using DataView = typename DataSet<DataType>::DataView;

    const DataSet<DataType>& dataSource;
    std::unordered_map<size_t, SplittingVector> splittingVectors;
    //DistType projectionOffset;

    angular_scheme(const DataSet<DataType>& data) : dataSource(data), splittingVectors(){};

    auto operator()(size_t splitIndex, std::pair<size_t, size_t> splittingPoints){

        splittingVectors[splitIndex] = angular_splitting_plane<DataView, OffSetType>(dataSource[splittingPoints.first], dataSource[splittingPoints.second]);

        //};
        if constexpr(is_aligned_contiguous_range_v<SplittingVector>){
            
            return bind_angular_predicate(std::as_const(dataSource).begin(), SplittingView(splittingVectors[splitIndex]));
            
        } else {

            return bind_angular_predicate(std::as_const(dataSource).begin(), SplittingView(splittingVectors[splitIndex].begin(), splittingVectors[splitIndex].end()));
            
        }

        
    };

    auto operator()(size_t splitIndex, TransformTag) const {
        const SplittingVector& splittingVec = splittingVectors.at(splitIndex);

        if constexpr(is_aligned_contiguous_range_v<SplittingVector>){
            
            return bind_angular_predicate(std::as_const(dataSource).begin(), SplittingView(splittingVec));
            
        } else {

            return bind_angular_predicate(std::as_const(dataSource).begin(), SplittingView(splittingVec.begin(), splittingVec.end()));
            
        }
        
    };
    
};

template<typename DataType, typename SplittingVector>
struct parallel_angular_scheme{
    using OffSetType = typename SplittingVector::value_type;
    using SplittingView = typename DefaultDataView<SplittingVector>::ViewType;
    using SplittingVectors = std::unordered_map<size_t, SplittingVector>;

    using ParallelScheme = std::true_type;
    using SerialScheme = std::false_type;
    using DataView = typename DataSet<DataType>::DataView;

    const DataSet<DataType>& dataSource;
    std::unordered_map<std::size_t, SplittingVector> splittingVectors{};

    AsyncQueue<std::tuple<std::size_t, SplittingVector>> generatedVectors{};

    parallel_angular_scheme(const DataSet<DataType>& data) : dataSource(data), splittingVectors(){};

    auto operator()(size_t splitIndex, std::pair<size_t, size_t> splittingPoints){

        auto splittingVector = angular_splitting_plane<DataView, OffSetType>(dataSource[splittingPoints.first], dataSource[splittingPoints.second]);

        if constexpr(is_aligned_contiguous_range_v<SplittingVector>){
            
            auto comparisonFunction =
                bind_angular_predicate(std::as_const(dataSource).begin(), SplittingView(splittingVectors[splitIndex]));
            
            generatedVectors.Put({splitIndex, std::move(splittingVector)});
            return comparisonFunction;

            
        } else {

            auto comparisonFunction =
                bind_angular_predicate(std::as_const(dataSource).begin(), SplittingView(splittingVectors[splitIndex].begin(), splittingVectors[splitIndex].end()));
            
            generatedVectors.Put({splitIndex, std::move(splittingVector)});
            return comparisonFunction;
        }

        
    };

    auto operator()(size_t splitIndex, TransformTag) const {
        const SplittingVector& splittingVec = splittingVectors.at(splitIndex);

        if constexpr(is_aligned_contiguous_range_v<SplittingVector>){
            
            return bind_angular_predicate(std::as_const(dataSource).begin(), SplittingView(splittingVec));
            
        } else {

            return bind_angular_predicate(std::as_const(dataSource).begin(), SplittingView(splittingVec.begin(), splittingVec.end()));
            
        }
        
    };

    void ConsumeNewVectors(){
        std::list<std::tuple<size_t, SplittingVector>> newVecs = generatedVectors.TryTakeAll();
        for (auto& newVec: newVecs){
            splittingVectors[std::get<0>(newVec)] = std::move(std::get<1>(newVec));
        }
    }

    template<typename Predicate>
    void ConsumeNewVectors(Predicate&& pred){
        if (!pred(generatedVectors.GetCount())) return;
        std::list<std::tuple<size_t, SplittingVector>> newVecs = generatedVectors.TryTakeAll();
        for (auto& newVec: newVecs){
            splittingVectors[std::get<0>(newVec)] = std::move(std::get<1>(newVec));
        }
    }
    
};



template<splitting_scheme SchemeType, typename DataType, typename SplittingVector>
struct serial_splitting_scheme;

template<typename DataType, typename SplittingVector>
struct serial_splitting_scheme<splitting_scheme::euclidean, DataType, SplittingVector>{
    using type = euclidean_scheme<DataType, SplittingVector>;
};

template<typename DataType, typename SplittingVector>
struct serial_splitting_scheme<splitting_scheme::angular, DataType, SplittingVector>{
    using type = angular_scheme<DataType, SplittingVector>;
};

template<splitting_scheme SchemeType, typename DataType, typename SplittingVector>
using pick_serial_scheme = typename serial_splitting_scheme<SchemeType, DataType, SplittingVector>::type;


template<splitting_scheme SchemeType, typename DataType, typename SplittingVector>
struct parallel_splitting_scheme;

template<typename DataType, typename SplittingVector>
struct parallel_splitting_scheme<splitting_scheme::euclidean, DataType, SplittingVector>{
    using type = parallel_euclidean_scheme<DataType, SplittingVector>;
};

template<typename DataType, typename SplittingVector>
struct parallel_splitting_scheme<splitting_scheme::angular, DataType, SplittingVector>{
    using type = parallel_angular_scheme<DataType, SplittingVector>;
};

template<splitting_scheme SchemeType, typename DataType, typename SplittingVector>
using pick_parallel_scheme = typename parallel_splitting_scheme<SchemeType, DataType, SplittingVector>::type;

template<splitting_scheme SchemeType, typename DataType, typename SplittingVector>
struct borrowed_splitting_scheme;

template<typename DataType, typename SplittingVector>
struct borrowed_splitting_scheme<splitting_scheme::euclidean, DataType, SplittingVector>{
    template<typename... Args>
    auto bind(Args&&... args){
        return borrowed_euclidean(std::forward<Args>(args)...);
    }
};


}
#endif //RPT_SPLITTINGSCHEME_HPP