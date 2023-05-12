/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef INTEGRATIONTESTUTILITIES_HPP
#define INTEGRATIONTESTUTILITIES_HPP

#include <filesystem>
#include <utility>

#include "NND/FunctorErasure.hpp"
#include "NND/Index.hpp"
#include "NND/MetricHelpers.hpp"
#include "NND/Search.hpp"

namespace nnd::test{

template<typename Metric, typename Distance>
struct dataset_properties{
    using distance_type = Distance;
    using metric_type = Metric;
    [[no_unique_address]] Metric metric;
    hyper_parameters index_params;
    std::size_t vector_length;
    std::size_t number_of_vectors;
};

struct truth_file{
    std::filesystem::path file;
    std::size_t neighbors;
    std::size_t entries;
};

template<typename Metric, typename Distance>
struct index_build_options{
    using distance_type = Distance;
    using metric_type = Metric;
    std::string dataset_name;
    dataset_properties<Metric, Distance> dataset_props;
    std::filesystem::path dataset_file;
    std::filesystem::path index_location;
    std::filesystem::path testset_file;
    truth_file ground_truth;
};

enum struct execution{
    do_serial,
    do_parallel
};

template<execution execution_strategy, typename DatasetProps>
auto run_test_build(
    const DatasetProps& dataset_props,
    const std::filesystem::path& dataset_file
    )->std::pair<
        std::chrono::nanoseconds,
        nnd::index<typename DatasetProps::distance_type>
    >{
    
    using distance_type = typename DatasetProps::distance_type;
    DataSet<distance_type> training_data(dataset_file, dataset_props.vector_length, dataset_props.number_of_vectors);
    
    std::chrono::time_point<std::chrono::steady_clock> build_start = std::chrono::steady_clock::now();
    
    nnd::index built_index = [&]{ 
        switch (execution_strategy){
            case execution::do_serial:
                return build_index(training_data, dataset_props.metric, dataset_props.index_params);
            case execution::do_parallel:
                return build_index(training_data, dataset_props.metric, std::thread::hardware_concurrency(), dataset_props.index_params);
        }                         
    }();
    std::chrono::time_point<std::chrono::steady_clock> build_end = std::chrono::steady_clock::now();
    
    return {build_end - build_start, std::move(built_index)};
}


// num_significant_neighbors <= std::size(ground_truth[i]) for all i < std::size(ground_truth)
float compute_recall(auto&& results, auto&& ground_truth, auto&& num_significant_neighbors){
    std::size_t correct_neighbors = 0;
    //std::vector<std::size_t> correct_neighbors_per_index(results.size());
    for (size_t i = 0; i < std::size(results); ++i) {
        const auto& result = results[i];
        auto begin_correct = std::begin(ground_truth[i]);
        auto end_correct = begin_correct + num_significant_neighbors;
        for (size_t j = 0; j < std::size(result); ++j) {
            const auto& neighbor = result[j];
            auto find_result = std::find(begin_correct, end_correct, neighbor);
            correct_neighbors += (find_result != end_correct);
        }
        
    }
    
    return correct_neighbors / double(num_significant_neighbors * std::size(ground_truth));
}

template<typename Metric, typename DistanceType>
void search_prep(index<DistanceType>& index, const nnd::search_parameters& search_params, const nnd::DataSet<DistanceType>& test_set){

    for (auto& context : index.query_contexts) {
        context.querySearchDepth = index.search_params.search_depth;
        context.querySize = index.search_params.search_neighbors;
    }
}

template<typename Metric, typename DistanceType>
ann::dual_vector<std::chrono::nanoseconds, float> test_search(const nnd::index<DistanceType>& index, const nnd::search_parameters& search_params, const nnd::DataSet<DistanceType>& test_set, auto&& ground_truth){
    
    
    ann::dual_vector<std::chrono::nanoseconds, float> time_and_recall(10);

    for (size_t i = 0; i < 10; i += 1) {
        std::chrono::time_point<std::chrono::steady_clock> runStart = std::chrono::steady_clock::now();

        auto results = nnd::search<Metric>(test_set, index, std::thread::hardware_concurrency());
                        //: nnd::search(test_set, index);
        

        std::chrono::time_point<std::chrono::steady_clock> runEnd = std::chrono::steady_clock::now();
        std::chrono::nanoseconds runTime = runEnd - runStart;
        
        float recall = compute_recall(results, ground_truth, search_params.search_neighbors);

        time_and_recall[i] = std::pair{runTime, recall};
    }
    
    return time_and_recall;
    
}

}

#endif