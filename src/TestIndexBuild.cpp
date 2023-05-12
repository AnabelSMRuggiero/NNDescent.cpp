/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#include <chrono>
#include <cstddef>
#include <filesystem>

#include "NND/Index.hpp"
#include "NND/MetricHelpers.hpp"
#include "IntegrationTestUtilities.hpp"

const nnd::test::index_build_options mnist_fashion{
    .dataset_name = "MNIST Fashion", 
    .dataset_props = nnd::test::dataset_properties<nnd::euclidean_metric_pair, float>{
        .metric = {},
        .index_params = {
            .split_params = {
                .split_threshold = 205,
                .child_threshold = 123,
                .max_tree_size = 287,
                .max_retry = std::size_t(-1),
                .max_split_fraction = 0.0f
            },
            .index_params = {
                .block_graph_neighbors = 10,
                .COM_neighbors = 10,
                .nearest_node_neighbors = 15,
                .query_depth = 8
            },
            .search_params = {
                .search_neighbors = 10,
                .search_depth = 6,
                .max_searches_queued = 5,
            }
        },
        .vector_length = 28*28,
        .number_of_vectors = 60'000
    },
    .dataset_file = "./TestData/MNIST-Fashion-Train.bin",
    .testset_file = "./TestData/MNIST-Fashion-Test.bin",
    .ground_truth ={
        .file = "./TestData/MNIST-Fashion-Neighbors.bin",
        .neighbors = 100,
        .entries = 10'000
    }
};


const nnd::test::index_build_options sift{
    .dataset_name = "SIFT",
    .dataset_props = nnd::test::dataset_properties<nnd::euclidean_metric_pair, float>{
            .metric = {},
            .index_params = {
                .split_params = {
                    .split_threshold = 2500,
                    .child_threshold = 1500,
                    .max_tree_size = 3500,
                    .max_retry = 50,
                    .max_split_fraction = 0.0f
                },
                .index_params = {
                    .block_graph_neighbors = 10,
                    .COM_neighbors = 10,
                    .nearest_node_neighbors = 15,
                    .query_depth = 8
                },
                .search_params = {
                    .search_neighbors = 10,
                    .search_depth = 6,
                    .max_searches_queued = 5,
                }
            },
        .vector_length = 128,
        .number_of_vectors = 1'000'000
    },
    .dataset_file = "./TestData/SIFT-Train.bin",
    .testset_file = "./TestData/SIFT-Test.bin",
    .ground_truth ={
        .file = "./TestData/SIFT-Neighbors.bin",
        .neighbors = 100,
        .entries = 10'000
    }
};

const nnd::test::index_build_options ny_times{
    .dataset_name = "NY Times",
    .dataset_props = nnd::test::dataset_properties<nnd::inner_product_pair, float>{
        .metric = {},
        .index_params = {
            .split_params = {
                .split_threshold = 725,
                .child_threshold = 435,
                .max_tree_size = 1160,
                .max_retry = 50,
                .max_split_fraction = 0.4f
            },
            .index_params = {
                .block_graph_neighbors = 12,
                .COM_neighbors = 10,
                .nearest_node_neighbors = 15,
                .query_depth = 8
            },
            .search_params = {
                .search_neighbors = 10,
                .search_depth = 6,
                .max_searches_queued = 5,
            }
        },
        .vector_length = 256,
        .number_of_vectors = 290'000
    },
    .dataset_file = "./TestData/NYTimes-Angular-Train.bin",
    .testset_file = "./TestData/NYTimes-Angular-Test.bin",
    .ground_truth ={
        .file = "./TestData/NYTimes-Angular-Neighbors.bin",
        .neighbors = 100,
        .entries = 10'000
    }
};

template<nnd::test::execution execution_strategy>
constexpr auto build_and_print_runtime = [] (auto&& options){
    auto [run_time, index] = nnd::test::run_test_build<execution_strategy>(options.dataset_props, options.dataset_file);
    std::cout 
        << "Time to build index for "
        << options.dataset_name
        << ": "
        << std::chrono::duration_cast<std::chrono::duration<float>>(run_time)
        << "\n";
};

template<nnd::test::execution execution_strategy>
constexpr auto test_and_print = [](auto&& options){
    using props_type = std::remove_cvref_t<decltype(options)>;
    auto [run_time, index] = nnd::test::run_test_build<execution_strategy>(options.dataset_props, options.dataset_file);
    std::cout 
        << "Time to build index for "
        << options.dataset_name
        << ": "
        << std::chrono::duration_cast<std::chrono::duration<float>>(run_time)
        << "\n";

    nnd::DataSet<typename props_type::distance_type> test_data(options.testset_file, options.dataset_props.vector_length, options.dataset_props.number_of_vectors);
    nnd::DataSet<std::uint32_t> test_neighbors(options.ground_truth.file, options.ground_truth.neighbors, options.ground_truth.entries);

    auto results = nnd::test::test_search<typename props_type::metric_type>(index, options.dataset_props.index_params.search_params, test_data, test_neighbors);
    double average_recall = 0.0;
    std::chrono::nanoseconds average_time{0};

    for (const auto& [time, recall] : results){
        average_recall += recall;
        average_time += time;
    }
    average_recall /= std::size(results);
    average_time /= std::size(results);
    std::cout
        << "Number of trials: "
        << std::size(results)
        << "\n"
        << "Average recall: "
        << average_recall
        << "\n"
        << "Average runtime (s):"
        << std::chrono::duration_cast<std::chrono::duration<float>>(average_time)
        << "\n";
};

int main(){
    test_and_print<nnd::test::execution::do_serial>(mnist_fashion);
    test_and_print<nnd::test::execution::do_parallel>(sift);
    test_and_print<nnd::test::execution::do_parallel>(ny_times);
}