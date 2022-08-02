# NNDescent.cpp: Copyright (c) Anabel Ruggiero
# At the time of writting, this code is unreleased and not published under a license.
# As a result, I currently retain all legal rights I am legally entitled to.
# 
# I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
# Please refer to the project repo for any updates regarding liscensing.
# https://github.com/AnabelSMRuggiero/NNDescent.cpp

add_subdirectory(ann)

set(
    GRAPH_STRUCTURE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/GraphStructures/CachingFunctor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/GraphStructures/CachableVertex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/GraphStructures/DirectedGraph.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/GraphStructures/Graph.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/GraphStructures/GraphVertex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/GraphStructures/UndirectedGraph.hpp
)

set(
    PARALLEL_ALGORITHM_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/Parallel-Algorithm/FreeFunctions.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/Parallel-Algorithm/GraphComparisonTask.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/Parallel-Algorithm/GraphUpdateTask.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/Parallel-Algorithm/InitJoinTask.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/Parallel-Algorithm/NearestNodesTask.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/Parallel-Algorithm/ParallelizationObjects.hpp
)

set(
    NND_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/BlockwiseAlgorithm.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/FunctorErasure.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/FunctorErasureOld.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/GraphInitialization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/GraphStructures.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/Index.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/MemoryInternals.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/MetaGraph.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/MetricHelpers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/RNG.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/Search.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/SearchContexts.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/SubGraphQuerying.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/Type.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NND/UtilityFunctions.hpp
)

set(
    PARALLELIZATION_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Parallelization/AsyncQueue.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Parallelization/TaskQueuer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Parallelization/ThreadPool.hpp
)

set(
    RP_TREES_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/RPTrees/Forest.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPTrees/SplittingScheme.hpp
)

add_library(nndescent INTERFACE)

target_sources(
    nndescent
    INTERFACE
        ${GRAPH_STRUCTURE_HEADERS}
        ${PARALLEL_ALGORITHM_HEADERS}
        ${NND_HEADERS}
        ${PARALLELIZATION_HEADERS}
        ${RP_TREES_HEADERS}
)

target_include_directories(
    nndescent
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/NND
        ${CMAKE_CURRENT_SOURCE_DIR}/Parallelization
        ${CMAKE_CURRENT_SOURCE_DIR}/RPTrees
)

target_link_libraries(
    nndescent
    INTERFACE anns_numerical_nonsense
)

target_compile_features(nndescent INTERFACE cxx_std_20)

add_executable(
    nndescent_index_build
    nndescent.cpp
)

target_link_libraries(
    nndescent_index_build
    nndescent
    anns_numerical_nonsense
)

# Todo: set up ways to detect instruction set during build
if(MSVC)
    target_compile_options(nndescent_index_build PRIVATE "/arch:AVX2")
endif()