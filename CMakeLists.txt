# NNDescent.cpp: Copyright (c) Anabel Ruggiero
# At the time of writting, this code is unreleased and not published under a license.
# As a result, I currently retain all legal rights I am legally entitled to.
# 
# I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
# Please refer to the project repo for any updates regarding liscensing.
# https://github.com/AnabelSMRuggiero/NNDescent.cpp

cmake_minimum_required(VERSION 3.24)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/CMake)

project(nndescent.cpp)

#add_compile_options("-fsanitize=address" "-Zi")
#add_link_options("-fsanitize=address")

add_subdirectory(ext)

add_subdirectory(src)