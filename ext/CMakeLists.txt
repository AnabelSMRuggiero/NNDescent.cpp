# NNDescent.cpp: Copyright (c) Anabel Ruggiero
# At the time of writting, this code is unreleased and not published under a license.
# As a result, I currently retain all legal rights I am legally entitled to.
# 
# I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
# Please refer to the project repo for any updates regarding liscensing.
# https://github.com/AnabelSMRuggiero/NNDescent.cpp

include(FetchContent)

FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2
    GIT_TAG v3.1.0
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Catch2
)

FetchContent_MakeAvailable(catch2)

#add_subdirectory(catch2)