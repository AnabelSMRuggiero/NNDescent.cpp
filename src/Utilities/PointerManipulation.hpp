/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_POINTERMANIPULATION_HPP
#define NND_POINTERMANIPULATION_HPP

#include <type_traits>

namespace nnd{
    //Helper for casting between char/byte and another type
    template <typename RetType, typename ArgType>
    std::remove_pointer_t<RetType>* PtrCast(ArgType* ptrToCast){
        // allows either PtrCast<char> or PtrCast<char*> to work equivalently
        // PtrCast<Type**> should still work properly, but why
        using RetPtr = std::remove_pointer_t<RetType>*;
        if constexpr (std::is_const_v<RetType> && std::is_volatile_v<RetType>){
            return static_cast<RetPtr>(static_cast<const volatile void*>(ptrToCast));
        } else if (std::is_const_v<RetType>){
            return static_cast<RetPtr>(static_cast<const void*>(ptrToCast));
        } else if (std::is_volatile_v<RetType>){
            return static_cast<RetPtr>(static_cast<volatile void*>(ptrToCast));
        } else{
            return static_cast<RetPtr>(static_cast<void*>(ptrToCast));
        }
    }
}

#endif
