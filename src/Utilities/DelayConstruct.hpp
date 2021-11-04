/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_DELAYCONSTRUCT_HPP
#define NND_DELAYCONSTRUCT_HPP

#include <type_traits>
#include <concepts>

namespace nnd{


namespace internal{

template<typename DelayType, typename Functor>
struct DelayConstructHelper{
    
    Functor&& func;
    
    

    operator DelayType() noexcept(noexcept(func())){
        return func();
    }
};

}

template<typename DelayType, std::invocable<> Functor>
    requires std::is_constructible_v<DelayType, std::invoke_result_t<Functor>>
auto DelayConstruct(Functor&& func) noexcept {
    return internal::DelayConstructHelper<DelayType, Functor>{std::forward<Functor>(func)};
}

template<typename DelayType, typename... Ts>
    requires std::is_constructible_v<DelayType, Ts...>
auto DelayConstruct(Ts&&... ts) noexcept {
    auto constructor = [&](){
        return DelayType(std::forward<Ts>(ts)...);
    };
    return DelayConstruct<DelayType>(constructor);
}
}

#endif