/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_UNIQUEFUNCTION_HPP
#define NND_UNIQUEFUNCTION_HPP

#include <memory>
#include <utility>

#include "Type.hpp"

// A move-only, type erased function

namespace nnd{

template<typename>
struct UniqueFunction;

template<typename R, typename... Args>
struct UniqueFunction<R(Args...)>{

    UniqueFunction() = default;

    UniqueFunction(UniqueFunction&&) = default;

    template<IsNot<UniqueFunction> Functor>
    UniqueFunction(Functor&& target): ptrToFunctor(std::make_unique<ConcreteFunction<Functor>>(std::move(target))){};

    template<IsNot<R(Args...)> WrongSignature>
    UniqueFunction(UniqueFunction<WrongSignature>&&) = delete;

    private:
    struct AbstractFunction{
        virtual ~AbstractFunction(){};
        virtual R operator()(Args... args) = 0;
    };

    template<typename Functor>
    struct ConcreteFunction final: AbstractFunction{
        Functor target;

        ConcreteFunction(Functor&& target): target(std::move(target)){};

        ConcreteFunction(ConcreteFunction&& other): target(std::move(other.target)){};

        R operator()(Args... args){
            return target(args...);
        }
    };

    std::unique_ptr<AbstractFunction> ptrToFunctor;


};

}

#endif