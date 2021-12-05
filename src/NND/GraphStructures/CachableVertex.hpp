/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_VERTEXCACHE_HPP
#define NND_VERTEXCACHE_HPP


#include <optional>
#include <type_traits>
#include <vector>
#include <concepts>

#include "GraphVertex.hpp"
#include "../Type.hpp"

namespace nnd{

template<typename ResourceType, typename Handler>
struct UniqueResource{

    std::optional<ResourceType> resource = std::nullopt;
    [[no_unique_address]] Handler resourceHandler{};

    UniqueResource() = default;

    UniqueResource(ResourceType&& resource): resource(std::move(resource)) {}

    UniqueResource(ResourceType&& resource, Handler& handler): resource(std::move(resource)), resourceHandler(handler) {}

    UniqueResource(ResourceType&& resource, Handler&& handler): resource(std::move(resource)), resourceHandler(std::move(handler)) {}

    UniqueResource(const UniqueResource&) = delete;

    UniqueResource& operator=(const UniqueResource&) = delete;

    UniqueResource(UniqueResource&& other) requires std::is_move_constructible_v<Handler>:
        resource{std::move(other.resource)}, resourceHandler{std::move(other.resourceHandler)} {
            other.resource = std::nullopt;
    }

    UniqueResource(UniqueResource&& other) 
        requires (std::is_copy_constructible_v<Handler> && !std::is_move_constructible_v<Handler>):
        resource{std::move(other.resource)}, resourceHandler{other.resourceHandler} {
            other.resource = std::nullopt;
    }

    UniqueResource(UniqueResource&& other) 
        requires (!std::is_copy_constructible_v<Handler> && !std::is_move_constructible_v<Handler>):
        resource{std::move(other.resource)}, resourceHandler{} {
            other.resource = std::nullopt;
    }

    UniqueResource& operator=(UniqueResource&& other){
        Release();
        resource.emplace(std::move(other.resource));
        if constexpr (std::is_move_assignable_v<Handler>){
            resourceHandler = std::move(other.resourceHandler);
        } else if constexpr (std::is_copy_assignable_v<Handler>){
            resourceHandler = other.resourceHandler;
        }
    }

    ~UniqueResource(){
        Release();
    }

    void Release(){
        if (resource){
            resourceHandler(std::move(*resource));
        }
    }

    ResourceType* operator->(){
        return resource.operator->();
    }

    ResourceType& operator*(){
        return *resource;
    }

};

template<typename ValueType>
struct DefaultInitalizer{

    ValueType operator()(){
        return ValueType{};
    }

};

struct NullReinitalizer{

    auto&& operator()(auto&& obj){ return std::move(obj);};

};

template<typename Cache>
struct CacheHandler{
    using ResourceType = typename Cache::CachedType;
    Cache& source;

    void operator()(ResourceType&& resourceToReturn){
        source.Put(std::move(resourceToReturn));
    }

};
/*
template<typename Cache, typename Reinializer = NullReinitalizer>
struct CacherGenerator{
    using Derived = typename Cache;
    using ResourceType = typename Cache::CachedType;

    [[no_unique_address]] Reinializer reinit{};
    //Cache& source;

    auto Handler(){
        auto handler = [&](ResourceType&& resourceToReturn){
            reinit(resourceToReturn);
            static_cast<Derived&>(*this).Put(std::move(resourceToReturn));
        };
        return handler;
    }

};
*/
template<typename Cache>
struct CacherGenerator{
    using Derived = typename Cache;
    using ResourceType = typename Cache::CachedType;

    //[[no_unique_address]] Reinializer reinit{};
    //Cache& source;

    auto Handler(auto reinit){
        auto handler = [&, reinit](ResourceType&& resourceToReturn){
            static_cast<Derived&>(*this).Put(std::move(reinit(resourceToReturn)));
        };
        return handler;
    }

};


template<std::movable ValueType, template<typename> typename HandlerGenerator, typename Reinitalizer, typename Initalizer>
struct BasicCache : public HandlerGenerator<BasicCache>{
    using HandlerGenerator<BasicCache>::Handler();
    using value_type = ValueType;
    using CachedType = ValueType;
    using Cachable = UniqueResource<ValueType, decltype(Handler(std::declval<Reinitalizer>()))>;
    
    std::vector<ValueType> cache;
    [[no_unique_address]] Reinitalizer reinit{};
    [[no_unique_address]] Initalizer init{};

    //Cache(Reinitalizer& reinit, Initalizer& init): reinit{reinit}, init{init} {}

    Cachable Take(){
        if (cache.size() == 0){
            return Cachable(init(), Handler(reinit));
        } else {
            Cachable temp(std::move(cache.back()), Handler(reinit));
            cache.pop_back();
            return std::move(temp);
        }
    }

    void Put(CachedType&& returning){
        cache.push_back(std::move(returning));
        reinit(cache.back());
    }

    void Release(){
        cache.resize(0);
        cache.shrink_to_fit();
    }

};

/*
template<std::movable ValueType, typename Reinitalizer = NullReinitalizer, typename Initalizer = DefaultInitalizer<ValueType>>
struct Cache{
    using value_type = ValueType;
    using CachedType = ValueType;
    using Cachable = UniqueResource<ValueType, CacheHandler<Cache>>;
    
    std::vector<ValueType> cache;
    [[no_unique_address]] Reinitalizer reinit{};
    [[no_unique_address]] Initalizer init{};

    //Cache(Reinitalizer& reinit, Initalizer& init): reinit{reinit}, init{init} {}

    Cachable Take(){
        if (cache.size() == 0){
            return Cachable(init(), CacheHandler<Cache>{*this});
        } else {
            Cachable temp(std::move(cache.back()), CacheHandler<Cache>{*this});
            cache.pop_back();
            return std::move(temp);
        }
    }

    void Put(CachedType&& returning){
        cache.push_back(std::move(returning));
        reinit(cache.back());
    }

    void Release(){
        cache.resize(0);
        cache.shrink_to_fit();
    }

};
*/

template<typename ValueType, typename Reinitalizer, typename Initalizer = DefaultInitalizer<ValueType>>
using LocalCache = BasicCache<ValueType, CacherGenerator, Reinitalizer, Initalizer>;

template<typename DistType>
auto MakeVertexCache(const size_t numNeighbors = 0){
    auto initalizer = [=]()->GraphVertex<DataIndex_t, DistType>{
        return GraphVertex<DataIndex_t, DistType>(numNeighbors);
    };

    auto reinitalizer = [](GraphVertex<DataIndex_t, DistType>& returningVertex){
        returningVertex.resize(0);
    };

    return LocalCache<GraphVertex<DataIndex_t, DistType>, decltype(reinitalizer), decltype(initalizer)>{std::vector<GraphVertex<DataIndex_t, DistType>>(), reinitalizer, initalizer};
};

template<typename Cache>
struct ThreadCacherGenerator;

namespace internal{



}


template<typename Cache>
struct ThreadCacherGenerator{
    using Derived = typename Cache;
    using ResourceType = typename Cache::CachedType;

    //[[no_unique_address]] Reinializer reinit{};
    //Cache& source;

    auto Handler(auto reinit){
        auto handler = [&, reinit](ResourceType&& resourceToReturn){
            static_cast<Derived&>(*this).Put(std::move(reinit(resourceToReturn)));
        };
        return handler;
    }

};

template<typename DistType>
using CachableVertex = decltype(MakeVertexCache<DistType>())::Cachable;

}


#endif