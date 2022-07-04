#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include <filesystem>

#include "ann/AlignedMemory/ContainerHelpers.hpp"



namespace ann{

template<typename FirstType, typename SecondType>
struct layout_order{
    using first = std::conditional_t<alignof(FirstType) >= alignof(SecondType),
                                     FirstType,
                                     SecondType>;
    using second = std::conditional_t<alignof(FirstType) >= alignof(SecondType),
                                      SecondType,
                                      FirstType>;

};

template<typename Allocator>
struct bind_allocator{
    using allocator_type = Allocator;
    using alloc_traits = std::allocator_traits<Allocator>;

    private:
    template<typename Pointer>
    using element = typename std::pointer_traits<Pointer>::element_type;
    public:

    template<typename Pointer, typename... Types>
    static constexpr bool noexcept_construct = detect_noexcept_construct<element<Pointer>, allocator_type, Types...>();

    inline static constexpr auto construct =
        []<typename Pointer, typename... Types>(allocator_type& alloc, Pointer ptr, Types&&... args) noexcept(noexcept_construct<Pointer, Types...>) requires constructible_by_alloc<element<Pointer>, allocator_type, Types...> {
        alloc_traits::construct(alloc, ptr, std::forward<Types>(args)...);
    };

    inline static constexpr auto bind_construct = []<typename... Types>(allocator_type& alloc, Types&&... args) noexcept{
        return [&] <typename Pointer> (Pointer ptr) noexcept(noexcept_construct<Pointer, Types...>){
            alloc_traits::construct(alloc, ptr, std::forward<Types>(args)...);
        };
    };
    
    
    inline static constexpr auto bind_copy_construct = [](allocator_type & alloc) noexcept -> auto {
        return [&alloc]<typename Pointer, typename IterOther>(Pointer to, IterOther&& from) noexcept(noexcept_construct<std::iter_reference_t<IterOther>>)
            requires constructible_by_alloc<element<Pointer>, allocator_type, const element<Pointer>&> { 
                construct(alloc, to, *from); 
        };
    };

    inline static constexpr auto bind_move_construct = [](allocator_type & alloc) noexcept -> auto {
        return [&alloc](pointer to, auto&& from) noexcept(noexcept_construct<value_type&&>) 
            requires constructible_by_alloc<element<Pointer>, allocator_type, element<Pointer>&&> { 
                construct(alloc, to, std::ranges::iter_move(from)); 
        };
    };

    template<typename Range>
    static constexpr auto bind_forward_construct = [](allocator_type & alloc) noexcept -> auto {
        return [&alloc] <typename Pointer> (Pointer to, auto&& from)
                    noexcept(noexcept_construct<range_forward_reference<Range>>) 
                    requires constructible_by_alloc<element<Pointer>, allocator_type, range_forward_reference<Range>> { 
            construct(alloc, to, iter_forward_like<Range>(from)); 
        };
    };
    
};


template<typename FirstType, typename SecondType, typename Allocator = std::allocator<std::pair<FirstType, SecondType>>>
struct dual_vector{
    using value_type = std::pair<FirstType, SecondType>;
    using reference = std::pair<FirstType&, SecondType&>;
    using const_reference =  std::pair<const FirstType&, const SecondType&>;
    using allocator_type = Allocator;

    private:
    using alloc = typename std::allocator_traits<allocator_type>::template rebind_alloc<std::byte>;
    using alloc_traits = std::allocator_traits<alloc>;
    using pointer = typename alloc_traits::pointer;

    using layout = layout_order<FirstType, SecondType>;
    using first = typename layout_order::first;
    using alloc_first = typename std::allocator_traits<allocator_type>::template rebind_alloc<first>;
    using alloc_traits_first = std::allocator_traits<alloc_first>;
    using pointer_first = typename alloc_traits_first::pointer;

    using second = typename layout_order::second;
    using alloc_second = typename std::allocator_traits<allocator_type>::template rebind_alloc<second>;
    using alloc_traits_second = std::allocator_traits<alloc_second>;
    using pointer_second = typename alloc_traits_second::pointer;

    using bind = bind_allocator<alloc>;

    static constexpr std::size_t num_arrays = 2;

    static constexpr sum_of_sizes = sizeof(FirstType) + sizeof(SecondType);

    static constexpr adjust_alloc_size = [](std::size_t size){ return size * sum_of_sizes/sizeof(typename layout::first) + 1; };

    static constexpr auto allocate = [](alloc_first& allocator, std::size_t size)->pointer_first{
        std::size_t array_elements = adjust_alloc_size(size);
        auto array_ptr = alloc_traits_first::allocate(allocator, array_elements);
        // In the context of language rules, we really need an array of bytes, but the allocator for the first type
        // returns a pointer to an array of first. We need to use the allocator for the first type to get alignment
        // for the overall buffer correct.
        new ((void*) std::to_address(array_ptr)) std::byte[array_elements * sizeof(FirstType)];
        return array_ptr;
    };

    static constexpr auto deallocate = [](alloc_first& allocator, pointer_first buffer, std::size_t size)->void{
        alloc_traits_first::deallocate(allocator, buffer, adjust_alloc_size(size));
    };

    public:

    constexpr dual_vector() = default;

    constexpr dual_vector(std::size_t size_in, const allocator_type& alloc = {}): 
        allocator{alloc},
        buffer{allocate(alloc, size_in)},
        capacity{size_in},
        size{size_in} {
            initalize(bind::bind_construct(allocator));
    }

    constexpr dual_vector(std::size_t size_in, std::convertible_to<value_type> auto&& pair_like, const allocator_type& alloc = {}): 
        allocator{alloc},
        buffer{allocate(alloc, size_in)},
        capacity{size_in},
        size{size_in} {
            const auto& [first, second] = pair_like;
            initalize(bind::bind_construct(allocator, first), bind::bind_construct(allocator, second));
    }

    constexpr ~dual_vector(){
        if(buffer != nullptr){
            deallocate(allocator, buffer, size);
        }
    }

    private:
    /*
    template<std::invocable<pointer> Functor>
        requires(std::invocable<Functor, pointer_first> 
                 and std::invocable<Functor, pointer_second>
                 and noexcept(initializer(std::declval<pointer_first>()) 
                              and initializer(std::declval<pointer_second>())))
    void initalize(Functor&& initalizer) noexcept {



        auto first_current = begin<0>();
        auto second_current = begin<1>();

        
        for (; first_current != end<0>(); ++first_current) {
            initalizer(first_current);
        }
        for (; second_current != end<1>(); ++second_current) {
            initalizer(second_current);
        }
    }
    */
    
    template<typename Functor, typename Functor>
        requires (std::invocable<Functor, pointer_first> and std::invocable<Functor, pointer_second>)
    void initalize(Functor&& initializer)   {
        initalize(initializer, initializer);
    }

    template<typename FirstFunctor, typename SecondFunctor>
        requires (std::invocable<FirstFunctor, pointer_first> and std::invocable<SecondFunctor, pointer_second>)
    void initalize(FirstFunctor&& first_initializer, SecondFunctor&& second_initializer)   {


        auto first_begin = begin_first();
        auto first_current = first_begin;
        auto first_end = end_first();
        //auto second_current = begin<1>();

        try {
            for (; first_current != first_end; ++first_current) {
                first_initializer(first_current);
            }
            auto second_begin = begin_second();
            auto second_current = second_begin;
            auto second_end = end_second();
            try{
                for (; second_current != second_end; ++second_current) {
                    second_initializer(second_current);
                }
            } catch(...){
                for (; second_current != second_begin; second_current -= 1) {
                    alloc_traits_first::destroy(allocator, second_current - 1);
                }
                throw;
            }
        } catch (...) {
            for (; first_current != first_begin; first_current -= 1) {
                alloc_traits_first::destroy(allocator, first_current - 1);
            }
            deallocate(allocator, buffer, size);
            buffer = nullptr;
            size = 0;
            throw;
        }
    }
    /*
    template<std::size_t index, Functor>
    void initalize_array(Functor&& initalizer) {
        static_assert(index < num_arrays);
        auto begin = begin_internal<index>();
        auto current = begin;
        auto end = end_internal<index>();
        try {
            for (; current != end<index>(); ++current) {
                initalizer(current);
            }
            if constexpr (index+1<num_arrays){
                initalize_array(std::forward<Functor>(initalizer));
            }
        } catch (...) {
            for (; current != begin<index>(); current -= 1) {
                alloc_traits_first::destroy(allocator, current - 1);
            }
        }
    }
    */
    pointer_first begin_first(){
        return buffer;
    }

    pointer_first end_first(){
        return buffer + size;
    }

    pointer_second begin_second(){
        first* raw_ptr = std::to_address(buffer);
        return std::launder((second *) raw_ptr + capacity)
    }

    pointer_second end_second(){
        return begin_second() + size;
    }
    
    


    

    private:
    [[no_unique_address]] alloc_first allocator;
    pointer_first buffer;
    std::size_t capacity;
    std::size_t size;
};

}



int main(){

}