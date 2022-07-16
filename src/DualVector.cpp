#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include <filesystem>
#include <vector>
#include <xutility>

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

    inline static constexpr auto bind_destroy = []<typename... Types>(allocator_type& alloc) noexcept{
        return [&] <typename Pointer> (Pointer ptr) noexcept{
            alloc_traits::destroy(alloc, ptr);
        };
    };
    
    
    inline static constexpr auto bind_copy_construct = [](allocator_type & alloc) noexcept -> auto {
        return [&alloc]<typename Pointer, typename IterOther>(Pointer to, IterOther&& from) noexcept(noexcept_construct<std::iter_reference_t<IterOther>>)
            requires constructible_by_alloc<element<Pointer>, allocator_type, const element<Pointer>&> { 
                construct(alloc, to, *from); 
        };
    };

    inline static constexpr auto bind_move_construct = [](allocator_type & alloc) noexcept -> auto {
        return [&alloc]<typename Pointer, typename IterOther>(Pointer to, IterOther&& from) noexcept(noexcept_construct<std::iter_rvalue_reference_t<IterOther>>) 
            requires constructible_by_alloc<element<Pointer>, allocator_type, std::iter_rvalue_reference_t<IterOther>> { 
                construct(alloc, to, std::ranges::iter_move(from)); 
        };
    };

    template<typename Range>
    inline static constexpr auto bind_forward_construct = [](allocator_type & alloc) noexcept -> auto {
        return [&alloc] <typename Pointer> (Pointer to, auto&& from)
                    noexcept(noexcept_construct<range_forward_reference<Range>>) 
                    requires constructible_by_alloc<element<Pointer>, allocator_type, range_forward_reference<Range>> { 
            construct(alloc, to, iter_forward_like<Range>(from)); 
        };
    };
    
};
template<typename OtherIter, typename Pointer, typename Alloc, auto Binder>
constexpr bool bound_alloc_operation_check = requires(OtherIter&& iter, Pointer&& pointer, Alloc&& alloc){
    Binder(alloc)(pointer, iter);
};
template<typename OtherIter, typename Pointer, typename Alloc>
concept bound_copyable = //bound_alloc_operation_check<OtherIter, Pointer, Alloc, bind_allocator<Alloc>::bind_copy_construct>;
requires(OtherIter&& iter, Pointer&& pointer, Alloc&& alloc){
    typename bind_allocator<Alloc>;
    bind_allocator<Alloc>::bind_copy_construct(alloc)(pointer, iter);
};
template<std::input_iterator InIter, std::sentinel_for<InIter> InSent, typename OutPtr, typename Alloc>
auto uninitialized_alloc_copy(InIter in_iter, InSent in_sent, OutPtr out_ptr, Alloc& allocator){
    OutPtr out_begin = out_ptr;
    auto bound_copy = bind_allocator<Alloc>::bind_copy_construct(allocator);
    try{
        for (; in_iter != in_sent; ++out_ptr, ++in_iter ){
            bound_copy(out_ptr, in_iter);
        }
    } catch(...){
        auto bound_destroy = bind_allocator<Alloc>::bind_destroy(allocator);
        while( out_ptr!=out_begin){
            bound_destroy(--out_ptr);
        }
        throw;
    }
    return std::ranges::in_out_result<InIter, OutPtr>{in_iter, out_ptr};
};

template<std::input_iterator InIter, std::sentinel_for<InIter> InSent, typename OutPtr, typename Alloc>
    requires std::is_trivially_copyable_v<std::iter_value_t<OutPtr>>
auto uninitialized_alloc_copy(InIter in_iter, InSent in_sent, OutPtr out_ptr, Alloc& allocator){
    std::ranges::copy(in_iter, in_sent, out_ptr);
};

template<std::input_iterator InIter, std::sentinel_for<InIter> InSent, typename OutPtr, typename Alloc>
auto uninitialized_alloc_move(InIter in_iter, InSent in_sent, OutPtr out_ptr, Alloc& allocator){
    OutPtr out_begin = out_ptr;
    auto bound_move = bind_allocator<Alloc>::bind_move_construct(allocator);
    try{
        for (; in_iter != in_sent; ++out_ptr, ++in_iter ){
            bound_move(out_ptr, in_iter);
        }
    } catch(...){
        auto bound_destroy = bind_allocator<Alloc>::bind_destroy(allocator);
        while( out_ptr!=out_begin){
            bound_destroy(--out_ptr);
        }
        throw;
    }
    return std::ranges::in_out_result<InIter, OutPtr>{in_iter, out_ptr};
};

template<typename Pointer, typename Alloc>
auto alloc_destroy(Pointer begin, Pointer end, Alloc& alloc){
    auto bound_destroy = bind_allocator<Alloc>::bind_destroy(alloc);
    for(; begin != end; ++begin){
        bound_destroy(begin);
    }
}

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
    using first = typename layout::first;
    using alloc_first = typename std::allocator_traits<allocator_type>::template rebind_alloc<first>;
    using alloc_traits_first = std::allocator_traits<alloc_first>;
    using pointer_first = typename alloc_traits_first::pointer;
    using const_pointer_first = typename alloc_traits_first::const_pointer;

    using second = typename layout::second;
    using alloc_second = typename std::allocator_traits<allocator_type>::template rebind_alloc<second>;
    using alloc_traits_second = std::allocator_traits<alloc_second>;
    using pointer_second = typename alloc_traits_second::pointer;
    using const_pointer_second = typename alloc_traits_second::const_pointer;

    using bind = bind_allocator<alloc>;

    static constexpr std::size_t num_arrays = 2;

    static constexpr std::size_t sum_of_sizes = sizeof(FirstType) + sizeof(SecondType);

    static constexpr auto adjust_alloc_size = [](std::size_t size){ return size * sum_of_sizes/sizeof(typename layout::first) + 1; };

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
        buffer{allocate(allocator, size_in)},
        capacity{size_in},
        size{size_in} {
            const auto& [first, second] = pair_like;
            initalize(bind::bind_construct(allocator, first), bind::bind_construct(allocator, second));
    }

    constexpr dual_vector(const dual_vector& other):
        allocator{alloc_traits_first::select_on_container_copy_construction(other.allocator)},
        buffer{allocate(allocator, other.size)},
        capacity{other.size},
        size{other.size} {}

    constexpr dual_vector(dual_vector&& other) noexcept:
        allocator{other.allocator},
        buffer{std::exchange(other.buffer, nullptr)},
        capacity{std::exchange(other.capacity, 0)},
        size{std::exchange(other.size, 0)} {}


    constexpr ~dual_vector(){
        if(buffer != nullptr){
            destroy_elements();
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
    
    template<typename Functor>
        requires (std::invocable<Functor&, pointer_first> and std::invocable<Functor&, pointer_second>)
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
            auto second_begin = unclean_begin_second();
            auto second_current = second_begin;
            auto second_end = unclean_end_second();
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

    void destroy_elements(){
        alloc_destroy(begin_first(), end_first(), allocator);
        alloc_destroy(begin_second(), end_second(), allocator);
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

    pointer_second unclean_begin_second(){
        first* raw_ptr = std::to_address(buffer);
        return (second *) (raw_ptr + capacity);
    }

    pointer_second unclean_end_second(){
        return unclean_begin_second() + size;
    }

    pointer_second begin_second(){
        return std::launder(std::to_address(unclean_begin_second()));
    }

    pointer_second end_second(){
        return begin_second() + size;
    }
    
    static constexpr bool alloc_always_equal = alloc_traits::is_always_equal::value;
    static constexpr bool alloc_prop_copyassign = alloc_traits::propagate_on_container_copy_assignment::value && !alloc_always_equal;
    static constexpr bool alloc_prop_moveassign = alloc_traits::propagate_on_container_move_assignment::value && !alloc_always_equal;

    static constexpr bool nonprop_nothrow_element_move =
            !(alloc_always_equal || alloc_prop_moveassign) && std::is_nothrow_move_constructible_v<value_type>;
    template<bool propagate>
        requires propagate
    alloc_first alloc_assign_select(const dual_vector& other){
        return other.allocator;
    }
    template<bool propagate>
        requires (!propagate)
    alloc_first alloc_assign_select(const dual_vector& other){
        return allocator;
    }
    pointer_first new_copy_assign(alloc_first& new_alloc, const dual_vector& other){
        pointer_first new_buffer = allocate(new_alloc, other.size);
        pointer_first first_begin = new_buffer;
        pointer_second second_begin = (second *) (std::to_address(new_buffer) + other.size);
        try{
            std::tie(std::ignore, new_buffer) = uninitialized_alloc_copy(other.begin_first(), other.end_first(), new_buffer, new_alloc);
            uninitialized_alloc_copy(other.begin_second(), other.end_second(), second_begin, new_alloc);
        } catch(...){
            auto bound_destroy = bind::bind_destroy(new_alloc);
            while(new_buffer != first_begin){
                bound_destroy(--new_buffer);
            }
            deallocate(new_alloc, first_begin, other.size);
            throw;
        }
        return first_begin; 
    }
    public:
    dual_vector& operator=(const dual_vector& other) requires (alloc_always_equal || alloc_prop_copyassign) {
        if (capacity < other.size){
            alloc_first new_alloc = alloc_assign_select<alloc_prop_copyassign>(other);
            
            pointer_first new_buffer = new_copy_assign(new_alloc, other);
            destroy_elements();
            deallocate(allocator, buffer, capacity);
            allocator = new_alloc;
            buffer = new_buffer;
            size = other.size;
            capacity = other.size;
        } else {
            alloc_first new_alloc = alloc_assign_select<alloc_prop_copyassign>(other);
        }
        return *this;
    }

    dual_vector& operator=(dual_vector&& other);
    

    private:
    [[no_unique_address]] alloc_first allocator;
    pointer_first buffer;
    std::size_t capacity;
    std::size_t size;
};

}



int main(){

}