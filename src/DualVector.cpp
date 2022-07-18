#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
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

    static constexpr bool reversed = alignof(FirstType) >= alignof(SecondType);

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

template<typename FirstPointer, typename SecondPointer>
struct dual_vector_iterator{
    private:

    using first_pointer = FirstPointer;
    using second_pointer = SecondPointer;

    template<typename Pointer>
    using element = typename std::pointer_traits<Pointer>::element_type;

    using first = element<first_pointer>;
    using second = element<second_pointer>;

    

    public:
    
    using value_type = std::pair<first, second>;
    using difference_type = std::ptrdiff_t;
    using reference = std::pair<first&, second&>;
    using const_reference = std::pair<const first&, const second&>;

    first_pointer first_ptr;
    second_pointer second_ptr;

    public:

    constexpr dual_vector_iterator& operator++(){
        ++first_ptr;
        ++second_ptr;
        return *this;
    }

    constexpr dual_vector_iterator operator++(int){
        dual_vector_iterator copy = *this;
        ++*this;
        return copy;
    }

    constexpr dual_vector_iterator& operator--(){
        --first_ptr;
        --second_ptr;
        return *this;
    }

    constexpr dual_vector_iterator operator--(int){
        dual_vector_iterator copy = *this;
        --*this;
        return copy;
    }

    constexpr dual_vector_iterator operator+(std::ptrdiff_t inc) const {
        dual_vector_iterator copy{first_ptr + inc, second_ptr + inc};
        return copy;
    }

    constexpr dual_vector_iterator operator-(std::ptrdiff_t inc) const {
        dual_vector_iterator copy{first_ptr - inc, second_ptr - inc};
        return copy;
    }

    constexpr std::ptrdiff_t operator-(const dual_vector_iterator& other) const{
        return first_ptr - other.first_ptr;
    }
    
    constexpr bool operator==(const dual_vector_iterator& other) const {
        return first_ptr == other.first_ptr;
    }
    
    constexpr reference operator*() const{
        return reference{*first_ptr, *second_ptr};
    }

    constexpr reference operator[](size_t i) const {
        return *(*this + i);
    }

    constexpr dual_vector_iterator& operator+=(std::ptrdiff_t inc){
        *this = *this + inc;
        return *this;
    }

    constexpr dual_vector_iterator& operator-=(std::ptrdiff_t inc){
        *this = *this - inc;
        return *this;
    }

    constexpr auto operator<=>(const dual_vector_iterator& rhs) const {
        return first_ptr <=> rhs.first_ptr;
    }

};

template<typename FirstPointer, typename SecondPointer>
dual_vector_iterator<FirstPointer, SecondPointer> operator+(std::ptrdiff_t inc, const dual_vector_iterator<FirstPointer, SecondPointer>& iter){
    return iter + inc;
}

template<typename FirstPointer, typename SecondPointer>
dual_vector_iterator<FirstPointer, SecondPointer> operator-(std::ptrdiff_t inc, const dual_vector_iterator<FirstPointer, SecondPointer>& iter){
    return iter - inc;
}


template<typename FirstType, typename SecondType, typename Allocator = std::allocator<std::pair<FirstType, SecondType>>>
struct dual_vector{
    using value_type = std::pair<FirstType, SecondType>;
    using reference = std::pair<FirstType&, SecondType&>;
    using const_reference =  std::pair<const FirstType&, const SecondType&>;
    using allocator_type = Allocator;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    

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

    public:
    using iterator = dual_vector_iterator<pointer_first, pointer_second>;
    using const_iterator = dual_vector_iterator<const_pointer_first, const_pointer_second>;
    private:

    static constexpr std::size_t num_arrays = 2;

    static constexpr std::size_t sum_of_sizes = sizeof(FirstType) + sizeof(SecondType);

    static constexpr auto adjust_alloc_size = [](std::size_t size){ return size * sum_of_sizes/sizeof(typename layout::first) + 1; };

    static constexpr auto allocate = [](alloc_first& allocator, std::size_t size)->pointer_first{
        std::size_t array_elements = adjust_alloc_size(size);
        auto array_ptr = alloc_traits_first::allocate(allocator, array_elements);
        // In the context of language rules, we really need an array of bytes, but the allocator for the first type
        // returns a pointer to an array of first. We need to use the allocator for the first type to get alignment
        // for the overall buffer correct.
        std::byte* buffer_ptr = new ((void*) std::to_address(array_ptr)) std::byte[array_elements * sizeof(FirstType)];
        //first* first_array = new(buffer_ptr) first[]
        return array_ptr;
    };

    static constexpr auto deallocate = [](alloc_first& allocator, pointer_first buffer, std::size_t size)->void{
        alloc_traits_first::deallocate(allocator, buffer, adjust_alloc_size(size));
    };

    template<typename Value, typename... Types>
    inline static constexpr bool noexcept_construct = detect_noexcept_construct<Value, allocator_type, Types...>();

    static pointer_second to_begin_second(pointer_first begin_first, size_type buffer_capacity){
        first* end_of_first = std::to_address(begin_first) + buffer_capacity;
        second (* array_pointer)[] = std::launder((second(*)[]) end_of_first);
        return (second*) *array_pointer;
    }
    public:

    constexpr dual_vector() = default;

    constexpr dual_vector(std::size_t size_in, const allocator_type& alloc = {}): 
        allocator{alloc},
        buffer{allocate(alloc, size_in)},
        buffer_capacity{size_in},
        array_size{size_in} {
            initalize(bind::bind_construct(allocator));
    }

    constexpr dual_vector(std::size_t size_in, std::convertible_to<value_type> auto&& pair_like, const allocator_type& alloc = {}): 
        allocator{alloc},
        buffer{allocate(allocator, size_in)},
        buffer_capacity{size_in},
        array_size{size_in} {
            const auto& [first, second] = pair_like;
            if constexpr (layout::reversed){
                initalize(bind::bind_construct(allocator, second), bind::bind_construct(allocator, first));
            }else{
                initalize(bind::bind_construct(allocator, first), bind::bind_construct(allocator, second));
            }
    }

    constexpr dual_vector(const dual_vector& other):
        allocator{alloc_traits_first::select_on_container_copy_construction(other.allocator)},
        buffer{allocate(allocator, other.array_size)},
        buffer_capacity{other.array_size},
        array_size{other.array_size} {}

    constexpr dual_vector(dual_vector&& other) noexcept:
        allocator{other.allocator},
        buffer{std::exchange(other.buffer, nullptr)},
        buffer_capacity{std::exchange(other.buffer_capacity, 0)},
        array_size{std::exchange(other.array_size, 0)} {}


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
            deallocate(allocator, buffer, buffer_capacity);
            buffer = nullptr;
            array_size = 0;
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

    const_pointer_first begin_first() const {
        return buffer;
    }

    pointer_first end_first(){
        return buffer + array_size;
    }

    const_pointer_first end_first() const {
        return buffer + array_size;
    }

    pointer_second begin_second(){
        return to_begin_second(buffer, buffer_capacity);
    }

    const_pointer_second begin_second() const {
        return to_begin_second(buffer, buffer_capacity);
    }

    pointer_second end_second(){
        return begin_second() + array_size;
    }
    
    const_pointer_second end_second() const {
        return begin_second() + array_size;
    }

    static constexpr bool alloc_always_equal = alloc_traits::is_always_equal::value;
    static constexpr bool alloc_prop_copyassign = alloc_traits::propagate_on_container_copy_assignment::value && !alloc_always_equal;
    static constexpr bool alloc_prop_moveassign = alloc_traits::propagate_on_container_move_assignment::value && !alloc_always_equal;

    static constexpr bool nonprop_nothrow_element_move =
            !(alloc_always_equal || alloc_prop_moveassign) && std::is_nothrow_move_constructible_v<value_type>;

    static constexpr bool copy_over = std::is_nothrow_copy_constructible_v<first> && std::is_nothrow_copy_constructible_v<second>
                                    && std::is_nothrow_copy_assignable_v<first> && std::is_nothrow_assignable_v<second>;
    template<typename Type>
    static constexpr bool element_no_throw_move = std::uses_allocator_v<Type, allocator_type> ? std::is_nothrow_constructible_v<Type, Type&&, const allocator_type&>
                                                                                              : std::is_nothrow_move_constructible_v<Type> 
                                                  && std::is_nothrow_move_assignable_v<Type>;

    static constexpr bool move_over = element_no_throw_move<first> && element_no_throw_move<second>;
                                                //&& std::is_nothrow_move_constructible_v<second> && std::is_nothrow_move_assignable_v<second>;
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
        pointer_first new_buffer = allocate(new_alloc, other.array_size);
        pointer_first first_begin = new_buffer;
        pointer_second second_begin = (second *) (std::to_address(new_buffer) + other.array_size);
        try{
            std::tie(std::ignore, new_buffer) = uninitialized_alloc_copy(other.begin_first(), other.end_first(), new_buffer, new_alloc);
            uninitialized_alloc_copy(other.begin_second(), other.end_second(), second_begin, new_alloc);
        } catch(...){
            auto bound_destroy = bind::bind_destroy(new_alloc);
            while(new_buffer != first_begin){
                bound_destroy(--new_buffer);
            }
            deallocate(new_alloc, first_begin, other.array_size);
            throw;
        }
        return first_begin; 
    }

    void do_copy_assign(alloc_first& new_alloc, const dual_vector& other){
        pointer_first new_buffer = new_copy_assign(new_alloc, other);
        destroy_elements();
        deallocate(allocator, buffer, buffer_capacity);
        allocator = new_alloc;
        buffer = new_buffer;
        array_size = other.array_size;
        buffer_capacity = other.array_size;
    }

    template<typename Pointer>
    void copy_over_array(Pointer from_begin, Pointer from_end, Pointer to_begin, Pointer to_end) requires std::is_trivially_copyable_v<std::iter_value_t<Pointer>>{
        std::memcpy(to_begin, from_begin, from_end-from_begin);
    }

    template<typename Pointer>
    void copy_over_array(Pointer from_begin, Pointer from_end, Pointer to_begin, Pointer to_end) {
        std::size_t from_size = from_end - from_begin;
        std::size_t to_size = to_end - to_begin;
        if (from_size > to_size){
            std::copy(from_begin, from_begin + to_size, to_begin);
            std::uninitialized_copy(from_begin + to_size, from_end, to_end);

        } else {
            std::copy(from_begin, from_begin + from_size, to_begin);
            std::destroy(to_begin+from_size, to_end);
        }
    }

    void copy_over_buffer(const dual_vector& other){
        copy_over_array(other.begin_first(), other.end_first(), begin_first(), end_first());
        copy_over_array(other.begin_second(), other.end_second(), begin_second(), end_second());
    }

    template<typename Pointer>
    void move_over_array(Pointer from_begin, Pointer from_end, Pointer to_begin, Pointer to_end) requires std::is_trivially_copyable_v<std::iter_value_t<Pointer>>{
        std::memcpy(to_begin, from_begin, from_end-from_begin);
    }

    template<typename Pointer>
    void move_over_array(Pointer from_begin, Pointer from_end, Pointer to_begin, Pointer to_end) {
        std::size_t from_size = from_end - from_begin;
        std::size_t to_size = to_end - to_begin;
        if (from_size > to_size){
            std::move(from_begin, from_begin + to_size, to_begin);
            uninitialized_alloc_move(from_begin + to_size, from_end, to_end, allocator);

        } else {
            std::move(from_begin, from_begin + from_size, to_begin);
            std::destroy(to_begin+from_size, to_end);
        }
    }

    void move_over_buffer(dual_vector& other){
        move_over_array(other.begin_first(), other.end_first(), begin_first(), end_first());
        move_over_array(other.begin_second(), other.end_second(), begin_second(), end_second());
    }

    template<typename Type>
    static constexpr bool copy_relocate = !std::is_nothrow_move_constructible_v<Type> && std::is_copy_constructible_v<Type>;

    void relocate(size_type new_capacity){
        pointer_first new_buffer = allocate(allocator, new_capacity);
        pointer_first constructed_first = new_buffer;
        pointer_second constructed_second = to_begin_second(new_buffer, new_capacity);
        try{
            if constexpr (copy_relocate<first>){
                constructed_first = std::uninitialized_copy(begin_first(), end_first(), new_buffer);
            }
            if constexpr (copy_relocate<second>){
                constructed_second = std::uninitialized_copy(begin_second(), end_second(), to_begin_second(new_buffer, new_capacity));
            }
            if constexpr (!copy_relocate<first>){
                constructed_first = std::uninitialized_move(begin_first(), end_first(), new_buffer);
            }
            if constexpr (!copy_relocate<second>){
                constructed_second = std::uninitialized_move(begin_second(), end_second(), to_begin_second(new_buffer, new_capacity));
            }
        }
        catch(...){
            alloc_destroy(new_buffer, constructed_first, allocator);
            alloc_destroy(to_second_begin(new_buffer, new_capacity), constructed_second, allocator);
            deallocate(allocator, new_buffer, new_capacity);
            throw;
        }
        destroy_elements();
        deallocate(allocator, buffer, buffer_capacity);
        buffer = new_buffer;
        buffer_capacity = new_capacity;
    }
    
    template<typename ValueType, typename OtherType>
    static constexpr bool move_insert = std::is_rvalue_reference_v<OtherType&&> && noexcept_construct<ValueType, OtherType&&>;

    template<std::convertible_to<first> FirstArg, std::convertible_to<second> SecondArg>
    void insert_at(pointer_first first_location, FirstArg&& first_arg, pointer_second second_location, SecondArg&& second_arg){
        auto copy_construct = bind::bind_copy_construct(allocator);
        if constexpr (!move_insert<first, FirstArg>){
            copy_construct(first_location, first_arg);
        }
        if constexpr (!move_insert<second, SecondArg>){
            try{
                copy_construct(second_location, second_arg);
            } catch(...){
                if constexpr(!move_insert<first, FirstArg>){
                    alloc_traits_first::destroy(allocator, first_location);
                }
                throw;
            }
        }
        auto forward_construct = bind::bind_foward_construct(allocator);
        if constexpr (move_insert<first, FirstArg>){
            forward_construct(first_location, std::forward<FirstArg>(first_arg));
        }
        if constexpr (move_insert<second, SecondArg>){
            forward_construct(second_location, std::forward<SecondArg>(second_arg));
        }
    }

    template<std::convertible_to<first> FirstArg, std::convertible_to<second> SecondArg>
    void insert_relocate(size_type new_capacity, size_type insert_index, FirstArg&& first_arg, SecondArg&& second_arg){
        pointer_first new_buffer = allocate(allocator, new_capacity);
        pointer_first constructed_first = new_buffer;
        pointer_second constructed_second = to_begin_second(new_buffer, new_capacity);
        bool first_inserted = false;
        bool second_inserted = false;
        try{
            if constexpr (copy_relocate<first>){
                constructed_first = std::uninitialized_copy(begin_first(), begin_first() + insert_index, new_buffer);

                constructed_first = std::uninitialized_copy(begin_first() + insert_index, end_first(), new_buffer + insert_index + 1);
            }
            if constexpr (copy_relocate<second>){
                constructed_second = std::uninitialized_copy(begin_second(), begin_second() + insert_index, to_begin_second(new_buffer, new_capacity));

                constructed_second = std::uninitialized_copy(begin_second() + insert_index, end_second(), to_begin_second(new_buffer, new_capacity) + insert_index + 1);
            }
            insert_at(new_buffer + insert_index, std::forward<FirstArg>(first_arg), to_begin_second(new_buffer, new_capacity) + insert_index, std::forward<SecondArg>(second_arg));
            if constexpr (!copy_relocate<first>){
                constructed_first = std::uninitialized_move(begin_first(), end_first(), new_buffer);
            }
            if constexpr (!copy_relocate<second>){
                constructed_second = std::uninitialized_move(begin_first(), end_first(), to_begin_second(new_buffer, new_capacity));
            }
        }
        catch(...){
            if (constructed_first-new_buffer > insert_index){
                alloc_destroy(new_buffer + insert_index + 1, constructed_first, allocator);
                constructed_first = new_buffer + insert_index;
            }
            alloc_destroy(new_buffer, constructed_first, allocator);
            auto new_begin_second = to_second_begin(new_buffer, new_capacity);
            if (constructed_second-new_begin_second > insert_index){
                alloc_destroy(new_begin_second + insert_index + 1, constructed_second, allocator);
                constructed_second = new_begin_second + insert_index;
            }
            alloc_destroy(new_begin_second, constructed_second, allocator);
            
            //alloc_destroy(new_buffer, constructed_first, allocator);
            //alloc_destroy(to_second_begin(new_buffer, new_capacity), constructed_second, allocator);
            deallocate(allocator, new_buffer, new_capacity);
            throw;
        }
        destroy_elements();
        deallocate(allocator, buffer, buffer_capacity);
        buffer = new_buffer;
        buffer_capacity = new_capacity;
    }

    public:
    dual_vector& operator=(const dual_vector& other) {
        if (this != &other){
            alloc_first new_alloc = alloc_assign_select<alloc_prop_copyassign>(other);
            do_copy_assign(new_alloc, other);
        }
        return *this;
    }

    dual_vector& operator=(const dual_vector& other) requires copy_over {
        if(this != &other){
            if (buffer_capacity < other.array_size){
                alloc_first new_alloc = alloc_assign_select<alloc_prop_copyassign>(other);
                do_copy_assign(new_alloc, other);
            } else {
                alloc_first new_alloc = alloc_assign_select<alloc_prop_copyassign>(other);
                if (new_alloc == allocator){
                    copy_over_buffer(other);
                    array_size = other.array_size;
                } else {
                    do_copy_assign(new_alloc, other);
                }
            }
        }
        return *this;
    }

    dual_vector& operator=(dual_vector&& other) noexcept requires (alloc_always_equal || alloc_prop_moveassign) {
        alloc_first old_allocator = allocator;
        pointer_first old_buffer = three_way_exchange(buffer, other.buffer, nullptr); // old_buffer == nullptr on self-assign
        size_type old_size = three_way_exchange(array_size, other.array_size, 0);
        size_type old_capacity = three_way_exchange(buffer_capacity, other.buffer_capacity, 0);
        
        deallocate(old_allocator, old_buffer, buffer_capacity);
        allocator = alloc_assign_select<alloc_prop_moveassign>(other);
        return *this;
    }
    
    dual_vector& operator=(dual_vector&& other) requires (move_over) {
        if (allocator == other.allocator){
            pointer_first old_buffer = three_way_exchange(buffer, other.buffer, nullptr); // old_buffer == nullptr on self-assign
            size_type old_size = three_way_exchange(array_size, other.array_size, 0);
            size_type old_capacity = three_way_exchange(buffer_capacity, other.buffer_capacity, 0);
            alloc_destroy(old_buffer, old_buffer+old_size, allocator);
            alloc_destroy(to_begin_second(old_buffer, old_capacity), to_begin_second(old_buffer, old_capacity)+old_size, allocator);
            deallocate(allocator, old_buffer, buffer_capacity);
        } else {
            if (buffer_capacity < other.array_size){
                pointer_first new_buffer = allocate(allocator, other.array_size);
                uninitialized_alloc_move(begin_first(), end_first(), new_buffer, allocator);
                uninitialized_alloc_move(begin_second(), end_second(), (second *)(std::to_address(new_buffer)+other.array_size), allocator);
                destroy_elements();
                deallocate(allocator, buffer, buffer_capacity);
                buffer = new_buffer;
                array_size = other.array_size;
                buffer_capacity = other.array_size;
            }

            move_over_buffer(other);
        }
        return *this;
    }

    dual_vector& operator=(dual_vector&& other){
        if (allocator == other.allocator){
            pointer_first old_buffer = three_way_exchange(buffer, other.buffer, nullptr); // old_buffer == nullptr on self-assign
            size_type old_size = three_way_exchange(array_size, other.array_size, 0);
            size_type old_capacity = three_way_exchange(buffer_capacity, other.buffer_capacity, 0);
            
            deallocate(allocator, old_buffer, buffer_capacity);
        } else {
            
            pointer_first new_buffer = allocate(allocator, other.array_size);
            uninitialized_alloc_move(begin_first(), end_first(), new_buffer, allocator);
            uninitialized_alloc_move(begin_second(), end_second(), (second *)(std::to_address(new_buffer)+other.array_size), allocator);
            deallocate(allocator, buffer, buffer_capacity);
            buffer = new_buffer;
            array_size = other.array_size;
            buffer_capacity = other.array_size;
            

            //move_over_buffer(other);
        }
        return *this;
    }

    allocator_type get_allocator(){
        return allocator;
    }

    reference operator[](size_type index){
        return begin()[index];
    }

    const_reference operator[](size_type index) const {
        return begin()[index];
    }

    reference front(){
        return *begin();
    }

    const_reference front() const {
        return *begin();
    }

    reference back(){
        return *--end();
    }

    const_reference back() const {
        return *--end();
    }

    iterator begin(){
        if constexpr (layout::reversed){
            return {begin_second(), begin_first()};
        } else {
            return {begin_first(), begin_second()};
        }
    }

    const_iterator begin() const {
        if constexpr (layout::reversed){
            return {begin_second(), begin_first()};
        } else {
            return {begin_first(), begin_second()};
        }
    }

    iterator end(){
        if constexpr (layout::reversed){
            return {end_second(), end_first()};
        } else {
            return {end_first(), end_second()};
        }
    }

    const_iterator end() const {
        if constexpr (layout::reversed){
            return {end_second(), end_first()};
        } else {
            return {end_first(), end_second()};
        }
    }

    bool empty() const{
        return array_size == 0;
    }

    size_type size() const {
        return array_size;
    }

    difference_type max_size() const {
        return std::numeric_limits<difference_type>::max();
    }

    // void reserve()

    size_type capacity(){
        return buffer_capacity;
    }

    // void shrink_to_fit()

    private:
    [[no_unique_address]] alloc_first allocator;
    pointer_first buffer;
    size_type buffer_capacity;
    size_type array_size;
};

}



int main(){

}