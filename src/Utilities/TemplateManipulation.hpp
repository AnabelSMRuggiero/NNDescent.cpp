template<typename... Types>
struct parameter_pack{

};

template< template<typename...> typename TemplateParam, typename... Pack>
consteval parameter_pack<Pack...> extract_pack(const TemplateParam<Pack...>&){
    return parameter_pack<Pack...>{};
}

template<typename... Types>
struct pack_reverser{


    template<typename... OpTypes>
    constexpr pack_reverser<OpTypes..., Types...> operator<<=(pack_reverser<OpTypes...>){
        return pack_reverser<OpTypes..., Types...>{};
    }
};

template<typename... Types>
consteval auto reverse_pack(parameter_pack<Types...>){
    return extract_pack((pack_reverser<Types>{} <<= ...));
}

template<typename... Pack>
using reverse_pack_t = decltype(reverse_pack(parameter_pack<Pack...>{}));

template<template <typename...> typename Parameter>
struct template_parameter{
    //template<typename... Pack>
    //using type = Parameter<Pack...>;
};

template<typename Type>
struct type_obj{
    using type = Type;
};

template<template <typename...> typename TemplateParam, typename... Pack>
consteval auto apply_pack(template_parameter<TemplateParam>, parameter_pack<Pack...>){
    return type_obj<TemplateParam<Pack...>>{};
}

template<template <typename...> typename Template, typename... FrontArgs>
struct front_bind{

    template<typename... Args>
    using apply = parameter_pack<FrontArgs..., Args...>;
    
    template<typename... Args>
    using type = typename decltype(apply_pack(template_parameter<Template>{}, apply<Args...>{}))::type;
    
};

template<template <typename...> typename Template, typename... BackArgs>
struct back_bind{

    template<typename... Args>
    using apply = parameter_pack<Args..., BackArgs...>;
    
    template<typename... Args>
    using type = typename decltype(apply_pack(template_parameter<Template>{}, apply<Args...>{}))::type;
    
};

template<template <typename...> typename TemplateParam, typename... Pack>
consteval auto bind_to_front(template_parameter<TemplateParam>, parameter_pack<Pack...>){
    return template_parameter<front_bind<TemplateParam, Pack...>::template type>{};
}

template<template <typename...> typename TemplateParam, typename... Pack>
consteval auto bind_to_back(template_parameter<TemplateParam>, parameter_pack<Pack...>){
    return template_parameter<back_bind<TemplateParam, Pack...>::template type>{};
}

template<template <typename...> typename TemplateParam, typename... Pack>
using apply_pack_t = typename decltype(apply_pack(template_parameter<TemplateParam>{}, parameter_pack<Pack...>{}))::type;