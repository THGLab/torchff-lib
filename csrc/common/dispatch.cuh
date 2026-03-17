#ifndef TORCHFF_DISPATCH_CUH
#define TORCHFF_DISPATCH_CUH

#define DISPATCH_BOOL(VALUE, NAME, ...)            \
    do {                                           \
        if (VALUE) {                               \
            constexpr bool NAME = true;             \
            __VA_ARGS__();                          \
        } else {                                    \
            constexpr bool NAME = false;            \
            __VA_ARGS__();                          \
        }                                           \
    } while (0)

#define DISPATCH_RANK(VALUE, NAME, ...)            \
    do {                                           \
        if ((VALUE) == 0) {                         \
            constexpr int NAME = 0;                 \
            __VA_ARGS__();                          \
        } else if ((VALUE) == 1) {                  \
            constexpr int NAME = 1;                 \
            __VA_ARGS__();                          \
        } else {                                    \
            constexpr int NAME = 2;                 \
            __VA_ARGS__();                          \
        }                                           \
    } while (0)

#endif /* TORCHFF_DISPATCH_CUH */
