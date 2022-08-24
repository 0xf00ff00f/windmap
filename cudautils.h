#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                                                               \
    {                                                                                                                  \
        const cudaError_t error = call;                                                                                \
        if (error != cudaSuccess)                                                                                      \
        {                                                                                                              \
            std::fprintf(stderr, "CUDA error in `%s' line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));   \
            std::abort();                                                                                              \
        }                                                                                                              \
    }

namespace CudaUtils
{

template<class T>
struct Allocator
{
    using value_type = T;

    T *allocate(std::size_t n)
    {
        T *p;
        CUDA_CHECK(cudaMallocManaged(&p, n * sizeof(T)));
        return p;
    }

    void deallocate(T *p, std::size_t) noexcept { CUDA_CHECK(cudaFree(p)); }
};

template<class T, class U>
bool operator==(const Allocator<T> &, const Allocator<U> &) noexcept
{
    return true;
}

template<class T, class U>
bool operator!=(const Allocator<T> &lhs, const Allocator<U> &rhs) noexcept
{
    return !(lhs == rhs);
}

template<typename T>
using CuVector = std::vector<T, Allocator<T>>;

}
