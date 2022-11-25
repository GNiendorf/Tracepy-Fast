#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <alpaka/alpaka.hpp>

using Data = std::float_t;
using Dim = alpaka::DimInt<1u>;
using Idx = std::size_t;

// - AccGpuCudaRt
// - AccCpuThreads
// - AccCpuFibers
// - AccCpuSerial
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
#elif ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
    using Acc = alpaka::AccCpuThreads<Dim, Idx>;
#elif ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
    using Acc = alpaka::AccCpuFibers<Dim, Idx>;
#else
    using Acc = alpaka::AccCpuSerial<Dim, Idx>;
#endif

// Non blocking significantly increases timing, .000037 -> 0.00013 s
using QueueProperty = alpaka::Blocking;
using QueueAcc = alpaka::Queue<Acc, QueueProperty>;

// Get the host device for allocating memory on the host.
using DevHost = alpaka::DevCpu;
using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;

// Define the accelerator buffer.
using BufAcc = alpaka::Buf<Acc, Data, Dim, Idx>;

#endif