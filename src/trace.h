#ifndef TRACE_H
#define TRACE_H

#include "constants.h"
#include "transforms.cpp"
#include "ray.cpp"

#include <chrono>
#include <iostream>
#include <random>
#include <typeinfo>
#include <cmath>

auto const devHost = alpaka::getDevByIdx<DevHost>(0u);
auto const devAcc = alpaka::getDevByIdx<Acc>(0u);

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    Idx const elementsPerThread = 1u;
#else
    Idx const elementsPerThread = 20000u;
#endif

#endif