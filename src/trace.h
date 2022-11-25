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

#endif