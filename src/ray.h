#ifndef RAY_H
#define RAY_H

#include "constants.h"

template<typename TAcc>
ALPAKA_FN_ACC ALPAKA_FN_INLINE float conicsFunc(
    TAcc const& acc,
    float Sc,
    float Skappa,
    float rho,
    float Zi);

template<typename TAcc>
ALPAKA_FN_ACC ALPAKA_FN_INLINE void findIntersection(
    TAcc const& acc,
    float* X,
    float* Y,
    float* Z,
    float* D0,
    float* D1,
    float* D2,
    float Skappa,
    float Sc,
    float SDiam,
    float& rho,
    float& func,
    float& E,
    float& deriv,
    float normal[3],
    int rayidx);

ALPAKA_FN_ACC ALPAKA_FN_INLINE void reflection(
    float* D0,
    float* D1,
    float* D2,
    float normal[3],
    const float a,
    int rayidx);

template<typename TAcc>
ALPAKA_FN_ACC ALPAKA_FN_INLINE void refraction(
    TAcc const& acc,
    float* D0,
    float* D1,
    float* D2,
    float* X,
    float normal[3],
    const float a,
    const float b,
    const float mu,
    int rayidx);

template<typename TAcc>
ALPAKA_FN_ACC ALPAKA_FN_INLINE void interact(
    TAcc const& acc,
    float* X,
    float* D0,
    float* D1,
    float* D2,
    float rayN,
    float surfaceN,
    int intertype,
    float& deriv,
    float normal[3],
    int rayidx);

#endif