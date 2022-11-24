#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include "constants.h"

ALPAKA_FN_ACC ALPAKA_FN_INLINE void transform(
    float* R,
    float* X,
    float* Y,
    float* Z,
    float* D0,
    float* D1,
    float* D2,
    const float SX,
    const float SY,
    const float SZ,
    int rayidx);

ALPAKA_FN_ACC ALPAKA_FN_INLINE void labFrame(
    float* R,
    float* X,
    float* Y,
    float* Z,
    float* D0,
    float* D1,
    float* D2,
    const float SX,
    const float SY,
    const float SZ,
    int rayidx);

ALPAKA_FN_HOST ALPAKA_FN_INLINE void genRot(
    Data* R,
    const float alpha,
    const float beta,
    const float gamma);

#endif