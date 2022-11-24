#include "transforms.h"

ALPAKA_FN_HOST ALPAKA_FN_INLINE void genRot(
    Data* R,
    const float alpha,
    const float beta,
    const float gamma)
{
    // Generate the rotation matrix from surface rotation angles alpha, beta, gamma.
    float cosAlpha = std::cos(alpha); float sinAlpha = std::sin(alpha);
    float cosBeta = std::cos(beta); float sinBeta = std::sin(beta);
    float cosGamma = std::cos(gamma); float sinGamma = std::sin(gamma);

    R[0] = cosAlpha * cosGamma + sinAlpha * sinBeta * sinGamma;
    R[1] = -cosBeta * sinGamma;
    R[2] = -sinAlpha * cosGamma + cosAlpha * sinBeta * sinGamma;
    R[3] = cosAlpha * sinGamma - sinAlpha * sinBeta * cosGamma;
    R[4] = cosBeta * cosGamma;
    R[5] = -sinAlpha * sinGamma - cosAlpha * sinBeta * cosGamma;
    R[6] = sinAlpha * cosBeta;
    R[7] = sinBeta;
    R[8] = cosAlpha * cosBeta;
}

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
    int rayidx)
{
    // Transformed ray coordinates.
    X[rayidx] = R[0]*(X[rayidx]-SX) + R[1]*(Y[rayidx]-SY) + R[2]*(Z[rayidx]-SZ);
    Y[rayidx] = R[3]*(X[rayidx]-SX) + R[4]*(Y[rayidx]-SY) + R[5]*(Z[rayidx]-SZ);
    Z[rayidx] = R[6]*(X[rayidx]-SX) + R[7]*(Y[rayidx]-SY) + R[8]*(Z[rayidx]-SZ);

    // Transformed ray directions.
    D0[rayidx] = R[0]*D0[rayidx] + R[1]*D1[rayidx] + R[2]*D2[rayidx];
    D1[rayidx] = R[3]*D0[rayidx] + R[4]*D1[rayidx] + R[5]*D2[rayidx];
    D2[rayidx] = R[6]*D0[rayidx] + R[7]*D1[rayidx] + R[8]*D2[rayidx];
}

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
    int rayidx)
{
    // Transform coordinates back to lab frame.
    X[rayidx] = (R[0]*X[rayidx] + R[3]*Y[rayidx] + R[6]*Z[rayidx]) + SX;
    Y[rayidx] = (R[1]*X[rayidx] + R[4]*Y[rayidx] + R[7]*Z[rayidx]) + SY;
    Z[rayidx] = (R[2]*X[rayidx] + R[5]*Y[rayidx] + R[8]*Z[rayidx]) + SZ;

    // Transform directions back to lab frame.
    D0[rayidx] = R[0]*D0[rayidx] + R[3]*D1[rayidx] + R[6]*D2[rayidx];
    D1[rayidx] = R[1]*D0[rayidx] + R[4]*D1[rayidx] + R[7]*D2[rayidx];
    D2[rayidx] = R[2]*D0[rayidx] + R[5]*D1[rayidx] + R[8]*D2[rayidx];
}