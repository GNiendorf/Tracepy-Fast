#include "intersection.h"

template<typename TAcc>
ALPAKA_FN_ACC ALPAKA_FN_INLINE float conics_func(
    TAcc const& acc,
    float Sc,
    float Skappa,
    float rho,
    float Zi)
{
    // Conic equation.
    float func = Zi - Sc*(rho*rho)/(1 + alpaka::math::sqrt(acc, (1-Skappa*(Sc*Sc)*(rho*rho))));
    return func;
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

ALPAKA_FN_ACC ALPAKA_FN_INLINE void lab_frame(
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

template<typename TAcc>
ALPAKA_FN_ACC ALPAKA_FN_INLINE void find_intersection(
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
    int rayidx)
{
    // Initial guesses. See Spencer, Murty for explanation.
    float s_0 = -Z[rayidx] / D2[rayidx];
    float X_1 = X[rayidx] + D1[rayidx] * s_0;
    float Y_1 = Y[rayidx] + D2[rayidx] * s_0;
    float s_j[2] = {0.f, 0.f};

    //Initial error and iterator.
    float error = 1.f;
    int n_iter = 0u;

    //Max iterations allowed.
    int n_max = 1e4;
    float Xi, Yi, Zi;
    while (error > 1e-6 and n_iter < n_max)
    {
        // Iterated coordinates.
        Xi = X_1 + D0[rayidx] * s_j[0];
        Yi = Y_1 + D1[rayidx] * s_j[0];
        Zi = 0.f + D2[rayidx] * s_j[0];

        rho = alpaka::math::sqrt(acc, Xi*Xi + Yi*Yi);

        // Function value and derivative at iterated point.
        func = conics_func(acc, Sc, Skappa, rho, Zi);

        // See Spencer, Murty section on rotational surfaces for definition of E.
        E = Sc / alpaka::math::sqrt(acc, (1 - Skappa * (Sc*Sc) * (rho*rho)));
        normal[0] = -Xi * E;
        normal[1] = -Yi * E;
        normal[2] = 1.f;

        deriv = normal[0]*D0[rayidx] + normal[1]*D1[rayidx] + normal[2]*D2[rayidx];

        // Newton-raphson method
        s_j[0] = s_j[1];
        s_j[1] = s_j[1] - func / deriv;

        // Error is how far f(X, Y, Z) is from 0.
        error = alpaka::math::abs(acc, func);

        n_iter += 1;
    }

    // Check prevents rays from propagating backwards.
    float bcheck = (Xi - X[rayidx]) * D0[rayidx] + (Yi - Y[rayidx]) * D1[rayidx] + (Zi - Z[rayidx]) * D2[rayidx];
    if (n_iter == n_max || s_0 + s_j[0] < 0.f || bcheck < 0.f)
    {
        // Dummy values for now. This implies that the algo did not converge.
        X[rayidx] = -9999.f;
    } else {
        // Algorithm converged. Update position of rays.
        X[rayidx] = Xi;
        Y[rayidx] = Yi;
        Z[rayidx] = Zi;
    }
}

ALPAKA_FN_ACC ALPAKA_FN_INLINE void reflection(
    float* D0,
    float* D1,
    float* D2,
    float normal[3],
    const float a,
    int rayidx)
{
    D0[rayidx] = D0[rayidx] - 2.f * a * normal[0];
    D1[rayidx] = D1[rayidx] - 2.f * a * normal[1];
    D2[rayidx] = D2[rayidx] - 2.f * a * normal[2];
}

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
    int rayidx)
{
    // Should be put into its own seperate function eventually.
    float G[2] = {-b / (2.f * a), -b / (2.f * a)};

    // Initial error.
    float error = 1.f;

    // Initial and xax iterations allowed.
    int niter = 0u;
    int nmax = 1e5;
    while (error > 1.e-15f and niter < nmax)
    {
        // Newton-raphson method
        G[0] = G[1];
        G[1] = ((G[1] * G[1]) - b) / (2.f * (G[1] + a));

        // See Spencer, Murty for where this is inspired by.
        error = alpaka::math::abs(acc, (G[1] * G[1]) + 2.f * a * G[1] + b);
        niter += 1;
    }

    // Failed to converge.
    if (niter == nmax)
    {
        // Dummy values for now. This implies that the refraction algo did not converge.
        X[rayidx] = -9999.f;
    }

    // Update direction and index of refraction of the current material.
    D0[rayidx] = mu * D0[rayidx] + G[1] * normal[0];
    D1[rayidx] = mu * D1[rayidx] + G[1] * normal[1];
    D2[rayidx] = mu * D2[rayidx] + G[1] * normal[2];
}

template<typename TAcc>
ALPAKA_FN_ACC ALPAKA_FN_INLINE void interaction(
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
    int rayidx)
{
    float surf_norm = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    float mu = rayN / surfaceN;
    float a = mu * deriv / surf_norm;
    float b = (mu * mu - 1) / surf_norm;

    if (intertype == 0)
    {
        // Do nothing for now, the null interaction.
    }
    // The first condition is needed for total internal reflection.
    else if (b > a * a or intertype == 1)
    {
        reflection(D0, D1, D2, normal, a, rayidx);
    }
    else if (intertype == 2)
    {
        refraction(acc, D0, D1, D2, X, normal, a, b, mu, rayidx);
    }
    else
    {
        printf("Warning! No interaction or incorrect interaction type specified.");
    }
}



class traceKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        float* X,
        float* Y,
        float* Z,
        float* D0,
        float* D1,
        float* D2,
        float SX,
        float SY,
        float SZ,
        float* R,
        float Skappa,
        float Sc,
        float SDiam,
        float surfaceN,
        float rayN,
        int intertype,
        TIdx const& nRays) const -> void
    {

        TIdx const gridThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        TIdx const threadElemExtent(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        TIdx const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        if(threadFirstElemIdx < nRays)
        {

            TIdx const threadLastElemIdx(threadFirstElemIdx + threadElemExtent);
            TIdx const threadLastElemIdxClipped((nRays > threadLastElemIdx) ? threadLastElemIdx : nRays);

            for(TIdx i(threadFirstElemIdx); i < threadLastElemIdxClipped; ++i)
            {
                // Don't waste time tracing rays that failed previously.
                if (X[i] == -9999.f)
                    //printf("Ignoring previous ray that failed!");
                    continue;

                // Transform coordinate system to surface coordinate system.
                transform(R, X, Y, Z, D0, D1, D2, SX, SY, SZ, i);

                // Find intersection coordinates between ray and surface.
                float rho, func, E, deriv; float normal[3];
                find_intersection(acc, X, Y, Z, D0, D1, D2, Skappa, Sc, SDiam, rho, func, E, deriv, normal, i);

                if (X[i] == -9999.f)
                    //printf("Failure to find intersection!");
                    continue;

                // Find new ray directions from interaction with surface
                interaction(acc, X, D0, D1, D2, rayN, surfaceN, intertype, deriv, normal, i);

                if (X[i] == -9999.f)
                    //printf("Failure with interaction!");
                    continue;

                // Transform coordinate system back to the lab frame.
                lab_frame(R, X, Y, Z, D0, D1, D2, SX, SY, SZ, i);
            }
        }
    }
};

int main()
{

    int nrays = 100000u;
    alpaka::Vec<Dim, Idx> const extent(nrays);

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    Idx const blocksPerGrid = 5;
    Idx const threadsPerBlock = 512;
    auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{blocksPerGrid, threadsPerBlock, elementsPerThread};
#else
    // Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::WorkDivMembers<Dim, Idx> const workDiv(alpaka::getValidWorkDiv<Acc>(
        devAcc,
        extent,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));
#endif

    BufHost bufHostX(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostY(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostZ(alpaka::allocBuf<Data, Idx>(devHost, extent));

    BufHost bufHostD0(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostD1(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostD2(alpaka::allocBuf<Data, Idx>(devHost, extent));

    Data* pBufHostX(alpaka::getPtrNative(bufHostX));
    Data* pBufHostY(alpaka::getPtrNative(bufHostY));
    Data* pBufHostZ(alpaka::getPtrNative(bufHostZ));

    Data* pBufHostD0(alpaka::getPtrNative(bufHostD0));
    Data* pBufHostD1(alpaka::getPtrNative(bufHostD1));
    Data* pBufHostD2(alpaka::getPtrNative(bufHostD2));

    // Generate random ray positions between -1 and 1 for each coordinate.
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_real_distribution<Data> dist(-1, 1);

    for(Idx i(0); i < nrays; ++i)
    {
        pBufHostX[i] = dist(eng);
        pBufHostY[i] = dist(eng);
        pBufHostZ[i] = dist(eng);

        pBufHostD0[i] = 0.f;
        pBufHostD1[i] = 0.f;
        pBufHostD2[i] = 1.f;
    }

    BufAcc bufAccX(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccY(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccZ(alpaka::allocBuf<Data, Idx>(devAcc, extent));

    BufAcc bufAccD0(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccD1(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccD2(alpaka::allocBuf<Data, Idx>(devAcc, extent));

    // Define our queue.
    QueueAcc queue(devAcc);

    // Copy Host -> Acc
    alpaka::memcpy(queue, bufAccX, bufHostX);
    alpaka::memcpy(queue, bufAccY, bufHostY);
    alpaka::memcpy(queue, bufAccZ, bufHostZ);

    alpaka::memcpy(queue, bufAccD0, bufHostD0);
    alpaka::memcpy(queue, bufAccD1, bufHostD1);
    alpaka::memcpy(queue, bufAccD2, bufHostD2);

    // Hard-coded surface parameters for now.
    float Skappa = 0.f;
    float Sc = 0.05f;
    float SDiam = 10.f;
    float RayN = 1.f;
    float SurfaceN = 1.5f;
    int intertype = 1u;
    float alpha = 0.f;
    float beta = M_PI;
    float gamma = 0.f;
    float SX = 0.f;
    float SY = 0.f;
    float SZ = 10.f;

    // Calculate on the CPU, only needs to be done once. Such awful code though.
    alpaka::Vec<Dim, Idx> R_dim = 9u;
    BufHost bufHostR(alpaka::allocBuf<Data, Idx>(devHost, R_dim));
    Data* pBufHostR(alpaka::getPtrNative(bufHostR));
    BufAcc bufAccR(alpaka::allocBuf<Data, Idx>(devAcc, R_dim));

    pBufHostR[0] = std::cos(alpha)*std::cos(gamma)+std::sin(alpha)*std::sin(beta)*std::sin(gamma);
    pBufHostR[1] = -std::cos(beta)*std::sin(gamma);
    pBufHostR[2] = -std::sin(alpha)*std::cos(gamma)+std::cos(alpha)*std::sin(beta)*std::sin(gamma);
    pBufHostR[3] = std::cos(alpha)*std::sin(gamma)-std::sin(alpha)*std::sin(beta)*std::cos(gamma);
    pBufHostR[4] = std::cos(beta)*std::cos(gamma);
    pBufHostR[5] = -std::sin(alpha)*std::sin(gamma)-std::cos(alpha)*std::sin(beta)*std::cos(gamma);
    pBufHostR[6] = std::sin(alpha)*std::cos(beta);
    pBufHostR[7] = std::sin(beta);
    pBufHostR[8] = std::cos(alpha)*std::cos(beta);

    alpaka::memcpy(queue, bufAccR, bufHostR);

    // Create the trace kernel execution task.
    traceKernel trace_kernel;
    auto const traceKernelTask = alpaka::createTaskKernel<Acc>(
        workDiv,
        trace_kernel,
        alpaka::getPtrNative(bufAccX),
        alpaka::getPtrNative(bufAccY),
        alpaka::getPtrNative(bufAccZ),
        alpaka::getPtrNative(bufAccD0),
        alpaka::getPtrNative(bufAccD1),
        alpaka::getPtrNative(bufAccD2),
        SX,
        SY,
        SZ,
        alpaka::getPtrNative(bufAccR),
        Skappa,
        Sc,
        SDiam,
        SurfaceN,
        RayN,
        intertype,
        nrays);

    const auto beginT = std::chrono::high_resolution_clock::now();

    alpaka::enqueue(queue, traceKernelTask);
    alpaka::wait(queue);

    // Change eventually.
    if (intertype == 2) {
        RayN = SurfaceN;
    }

    const auto endT = std::chrono::high_resolution_clock::now();

    // Copy back the results.
    alpaka::memcpy(queue, bufHostX, bufAccX);
    alpaka::memcpy(queue, bufHostY, bufAccY);
    alpaka::memcpy(queue, bufHostZ, bufAccZ);
    alpaka::wait(queue);

    std::cout << "Time for kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's' << std::endl;
    std::cout << "Number of rays: " << nrays << std::endl;

}