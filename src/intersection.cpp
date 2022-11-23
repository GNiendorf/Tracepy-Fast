#include "intersection.h"

template<typename TAcc>
ALPAKA_FN_ACC float conics_func(
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

class InteractionKernel
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
        float Skappa,
        float Sc,
        float SDiam,
        float rayN,
        float surfaceN,
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
                float rho = alpaka::math::sqrt(acc, X[i]*X[i] + Y[i]*Y[i]);
                float E = Sc / alpaka::math::sqrt(acc, (1-Skappa*(Sc*Sc)*(rho*rho)));
                float normal[3] = {-X[i]*E, -Y[i]*E, 1.f};
                float deriv = normal[0]*D0[i] + normal[1]*D1[i] + normal[2]*D2[i];
                float surf_norm = normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2];

                float mu = rayN / surfaceN;
                float a = mu*deriv/surf_norm;
                float b = (mu*mu-1)/surf_norm;
                if (intertype == 0)
                {
                    // Do nothing for now, the null interaction.
                }
                else if (b > a*a or intertype == 1)
                {
                    D0[i] = D0[i]-2.f*a*normal[0];
                    D1[i] = D1[i]-2.f*a*normal[1];
                    D2[i] = D2[i]-2.f*a*normal[2];
                }
                else if (intertype == 2)
                {
                    // Should be put into its own seperate function eventually.
                    float G[2] = {-b/(2.f*a), -b/(2.f*a)};

                    // Initial error.
                    float error = 1.f;

                    // Initial and xax iterations allowed.
                    int niter = 0u;
                    int nmax = 1e5;
                    while (error > 1.e-15f and niter < nmax)
                    {
                        // Newton-raphson method
                        G[0] = G[1];
                        G[1] = ((G[1]*G[1])-b)/(2.f*(G[1]+a));

                        // See Spencer, Murty for where this is inspired by.
                        error = alpaka::math::abs(acc, (G[1]*G[1])+2.f*a*G[1]+b);
                        niter += 1;
                    }
                    // Failed to converge.
                    if (niter==nmax)
                    {
                        // Dummy values for now. This implies that the refraction algo did not converge.
                        X[i] = -9999.f;
                        Y[i] = -9999.f;
                        Z[i] = -9999.f;
                    }
                    // Update direction and index of refraction of the current material.
                    D0[i] = mu*D0[i]+G[1]*normal[0];
                    D1[i] = mu*D1[i]+G[1]*normal[1];
                    D2[i] = mu*D2[i]+G[1]*normal[2];
                }
                else
                {
                    printf("Warning! No interaction or incorrect interaction type specified.");
                }
            }
        }
    }
};

class IntersectionKernel
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
                // Transformed ray coordinates.
                X[i] = R[0]*(X[i]-SX) + R[1]*(Y[i]-SY) + R[2]*(Z[i]-SZ);
                Y[i] = R[3]*(X[i]-SX) + R[4]*(Y[i]-SY) + R[5]*(Z[i]-SZ);
                Z[i] = R[6]*(X[i]-SX) + R[7]*(Y[i]-SY) + R[8]*(Z[i]-SZ);

                // Transformed ray directions.
                D0[i] = R[0]*D0[i] + R[1]*D1[i] + R[2]*D2[i];
                D1[i] = R[3]*D0[i] + R[4]*D1[i] + R[5]*D2[i];
                D2[i] = R[6]*D0[i] + R[7]*D1[i] + R[8]*D2[i];

                // Initial guesses. See Spencer, Murty for explanation
                float s_0 = -Z[i]/D2[i];
                float X_1 = X[i]+D1[i]*s_0;
                float Y_1 = Y[i]+D2[i]*s_0;
                float s_j[2] = {0.f, 0.f};

                //Initial error.
                float error = 1.f;
                int n_iter = 0u;
                //Max iterations allowed.
                int n_max = 1e4;
                float Xi, Yi, Zi;
                while (error > 1e-6 and n_iter < n_max)
                {
                    // Iterated coordinates.
                    Xi = X_1 + D0[i]*s_j[0];
                    Yi = Y_1 + D1[i]*s_j[0];
                    Zi = 0.f + D2[i]*s_j[0];

                    float rho = alpaka::math::sqrt(acc, Xi*Xi + Yi*Yi);

                    // Function value and derivative at iterated point.
                    float func = conics_func(acc, Sc, Skappa, rho, Zi);

                    // See Spencer, Murty section on rotational surfaces for definition of E.
                    float E = Sc / alpaka::math::sqrt(acc, (1-Skappa*(Sc*Sc)*(rho*rho)));
                    float normal[3] = {-Xi*E, -Yi*E, 1.f};

                    float deriv = normal[0]*D0[i] + normal[1]*D1[i] + normal[2]*D2[i];

                    // Newton-raphson method
                    s_j[0] = s_j[1];
                    s_j[1] = s_j[1]-func/deriv;

                    // Error is how far f(X, Y, Z) is from 0.
                    error = alpaka::math::abs(acc, func);

                    n_iter += 1;
                }

                // Check prevents rays from propagating backwards.
                float bcheck = (Xi-X[i])*D0[i] + (Yi-Y[i])*D1[i] + (Zi-Z[i])*D2[i];
                if (n_iter == n_max || s_0+s_j[0] < 0.f || bcheck < 0.f)
                {
                    // Dummy values for now. This implies that the algo did not converge.
                    // Most likely because the ray did not intersect with the surface.
                    printf("Failure to find intersection!");
                    X[i] = -9999.f;
                    Y[i] = -9999.f;
                    Z[i] = -9999.f;
                } else {
                    // Algorithm converged. Update position of rays.
                    X[i] = Xi;
                    Y[i] = Yi;
                    Z[i] = Zi;
                }
            }
        }
    }
};

auto main() -> int
{
    // - AccGpuCudaRt
    // - AccCpuThreads
    // - AccCpuFibers
    // - AccCpuSerial
    using Data = std::float_t;
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;
    using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
    // Non blocking significantly increases timing, .000037 -> 0.00013 s
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    Idx const elementsPerThread(1u);

    QueueAcc queue(devAcc);

    int nrays = 100000u;
    Idx const numRayEle(nrays);
    alpaka::Vec<Dim, Idx> const extent(numRayEle);

/*
    // Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::WorkDivMembers<Dim, Idx> const workDiv(alpaka::getValidWorkDiv<Acc>(
        devAcc,
        extent,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));
*/

    Idx const blocksPerGrid = 5;
    Idx const threadsPerBlock = 512;
    auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{blocksPerGrid, threadsPerBlock, elementsPerThread};

    // Get the host device for allocating memory on the host.
    using DevHost = alpaka::DevCpu;
    auto const devHost = alpaka::getDevByIdx<DevHost>(0u);

    using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;
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

    for(Idx i(0); i < numRayEle; ++i)
    {
        pBufHostX[i] = dist(eng);
        pBufHostY[i] = dist(eng);
        pBufHostZ[i] = dist(eng);

        pBufHostD0[i] = 0.f;
        pBufHostD1[i] = 0.f;
        pBufHostD2[i] = 1.f;
    }

    using BufAcc = alpaka::Buf<Acc, Data, Dim, Idx>;
    BufAcc bufAccX(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccY(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccZ(alpaka::allocBuf<Data, Idx>(devAcc, extent));

    BufAcc bufAccD0(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccD1(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccD2(alpaka::allocBuf<Data, Idx>(devAcc, extent));

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
    Idx const Rele(9);
    alpaka::Vec<Dim, Idx> const extent2(Rele);
    BufHost bufHostR(alpaka::allocBuf<Data, Idx>(devHost, extent2));
    Data* pBufHostR(alpaka::getPtrNative(bufHostR));
    BufAcc bufAccR(alpaka::allocBuf<Data, Idx>(devAcc, extent2));

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

    // Create the intersection kernel execution task.
    IntersectionKernel intersect_kernel;
    auto const intersectionKernelTask = alpaka::createTaskKernel<Acc>(
        workDiv,
        intersect_kernel,
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
        numRayEle);

    // Create the interaction kernel execution task.
    InteractionKernel interact_kernel;
    auto const interactionKernelTask = alpaka::createTaskKernel<Acc>(
        workDiv,
        interact_kernel,
        alpaka::getPtrNative(bufAccX),
        alpaka::getPtrNative(bufAccY),
        alpaka::getPtrNative(bufAccZ),
        alpaka::getPtrNative(bufAccD0),
        alpaka::getPtrNative(bufAccD1),
        alpaka::getPtrNative(bufAccD2),
        Skappa,
        Sc,
        SDiam,
        RayN,
        SurfaceN,
        intertype,
        numRayEle);

    // Change eventually
    if (intertype == 2) {
        RayN = SurfaceN;
    }
    

    const auto beginT = std::chrono::high_resolution_clock::now();

    alpaka::enqueue(queue, intersectionKernelTask);
    alpaka::wait(queue);

    alpaka::enqueue(queue, interactionKernelTask);
    alpaka::wait(queue);

    const auto endT = std::chrono::high_resolution_clock::now();

    std::cout << "Time for kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's' << std::endl;
    std::cout << "Number of rays: " << nrays << std::endl;

}