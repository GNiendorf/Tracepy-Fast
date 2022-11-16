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
                if (intertype == 0) {} 
                // Needed for total internal reflection even if typeof is refraction.
                else if (b > pow(a, 2) or intertype == 1) {
                    D0[i] = D0[i]-2.f*a*normal[0];
                    D1[i] = D1[i]-2.f*a*normal[1];
                    D2[i] = D2[i]-2.f*a*normal[2];
                } else if (intertype == 2) {
                    //float* D_new = refraction(D, normal, a, b);
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

                    if (rho > SDiam/2.f)
                        printf("Not on surface!: rho:%f, Radius:%f\n", rho, SDiam/2.f);

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
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    Idx const elementsPerThread(1u);

    QueueAcc queue(devAcc);

    int nrays = 100000u;
    Idx const numElements(nrays);
    alpaka::Vec<Dim, Idx> const extent(numElements);

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

    for(Idx i(0); i < numElements; ++i)
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

    // Instantiate the kernel function object
    IntersectionKernel kernel;

    // Hard-coded surface parameters for now.
    float Skappa = 0.f;
    float Sc = 0.05f;
    float SDiam = 10.f;

    // Create the kernel execution task.
    auto const taskKernel = alpaka::createTaskKernel<Acc>(
        workDiv,
        kernel,
        alpaka::getPtrNative(bufAccX),
        alpaka::getPtrNative(bufAccY),
        alpaka::getPtrNative(bufAccZ),
        alpaka::getPtrNative(bufAccD0),
        alpaka::getPtrNative(bufAccD1),
        alpaka::getPtrNative(bufAccD2),
        Skappa,
        Sc,
        SDiam,
        numElements);

    // Enqueue the kernel execution task
    {
        const auto beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);

        alpaka::wait(queue); // wait in case we are using an asynchronous queue to time actual kernel runtime

        const auto endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                    << std::endl;
        std::cout << "Number of rays: " << nrays << std::endl;
    }

}