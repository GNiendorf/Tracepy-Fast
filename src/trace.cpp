#include "trace.h"

class traceKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TIdx>
    ALPAKA_FN_ACC void operator()(
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
        TIdx const& nRays) const
    {

        TIdx const globalThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        TIdx const gridThreadExtent(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u]);

        for(TIdx i = globalThreadIdx; i < nRays; i += gridThreadExtent)
        {
            // Don't waste time tracing rays that failed previously.
            if (X[i] == -9999.f)
            {
                //printf("Ignoring previous ray that failed!");
                continue;
            }

            // Transform coordinate system to surface coordinate system.
            transform(R, X, Y, Z, D0, D1, D2, SX, SY, SZ, i);

            // Find intersection coordinates between ray and surface.
            float rho, func, E, deriv; float normal[3];
            findIntersection(acc, X, Y, Z, D0, D1, D2, Skappa, Sc, SDiam, rho, func, E, deriv, normal, i);

            if (X[i] == -9999.f)
            {
                //printf("Failure to find intersection!");
                continue;
            }

            // Find new ray directions from interaction with surface
            interact(acc, X, D0, D1, D2, rayN, surfaceN, intertype, deriv, normal, i);

            if (X[i] == -9999.f)
            {
                //printf("Failure with interaction!");
                continue;
            }

            // Transform coordinate system back to the lab frame.
            labFrame(R, X, Y, Z, D0, D1, D2, SX, SY, SZ, i);
        }
    }
};

int main()
{

    const int nrays = 1000000u;
    alpaka::Vec<Dim, Idx> const extent(nrays);

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    Idx const elementsPerThread = 1u;
#else
    // This parameter doesn't affect timing very much.
    Idx const elementsPerThread = nrays / 10u;
#endif

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

    // Code below this line will change when tracing through multiple surfaces.

    // Curent index of refraction for the rays.
    float RayN = 1.f;

    // Hard-coded surface parameters for now.
    float Skappa = 0.f;
    float Sc = 0.05f;
    float SDiam = 10.f;
    float SurfaceN = .1f;
    int intertype = 2u;
    float alpha = 0.f;
    float beta = M_PI;
    float gamma = 0.f;
    float SX = 0.f;
    float SY = 0.f;
    float SZ = 10.f;

    // Calculate on the CPU, only needs to be done once for each surface.
    alpaka::Vec<Dim, Idx> const R_dim = 9u;
    BufHost bufHostR(alpaka::allocBuf<Data, Idx>(devHost, R_dim));
    Data* pBufHostR(alpaka::getPtrNative(bufHostR));

    genRot(pBufHostR, alpha, beta, gamma);

    BufAcc bufAccR(alpaka::allocBuf<Data, Idx>(devAcc, R_dim));
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


    alpaka::enqueue(queue, traceKernelTask);
    alpaka::wait(queue);

    // This only needs to be done once per surface.
    // How to deal with internal reflection?
    // Fix : Put this in the kernel code.
    if (intertype == 2)
        RayN = SurfaceN;
    const auto beginT = std::chrono::high_resolution_clock::now();
    // Copy back the results.
    alpaka::memcpy(queue, bufHostX, bufAccX);
    alpaka::memcpy(queue, bufHostY, bufAccY);
    alpaka::memcpy(queue, bufHostZ, bufAccZ);
    alpaka::wait(queue);
    const auto endT = std::chrono::high_resolution_clock::now();

    std::cout << "Time for kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's' << std::endl;
    std::cout << "Number of rays: " << nrays << std::endl;

}