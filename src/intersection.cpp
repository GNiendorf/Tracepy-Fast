#include <alpaka/alpaka.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <typeinfo>

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
        TIdx const& numElements) const -> void
    {

        TIdx const gridThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        TIdx const threadElemExtent(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        TIdx const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        if(threadFirstElemIdx < numElements)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            TIdx const threadLastElemIdx(threadFirstElemIdx + threadElemExtent);
            TIdx const threadLastElemIdxClipped((numElements > threadLastElemIdx) ? threadLastElemIdx : numElements);

            for(TIdx i(threadFirstElemIdx); i < threadLastElemIdxClipped; ++i)
            {
                // Hard-coded ray and surface parameters for now.
                float D[3] = {0.f, 0.f, 1.f};
                float Skappa = 0.f;
                float Sc = 0.05f;
                float SDiam = 10.f;

                // Initial guesses. See Spencer, Murty for explanation
                float s_0 = -Z[i]/D[2];
                float X_1 = X[i]+D[1]*s_0;
                float Y_1 = Y[i]+D[2]*s_0;
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
                    Xi = X_1 + D[0]*s_j[0];
                    Yi = Y_1 + D[1]*s_j[0];
                    Zi = 0.f + D[2]*s_j[0];

                    float rho = alpaka::math::sqrt(acc, Xi*Xi + Yi*Yi);

                    if (rho > SDiam/2.f)
                        printf("Not on surface!: rho:%f, Radius:%f\n", rho, SDiam/2.f);

                    // Conic equation.
                    float func = Zi - Sc*(rho*rho)/(1 + alpaka::math::sqrt(acc, (1-Skappa*(Sc*Sc)*(rho*rho))));
                    // See Spencer, Murty section on rotational surfaces for definition of E.
                    float E = Sc / alpaka::math::sqrt(acc, (1-Skappa*(Sc*Sc)*(rho*rho)));
                    float normal[3] = {-Xi*E, -Yi*E, 1.f};
                    float deriv = normal[0]*D[0] + normal[1]*D[1] + normal[2]*D[2];

                    // Newton-raphson method
                    s_j[0] = s_j[1];
                    s_j[1] = s_j[1]-func/deriv;

                    // Error is how far f(X, Y, Z) is from 0.
                    error = alpaka::math::abs(acc, func);

                    n_iter += 1;
                }

                // Check prevents rays from propagating backwards.
                float bcheck = (Xi-X[i])*D[0] + (Yi-Y[i])*D[1] + (Zi-Z[i])*D[2];
                if (n_iter == n_max || s_0+s_j[0] < 0.f || bcheck < 0.f)
                {
                    // Dummy values for now. This implies that the algo did not converge.
                    // Most likely because the ray did not intersect with the surface.
                    X[i] = -10.f;
                    Y[i] = -10.f;
                    Z[i] = -10.f;
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
    // Define the index domain
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;

    // Define the accelerator
    //
    // - AccGpuCudaRt
    // - AccCpuThreads
    // - AccCpuFibers
    // - AccCpuSerial
    using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;

    // Select a device
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);

    // Create a queue on the device
    QueueAcc queue(devAcc);

    // Define the work division
    int nrays = 100000u;
    Idx const numElements(nrays);
    Idx const elementsPerThread(8u);
    alpaka::Vec<Dim, Idx> const extent(numElements);

    // Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::WorkDivMembers<Dim, Idx> const workDiv(alpaka::getValidWorkDiv<Acc>(
        devAcc,
        extent,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

    // Define the buffer element type
    using Data = std::float_t;

    // Get the host device for allocating memory on the host.
    using DevHost = alpaka::DevCpu;
    auto const devHost = alpaka::getDevByIdx<DevHost>(0u);

    // Allocate 3 host memory buffers
    using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;
    BufHost bufHostX(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostY(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostZ(alpaka::allocBuf<Data, Idx>(devHost, extent));

    // Initialize the host input vectors A and B
    Data* pBufHostX(alpaka::getPtrNative(bufHostX));
    Data* pBufHostY(alpaka::getPtrNative(bufHostY));
    Data* pBufHostZ(alpaka::getPtrNative(bufHostZ));

    // Generate random ray positions between -1 and 1 for each coordinate.
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_real_distribution<Data> dist(-1, 1);

    for(Idx i(0); i < numElements; ++i)
    {
        pBufHostX[i] = dist(eng);
        pBufHostY[i] = dist(eng);
        pBufHostZ[i] = dist(eng);
    }

    // Allocate 3 buffers on the accelerator
    using BufAcc = alpaka::Buf<Acc, Data, Dim, Idx>;
    BufAcc bufAccX(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccY(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccZ(alpaka::allocBuf<Data, Idx>(devAcc, extent));

    // Copy Host -> Acc
    alpaka::memcpy(queue, bufAccX, bufHostX);
    alpaka::memcpy(queue, bufAccY, bufHostY);
    alpaka::memcpy(queue, bufAccZ, bufHostZ);

    // Instantiate the kernel function object
    IntersectionKernel kernel;

    // Create the kernel execution task.
    auto const taskKernel = alpaka::createTaskKernel<Acc>(
        workDiv,
        kernel,
        alpaka::getPtrNative(bufAccX),
        alpaka::getPtrNative(bufAccY),
        alpaka::getPtrNative(bufAccZ),
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