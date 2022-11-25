#include "ray.h"

template<typename TAcc>
ALPAKA_FN_ACC ALPAKA_FN_INLINE float conicsFunc(
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
    int rayidx)
{
    // Initial guesses. See Spencer, Murty for explanation.
    float s_0 = -Z[rayidx] / D2[rayidx];
    float X_1 = X[rayidx] + D0[rayidx] * s_0;
    float Y_1 = Y[rayidx] + D1[rayidx] * s_0;
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
        func = conicsFunc(acc, Sc, Skappa, rho, Zi);

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
    else if (b > a * a || intertype == 1)
    {
        reflection(D0, D1, D2, normal, a/mu, rayidx);
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