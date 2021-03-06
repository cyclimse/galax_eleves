#ifdef GALAX_MODEL_CPU_FAST

#include "Model_CPU_fast.hpp"

#include <cmath>

#include <eve/function/rsqrt.hpp>
#include <omp.h>

Model_CPU_fast ::Model_CPU_fast(const Initstate &initstate, Particles &particles)
    : Model_CPU(initstate, particles)
{
}

void Model_CPU_fast ::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

    constexpr bool use_rsqrt = true;

// OMP  version
#pragma omp parallel for
    for (int i = 0; i < n_particles / 2; i++)
    {
        for (int j = 0; j < i; j++)
        {
            const float diffx = particles.x[j] - particles.x[i];
            const float diffy = particles.y[j] - particles.y[i];
            const float diffz = particles.z[j] - particles.z[i];

            float dij = diffx * diffx + diffy * diffy + diffz * diffz;

            if (dij < 1.0f)
            {
                dij = 10.0f;
            }
            else
            {
                if constexpr (use_rsqrt)
                {
                    dij = dij * dij * dij;
                    dij = 10.0f * eve::rsqrt(dij);
                }
                else
                {
                    dij = std::sqrt(dij);
                    dij = 10.0 / (dij * dij * dij);
                }
            }

            accelerationsx[i] += diffx * dij * initstate.masses[j];
            accelerationsy[i] += diffy * dij * initstate.masses[j];
            accelerationsz[i] += diffz * dij * initstate.masses[j];

            accelerationsx[j] -= diffx * dij * initstate.masses[i];
            accelerationsy[j] -= diffy * dij * initstate.masses[i];
            accelerationsz[j] -= diffz * dij * initstate.masses[i];
        }

        auto invi = n_particles - 1 - i;
        for (int j = 0; j < invi; j++)
        {
            const float diffx = particles.x[j] - particles.x[invi];
            const float diffy = particles.y[j] - particles.y[invi];
            const float diffz = particles.z[j] - particles.z[invi];

            float dij = diffx * diffx + diffy * diffy + diffz * diffz;

            if (dij < 1.0f)
            {
                dij = 10.0f;
            }
            else
            {
                if constexpr (use_rsqrt)
                {
                    dij = dij * dij * dij;
                    dij = 10.0f * eve::rsqrt(dij);
                }
                else
                {
                    dij = std::sqrt(dij);
                    dij = 10.0 / (dij * dij * dij);
                }
            }

            accelerationsx[invi] += diffx * dij * initstate.masses[j];
            accelerationsy[invi] += diffy * dij * initstate.masses[j];
            accelerationsz[invi] += diffz * dij * initstate.masses[j];

            accelerationsx[j] -= diffx * dij * initstate.masses[invi];
            accelerationsy[j] -= diffy * dij * initstate.masses[invi];
            accelerationsz[j] -= diffz * dij * initstate.masses[invi];
        }
    }

#pragma omp parallel for
    for (int i = 0; i < n_particles; i++)
    {
        velocitiesx[i] += accelerationsx[i] * 2.0f;
        velocitiesy[i] += accelerationsy[i] * 2.0f;
        velocitiesz[i] += accelerationsz[i] * 2.0f;
        particles.x[i] += velocitiesx[i] * 0.1f;
        particles.y[i] += velocitiesy[i] * 0.1f;
        particles.z[i] += velocitiesz[i] * 0.1f;
    }

    // OMP + MIPP version
    // #pragma omp parallel for
    //     for (int i = 0; i < n_particles; i += mipp::N<float>())
    //     {
    //         // load registers body i
    //         const mipp::Reg<float> rposx_i = &particles.x[i];
    //         const mipp::Reg<float> rposy_i = &particles.y[i];
    //         const mipp::Reg<float> rposz_i = &particles.z[i];
    //               mipp::Reg<float> raccx_i = &accelerationsx[i];
    //               mipp::Reg<float> raccy_i = &accelerationsy[i];
    //               mipp::Reg<float> raccz_i = &accelerationsz[i];
    //     }
}

#endif // GALAX_MODEL_CPU_FAST
