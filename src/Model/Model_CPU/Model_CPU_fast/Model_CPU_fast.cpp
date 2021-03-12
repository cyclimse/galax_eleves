#ifdef GALAX_MODEL_CPU_FAST

#include "Model_CPU_fast.hpp"

#include <cmath>
#include <algorithm>

#include <eve/function/sub.hpp>
#include <eve/function/rsqrt.hpp>
#include <eve/constant/one.hpp>
#include <eve/function/hypot.hpp>
#include <eve/function/max.hpp>
#include <eve/function/mul.hpp>
#include <eve/function/div.hpp>
#include <eve/function/reduce.hpp>
#include <eve/wide.hpp>
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
        eve::wide<float, eve::fixed<8L>> regis_x;
        eve::wide<float, eve::fixed<8L>> regis_y;
        eve::wide<float, eve::fixed<8L>> regis_z;
        eve::wide<float, eve::fixed<8L>> regis_dij;
        eve::wide<float, eve::fixed<8L>> regis_m;

        for (int j = 0; j < i - i % 8; j += 8)
        {
            std::copy(particles.x.cbegin() + j, particles.x.cbegin() + j + 8, regis_x.begin());
            std::copy(particles.y.cbegin() + j, particles.y.cbegin() + j + 8, regis_y.begin());
            std::copy(particles.z.cbegin() + j, particles.z.cbegin() + j + 8, regis_z.begin());

            regis_x -= particles.x[i] * eve::one(as(regis_x));
            regis_y -= particles.y[i] * eve::one(as(regis_y));
            regis_z -= particles.z[i] * eve::one(as(regis_z));

            regis_dij = eve::hypot(regis_x, regis_y, regis_z);
            regis_dij = eve::max(1.0f, regis_dij);

            regis_dij = eve::mul(eve::mul(regis_dij, regis_dij), regis_dij);
            regis_dij = eve::div(10.0f, regis_dij);

            std::copy(initstate.masses.cbegin() + j, initstate.masses.cbegin() + j + 8, regis_m.begin());

            regis_x = eve::mul(regis_x, regis_dij);
            regis_y = eve::mul(regis_y, regis_dij);
            regis_z = eve::mul(regis_z, regis_dij);

            accelerationsx[i] += eve::reduce(eve::mul(regis_x, regis_m), std::plus<>{});
            accelerationsy[i] += eve::reduce(eve::mul(regis_y, regis_m), std::plus<>{});
            accelerationsz[i] += eve::reduce(eve::mul(regis_z, regis_m), std::plus<>{});

            std::copy(accelerationsx.cbegin() + j, accelerationsx.cbegin() + j + 8, regis_m.begin());
            regis_m -= initstate.masses[i] * regis_x;
            std::copy(regis_m.begin(), regis_m.begin() + 8, accelerationsx.begin() + j);

            std::copy(accelerationsy.cbegin() + j, accelerationsy.cbegin() + j + 8, regis_m.begin());
            regis_m -= initstate.masses[i] * regis_y;
            std::copy(regis_m.begin(), regis_m.begin() + 8, accelerationsy.begin() + j);

            std::copy(accelerationsz.cbegin() + j, accelerationsz.cbegin() + j + 8, regis_m.begin());
            regis_m -= initstate.masses[i] * regis_z;
            std::copy(regis_m.begin(), regis_m.begin() + 8, accelerationsz.begin() + j);
        }

        for (int j = i - i % 8; j < i; j++)
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
        for (int j = 0; j < invi - invi % 8; j += 8)
        {
            std::copy(particles.x.cbegin() + j, particles.x.cbegin() + j + 8, regis_x.begin());
            std::copy(particles.y.cbegin() + j, particles.y.cbegin() + j + 8, regis_y.begin());
            std::copy(particles.z.cbegin() + j, particles.z.cbegin() + j + 8, regis_z.begin());

            regis_x -= particles.x[invi] * eve::one(as(regis_x));
            regis_y -= particles.y[invi] * eve::one(as(regis_y));
            regis_z -= particles.z[invi] * eve::one(as(regis_z));

            regis_dij = eve::hypot(regis_x, regis_y, regis_z);
            regis_dij = eve::max(1.0f, regis_dij);

            regis_dij = eve::mul(eve::mul(regis_dij, regis_dij), regis_dij);
            regis_dij = eve::div(10.0f, regis_dij);

            std::copy(initstate.masses.cbegin() + j, initstate.masses.cbegin() + j + 8, regis_m.begin());

            regis_x = eve::mul(regis_x, regis_dij);
            regis_y = eve::mul(regis_y, regis_dij);
            regis_z = eve::mul(regis_z, regis_dij);

            accelerationsx[invi] += eve::reduce(eve::mul(regis_x, regis_m), std::plus<>{});
            accelerationsy[invi] += eve::reduce(eve::mul(regis_y, regis_m), std::plus<>{});
            accelerationsz[invi] += eve::reduce(eve::mul(regis_z, regis_m), std::plus<>{});

            std::copy(accelerationsx.cbegin() + j, accelerationsx.cbegin() + j + 8, regis_m.begin());
            regis_m -= initstate.masses[invi] * regis_x;
            std::copy(regis_m.begin(), regis_m.begin() + 8, accelerationsx.begin() + j);

            std::copy(accelerationsy.cbegin() + j, accelerationsy.cbegin() + j + 8, regis_m.begin());
            regis_m -= initstate.masses[invi] * regis_y;
            std::copy(regis_m.begin(), regis_m.begin() + 8, accelerationsy.begin() + j);

            std::copy(accelerationsz.cbegin() + j, accelerationsz.cbegin() + j + 8, regis_m.begin());
            regis_m -= initstate.masses[invi] * regis_z;
            std::copy(regis_m.begin(), regis_m.begin() + 8, accelerationsz.begin() + j);
        }

        for (int j = invi - invi % 8; j < invi; j++)
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
