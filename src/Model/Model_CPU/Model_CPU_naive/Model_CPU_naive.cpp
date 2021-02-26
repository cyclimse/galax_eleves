#include <cmath>

#include "Model_CPU_naive.hpp"

Model_CPU_naive ::Model_CPU_naive(const Initstate &initstate, Particles &particles)
	: Model_CPU(initstate, particles)
{
}

void Model_CPU_naive ::step()
{
	std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
	std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
	std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

	for (int i = 0; i < n_particles; i++)
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
				dij = std::sqrt(dij);
				dij = 10.0f / (dij * dij * dij);
			}

			accelerationsx[i] += diffx * dij * initstate.masses[j];
			accelerationsy[i] += diffy * dij * initstate.masses[j];
			accelerationsz[i] += diffz * dij * initstate.masses[j];

			accelerationsx[j] -= diffx * dij * initstate.masses[i];
			accelerationsy[j] -= diffy * dij * initstate.masses[i];
			accelerationsz[j] -= diffz * dij * initstate.masses[i];
		}
	}

	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx[i] * 0.1f;
		particles.y[i] += velocitiesy[i] * 0.1f;
		particles.z[i] += velocitiesz[i] * 0.1f;
	}
}