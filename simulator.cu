#include "simulator.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <random>
#include <cassert>
#include <cstdio>
#include <cstdlib>

// latitude: n/s, -pi/2 to pi/2
// longitude: e/w, -pi to pi
__device__ glm::vec3 latLonToCartesian(float lat, float lon)
{
    const auto r = glm::cos(lat);
    const auto x = r * glm::cos(lon);
    const auto y = r * glm::sin(lon);
    const auto z = glm::sin(lat);
    return {x, y, z};
}

__device__ glm::vec2 cartesianToLatLon(const glm::vec3 &position)
{
    float lon = glm::atan(position.y, position.x);
    float lat = glm::asin(position.z);
    return {lat, lon};
}

__global__ void updateParticle(Particle *particles, int particleCount, glm::vec2 *windMap, int windMapWidth,
                               int windMapHeight)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < particleCount; i += stride)
    {
        auto &particle = particles[i];

        const auto lat = particle.position.x;
        const auto lon = particle.position.y;
        auto position = latLonToCartesian(lat, lon);

        const auto windSpeed = [=] {
            const auto x =
                static_cast<int>(((lon + glm::pi<float>()) / (2.0f * glm::pi<float>())) * windMapWidth) % windMapWidth;
            const auto y =
                windMapHeight - 1 -
                static_cast<int>(((lat + glm::half_pi<float>()) / glm::pi<float>()) * windMapHeight) % windMapHeight;
            return windMap[y * windMapWidth + x];
        }();
        auto speed = particle.speed + 0.02f * windSpeed;

        // XXX check this
        auto n = glm::normalize(position);
        auto u = glm::cross(n, glm::vec3(0, 0, 1));
        auto v = glm::cross(n, u);
        position += v * speed.y + u * speed.x;
        position = glm::normalize(position);

        particle.position = cartesianToLatLon(position);

        particle.history[particle.historySize % Particle::MaxHistorySize] = particle.position;
        particle.historySize++;
    }
}

void Simulator::update()
{
    constexpr auto BlockSize = 256;
    constexpr auto NumBlocks = (ParticleCount + BlockSize - 1) / BlockSize;
    updateParticle<<<NumBlocks, BlockSize>>>(m_particles, ParticleCount, m_windMap, m_windMapWidth, m_windMapHeight);
    cudaDeviceSynchronize();
}

Simulator::Simulator(const glm::vec2 *windMap, int windMapWidth, int windMapHeight)
    : m_windMapWidth(windMapWidth)
    , m_windMapHeight(windMapHeight)
{
    // initialize particles
    auto rv = cudaMallocManaged(&m_particles, ParticleCount * sizeof(Particle));
    if (rv != cudaSuccess)
    {
        fprintf(stderr, "cudaMallocManaged failed (%d): %s\n", rv, cudaGetErrorString(rv));
        abort();
    }

    std::mt19937 eng;
    std::uniform_real_distribution<> dist(0, 1);
    std::for_each(m_particles, m_particles + ParticleCount, [&](Particle &particle) {
        const auto lat = dist(eng) * glm::pi<float>() - glm::half_pi<float>();
        const auto lon = dist(eng) * 2.0f * glm::pi<float>() - glm::pi<float>();
        particle.position = glm::vec2(lat, lon);
        particle.speed = glm::vec2(0, 0);
    });

    // initialize wind map
    cudaMallocManaged(&m_windMap, windMapWidth * windMapHeight * sizeof(glm::vec2));
    std::copy(windMap, windMap + windMapWidth * windMapHeight, m_windMap);
}

Simulator::~Simulator()
{
    cudaFree(m_windMap);
    cudaFree(m_particles);
}
