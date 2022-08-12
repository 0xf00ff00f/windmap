#include "simulator.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <random>
#include <cassert>
#include <cstdio>
#include <cstdlib>

// latitude: n/s, -pi/2 to pi/2
// longitude: e/w, -pi to pi
glm::vec3 latLonToCartesian(float lat, float lon)
{
    const auto r = glm::cos(lat);
    const auto x = r * glm::cos(lon);
    const auto y = r * glm::sin(lon);
    const auto z = glm::sin(lat);
    return {x, y, z};
}

glm::vec2 cartesianToLatLon(const glm::vec3 &position)
{
    float lon = glm::atan(position.y, position.x);
    float lat = glm::asin(position.z);
    return {lat, lon};
}

glm::vec2 Simulator::windSpeed(float lat, float lon) const
{
    // Q_ASSERT(lon >= -M_PI && lon < M_PI);
    const auto x =
        static_cast<int>(((lon + glm::pi<float>()) / (2.0f * glm::pi<float>())) * m_windMapWidth) % m_windMapWidth;
    // Q_ASSERT(lat >= -M_PI / 2 && lat < M_PI / 2);
    const auto y =
        m_windMapHeight - 1 -
        static_cast<int>(((lat + glm::half_pi<float>()) / glm::pi<float>()) * m_windMapHeight) % m_windMapHeight;
    return m_windMap[y * m_windMapWidth + x];
}

void Simulator::update()
{
    std::for_each(m_particles, m_particles + ParticleCount, [this](auto &particle) {
        const auto lat = particle.position.x;
        const auto lon = particle.position.y;
        auto position = latLonToCartesian(lat, lon);

        auto speed = particle.speed + 0.02f * windSpeed(lat, lon);

        // XXX check this
        auto n = glm::normalize(position);
        auto u = glm::cross(n, glm::vec3(0, 0, 1));
        auto v = glm::cross(n, u);
        position += v * speed.y + u * speed.x;
        position = glm::normalize(position);

        particle.position = cartesianToLatLon(position);

        particle.history[particle.historySize % Particle::MaxHistorySize] = particle.position;
        particle.historySize++;
    });
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
