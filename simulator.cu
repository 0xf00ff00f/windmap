#include "simulator.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <random>
#include <cassert>

namespace
{

struct Vertex
{
    glm::vec3 position;
    glm::vec4 color;
};
}

__device__ constexpr auto TrailColor = glm::vec3(0, 1, 1);

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
                               int windMapHeight, Vertex *vertices)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < particleCount; i += stride)
    {
        auto &particle = particles[i];

        // update particle

        if (particle.time > particle.period)
        {
            particle.position = particle.initialPosition;
            particle.historySize = 0;
            particle.time = 0;
        }

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
        const auto speed = 0.02f * windSpeed;

        // XXX check this
        auto n = glm::normalize(position);
        auto u = glm::cross(n, glm::vec3(0, 0, 1));
        auto v = glm::cross(n, u);
        position += v * speed.y + u * speed.x;
        position = glm::normalize(position);

        particle.position = cartesianToLatLon(position);
        particle.history[particle.historySize % Particle::MaxHistorySize] = { position, glm::length(windSpeed) };
        ++particle.historySize;
        ++particle.time;

        // update trail vertices

        Vertex *p = &vertices[i * Particle::MaxHistorySize];
        int historySize = glm::min(particle.historySize, Particle::MaxHistorySize);
        int head = particle.historySize % Particle::MaxHistorySize;
        constexpr auto Distance = 1.005f;
        for (int i = 0; i < historySize; ++i)
        {
            head = (head + (Particle::MaxHistorySize - 1)) % Particle::MaxHistorySize;
            const auto speed = particle.history[head].speed;
            const auto position = Distance * particle.history[head].position;
            const auto alpha = speed * (1.0f - static_cast<float>(i) / (historySize - 1));
            const auto color = glm::vec4(TrailColor, alpha);
            *p++ = Vertex{position, color};
        }
        if (historySize < Particle::MaxHistorySize)
        {
            const auto position = Distance * particle.history[head].position;
            const auto color = glm::vec4(TrailColor, 0);
            const auto vertex = Vertex{position, color};
            for (int i = historySize; i < Particle::MaxHistorySize; ++i)
                *p++ = vertex;
        }
    }
}

void Simulator::update()
{
    constexpr auto BlockSize = 256;
    constexpr auto NumBlocks = (ParticleCount + BlockSize - 1) / BlockSize;

    CUDA_CHECK(cudaGraphicsMapResources(1, &m_vboResource, 0));
    Vertex *vertices;
    size_t numBytes;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&vertices), &numBytes, m_vboResource));

    updateParticle<<<NumBlocks, BlockSize>>>(m_particles.data(), ParticleCount, m_windMap.data(), m_windMapWidth,
                                             m_windMapHeight, vertices);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_vboResource, 0));
}

Simulator::Simulator(const glm::vec2 *windMap, int windMapWidth, int windMapHeight, GLuint vbo)
    : m_windMapWidth(windMapWidth)
    , m_windMapHeight(windMapHeight)
{
    // initialize particles
    m_particles.resize(ParticleCount);

    std::mt19937 eng;
    std::uniform_real_distribution<> dist(0, 1);
    std::uniform_int_distribution<> period_dist(100, 200);
    for (auto &particle : m_particles)
    {
        const auto lat = dist(eng) * glm::pi<float>() - glm::half_pi<float>();
        const auto lon = dist(eng) * 2.0f * glm::pi<float>() - glm::pi<float>();
        particle.initialPosition = particle.position = glm::vec2(lat, lon);
        particle.period = period_dist(eng);
        particle.time = 0;
    };

    // initialize wind map
    m_windMap.resize(windMapWidth * windMapHeight);
    std::copy(windMap, windMap + windMapWidth * windMapHeight, m_windMap.begin());

    // initialize VBO
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_vboResource, vbo, cudaGraphicsMapFlagsWriteDiscard));
}

Simulator::~Simulator() = default;
