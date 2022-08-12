#pragma once

#include <cuda.h>
#include <cuda_gl_interop.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct Particle
{
    glm::vec2 position; // polar
    glm::vec2 speed;    // polar

    static constexpr auto MaxHistorySize = 20;

    glm::vec2 history[MaxHistorySize];
    int historySize = 0;
};

class Simulator
{
public:
    Simulator(const glm::vec2 *windMap, int windMapWidth, int windMapHeight, GLuint vbo);
    ~Simulator();

    Simulator(const Simulator &) = delete;
    Simulator &operator=(const Simulator &) = delete;

    void update();

    static constexpr auto ParticleCount = 20000;

    Particle *m_particles = nullptr;

private:
    glm::vec2 windSpeed(float lat, float lon) const;

    glm::vec2 *m_windMap = nullptr;
    int m_windMapWidth;
    int m_windMapHeight;
    cudaGraphicsResource *m_vboResource = nullptr;
};
