#pragma once

#include <cuda.h>
#include <cuda_gl_interop.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

static constexpr auto ParticleCount = 20000;

struct Particle
{
    glm::vec2 initialPosition;
    glm::vec2 position; // polar
    int time, period;

    static constexpr auto MaxHistorySize = 40;

    glm::vec3 history[MaxHistorySize];
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

private:
    glm::vec2 windSpeed(float lat, float lon) const;

    Particle *m_particles = nullptr;
    glm::vec2 *m_windMap = nullptr;
    int m_windMapWidth;
    int m_windMapHeight;
    cudaGraphicsResource *m_vboResource = nullptr;
};
