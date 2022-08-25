#pragma once

#include "cudautils.h"

#include <cuda.h>
#include <cuda_gl_interop.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <tuple>
#include <array>

static constexpr auto ParticleCount = 80000;

struct Particle
{
    glm::vec2 initialPosition;
    glm::vec2 position; // polar
    int time, period;
    static constexpr auto MaxHistorySize = 60;
    struct PositionSpeed {
        glm::vec3 position;
        float speed;
    };
    PositionSpeed history[MaxHistorySize];
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
    CudaUtils::CuVector<Particle> m_particles;
    CudaUtils::CuVector<glm::vec2> m_windMap;
    int m_windMapWidth;
    int m_windMapHeight;
    cudaGraphicsResource *m_vboResource = nullptr;
};
