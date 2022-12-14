#pragma once

#define CUDA_PARTICLES

#if defined(CUDA_PARTICLES)
#include "simulator.h"
#else
static constexpr auto ParticleCount = 20000;
#endif

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QMatrix4x4>

class Camera;
class QOpenGLShaderProgram;
class QOpenGLTexture;
class QTimer;

#if !defined(CUDA_PARTICLES)
struct Particle
{
    QVector2D position; // polar
    int lifetime;
    void reset();

    static constexpr auto MaxHistorySize = 20;
    std::array<QVector2D, MaxHistorySize> history;
    int historySize = 0;
};
#endif

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
    explicit GLWidget(QWidget *parent = nullptr);
    ~GLWidget();

protected:
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void initializeGL() override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    void initParticles();
    void initProgram();
    void initBuffers();
    void updateParticles();
    QVector2D windSpeed(float lat, float lon) const;

    QOpenGLShaderProgram *m_programEarth = nullptr;
    int m_mvpUniformEarth = -1;
    QOpenGLVertexArrayObject m_vaoEarth;
    QOpenGLBuffer m_vboEarth;
    QOpenGLTexture *m_earthTexture = nullptr;
    QOpenGLShaderProgram *m_programParticles = nullptr;
    int m_mvpUniformParticles;
    QOpenGLVertexArrayObject m_vaoParticles;
    QOpenGLBuffer m_vboParticleVerts;
    QOpenGLBuffer m_vboParticleIndices;
    Camera *m_camera;
    QMatrix4x4 m_model;
    QMatrix4x4 m_projection;
    struct Vertex
    {
        QVector3D position;
        QVector2D texCoord;
    };
    std::vector<Vertex> m_earthVertices;
#if defined(CUDA_PARTICLES)
    std::unique_ptr<Simulator> m_simulator;
#else
    std::vector<Particle> m_particles;
#endif
    struct ParticleVertex
    {
        QVector3D position;
        QVector4D color;
    };
    QPoint m_lastMousePos;
    enum class CameraCommand
    {
        None,
        Rotate,
        Pan
    };
    CameraCommand m_cameraCommand = CameraCommand::None;
    QTimer *m_timer;
    QImage m_windImage;
};
