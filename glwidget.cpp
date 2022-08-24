#include "glwidget.h"

#include "camera.h"

#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QMouseEvent>
#include <QRandomGenerator>
#include <QTimer>

#include <cmath>
#include <set>

namespace
{

// latitude: n/s, -pi/2 to pi/2
// longitude: e/w, -pi to pi
QVector3D latLonToCartesian(float lat, float lon)
{
    const auto r = std::cos(lat);
    const auto x = r * std::cos(lon);
    const auto y = r * std::sin(lon);
    const auto z = std::sin(lat);
    return {x, y, z};
}

QVector2D cartesianToLatLon(const QVector3D &position)
{
    float lon = std::atan2(position.y(), position.x());
    float lat = std::asin(position.z());
    return {lat, lon};
}

constexpr auto MaxParticleVertices = ParticleCount * Particle::MaxHistorySize;
constexpr auto MaxParticleIndices = ParticleCount * 2 * (Particle::MaxHistorySize - 1);
}

#if !defined(CUDA_PARTICLES)
void Particle::reset()
{
    auto *rng = QRandomGenerator::global();
    const auto lat = rng->generateDouble() * M_PI - M_PI / 2;
    const auto lon = rng->generateDouble() * 2.0 * M_PI - M_PI;
    position = QVector2D(lat, lon);
    lifetime = rng->bounded(100, 200);
    historySize = 0;
}
#endif

GLWidget::GLWidget(QWidget *parent)
    : QOpenGLWidget(parent)
    , m_vboParticleIndices(QOpenGLBuffer::IndexBuffer)
    , m_camera(new Camera(this))
    , m_timer(new QTimer(this))
    , m_windImage(":/wind.png")
{
    Q_ASSERT(m_model.isIdentity());
    Q_ASSERT(m_projection.isIdentity());

    connect(m_camera, &Camera::viewMatrixChanged, this, [this] { update(); });

    m_timer->setInterval(10);
    connect(m_timer, &QTimer::timeout, this, [this] {
        updateParticles();
        update();
    });
    m_timer->start();
}

GLWidget::~GLWidget()
{
    makeCurrent();
    delete m_earthTexture;
    m_vboEarth.destroy();
    m_vaoEarth.destroy();
    delete m_programEarth;

    m_vboParticleVerts.destroy();
    m_vboParticleIndices.destroy();
    m_vaoParticles.destroy();
    delete m_programParticles;
    doneCurrent();
}

void GLWidget::paintGL()
{
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(2.0);

    QMatrix4x4 mvp = m_projection * m_camera->viewMatrix() * m_model;

    // earth

    m_earthTexture->bind();
    m_programEarth->bind();
    m_programEarth->setUniformValue(m_mvpUniformEarth, mvp);
    {
        QOpenGLVertexArrayObject::Binder vaoBinder(&m_vaoEarth);
        glDrawArrays(GL_TRIANGLES, 0, m_earthVertices.size());
    }

    // particles

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    m_programParticles->bind();
    m_programParticles->setUniformValue(m_mvpUniformParticles, mvp);

#if !defined(CUDA_PARTICLES)
    m_vboParticleVerts.bind();
    auto *particleVertices = reinterpret_cast<ParticleVertex *>(
        m_vboParticleVerts.mapRange(0, MaxParticleVertices * sizeof(ParticleVertex), QOpenGLBuffer::RangeWrite));
    Q_ASSERT(particleVertices);
    int vertexCount = 0;

    for (const auto &particle : m_particles)
    {
        int historySize = std::min(particle.historySize, Particle::MaxHistorySize);
        int head = particle.historySize % Particle::MaxHistorySize;
        for (int i = 0; i < historySize; ++i)
        {
            head = (head + (Particle::MaxHistorySize - 1)) % Particle::MaxHistorySize;
            const auto alpha = 1.0f - static_cast<float>(i) / (historySize - 1);
            const auto &p = particle.history[head];
            const auto position = 1.01f * latLonToCartesian(p.x(), p.y());
            const auto color = QVector4D(1, 0, 0, alpha);
            particleVertices[vertexCount++] = ParticleVertex{position, color};
        }
        for (int i = historySize; i < Particle::MaxHistorySize; ++i)
        {
            const auto &p = particle.history[head];
            const auto position = 1.01f * latLonToCartesian(p.x(), p.y());
            const auto color = QVector4D(1, 0, 0, 0);
            particleVertices[vertexCount++] = ParticleVertex{position, color};
        }
    }
    Q_ASSERT(vertexCount == MaxParticleVertices);

    m_vboParticleVerts.unmap();
    m_vboParticleVerts.release();
#endif

    {
        QOpenGLVertexArrayObject::Binder vaoBinder(&m_vaoParticles);
        glDrawElements(GL_LINES, MaxParticleIndices, GL_UNSIGNED_INT, nullptr);
    }

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
}

void GLWidget::initializeGL()
{
    initializeOpenGLFunctions();

    initProgram();
    initBuffers();
    initParticles();
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    m_lastMousePos = event->pos();
    m_cameraCommand = [event] {
        switch (event->button())
        {
        case Qt::LeftButton:
#if 0
            return CameraCommand::Pan;
        case Qt::RightButton:
#endif
            return CameraCommand::Rotate;
        default:
            return CameraCommand::None;
        }
    }();
    event->accept();
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    const auto offset = QVector2D(event->pos() - m_lastMousePos);

    switch (m_cameraCommand)
    {
    case CameraCommand::Rotate: {
        constexpr auto RotateSpeed = 0.2f;
        m_camera->panAboutViewCenter(-RotateSpeed * offset.x());
        m_camera->tiltAboutViewCenter(RotateSpeed * offset.y());
        break;
    }
    case CameraCommand::Pan: {
        constexpr auto PanSpeed = 0.01f;
        const auto l = (m_camera->viewCenter() - m_camera->position()).length();
        m_camera->translate(PanSpeed * QVector3D(-offset.x(), offset.y(), 0));
        break;
    }
    default:
        break;
    }

    m_lastMousePos = event->pos();
    event->accept();
}

void GLWidget::wheelEvent(QWheelEvent *event)
{
    constexpr auto ZoomSpeed = 0.01f;
    const auto dz = ZoomSpeed * event->angleDelta().y();
    m_camera->zoom(dz);
}

void GLWidget::initParticles()
{
#if defined(CUDA_PARTICLES)
    std::vector<glm::vec2> windMap;
    windMap.reserve(m_windImage.width() * m_windImage.height());
    for (int y = 0; y < m_windImage.height(); ++y)
    {
        for (int x = 0; x < m_windImage.width(); ++x)
        {
            const auto sample = m_windImage.pixel(x, y);
            windMap.push_back(glm::vec2(static_cast<float>(qRed(sample)) / 255.0 - 0.5,
                                        static_cast<float>(qGreen(sample)) / 255.0 - 0.5));
        }
    }

    m_simulator.reset(
        new Simulator(windMap.data(), m_windImage.width(), m_windImage.height(), m_vboParticleVerts.bufferId()));
#else
    m_particles.resize(ParticleCount);
    for (auto &particle : m_particles)
        particle.reset();
#endif
}

void GLWidget::initProgram()
{
    auto createProgram = [](const QString &vs, const QString &fs) {
        auto *program = new QOpenGLShaderProgram;
        if (!program->addShaderFromSourceFile(QOpenGLShader::Vertex, vs))
            qWarning() << "Failed to add vertex shader:" << program->log();
        if (!program->addShaderFromSourceFile(QOpenGLShader::Fragment, fs))
            qWarning() << "Failed to add fragment shader:" << program->log();
        if (!program->link())
            qWarning() << "Failed to link program";
        return program;
    };

    m_programEarth = createProgram(":/earth.vert", ":/earth.frag");
    m_mvpUniformEarth = m_programEarth->uniformLocation("mvp");

    m_programParticles = createProgram(":/particle.vert", ":/particle.frag");
    m_mvpUniformParticles = m_programParticles->uniformLocation("mvp");
}

void GLWidget::initBuffers()
{
    // texture
    m_earthTexture = new QOpenGLTexture(QImage(":/earth.png").mirrored());

    // sphere mesh

    constexpr auto kRings = 20;
    constexpr auto kSlices = 20;

    for (int i = 0; i < kRings; ++i)
    {
        for (int j = 0; j < kSlices; ++j)
        {
            auto vertex = [](int i, int j) -> Vertex {
                const auto lat = i * M_PI / kRings - M_PI_2;
                const auto lon = j * 2.0f * M_PI / kSlices - M_PI;
                const auto position = latLonToCartesian(lat, lon);

                const auto u = static_cast<float>(i) / kRings;
                const auto v = static_cast<float>(j) / kSlices;
                const auto texCoord = QVector2D(v, u);

                return {position, texCoord};
            };

            const auto v0 = vertex(i, j);
            const auto v1 = vertex(i + 1, j);
            const auto v2 = vertex(i + 1, j + 1);
            const auto v3 = vertex(i, j + 1);

            m_earthVertices.push_back(v0);
            m_earthVertices.push_back(v1);
            m_earthVertices.push_back(v2);

            m_earthVertices.push_back(v2);
            m_earthVertices.push_back(v3);
            m_earthVertices.push_back(v0);
        }
    }

    {
        m_vaoEarth.create();
        m_vaoEarth.bind();

        m_vboEarth.create();
        m_vboEarth.bind();
        m_vboEarth.allocate(m_earthVertices.data(), m_earthVertices.size() * sizeof(Vertex));

        m_programEarth->enableAttributeArray(0); // position
        m_programEarth->enableAttributeArray(1); // texCoord
        m_programEarth->setAttributeBuffer(0, GL_FLOAT, offsetof(Vertex, position), 3, sizeof(Vertex));
        m_programEarth->setAttributeBuffer(1, GL_FLOAT, offsetof(Vertex, texCoord), 2, sizeof(Vertex));

        m_vaoEarth.release();
        m_vboEarth.release();
    }

    {
        m_vaoParticles.create();
        m_vaoParticles.bind();

        m_vboParticleVerts.create();
        m_vboParticleVerts.bind();
        m_vboParticleVerts.allocate(MaxParticleVertices * sizeof(ParticleVertex));

        m_vboParticleIndices.create();
        m_vboParticleIndices.bind();

        std::vector<GLuint> indices;
        indices.reserve(MaxParticleIndices);
        for (int i = 0; i < ParticleCount; ++i)
        {
            const auto base = i * Particle::MaxHistorySize;
            for (int j = 0; j < Particle::MaxHistorySize - 1; ++j)
            {
                indices.push_back(base + j);
                indices.push_back(base + j + 1);
            }
        }
        Q_ASSERT(indices.size() == MaxParticleIndices);
        m_vboParticleIndices.allocate(indices.data(), MaxParticleIndices * sizeof(GLuint));

        m_programParticles->enableAttributeArray(0); // position
        m_programParticles->enableAttributeArray(1); // color
        m_programParticles->setAttributeBuffer(0, GL_FLOAT, offsetof(ParticleVertex, position), 3,
                                               sizeof(ParticleVertex));
        m_programParticles->setAttributeBuffer(1, GL_FLOAT, offsetof(ParticleVertex, color), 4, sizeof(ParticleVertex));

        m_vaoParticles.release();
        m_vboParticleIndices.release();
        m_vboParticleVerts.release();
    }

    {
        QOpenGLVertexArrayObject::Binder vaoBinder(&m_vaoParticles);
        int binding;
        glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &binding);
        Q_ASSERT(binding == m_vboParticleIndices.bufferId());
    }
}

void GLWidget::resizeGL(int w, int h)
{
    m_projection.setToIdentity();
    m_projection.perspective(45.f, static_cast<qreal>(w) / static_cast<qreal>(h), 0.1f, 100.0f);
}

void GLWidget::updateParticles()
{
#if defined(CUDA_PARTICLES)
    m_simulator->update();
#else
    for (auto &particle : m_particles)
    {
        if (--particle.lifetime < 0)
        {
            particle.reset();
            continue;
        }

        const auto lat = particle.position.x();
        const auto lon = particle.position.y();
        auto position = latLonToCartesian(lat, lon);

        const auto speed = 0.02 * windSpeed(lat, lon);

        // XXX check this
        auto n = position.normalized();
        auto u = QVector3D::crossProduct(n, QVector3D(0, 0, 1));
        auto v = QVector3D::crossProduct(n, u);
        position += v * speed.y() + u * speed.x();
        position = position.normalized();

        particle.position = cartesianToLatLon(position);

        particle.history[particle.historySize % Particle::MaxHistorySize] = particle.position;
        particle.historySize++;
    }
#endif
}

QVector2D GLWidget::windSpeed(float lat, float lon) const
{
    // Q_ASSERT(lon >= -M_PI && lon < M_PI);
    const auto x = static_cast<int>(((lon + M_PI) / (2.0 * M_PI)) * m_windImage.width()) % m_windImage.width();
    // Q_ASSERT(lat >= -M_PI / 2 && lat < M_PI / 2);
    const auto y = m_windImage.height() - 1 -
                    static_cast<int>(((lat + M_PI / 2) / M_PI) * m_windImage.height()) % m_windImage.height();
    const auto sample = m_windImage.pixel(x, y);
    return QVector2D(static_cast<float>(qRed(sample)) / 255.0 - 0.5, static_cast<float>(qGreen(sample)) / 255.0 - 0.5);
}
