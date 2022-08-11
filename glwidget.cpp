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
// longitude: e/w, 0 to 2*pi
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

}

GLWidget::GLWidget(QWidget *parent)
    : QOpenGLWidget(parent)
    , m_camera(new Camera(this))
    , m_timer(new QTimer(this))
    , m_windImage(":/wind.png")
{
    Q_ASSERT(m_model.isIdentity());
    Q_ASSERT(m_projection.isIdentity());

    initParticles();

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

    m_vboParticles.destroy();
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
    m_programParticles->bind();
    m_programParticles->setUniformValue(m_mvpUniformParticles, mvp);

    m_vboParticles.bind();
    auto *particleVertices = reinterpret_cast<QVector3D *>(
        m_vboParticles.mapRange(0, m_particles.size() * sizeof(QVector3D), QOpenGLBuffer::RangeWrite));
    Q_ASSERT(particleVertices);
    std::transform(m_particles.begin(), m_particles.end(), particleVertices, [](const Particle &particle) {
        return latLonToCartesian(particle.position.x(), particle.position.y());
    });
    m_vboParticles.unmap();
    m_vboParticles.release();
    {
        QOpenGLVertexArrayObject::Binder vaoBinder(&m_vaoParticles);
        glDrawArrays(GL_POINTS, 0, m_particles.size());
    }
}

void GLWidget::initializeGL()
{
    initializeOpenGLFunctions();

    initProgram();
    initBuffers();
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    m_lastMousePos = event->pos();
    m_cameraCommand = [event] {
        switch (event->button())
        {
        case Qt::LeftButton:
            return CameraCommand::Pan;
        case Qt::RightButton:
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

void GLWidget::Particle::reset()
{
    auto *rng = QRandomGenerator::global();
    const auto lat = rng->generateDouble() * M_PI - M_PI / 2;
    const auto lon = rng->generateDouble() * 2.0 * M_PI - M_PI;
    position = QVector2D(lat, lon);
    speed = QVector2D(0, 0);
    lifetime = rng->bounded(200, 500);
}

void GLWidget::initParticles()
{
    constexpr auto kParticleCount = 20000;
    m_particles.resize(kParticleCount);
    for (auto &particle : m_particles)
        particle.reset();
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
        QOpenGLVertexArrayObject::Binder vaoBinder(&m_vaoEarth);

        m_vboEarth.create();
        m_vboEarth.bind();
        m_vboEarth.allocate(m_earthVertices.data(), m_earthVertices.size() * sizeof(Vertex));

        m_programEarth->enableAttributeArray(0); // position
        m_programEarth->enableAttributeArray(1); // texCoord
        m_programEarth->setAttributeBuffer(0, GL_FLOAT, offsetof(Vertex, position), 3, sizeof(Vertex));
        m_programEarth->setAttributeBuffer(1, GL_FLOAT, offsetof(Vertex, texCoord), 2, sizeof(Vertex));

        m_vboEarth.release();
    }

    {
        m_vaoParticles.create();
        QOpenGLVertexArrayObject::Binder vaoBinder(&m_vaoParticles);

        m_vboParticles.create();
        m_vboParticles.bind();
        m_vboParticles.allocate(m_particles.size() * sizeof(QVector3D));

        m_programParticles->enableAttributeArray(0);                                  // position
        m_programParticles->setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(QVector3D)); // position

        m_vboParticles.release();
    }
}

void GLWidget::resizeGL(int w, int h)
{
    m_projection.setToIdentity();
    m_projection.perspective(45.f, static_cast<qreal>(w) / static_cast<qreal>(h), 0.1f, 100.0f);
}

void GLWidget::updateParticles()
{
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

        auto speed = particle.speed + 0.02 * windSpeed(lat, lon);

        // XXX check this
        auto n = position.normalized();
        auto u = QVector3D::crossProduct(n, QVector3D(0, 0, 1));
        auto v = QVector3D::crossProduct(n, u);
        position += v * speed.y() + u * speed.x();
        position = position.normalized();

        particle.position = cartesianToLatLon(position);
    }
}

QVector2D GLWidget::windSpeed(float lat, float lon) const
{
    // Q_ASSERT(lon >= -M_PI && lon < M_PI);
    const float x = static_cast<int>(((lon + M_PI) / (2.0 * M_PI)) * m_windImage.width()) % m_windImage.width();
    // Q_ASSERT(lat >= -M_PI / 2 && lat < M_PI / 2);
    const float y = static_cast<int>(((lat + M_PI / 2) / M_PI) * m_windImage.height()) % m_windImage.height();
    const auto sample = m_windImage.pixel(x, y);
    return QVector2D(static_cast<float>(qGreen(sample)) / 255.0 - 0.5, static_cast<float>(qRed(sample)) / 255.0 - 0.5);
}
