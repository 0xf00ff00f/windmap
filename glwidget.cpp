#include "glwidget.h"

#include "camera.h"

#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QMouseEvent>

#include <cmath>
#include <set>

GLWidget::GLWidget(QWidget *parent)
    : QOpenGLWidget(parent)
    , m_camera(new Camera(this))
{
    Q_ASSERT(m_model.isIdentity());
    Q_ASSERT(m_projection.isIdentity());

    connect(m_camera, &Camera::viewMatrixChanged, this, [this] { update(); });
}

GLWidget::~GLWidget()
{
    makeCurrent();
    delete m_earthTexture;
    m_vboEarth.destroy();
    m_vaoEarth.destroy();
    delete m_program;
    doneCurrent();
}

void GLWidget::paintGL()
{
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);

    QMatrix4x4 mvp = m_projection * m_camera->viewMatrix() * m_model;

    m_earthTexture->bind();
    m_program->bind();
    m_program->setUniformValue(m_mvpUniform, mvp);

    {
        QOpenGLVertexArrayObject::Binder vaoBinder(&m_vaoEarth);
        glDrawArrays(GL_TRIANGLES, 0, m_earthVertices.size());
    }
}

void GLWidget::initializeGL()
{
    initializeOpenGLFunctions();

    initProgram();
    initBuffer();
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

void GLWidget::initProgram()
{
    m_program = new QOpenGLShaderProgram;
    if (!m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/earth.vert"))
        qWarning() << "Failed to add vertex shader:" << m_program->log();
    if (!m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/earth.frag"))
        qWarning() << "Failed to add fragment shader:" << m_program->log();
    if (!m_program->link())
        qWarning() << "Failed to link program";
    m_mvpUniform = m_program->uniformLocation("mvp");
}

void GLWidget::initBuffer()
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
                const auto phi = i * M_PI / kRings - M_PI_2;
                const auto theta = j * 2.0f * M_PI / kSlices;
                const auto r = std::cos(phi);
                const auto x = r * std::cos(theta);
                const auto y = r * std::sin(theta);
                const auto z = std::sin(phi);
                const auto position = QVector3D(x, y, z);

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

    m_vaoEarth.create();
    QOpenGLVertexArrayObject::Binder vaoBinder(&m_vaoEarth);

    m_vboEarth.create();
    m_vboEarth.bind();
    m_vboEarth.allocate(m_earthVertices.data(), m_earthVertices.size() * sizeof(Vertex));

    m_program->enableAttributeArray(0); // position
    m_program->enableAttributeArray(1); // texCoord
    m_program->setAttributeBuffer(0, GL_FLOAT, offsetof(Vertex, position), 3, sizeof(Vertex));
    m_program->setAttributeBuffer(1, GL_FLOAT, offsetof(Vertex, texCoord), 2, sizeof(Vertex));
    m_vboEarth.release();
}

void GLWidget::resizeGL(int w, int h)
{
    m_projection.setToIdentity();
    m_projection.perspective(45.f, static_cast<qreal>(w) / static_cast<qreal>(h), 0.1f, 100.0f);
}
