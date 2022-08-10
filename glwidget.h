#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QMatrix4x4>

class Camera;
class QOpenGLShaderProgram;
class QOpenGLTexture;

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
    void initProgram();
    void initBuffer();

    QOpenGLShaderProgram *m_program = nullptr;
    QOpenGLVertexArrayObject m_vaoEarth;
    QOpenGLBuffer m_vboEarth;
    QOpenGLTexture *m_earthTexture = nullptr;
    Camera *m_camera;
    QMatrix4x4 m_model;
    QMatrix4x4 m_projection;
    int m_mvpUniform = -1;
    struct Vertex
    {
        QVector3D position;
        QVector2D texCoord;
    };
    std::vector<Vertex> m_earthVertices;
    QPoint m_lastMousePos;
    enum class CameraCommand
    {
        None,
        Rotate,
        Pan
    };
    CameraCommand m_cameraCommand = CameraCommand::None;
};
