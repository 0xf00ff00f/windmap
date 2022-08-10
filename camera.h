#pragma once

#include <QObject>
#include <QVector3D>
#include <QMatrix4x4>

class Camera : public QObject
{
    Q_OBJECT
public:
    explicit Camera(QObject *parent = nullptr);
    ~Camera();

    QVector3D position() const { return m_position; }
    void setPosition(const QVector3D &position);

    QVector3D upVector() const { return m_upVector; }
    void setUpVector(const QVector3D &upVector);

    QVector3D viewCenter() const { return m_viewCenter; }
    void setViewCenter(const QVector3D &viewCenter);

    QMatrix4x4 viewMatrix() const { return m_viewMatrix; }

    QVector3D viewVector() const { return m_viewCenter - m_position; }

    void panAboutViewCenter(float angle);
    void tiltAboutViewCenter(float angle);
    void zoom(float dz);

    void translate(const QVector3D &offset);
    void rotate(float angle, const QVector3D &axis);

signals:
    void viewMatrixChanged(const QMatrix4x4 &viewMatrix);

private:
    QVector3D localToWorld(const QVector3D &v) const;
    void updateViewMatrix();

    QVector3D m_position = QVector3D(0, 0, 4);
    QVector3D m_upVector = QVector3D(0, 1, 0);
    QVector3D m_viewCenter = QVector3D(0, 0, 0);
    QMatrix4x4 m_viewMatrix;
};
