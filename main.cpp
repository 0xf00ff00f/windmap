#include "glwidget.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    GLWidget w;
    w.resize(800, 800);
    w.show();

    return app.exec();
}
