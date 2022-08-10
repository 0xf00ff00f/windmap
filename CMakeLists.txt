cmake_minimum_required(VERSION 3.1.0)

project(test)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 COMPONENTS Core Gui OpenGL OpenGLWidgets Widgets REQUIRED)

add_executable(test
    main.cpp
    camera.cpp
    camera.h
    glwidget.cpp
    glwidget.h
    test.qrc
)

target_link_libraries(test PUBLIC
    Qt::Core
    Qt::Gui
    Qt::OpenGL
    Qt::OpenGLWidgets
    Qt::Widgets
)
