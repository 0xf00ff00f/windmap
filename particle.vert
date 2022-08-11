#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

uniform mat4 mvp;

out vec2 vs_texCoord;
out vec4 vs_color;

void main(void)
{
    gl_Position = mvp * vec4(position, 1.0);
    vs_color = color;
}
