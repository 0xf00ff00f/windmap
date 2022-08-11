#version 330

layout(location = 0) in vec3 position;

uniform mat4 mvp;

out vec2 vs_texCoord;

void main(void)
{
    gl_Position = mvp * vec4(position, 1.0);
}