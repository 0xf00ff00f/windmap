#version 330

uniform sampler2D sampler;

in vec2 vs_texCoord;

out vec4 fragColor;

void main(void)
{
    fragColor = vec4(texture2D(sampler, vs_texCoord).rgb, 1.0);
}
