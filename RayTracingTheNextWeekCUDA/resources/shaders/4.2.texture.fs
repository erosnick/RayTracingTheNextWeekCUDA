#version 330 core
out vec4 FragColor;

in vec3 ourColor;
in vec2 TexCoord;

// texture samplers
uniform sampler2D result;

void main()
{
	// linearly interpolate between both textures (80% container, 20% awesomeface)
	FragColor = texture(result, TexCoord);
}