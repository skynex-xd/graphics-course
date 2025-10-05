#version 460

layout(push_constant) uniform VSPC {
    mat4  projView;
    uint  useIndices;
    uint  drawCount;
    uint  colorMode;
    uint  _pad;
} pc;

layout(location = 0) in  float vLifeRatio;
layout(location = 1) in  float vOrder;
layout(location = 0) out vec4  outColor;

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

void main()
{
    vec2  uv   = gl_PointCoord * 2.0 - 1.0;
    float r    = length(uv);
    float mask = smoothstep(1.0, 0.8, 1.0 - r);
    float t = pow(clamp(vOrder, 0.0, 1.0), 0.35);

    vec3 rgb;
    if (pc.colorMode == 2u) {
        rgb = hsv2rgb(vec3(0.05 + 0.90 * t, 0.85, 1.0));
    } else if (pc.colorMode == 1u) {
        rgb = mix(vec3(0.08), vec3(1.0, 0.0, 0.0), t);
    } else {
        rgb = vec3(1.0);
    }

    outColor = vec4(rgb, mask * vLifeRatio);
    if (outColor.a <= 0.001) discard;
}
