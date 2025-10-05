#version 450

layout(location=0) out vec4 outColor;

layout(push_constant) uniform PC {
    mat4 projView;
    vec4 posSize;
    vec4 colorAlpha;
    vec4 viewport;
} pc;

float hash12(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

void main(){
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r = length(uv);
    float aSigned = pc.colorAlpha.w;
    bool  isSnow  = (aSigned >= 0.0);
    float baseA   = abs(aSigned);
    float edge = smoothstep(1.0, 0.85, 1.0 - r*r);
    float mask = 0.0;

    if (isSnow) {
        float ang = atan(uv.y, uv.x);
        float spokes = abs(cos(6.0 * ang));
        float core = smoothstep(0.25, 0.05, r);
        float arms = smoothstep(0.88, 0.98, spokes) * smoothstep(1.2, 0.2, r);
        mask = clamp(core + arms, 0.0, 1.0);
        mask *= edge;
    } else {
        vec2 wuv = uv;
        float ang = atan(wuv.y, wuv.x);
        float wobble = 0.05 * cos(5.0 * ang) * (1.0 - r);
        float n = hash12(gl_PointCoord * 17.0) * 0.1;
        float radius = 0.9 + wobble + n;
        float circle = smoothstep(radius, radius - 0.15, r);
        mask = circle * edge;
    }

    float alpha = clamp(baseA * mask, 0.0, 1.0);
    if (alpha < 0.01) discard;

    outColor = vec4(pc.colorAlpha.xyz, alpha);
}
