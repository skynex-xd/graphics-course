#version 450

layout(push_constant) uniform PC {
    mat4 projView;
    vec4 posSize;
    vec4 colorAlpha;
    vec4 viewport;
} pc;

void main() {
    vec3  pos = pc.posSize.xyz;
    float sz  = pc.posSize.w;
    gl_Position = pc.projView * vec4(pos, 1.0);
    gl_PointSize = max(1.0, sz);
}
