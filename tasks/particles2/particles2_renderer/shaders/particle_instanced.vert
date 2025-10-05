#version 460

layout(push_constant) uniform VSPC {
    mat4  projView;
    uint  useIndices;
    uint  drawCount;
    uint  colorMode;
    uint  _pad;
} pc;

struct ParticleGpu { vec4 pos_size; vec4 vel_age; vec4 col_life; };

layout(set=0, binding=0, std430) readonly buffer Particles { ParticleGpu P[]; };
layout(set=0, binding=1, std430) readonly buffer Indices   { uint        I[]; };
layout(location = 0) out float vLifeRatio;
layout(location = 1) out float vOrder;

void main()
{
    uint srcId = (pc.useIndices != 0u) ? I[gl_InstanceIndex] : gl_InstanceIndex;

    vec3  worldPos = P[srcId].pos_size.xyz;
    float size     = P[srcId].pos_size.w;
    float lifeTotal = max(P[srcId].col_life.w, 1e-6);
    float age       = P[srcId].vel_age.w;
    vLifeRatio      = clamp(1.0 - age / lifeTotal, 0.0, 1.0);
    float denom = max(1.0, float(pc.drawCount) - 1.0);
    vOrder = clamp(float(gl_InstanceIndex) / denom, 0.0, 1.0);

    gl_Position = pc.projView * vec4(worldPos, 1.0);
    gl_PointSize = size;
}
