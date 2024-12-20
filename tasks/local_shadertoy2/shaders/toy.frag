#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 fragColor;

layout(binding = 0) uniform sampler2D iChannel0;
layout(binding = 1) uniform sampler2D iChannel1;

const vec3 eye = vec3(0, 0, 5);
const int maxSteps = 100;
const float eps = 0.01;

mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}

mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

float sdRoundBox(vec3 p, vec3 b, float r) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

float smin(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

float sdf(in vec3 p, in mat3 m, out bool isInner) {
    vec3 q = m * p;

    // Внешний квадратный контур
    float outerBox = sdRoundBox(q, vec3(1.2, 1.2, 0.3), 0.1);
    float innerBox = sdRoundBox(q, vec3(0.7, 0.7, 0.4), 0.1);
    float ring = max(outerBox, -innerBox);

    // Внутренний объект
    float innerObject = sdRoundBox(q, vec3(0.3), 0.1);

    isInner = innerObject < ring;
    return smin(ring, innerObject, 0.5);
}

float sdf(in vec3 p, in mat3 m) {
    bool dummy;
    return sdf(p, m, dummy);
}

vec3 trace(in vec3 from, in vec3 dir, out bool hit, in mat3 m, out bool isInner) {
    vec3 p = from;
    float totalDist = 0.0;
    hit = false;

    for(int steps = 0; steps < maxSteps; steps++) {
        float dist = sdf(p, m, isInner);

        if(dist < eps) {
            hit = true;
            break;
        }

        totalDist += dist;
        if(totalDist > 20.0) break;
        p += dist * dir;
    }

    return p;
}

vec3 generateNormal(vec3 p, in mat3 m) {
    vec2 e = vec2(eps, 0.0);
    return normalize(vec3(
        sdf(p + e.xyy, m) - sdf(p - e.xyy, m),
        sdf(p + e.yxy, m) - sdf(p - e.yxy, m),
        sdf(p + e.yyx, m) - sdf(p - e.yyx, m)
    ));
}

vec3 getRainbowColor(vec3 p) {
    float t = length(p) * 2.0;
    return 0.5 + 0.5 * cos(6.28318 * (t + vec3(0.0, 0.33, 0.67)));
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec2 resolution = vec2(1280.0, 720.0);
    vec2 mouse_pos = vec2(10.0, 25.0);

    fragColor = vec4(0.0, 0.0, 0.0, 1.0);

    bool hit;
    bool isInner;

    vec2 mouseNorm = mouse_pos.xy / resolution.xy;
    vec2 angles = (mouseNorm - 0.5) * 5.0;
    mat3 m = rotateY(angles.x) * rotateX(angles.y);

    vec2 uv = 2.0 * (vec2(fragCoord) / resolution.xy - 0.5);
    uv.x *= resolution.x/resolution.y;

    vec3 dir = normalize(vec3(uv, -1.5));
    vec3 p = trace(eye, dir, hit, m, isInner);

    if(hit) {
        vec3 n = generateNormal(p, m);
        vec3 viewDir = normalize(eye - p);
        float fresnel = pow(1.0 - max(0.0, dot(n, viewDir)), 2.0);

        vec3 norm = abs(n);
        norm /= (norm.x + norm.y + norm.z);

        if(isInner) {
            vec3 rainbowColor = getRainbowColor(p);
            fragColor = vec4(rainbowColor, 1.0);
            vec3 albedo =
                norm.x * texture(iChannel1, p.yz).rgb +
                norm.y * texture(iChannel1, p.xz).rgb +
                norm.z * texture(iChannel1, p.xy).rgb;
            fragColor = vec4(albedo, 1.0);
        } else {
            vec3 baseColor = mix(
                vec3(0.0, 0.8, 0.8),
                vec3(0.0, 0.5, 1.0),
                fresnel
            );
            vec3 albedo =
                norm.x * texture(iChannel0, p.yz).rgb +
                norm.y * texture(iChannel0, p.xz).rgb +
                norm.z * texture(iChannel0, p.xy).rgb;
            fragColor = vec4(baseColor * 0.5 + albedo * 0.5, 1.0);
        }
    }
}
