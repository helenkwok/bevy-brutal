// World-space surfaces (terrain chunks, voxel meshes): position + normal vertices, no instancing.
// Uses the same push block as the instanced pass (only `frame` is read).
import common;

[[vk::push_constant]] ConstantBuffer<Push> pc;

struct VSIn {
    [[vk::location(0)]] float3 pos;
    [[vk::location(1)]] float3 normal;
}

struct VSOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 normal;
    [[vk::location(1)]] float3 world;
}

[shader("vertex")]
VSOut vertexMain(VSIn v) {
    VSOut o;
    o.position = mul_cols(pc.frame.view_proj, float4(v.pos, 1.0));
    o.normal = v.normal;
    o.world = v.pos;
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOut i) : SV_Target0 {
    float3 n = normalize(i.normal);
    float3 sun = normalize(float3(0.45, 0.8, 0.35));
    float slope = 1.0 - saturate(n.y);
    float h = i.world.y;
    float3 sand = float3(0.76, 0.70, 0.50);
    float3 grass = float3(0.25, 0.45, 0.18);
    float3 rock = float3(0.42, 0.40, 0.38);
    float3 snow = float3(0.92, 0.93, 0.96);
    float3 base = lerp(sand, grass, smoothstep(-6.0, 0.0, h));
    base = lerp(base, snow, smoothstep(18.0, 28.0, h));
    base = lerp(base, rock, smoothstep(0.35, 0.6, slope));
    float lambert = max(dot(n, sun), 0.0);
    float sky = 0.5 + 0.5 * n.y;                  // hemispheric ambient (caves/overhangs get less)
    float3 c = base * (0.18 * sky + 0.82 * lambert) + float3(0.03, 0.04, 0.06) * sky;
    return float4(c, 1.0);
}
