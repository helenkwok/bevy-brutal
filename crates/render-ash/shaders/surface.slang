// World-space surfaces (terrain chunks, DEM chunks, voxel meshes): position + normal vertices, no
// instancing. Uses the same push block as the instanced pass (only `frame` is read, in the vertex
// stage: the push range is VERTEX-only, so the palette range reaches the fragment as a flat varying).
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
    [[vk::location(2)]] nointerpolation float2 range;
}

[shader("vertex")]
VSOut vertexMain(VSIn v) {
    VSOut o;
    o.position = mul_cols(pc.frame.view_proj, float4(v.pos, 1.0));
    o.normal = v.normal;
    o.world = v.pos;
    o.range = pc.frame.surface_range;
    return o;
}

// Noise terrain (±40 m): sand below 0, grass, snow from 18-28 m.
float3 noisePalette(float h) {
    float3 sand = float3(0.76, 0.70, 0.50);
    float3 grass = float3(0.25, 0.45, 0.18);
    float3 snow = float3(0.92, 0.93, 0.96);
    float3 base = lerp(sand, grass, smoothstep(-6.0, 0.0, h));
    return lerp(base, snow, smoothstep(18.0, 28.0, h));
}

// Real heights (metres, sea level = 0): water, then a hypsometric ramp over t = (h - lo) / (hi - lo).
// Coast sand -> dry plains -> pasture -> scrub/forest -> bare high ground. No snow.
float3 demPalette(float h, float2 range) {
    float3 water = float3(0.10, 0.20, 0.30);
    float3 sand = float3(0.70, 0.66, 0.52);
    float3 plains = float3(0.56, 0.55, 0.38);
    float3 pasture = float3(0.38, 0.47, 0.27);
    float3 scrub = float3(0.25, 0.34, 0.20);
    float3 bare = float3(0.48, 0.44, 0.36);
    float t = saturate((h - range.x) / max(range.y - range.x, 1.0));
    // sand only in the first ~4 m above the shore, independent of the range
    float3 c = lerp(sand, plains, smoothstep(0.5, 4.0, h));
    c = lerp(c, pasture, smoothstep(0.10, 0.35, t));
    c = lerp(c, scrub, smoothstep(0.35, 0.70, t));
    c = lerp(c, bare, smoothstep(0.80, 1.00, t));
    return lerp(water, c, smoothstep(0.0, 0.5, h));
}

[shader("fragment")]
float4 fragmentMain(VSOut i) : SV_Target0 {
    float3 n = normalize(i.normal);
    float3 sun = normalize(float3(0.45, 0.8, 0.35));
    float slope = 1.0 - saturate(n.y);
    float h = i.world.y;
    float3 rock = float3(0.42, 0.40, 0.38);
    bool dem = i.range.y > i.range.x;
    float3 base = dem ? demPalette(h, i.range) : noisePalette(h);
    // DEM: steeper threshold (31 m+ cells smooth real slopes; quarry walls / escarpments only)
    float2 rs = dem ? float2(0.45, 0.70) : float2(0.35, 0.6);
    base = lerp(base, rock, smoothstep(rs.x, rs.y, slope) * (dem && h < 0.5 ? 0.0 : 1.0));
    float lambert = max(dot(n, sun), 0.0);
    float sky = 0.5 + 0.5 * n.y;                  // hemispheric ambient (caves/overhangs get less)
    float3 c = base * (0.18 * sky + 0.82 * lambert) + float3(0.03, 0.04, 0.06) * sky;
    return float4(c, 1.0);
}
