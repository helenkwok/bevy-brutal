import common;

[[vk::push_constant]] ConstantBuffer<Push> pc;

struct VSIn {
    [[vk::location(0)]] float3 pos;
    [[vk::location(1)]] float3 normal;
}

struct VSOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 normal;
    [[vk::location(1)]] nointerpolation uint material;
}

[shader("vertex")]
VSOut vertexMain(VSIn v, uint instance: SV_VulkanInstanceID) {
    // SV_VulkanInstanceID == gl_InstanceIndex, which includes firstInstance (= mesh_base),
    // so it indexes the compacted `visible` list directly. (SV_InstanceID would exclude it.)
    uint idx = pc.visible[instance];
    Instance inst = pc.instances[idx];
    float4 world = mul_cols(inst.model, float4(v.pos, 1.0));
    VSOut o;
    o.position = mul_cols(pc.frame.view_proj, world);
    o.normal = inst.model[0].xyz * v.normal.x + inst.model[1].xyz * v.normal.y + inst.model[2].xyz * v.normal.z;
    o.material = inst.mesh_id * 7 + inst.material_id;
    return o;
}

float3 palette(uint k) {
    return 0.5 + 0.5 * cos(6.28318 * (float3(0.0, 0.33, 0.67) + float(k) * 0.137));
}

[shader("fragment")]
float4 fragmentMain(VSOut i) : SV_Target0 {
    float3 n = normalize(i.normal);
    float lambert = max(dot(n, normalize(float3(0.4, 0.8, 0.3))), 0.0);
    return float4(palette(i.material) * (0.25 + 0.75 * lambert), 1.0);
}
