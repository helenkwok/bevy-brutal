import common;

[[vk::push_constant]] ConstantBuffer<Push> pc;

// Pass 1 (count): frustum-cull one instance. Visible instances take a rank within their mesh
// (atomic on the per-mesh count); everything else records INVALID_SLOT.
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    Frame* f = pc.frame;
    if (i >= f.instance_count) return;
    Instance inst = pc.instances[i];
    uint slot = INVALID_SLOT;
    if (inst.mesh_id < f.mesh_count) {
        MeshInfo m = pc.meshes[inst.mesh_id];
        float3 c = mul_cols(inst.model, float4(m.sphere.xyz, 1.0)).xyz;
        float s = max(length(inst.model[0].xyz), max(length(inst.model[1].xyz), length(inst.model[2].xyz)));
        float r = m.sphere.w * s;
        bool inside = true;
        for (int p = 0; p < 6; ++p) {
            float4 pl = f.planes[p];
            if (dot(pl.xyz, c) + pl.w < -r) { inside = false; break; }
        }
        if (inside) slot = pc.counts[inst.mesh_id].increment();
    }
    pc.slots[i] = slot;
}
