import common;

[[vk::push_constant]] ConstantBuffer<Push> pc;

// Pass 3b (indirect args): one thread per mesh; meshes with >= 1 visible instance append a draw
// whose firstInstance is the mesh's region start in `visible`.
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint m = tid.x;
    Frame* f = pc.frame;
    if (m >= f.mesh_count) return;
    uint n = pc.counts[m].load();
    if (n == 0) return;
    MeshInfo mi = pc.meshes[m];
    uint d = pc.draw_count[0].increment();
    DrawCmd cmd;
    cmd.index_count = mi.index_count;
    cmd.instance_count = n;
    cmd.first_index = mi.first_index;
    cmd.vertex_offset = mi.vertex_offset;
    cmd.first_instance = pc.offsets[m];
    pc.draws[d] = cmd;
}
