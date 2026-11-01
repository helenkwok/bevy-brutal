import common;

[[vk::push_constant]] ConstantBuffer<Push> pc;

// Pass 3a (scatter): place each visible instance at offsets[mesh] + rank.
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.frame.instance_count) return;
    uint slot = pc.slots[i];
    if (slot == INVALID_SLOT) return;
    uint mesh = pc.instances[i].mesh_id;
    pc.visible[pc.offsets[mesh] + slot] = i;
}
