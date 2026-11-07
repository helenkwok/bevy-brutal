// Shared declarations for the instanced pass. Buffers are reached through buffer device
// addresses (Slang pointers -> SPIR-V PhysicalStorageBuffer); there are no descriptor sets.
// Matrices are stored as 4 float4 columns (glam's column-major `[[f32; 4]; 4]`) and multiplied
// explicitly, so there is no row-/column-major layout ambiguity.
module common;

public static const uint INVALID_SLOT = 0xFFFFFFFFu;

public struct Instance {        // == lowlevel_abi::InstanceGpu (80 bytes)
    public float4 model[4];     // columns
    public uint mesh_id;
    public uint material_id;
    public uint _pad0;
    public uint _pad1;
}

public struct MeshInfo {        // == render_ash::MeshInfoGpu (32 bytes)
    public uint index_count;
    public uint first_index;
    public int vertex_offset;
    public uint _pad;
    public float4 sphere;       // xyz = local center, w = radius
}

public struct DrawCmd {         // == VkDrawIndexedIndirectCommand (20 bytes)
    public uint index_count;
    public uint instance_count;
    public uint first_index;
    public int vertex_offset;
    public uint first_instance;
}

public struct Frame {           // == render_ash FrameGpu (176 bytes)
    public float4 view_proj[4]; // columns
    public float4 planes[6];    // normalized, inside when dot(n, p) + d >= 0
    public uint instance_count;
    public uint mesh_count;
    public float2 surface_range; // surface palette height range in m; y <= x: noise palette
}

public struct Push {            // == render_ash PushGpu (9 x u64 = 72 bytes); all per frame slot
    public Frame* frame;
    public Instance* instances;
    public MeshInfo* meshes;
    public Atomic<uint>* counts;     // per-mesh visible counts (zeroed each frame)
    public uint* offsets;            // exclusive prefix sum of counts (GPU scan)
    public uint* slots;              // per instance: rank within its mesh, or INVALID_SLOT if culled
    public uint* visible;            // compacted instance indices, grouped by mesh
    public DrawCmd* draws;           // compacted draw commands
    public Atomic<uint>* draw_count; // [0] = number of draws
}

public float4 mul_cols(float4 c[4], float4 v) {
    return c[0] * v.x + c[1] * v.y + c[2] * v.z + c[3] * v.w;
}
