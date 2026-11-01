// Multi-level exclusive prefix sum over u32 (reduce-then-scan / scan-then-propagate):
//   level k:  scanBlocks over tiles of 1024 (256 threads x 4), writing each tile's total to sums[k+1]
//   ...repeat on the tile totals until one tile remains, then...
//   addOffsets walks back down adding the scanned tile totals.
// Handles any n (0 is skipped on the host); 2 levels cover 1M, 3 levels cover 1G. Wrapping u32 math.
//
// Two tile kernels with identical results:
//   scanBlocks      shared-memory Hillis-Steele (portable fallback, no subgroup ops)
//   scanBlocksWave  subgroup arithmetic: WavePrefixSum/WaveActiveSum per subgroup, then subgroup 0
//                   scans the per-subgroup totals. Size-agnostic (lane count read at runtime, any
//                   power of two dividing 256); requires FULL subgroups, which the host guarantees
//                   with VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT (+ a required
//                   subgroup size when subgroupSizeControl allows it). Element ownership is derived
//                   from the SubgroupId / SubgroupLocalInvocationId builtins, so nothing assumes
//                   how LocalInvocationIndex maps onto subgroups.
static const uint WG = 256;
static const uint PER_THREAD = 4;
static const uint TILE = WG * PER_THREAD;

struct ScanPush {               // == render_ash ScanPushGpu (32 bytes)
    uint* input;
    uint* output;               // may alias input (in-place)
    uint* sums;                 // per-tile totals (written by scanBlocks if has_sums, read by addOffsets)
    uint n;
    uint has_sums;
}

[[vk::push_constant]] ConstantBuffer<ScanPush> sp;

groupshared uint s_data[WG];

[shader("compute")]
[numthreads(256, 1, 1)]
void scanBlocks(uint3 gid: SV_GroupID, uint3 ltid: SV_GroupThreadID) {
    uint t = ltid.x;
    uint base = gid.x * TILE + t * PER_THREAD;
    uint local[PER_THREAD];
    uint total = 0;
    [unroll]
    for (uint k = 0; k < PER_THREAD; ++k) {
        uint idx = base + k;
        uint x = idx < sp.n ? sp.input[idx] : 0;
        local[k] = total;           // thread-local exclusive
        total += x;
    }
    // Workgroup inclusive scan of per-thread totals (Hillis-Steele in shared memory).
    s_data[t] = total;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1; off < WG; off <<= 1) {
        uint add = t >= off ? s_data[t - off] : 0;
        GroupMemoryBarrierWithGroupSync();
        s_data[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint inclusive = s_data[t];
    uint exclusive = inclusive - total;
    [unroll]
    for (uint k = 0; k < PER_THREAD; ++k) {
        uint idx = base + k;
        if (idx < sp.n) sp.output[idx] = exclusive + local[k];
    }
    if (t == WG - 1 && sp.has_sums != 0) sp.sums[gid.x] = inclusive;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void addOffsets(uint3 gid: SV_GroupID, uint3 ltid: SV_GroupThreadID) {
    uint add = sp.sums[gid.x];      // exclusive-scanned tile totals
    uint base = gid.x * TILE + ltid.x * PER_THREAD;
    [unroll]
    for (uint k = 0; k < PER_THREAD; ++k) {
        uint idx = base + k;
        if (idx < sp.n) sp.output[idx] += add;
    }
}

// ---- subgroup path ----

groupshared uint s_sub[WG];     // per-subgroup totals, then their exclusive prefix (<= 256 subgroups)
groupshared uint s_tile_total;

uint subgroupId() {
    return spirv_asm {
        OpCapability GroupNonUniform;
        result:$$uint = OpLoad builtin(SubgroupId:uint);
    };
}

[shader("compute")]
[numthreads(256, 1, 1)]
void scanBlocksWave(uint3 gid: SV_GroupID) {
    uint lanes = WaveGetLaneCount();
    uint lane = WaveGetLaneIndex();
    uint sg = subgroupId();
    uint num_sg = WG / lanes;                     // full subgroups (host-guaranteed)
    uint logical = sg * lanes + lane;             // 0..255, contiguous per subgroup in lane order
    uint base = gid.x * TILE + logical * PER_THREAD;

    uint local[PER_THREAD];
    uint total = 0;
    [unroll]
    for (uint k = 0; k < PER_THREAD; ++k) {
        uint idx = base + k;
        uint x = idx < sp.n ? sp.input[idx] : 0;
        local[k] = total;
        total += x;
    }
    uint lane_prefix = WavePrefixSum(total);      // exclusive, in lane order
    uint sg_total = WaveActiveSum(total);
    if (lane == 0) s_sub[sg] = sg_total;
    GroupMemoryBarrierWithGroupSync();

    if (sg == 0) {                                // subgroup-uniform branch
        uint carry = 0;
        for (uint b = 0; b < num_sg; b += lanes) {
            uint i = b + lane;
            uint v = i < num_sg ? s_sub[i] : 0;
            uint p = WavePrefixSum(v);
            uint sum = WaveActiveSum(v);
            if (i < num_sg) s_sub[i] = carry + p;
            carry += sum;
        }
        if (lane == 0) s_tile_total = carry;
    }
    GroupMemoryBarrierWithGroupSync();

    uint exclusive = s_sub[sg] + lane_prefix;
    [unroll]
    for (uint k = 0; k < PER_THREAD; ++k) {
        uint idx = base + k;
        if (idx < sp.n) sp.output[idx] = exclusive + local[k];
    }
    if (logical == 0 && sp.has_sums != 0) sp.sums[gid.x] = s_tile_total;
}
