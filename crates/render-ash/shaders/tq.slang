// TurboQuant-style brick decoder (asset::tq, see starter.md "Learning from turbovec and
// TurboQuant"). One workgroup per 512-sample block, 256 threads, reading the verified TQ blob
// words straight from the uploaded asset bytes (no CPU decode). Per block: codes -> Lloyd-Max
// centroids -> transposed rotation (per round in reverse: unnormalised Walsh-Hadamard, signs,
// inverse permutation) -> floor(mean + a * v + 0.5) clamped to +-DQ_MAX.
// Bit-exact with asset::tq::TqView::decode: the same f32 adds/subs in the same butterfly pairs
// (pair results do not depend on the schedule) and `precise` blocks FMA contraction.

static const uint D = 512;
static const uint HEADER = 8;
static const uint ROUNDS = 2;
// u32 offsets in the tables buffer (== asset::tq::GPU_*)
static const uint T_PERM = 0;
static const uint T_SIGNS = ROUNDS * D;
static const uint T_CENT = T_SIGNS + ROUNDS * D / 32;
static const float DQ_MAX = 1048576.0;

struct TqPush {                // == render_ash::tq_gpu::TqPushGpu (40 bytes)
    uint* tables;
    uint* blobs;               // concatenated blobs, blob_words each
    int* out;                  // samples per blob
    uint blob_words;
    uint blocks;
    uint samples;
    uint bits;
};
[[vk::push_constant]] ConstantBuffer<TqPush> tp;

groupshared float sa[D];
groupshared float sb[D];

[numthreads(256, 1, 1)]
void decodeMain(uint3 gid: SV_GroupID, uint t: SV_GroupIndex)
{
    uint blk = gid.x;
    uint base = gid.y * tp.blob_words;
    uint bits = tp.bits;
    uint codes = base + HEADER + 2 * tp.blocks + blk * (D / 32) * bits;
    for (uint q = 0; q < 2; q++) {
        uint j = t + q * 256;
        uint bit = j * bits;
        uint w = bit >> 5;
        uint s = bit & 31;
        uint c = tp.blobs[codes + w] >> s;
        if (s + bits > 32) c |= tp.blobs[codes + w + 1] << (32 - s);
        c &= (1u << bits) - 1;
        sa[j] = asfloat(tp.tables[T_CENT + c]);
    }
    GroupMemoryBarrierWithGroupSync();
    for (int r = int(ROUNDS) - 1; r >= 0; r--) {
        for (uint h = 1; h < D; h *= 2) {
            uint i = (t / h) * 2 * h + (t % h);
            precise float a = sa[i];
            precise float b = sa[i + h];
            precise float sum = a + b;
            precise float dif = a - b;
            sa[i] = sum;
            sa[i + h] = dif;
            GroupMemoryBarrierWithGroupSync();
        }
        for (uint q = 0; q < 2; q++) {
            uint j = t + q * 256;
            float x = sa[j];
            if (((tp.tables[T_SIGNS + uint(r) * (D / 32) + j / 32] >> (j % 32)) & 1) != 0) x = -x;
            sb[tp.tables[T_PERM + uint(r) * D + j]] = x;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint q = 0; q < 2; q++) sa[t + q * 256] = sb[t + q * 256];
        GroupMemoryBarrierWithGroupSync();
    }
    float mean = asfloat(tp.blobs[base + HEADER + 2 * blk]);
    float a = asfloat(tp.blobs[base + HEADER + 2 * blk + 1]);
    for (uint q = 0; q < 2; q++) {
        uint j = t + q * 256;
        uint k = blk * D + j;
        if (k < tp.samples) {
            precise float p = a * sa[j];
            precise float x = mean + p;
            precise float y = x + 0.5;
            float rr = clamp(floor(y), -DQ_MAX, DQ_MAX);
            tp.out[gid.y * tp.samples + k] = int(rr);
        }
    }
}
