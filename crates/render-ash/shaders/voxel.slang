// GPU Transvoxel brick mesher (Prompt C2). Pure 32-bit integer port of voxel::mesh_brick:
// identical inputs (quantised i32 densities), identical integer formulas (edge_t, the global seam
// displacement PHI, lexicographic edge orientation), identical cell and triangle order, so the
// emitted brick-local sub-grid positions are bit-identical to the CPU mesher (tested). Normals are
// f32 sample gradients (compared within a tolerance). No shaderInt64: positions are brick-local
// (<= 64 cells * 64 lod-step * 256 sub = 2^20), PHI intermediates <= s^2 <= 2^28, edge_t <= 2^29.
//
// Cells of brick b (one thread each), CPU order:
//   regular     c = (k*n + j)*n + i                 for c < n^3
//   transition  c = n^3 + f*n^2 + cv*n + cu         (count 0 unless face f is in the mask)
// countMain writes per-cell triangle counts; the host runs the shared exclusive scan in place;
// emitMain writes non-indexed triangles at the scanned offsets: per vertex int3 pos + float3 normal.

static const int SUB = 256;
static const int T_STEPS = 256;
// u32 offsets in the tables buffer (== render_ash::voxel_gpu::pack_tables)
static const uint RC = 0;       // REGULAR_CELL_CLASS[256]
static const uint RD = 256;     // REGULAR_CELL_DATA[16] x (counts, 15 indices)
static const uint RV = 512;     // REGULAR_VERTEX_DATA[256][12]
static const uint TC = 3584;    // TRANSITION_CELL_CLASS[512]
static const uint TD = 4096;    // TRANSITION_CELL_DATA[56] x (counts, 36 indices)
static const uint TV = 6168;    // TRANSITION_VERTEX_DATA[512][12]
static const uint DESC_INTS = 17;

struct VoxPush {               // == render_ash::voxel_gpu::VoxPushGpu (56 bytes)
    uint* tables;
    int* bricks;               // DESC_INTS per brick, see BrickDesc
    int* samples;
    uint* counts;              // counts, scanned in place to offsets
    uint* out;                 // 6 u32 per vertex (int3 pos, float3 normal)
    uint n;
    uint cpb;                  // cells per brick = n^3 + 6 n^2
    uint pad0;
    uint pad1;
};
[[vk::push_constant]] ConstantBuffer<VoxPush> vp;

// BrickDesc ints: 0 s (lattice step), 1 log2(s*SUB), 2 transition mask, 3 has_phi,
// 4..6 phi lo, 7..9 phi hi (brick-local sub units, clamped), 10 sample offset, 11..16 face offsets.
int desc(uint b, uint i) { return vp.bricks[b * DESC_INTS + i]; }

int at(uint b, int3 c) {
    int m = int(vp.n) + 3;
    return vp.samples[desc(b, 10) + ((c.z + 1) * m + (c.y + 1)) * m + (c.x + 1)];
}

float3 grad(uint b, int3 c) {
    return float3(float(at(b, c + int3(1, 0, 0)) - at(b, c - int3(1, 0, 0))),
                  float(at(b, c + int3(0, 1, 0)) - at(b, c - int3(0, 1, 0))),
                  float(at(b, c + int3(0, 0, 1)) - at(b, c - int3(0, 0, 1))));
}

int edgeT(int da, int db) {
    int num = T_STEPS * da;
    int den = da - db;
    if (den < 0) { num = -num; den = -den; }
    return clamp((2 * num + den) / (2 * den), 1, T_STEPS - 1);
}

int3 phi(uint b, int3 p) {
    int log2s = desc(b, 1);
    int s = 1 << log2s;
    int3 lo = int3(desc(b, 4), desc(b, 5), desc(b, 6));
    int3 hi = int3(desc(b, 7), desc(b, 8), desc(b, 9));
    int g[3];
    [unroll] for (int a = 0; a < 3; ++a) g[a] = max(s - max(max(lo[a] - p[a], 0), p[a] - hi[a]), 0);
    int3 o;
    [unroll] for (int a = 0; a < 3; ++a) {
        int gb = g[(a + 1) % 3];
        int gc = g[(a + 2) % 3];
        int tp = max(s - abs(p[a] - hi[a]), 0);
        int tm = max(s - abs(p[a] - lo[a]), 0);
        int qp = (((tp * gb) >> log2s) * gc) >> (log2s + 2);
        int qm = (((tm * gb) >> log2s) * gc) >> (log2s + 2);
        o[a] = qp - qm;
    }
    return o;
}

bool lexLess(int3 a, int3 b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
}

// pa/pb: brick-local lattice points (LOD0 voxel units).
void vertexOf(uint b, int3 pa, int da, float3 ga, int3 pb, int db, float3 gb, bool displace,
              out int3 ip, out float3 nrm) {
    if (!lexLess(pa, pb)) {
        int3 tp = pa; pa = pb; pb = tp;
        int td = da; da = db; db = td;
        float3 tg = ga; ga = gb; gb = tg;
    }
    int t = edgeT(da, db);
    ip = pa * SUB + t * (pb - pa);   // SUB / T_STEPS == 1
    if (displace && desc(b, 3) != 0) ip += phi(b, ip);
    float tf = float(t) / float(T_STEPS);
    float3 g = float3(ga.x + (gb.x - ga.x) * tf, ga.y + (gb.y - ga.y) * tf, ga.z + (gb.z - ga.z) * tf);
    float gl = sqrt(g.x * g.x + g.y * g.y + g.z * g.z);
    nrm = gl > 0.0 ? float3(g.x / gl, g.y / gl, g.z / gl) : float3(0.0, 1.0, 0.0);
}

void writeVertex(uint slot, int3 ip, float3 nrm) {
    uint o = slot * 6;
    vp.out[o + 0] = asuint(ip.x);
    vp.out[o + 1] = asuint(ip.y);
    vp.out[o + 2] = asuint(ip.z);
    vp.out[o + 3] = asuint(nrm.x);
    vp.out[o + 4] = asuint(nrm.y);
    vp.out[o + 5] = asuint(nrm.z);
}

uint regularCell(uint b, uint c, bool emit, uint base) {
    int n = int(vp.n);
    int s = desc(b, 0);
    int3 cell = int3(int(c) % n, (int(c) / n) % n, int(c) / (n * n));
    int cd[8];
    uint cs = 0;
    [unroll] for (int q = 0; q < 8; ++q) {
        int3 k = cell + int3(q & 1, (q >> 1) & 1, (q >> 2) & 1);
        cd[q] = at(b, k);
        if (cd[q] < 0) cs |= 1u << q;
    }
    if (cs == 0 || cs == 255) return 0;
    uint cls = vp.tables[RC + cs];
    uint counts = vp.tables[RD + cls * 16];
    uint ntri = counts & 0xF;
    if (!emit) return ntri;
    int3 vpos[12];
    float3 vnrm[12];
    uint nv = counts >> 4;
    for (uint q = 0; q < nv; ++q) {
        uint e = vp.tables[RV + cs * 12 + q] & 0xFF;
        uint c0 = (e >> 4) & 0xF;
        uint c1 = e & 0xF;
        int3 k0 = cell + int3(int(c0 & 1), int((c0 >> 1) & 1), int((c0 >> 2) & 1));
        int3 k1 = cell + int3(int(c1 & 1), int((c1 >> 1) & 1), int((c1 >> 2) & 1));
        vertexOf(b, k0 * s, cd[c0], grad(b, k0), k1 * s, cd[c1], grad(b, k1), true, vpos[q], vnrm[q]);
    }
    for (uint t = 0; t < ntri; ++t) {
        [unroll] for (uint v = 0; v < 3; ++v) {
            uint vi = vp.tables[RD + cls * 16 + 1 + 3 * t + v];
            writeVertex((base + t) * 3 + v, vpos[vi], vnrm[vi]);
        }
    }
    return ntri;
}

static const uint BIT_ORDER[9] = { 0, 1, 2, 5, 8, 7, 6, 3, 4 };

uint transitionCell(uint b, uint c, bool emit, uint base) {
    int n = int(vp.n);
    uint f = c / uint(n * n);
    if ((uint(desc(b, 2)) & (1u << f)) == 0) return 0;
    uint r = c % uint(n * n);
    int cv = int(r) / n;
    int cu = int(r) % n;
    int axis = int(f / 2);
    int ua = (axis + 1) % 3;
    int va = (axis + 2) % 3;
    bool high = (f % 2) == 1;
    int s = desc(b, 0);
    int hs = s / 2;
    int fm = 2 * n + 1;
    int fo = desc(b, 11 + f);
    int sd[9];
    uint cs = 0;
    [unroll] for (uint q = 0; q < 9; ++q) {
        int su = int(q % 3), sv = int(q / 3);
        sd[q] = vp.samples[fo + (2 * cv + sv) * fm + (2 * cu + su)];
    }
    [unroll] for (uint bit = 0; bit < 9; ++bit) if (sd[BIT_ORDER[bit]] < 0) cs |= 1u << bit;
    if (cs == 0 || cs == 511) return 0;
    uint cls = vp.tables[TC + cs];
    uint counts = vp.tables[TD + (cls & 0x7F) * 37];
    uint ntri = counts & 0xF;
    if (!emit) return ntri;
    int3 sp[9];
    float3 sg[9];
    for (uint q = 0; q < 9; ++q) {
        int su = int(q % 3), sv = int(q / 3);
        int3 p = int3(0, 0, 0);
        p[axis] = high ? n * s : 0;
        p[ua] = cu * s + su * hs;
        p[va] = cv * s + sv * hs;
        sp[q] = p;
        int u0 = su / 2 * 2, u1 = (su + 1) / 2 * 2;
        int v0 = sv / 2 * 2, v1 = (sv + 1) / 2 * 2;
        int2 cc[4] = { int2(u0, v0), int2(u1, v0), int2(u0, v1), int2(u1, v1) };
        float3 g = float3(0.0, 0.0, 0.0);
        for (int w = 0; w < 4; ++w) {
            int3 k = int3(0, 0, 0);
            k[axis] = high ? n : 0;
            k[ua] = cu + cc[w].x / 2;
            k[va] = cv + cc[w].y / 2;
            float3 gc = grad(b, k);
            g = float3(g.x + 0.25 * gc.x, g.y + 0.25 * gc.y, g.z + 0.25 * gc.z);
        }
        sg[q] = g;
    }
    int3 vpos[12];
    float3 vnrm[12];
    uint nv = counts >> 4;
    static const uint HALF[4] = { 0, 2, 6, 8 };
    for (uint q = 0; q < nv; ++q) {
        uint e = vp.tables[TV + cs * 12 + q] & 0xFF;
        uint e0 = (e >> 4) & 0xF;
        uint e1 = e & 0xF;
        bool half = e0 >= 9;
        uint a = half ? HALF[e0 - 9] : e0;
        uint bb = half ? HALF[e1 - 9] : e1;
        vertexOf(b, sp[a], sd[a], sg[a], sp[bb], sd[bb], sg[bb], half, vpos[q], vnrm[q]);
    }
    bool flip = ((cls & 0x80) != 0) != high;
    for (uint t = 0; t < ntri; ++t) {
        uint i0 = vp.tables[TD + (cls & 0x7F) * 37 + 1 + 3 * t];
        uint i1 = vp.tables[TD + (cls & 0x7F) * 37 + 2 + 3 * t];
        uint i2 = vp.tables[TD + (cls & 0x7F) * 37 + 3 + 3 * t];
        if (flip) { uint tmp = i1; i1 = i2; i2 = tmp; }
        writeVertex((base + t) * 3 + 0, vpos[i0], vnrm[i0]);
        writeVertex((base + t) * 3 + 1, vpos[i1], vnrm[i1]);
        writeVertex((base + t) * 3 + 2, vpos[i2], vnrm[i2]);
    }
    return ntri;
}

uint cellOf(uint b, uint c, bool emit, uint base) {
    uint n3 = vp.n * vp.n * vp.n;
    return c < n3 ? regularCell(b, c, emit, base) : transitionCell(b, c - n3, emit, base);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void countMain(uint3 tid: SV_DispatchThreadID) {
    uint c = tid.x, b = tid.y;
    if (c >= vp.cpb) return;
    vp.counts[b * vp.cpb + c] = cellOf(b, c, false, 0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void emitMain(uint3 tid: SV_DispatchThreadID) {
    uint c = tid.x, b = tid.y;
    if (c >= vp.cpb) return;
    cellOf(b, c, true, vp.counts[b * vp.cpb + c]);
}
