// Quality-pass instance cull + per-view GPU LOD selection (BDA only).
// One "view" per pass that draws instances: 0 = camera, 1.. = sun cascades. Per (view, instance):
// sphere-vs-planes cull, then walk the instance's LOD chain (lod[mesh] = next mesh, switch
// distance, fade width, -) by camera distance:
//   mode 0 (camera): inside a fade band the instance is emitted twice, to both levels, with
//                    complementary dither thresholds (smooth crossfade, no popping);
//   mode 1 (hard):   one level, switched at the band centre (near shadow cascade);
//   mode 2 (coarsest): always the last level of the chain (impostors in far cascades).
// Draw slots: per view 2M of them, [0, M) for instances drawn whole and [M, 2M) for those
// inside a crossfade band (code != 127), so only the latter use the discard (dither) pipeline
// and everything else keeps early depth testing. The last level of a chain may set
// lod.w = asfloat(exact distance): camera-view instances of it nearer than that also go to the
// second slot (impostors there write per-pixel depth; farther ones use the card's depth).
// counts[view * 2M + slot] -> exclusive scan (scan.slang) -> scatter into `visible` with the
// dither code in bits 24..31 -> one fixed-slot draw per (view, slot) (instance_count may be 0).

static const uint NONE = 0xFFFFFFFFu;
// lod.x == VANISH: the chain ends in nothing (ground cover): beyond the switch the instance is
// not drawn; the camera view dithers it out across the fade band, shadow views stop hard
// (mode 1) or never draw it (mode 2, "coarsest" = nothing).
static const uint VANISH = 0xFFFFFFFEu;
static const uint MAX_VIEWS = 5;

struct QInstance { float4 model[4]; uint mesh_id; uint material_id; uint _p0; uint _p1; };
struct MeshInfo { uint index_count; uint first_index; int vertex_offset; uint _pad; float4 sphere; };
struct DrawCmd { uint index_count; uint instance_count; uint first_index; int vertex_offset; uint first_instance; };
struct View { float4 planes[6]; uint mode; float lod_scale; uint _p0; uint _p1; };

struct QCullFrame {            // == render_ash::quality::QCullFrameGpu
    QInstance* inst;
    MeshInfo* meshes;
    uint4* lod;                // per mesh: next (NONE = last), asfloat(switch m), asfloat(fade m), -
    Atomic<uint>* counts;      // view_count * 2 * mesh_count
    uint* offsets;             // exclusive scan of counts
    uint2* slots;              // (view * n + i) * 2 + k: (view * M + mesh, rank | code << 24) or NONE
    uint* visible;             // instance index | dither code << 24
    DrawCmd* draws;            // view_count * 2 * mesh_count, fixed slots
    uint n;
    uint mesh_count;
    uint view_count;
    uint _p;
    View views[MAX_VIEWS];
};
struct QCullPush { QCullFrame* f; };
[[vk::push_constant]] ConstantBuffer<QCullPush> pc;

// dither code: bit 7 = keep the upper part (h >= t), bits 0..6 = t * 127; 127 = keep all
uint code(float t, bool upper) { return uint(round(saturate(t) * 127.0)) | (upper ? 128u : 0u); }

static const uint ALPHA = 0x7FC0A1FAu; // == quality::LOD_ALPHA (lod.w): alpha-tested mesh

void emit(QCullFrame* f, uint v, uint i, uint k, uint mesh, uint c, bool exact) {
    bool alpha = f.lod[mesh].w == ALPHA;
    uint vm = v * 2u * f.mesh_count + mesh + ((c != 127u || exact || alpha) ? f.mesh_count : 0u);
    uint rank = f.counts[vm].increment();
    f.slots[(v * f.n + i) * 2 + k] = uint2(vm, rank | (c << 24));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cullMain(uint3 tid: SV_DispatchThreadID) {
    QCullFrame* f = pc.f;
    uint i = tid.x, v = tid.y;
    if (i >= f.n || v >= f.view_count) return;
    f.slots[(v * f.n + i) * 2] = uint2(NONE, 0);
    f.slots[(v * f.n + i) * 2 + 1] = uint2(NONE, 0);
    QInstance I = f.inst[i];
    if (I.mesh_id >= f.mesh_count) return;
    MeshInfo m = f.meshes[I.mesh_id];
    float3 c = (I.model[0] * m.sphere.x + I.model[1] * m.sphere.y + I.model[2] * m.sphere.z + I.model[3]).xyz;
    float s = max(length(I.model[0].xyz), max(length(I.model[1].xyz), length(I.model[2].xyz)));
    float r = m.sphere.w * s;
    View V = f.views[v];
    for (int p = 0; p < 6; ++p) {
        float4 pl = V.planes[p];
        if (dot(pl.xyz, c) + pl.w < -r) return;
    }
    // LOD by camera distance (camera-relative translation), the same in every view
    float d = length(I.model[3].xyz) * V.lod_scale;
    uint mesh = I.mesh_id;
    for (int lvl = 0; lvl < 4; ++lvl) {
        uint4 e = f.lod[mesh];
        if (e.x == NONE) break;
        float sw = asfloat(e.y), fw = asfloat(e.z);
        if (e.x == VANISH) {
            if (V.mode == 2u || d >= sw + (V.mode == 0u ? 0.5 * fw : 0.0)) return;
            if (V.mode == 0u && d > sw - 0.5 * fw && fw > 0.0) {
                emit(f, v, i, 0, mesh, code(1.0 - (d - (sw - 0.5 * fw)) / fw, false), false);
                return;
            }
            break;
        }
        if (V.mode == 2u) { mesh = e.x; continue; }
        if (V.mode == 1u) { if (d >= sw) { mesh = e.x; continue; } break; }
        if (d >= sw + 0.5 * fw) { mesh = e.x; continue; }
        if (d > sw - 0.5 * fw && fw > 0.0) {
            float t = (d - (sw - 0.5 * fw)) / fw; // 0 at the band start -> all current level
            emit(f, v, i, 0, mesh, code(1.0 - t, false), false);
            emit(f, v, i, 1, e.x, code(1.0 - t, true), false);
            return;
        }
        break;
    }
    float ex = asfloat(f.lod[mesh].w);
    emit(f, v, i, 0, mesh, 127u, V.mode == 0u && f.lod[mesh].x == NONE && d < ex);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void scatterMain(uint3 tid: SV_DispatchThreadID) {
    QCullFrame* f = pc.f;
    uint i = tid.x, v = tid.y;
    if (i >= f.n || v >= f.view_count) return;
    for (uint k = 0; k < 2; ++k) {
        uint2 s = f.slots[(v * f.n + i) * 2 + k];
        if (s.x == NONE) continue;
        f.visible[f.offsets[s.x] + (s.y & 0xFFFFFFu)] = i | (s.y & 0xFF000000u);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void drawsMain(uint3 tid: SV_DispatchThreadID) {
    QCullFrame* f = pc.f;
    uint vm = tid.x;
    if (vm >= f.view_count * 2u * f.mesh_count) return;
    MeshInfo mi = f.meshes[vm % f.mesh_count]; // (vm % 2M) % M == vm % M
    DrawCmd cmd;
    cmd.index_count = mi.index_count;
    cmd.instance_count = f.counts[vm].load();
    cmd.first_index = mi.first_index;
    cmd.vertex_offset = mi.vertex_offset;
    cmd.first_instance = f.offsets[vm];
    f.draws[vm] = cmd;
}
