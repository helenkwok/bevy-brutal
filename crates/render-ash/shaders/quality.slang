// Quality-pass terrain renderer (BDA only, no descriptors). Vertices are pulled through a
// pointer; shadow maps are depth images copied to a float buffer and filtered manually (PCF).
//   terrainVS/terrainFS : lit terrain (PBR-ish materials, triplanar detail, shadows, fog, ACES)
//   shadowVS            : depth-only into cascade `pc.cascade` (sun) or the headlamp (cave)
//   skyVS/skyFS         : fullscreen sky gradient + sun + clouds (mountains) / darkness (cave)
// Positions are camera-relative f32 (coords spirit); `cam_world` is added only for texturing.

struct QVertex {               // == render_ash::quality::QVertex (40 bytes)
    float3 pos;                // camera-relative metres
    float3 nrm;                // analytic field gradient normal
    float4 attr;               // x ao (multi-scale SDF), y cavity, z convexity, w flags (1 far ring)
};

struct QParams {               // == render_ash::quality::QParamsGpu
    float4 view_proj[4];       // reverse-Z, infinite far
    float4 inv_dir[4];         // clip -> camera-relative direction (inverse of view_proj w/o depth)
    float4 cam_world;          // xyz world, w pixel angle (rad)
    float4 sun_dir;            // xyz toward sun, w intensity
    float4 sun_color;          // rgb, w exposure
    float4 sky_zenith;         // rgb, w mode (0 mountains, 1 cave)
    float4 sky_horizon;        // rgb, w fog density at base height (1/m)
    float4 fog;                // x height falloff (1/m), y base height (world m), z max opacity, w sun inscatter
    float4 fog_color;          // rgb, w ambient strength
    float4 lamp_pos;           // camera-relative, w range
    float4 lamp_dir;           // xyz, w cos outer
    float4 lamp_color;         // rgb (intensity folded in), w cos inner
    float4 lantern_pos;        // camera-relative, w range
    float4 lantern_color;      // rgb
    float4 shadow_info;        // x res, y count, z normal-offset scale, w mode (0 sun ortho, 1 spot persp)
    float4 splits;             // far distance of each cascade
    float4 texel;              // world size of a shadow texel per cascade
    float4 cascade_vp[16];     // 4 matrices x 4 columns
    float4 atmo;               // x Mie density scale, y ground albedo, z aerial-perspective range (km), w enabled
    float4 atmo2;              // x grade contrast (mountains; 0 = off), y sun angular radius (rad), z Rayleigh scale, w grade white balance (cool, 0 = off)
    uint* gmap;                // Earth round 5 ground map (RGBA8: sRGB albedo, a = 7-bit weight | 0x80 water), null = off
    uint2 gmap_pad;
    uint4 gmap_lv[2];          // outer, inner: texels per side, mip count, word offset of mip 0
    float4 gmap_f;             // x, z world centre (m), outer / inner half extent (m)
};

struct QInstance {             // == lowlevel_abi::InstanceGpu (80 bytes; common.slang `Instance`)
    float4 model[4];           // columns, camera-relative
    uint mesh_id;
    uint material_id;          // per-instance random 0..1023 (colour variation)
    uint _p0;
    uint _p1;
};

struct QPush {                 // == render_ash::quality::QPushGpu (72 bytes)
    QParams* p;
    QVertex* v;
    float* shadow;             // count * res * res floats
    uint cascade;
    uint pad;
    QInstance* inst;           // instanced set (trees, boulders, tufts)
    uint* visible;             // GPU-compacted visible instance indices (cull.slang/scatter.slang)
    float4* atmo;              // atmosphere LUTs (see OFF_*), null when disabled
    float4* imp;               // impostor atlas header: per variant (centre, radius), (AABB half extents, 0), mesh space
    uint2* imp_tex;            // impostor texels [variant][frame][y][x] (render_ash::impostor)
};
[[vk::push_constant]] ConstantBuffer<QPush> pc;

float4 mulc(float4 c0, float4 c1, float4 c2, float4 c3, float4 v) { return c0 * v.x + c1 * v.y + c2 * v.z + c3 * v.w; }
float4 vp_main(float4 v) { QParams* p = pc.p; return mulc(p.view_proj[0], p.view_proj[1], p.view_proj[2], p.view_proj[3], v); }
// Earth round 3, orbit -> mountains descent ("composite layer" mode, lamp_pos.w = 1; the cave
// uses lamp_* for its headlamp, the default mountains view leaves them 0 so nothing below runs):
//   lamp_pos = (curvature centre x, z (m, world), 1 / (2 R) (1/m), 1)
//   lamp_dir = (coverage fade r0, r1 (m, horizontal from the centre), cirrus base y (m), 0)
// The flat world is bent onto the sphere for rasterisation only (clip position: y drops by
// d^2 / 2R); shading, shadows and aerial perspective keep the flat coordinates, so materials
// (altitude bands) and the shadow maps are those of the flat scene.
bool layer_mode() { return pc.p.lamp_pos.w > 0.5 && pc.p.sky_zenith.w < 0.5; }
float4 vp_view(float3 rel) {
    QParams* p = pc.p;
    if (layer_mode()) {
        float2 d = rel.xz + p.cam_world.xz - p.lamp_pos.xy;
        rel.y -= dot(d, d) * p.lamp_pos.z;
    }
    return vp_main(float4(rel, 1.0));
}
// cirrus base height (m): the camera's own altitude in the default view, fixed in layer mode
float cirrus_base() { return layer_mode() ? pc.p.lamp_dir.z : pc.p.cam_world.y; }
// output alpha: 1 in the default view; layer mode: terrain coverage x radial fade, sky 0
float layer_alpha(float3 world) {
    if (!layer_mode()) return 1.0;
    float d = length(world.xz - pc.p.lamp_pos.xy);
    return 1.0 - smoothstep(pc.p.lamp_dir.x, pc.p.lamp_dir.y, d);
}
float4 vp_casc(uint c, float4 v) { QParams* p = pc.p; return mulc(p.cascade_vp[c * 4], p.cascade_vp[c * 4 + 1], p.cascade_vp[c * 4 + 2], p.cascade_vp[c * 4 + 3], v); }

// ---------------------------------------------------------------- noise (value noise with derivatives)
float hash2(float2 p) {
    p = frac(p * float2(0.1031, 0.1030));
    p += dot(p, p.yx + 33.33);
    return frac((p.x + p.y) * p.x);
}
// value noise in [-1,1] with analytic gradient (xy) -> float3(v, dv/dx, dv/dy)
float3 vnoised(float2 x) {
    float2 i = floor(x);
    float2 f = x - i;
    float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    float a = hash2(i), b = hash2(i + float2(1, 0)), c = hash2(i + float2(0, 1)), d = hash2(i + float2(1, 1));
    float k1 = b - a, k2 = c - a, k4 = a - b - c + d;
    float v = a + k1 * u.x + k2 * u.y + k4 * u.x * u.y;
    float2 g = du * float2(k1 + k4 * u.y, k2 + k4 * u.x);
    return float3(2.0 * v - 1.0, 2.0 * g);
}
// band-limited fbm: weight octaves by footprint (metres per pixel) to avoid shimmer
float3 fbmd(float2 x, float scale, int oct, float foot) {
    float3 s = float3(0, 0, 0);
    float a = 0.5, f = 1.0 / scale;
    for (int o = 0; o < oct; o++) {
        float w = saturate((1.0 / f) / (foot * 3.0) - 1.0);
        if (w <= 0.0) break;
        float3 n = vnoised(x * f + float2(o * 17.1, o * 9.7));
        s += float3(n.x, n.yz * f) * a * w;
        a *= 0.5; f *= 2.07;
    }
    return s;
}

// triplanar: value + world-space gradient of a 2D fbm projected along each axis
float4 triplanar(float3 p, float3 w, float scale, int oct, float foot) {
    float3 nx = fbmd(p.zy, scale, oct, foot);
    float3 ny = fbmd(p.xz + 31.0, scale, oct, foot);
    float3 nz = fbmd(p.xy + 71.0, scale, oct, foot);
    float v = nx.x * w.x + ny.x * w.y + nz.x * w.z;
    float3 g = float3(0, nx.z, nx.y) * w.x + float3(ny.y, 0, ny.z) * w.y + float3(nz.y, nz.z, 0) * w.z;
    return float4(v, g);
}

float3 bump(float3 n, float3 g, float k) {
    float3 t = g - dot(g, n) * n;
    return normalize(n - k * t);
}

// ---------------------------------------------------------------- shading helpers
float3 aces(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}
// Mountains daylight grade (linear, before the tonemap): luminance-preserving cool white
// balance and a log-space contrast around mid grey (k > 0 deepens shade, lifts highlights).
// k = wb = 0 (the cave, GRADE=0) returns the input unchanged.
float3 grade(float3 c) {
    float k = pc.p.atmo2.x, wb = pc.p.atmo2.w;
    if (pc.p.sky_zenith.w > 0.5 || (k == 0.0 && wb == 0.0)) return c;
    const float3 LUMW = float3(0.2126, 0.7152, 0.0722);
    float3 w = float3(1.0 - 0.5 * wb, 1.0, 1.0 + wb);
    c *= w / dot(w, LUMW);
    float L = max(dot(c, LUMW), 1e-6);
    return c * (0.18 * pow(L / 0.18, 1.0 + k) / L);
}
float3 to_srgb(float3 c) {
    float3 lo = c * 12.92;
    float3 hi = 1.055 * pow(max(c, 0.0), 1.0 / 2.4) - 0.055;
    return lerp(hi, lo, step(c, float3(0.0031308)));
}
float D_ggx(float nh, float a) { float a2 = a * a; float d = nh * nh * (a2 - 1.0) + 1.0; return a2 / (3.14159 * d * d); }
float V_smith(float nv, float nl, float a) {
    float k = a * 0.5;
    return 0.25 / ((nv * (1.0 - k) + k) * (nl * (1.0 - k) + k));
}
float3 brdf(float3 n, float3 v, float3 l, float3 albedo, float rough, float spec) {
    float3 h = normalize(v + l);
    float nl = saturate(dot(n, l)), nv = max(dot(n, v), 1e-3), nh = saturate(dot(n, h)), vh = saturate(dot(v, h));
    float a = rough * rough;
    float F = spec + (1.0 - spec) * pow(1.0 - vh, 5.0);
    return (albedo * (1.0 - F) / 3.14159 + D_ggx(nh, a) * V_smith(nv, nl, a) * F) * nl;
}

float shadow_tap(uint c, int2 t, float z) {
    QParams* p = pc.p;
    int res = int(p.shadow_info.x);
    if (t.x < 0 || t.y < 0 || t.x >= res || t.y >= res) return 1.0;
    float d = pc.shadow[c * uint(res * res) + uint(t.y * res + t.x)];
    return z <= d ? 1.0 : 0.0;
}
// bilinear-weighted PCF over a (2r+2)^2 footprint
float shadow_pcf(uint c, float3 rel, float3 n, float nl) {
    QParams* p = pc.p;
    // normal offset grows at grazing angles in the mountain cascades (texels up to ~5 m there);
    // the cave keeps its original offset
    float k = p.sky_zenith.w < 0.5 ? 1.0 + 2.0 * (1.0 - nl) : 1.5 - nl;
    float3 q = rel + n * p.texel[c] * p.shadow_info.z * k;
    float4 s = vp_casc(c, float4(q, 1.0));
    float3 ndc = s.xyz / s.w;
    if (any(abs(ndc.xy) > 1.0) || ndc.z > 1.0 || ndc.z < 0.0) return 1.0;
    float res = p.shadow_info.x;
    float2 uv = (ndc.xy * 0.5 + 0.5) * res - 0.5;
    float2 b = floor(uv);
    float2 f = uv - b;
    float z = ndc.z - (p.shadow_info.w > 0.5 ? 4e-5 : 2e-5);
    float acc = 0.0, wsum = 0.0;
    for (int y = -1; y <= 2; y++) {
        for (int x = -1; x <= 2; x++) {
            float wx = x == -1 ? 1.0 - f.x : (x == 2 ? f.x : 1.0);
            float wy = y == -1 ? 1.0 - f.y : (y == 2 ? f.y : 1.0);
            acc += wx * wy * shadow_tap(c, int2(b) + int2(x, y), z);
            wsum += wx * wy;
        }
    }
    return acc / wsum;
}

float sun_shadow(float3 rel, float3 n, float nl, float dist) {
    QParams* p = pc.p;
    int count = int(p.shadow_info.y);
    for (int c = 0; c < count; c++) {
        if (dist < p.splits[c]) {
            float s = shadow_pcf(uint(c), rel, n, nl);
            // blend into the next cascade near the split (no visible cascade seam)
            float edge = p.splits[c] * 0.8;
            if (dist > edge) {
                float s2 = c + 1 < count ? shadow_pcf(uint(c + 1), rel, n, nl) : 1.0;
                s = lerp(s, s2, smoothstep(edge, p.splits[c], dist));
            }
            return s;
        }
    }
    return 1.0;
}

float3 sky_color(float3 dir) {
    QParams* p = pc.p;
    float3 sun = p.sun_dir.xyz;
    float e = dir.y;
    float3 zen = p.sky_zenith.rgb, hor = p.sky_horizon.rgb;
    float3 c = lerp(hor, zen, pow(saturate(e), 0.45));
    c = lerp(c, p.fog_color.rgb * 0.9, saturate(-e * 4.0));
    float mu = saturate(dot(dir, sun));
    c += p.sun_color.rgb * (0.10 * pow(mu, 8.0) + 0.35 * pow(mu, 64.0));
    return c;
}

// height + distance fog (analytic exponential height-fog integral), returns (inscatter, T)
float4 fog_mountains(float3 rel, float3 dir, float dist) {
    QParams* p = pc.p;
    float k = p.fog.x;
    float y0 = p.cam_world.y - p.fog.y;
    float dy = rel.y;
    float base = p.sky_horizon.w * exp(-k * y0);
    float kd = k * dy;
    float integ = abs(kd) > 1e-4 ? (1.0 - exp(-kd)) / kd : 1.0 - 0.5 * kd;
    float od = base * integ * dist;
    float T = max(exp(-od), 1.0 - p.fog.z);
    float mu = saturate(dot(dir, p.sun_dir.xyz));
    float3 ins = lerp(p.fog_color.rgb, p.sun_color.rgb * 0.55, p.fog.w * pow(mu, 6.0));
    // farther air takes the sky's horizon tint
    ins = lerp(ins, sky_color(normalize(float3(dir.x, max(dir.y, 0.0) * 0.3, dir.z))), 0.5);
    return float4(ins, T);
}


// ---------------------------------------------------------------- atmosphere (Hillaire 2020)
// "A Scalable and Production Ready Sky and Atmosphere Rendering Technique": transmittance LUT,
// multiple-scattering LUT (second-order isotropic approximation, Psi = L2 / (1 - f_ms)), sky-view
// LUT for the camera altitude, and an aerial-perspective froxel volume. Units: km, per-km
// coefficients (Earth-like Rayleigh, Mie, ozone). All LUTs live in one float4 buffer (BDA) and
// are filtered manually. Radiance is per unit sun illuminance; `sun_dir.w` scales it.
static const float PI = 3.14159265;
static const float RG = 6360.0;
static const float RT = 6460.0;
static const uint T_W = 256;
static const uint T_H = 64;
static const uint MS_N = 32;
static const uint SV_W = 192;
static const uint SV_H = 108;
static const uint AP_N = 32;
static const uint OFF_T = 0;
static const uint OFF_MS = 16384;
static const uint OFF_SV = 17408;
static const uint OFF_AP = 38144;
static const uint OFF_IRR = 70912;

struct Med { float3 rs; float ms; float3 ext; };
Med medium(float h) {
    QParams* p = pc.p;
    h = max(h, 0.0);
    float dr = exp(-h / 8.0);
    float dm = exp(-h / 1.2) * p.atmo.x;
    float doz = max(0.0, 1.0 - abs(h - 25.0) / 15.0);
    Med m;
    m.rs = float3(5.802e-3, 13.558e-3, 33.1e-3) * dr * p.atmo2.z;
    m.ms = 3.996e-3 * dm;
    // Mie: scattering 3.996e-3, extinction 4.40e-3 (absorption 0.444e-3, single-scattering albedo 0.9)
    m.ext = m.rs + m.ms + 0.444e-3 * dm + float3(0.650e-3, 1.881e-3, 0.085e-3) * doz;
    return m;
}
// nearest positive hit of a sphere centred at the origin, -1 if none
float ray_sphere(float3 ro, float3 rd, float R) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - R * R;
    float disc = b * b - c;
    if (disc < 0.0) return -1.0;
    float sq = sqrt(disc);
    float t0 = -b - sq, t1 = -b + sq;
    if (t0 > 0.0) return t0;
    if (t1 > 0.0) return t1;
    return -1.0;
}
float phase_r(float c) { return 3.0 / (16.0 * PI) * (1.0 + c * c); }
float phase_m(float c) {
    float g = 0.8;
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + c * c) / pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5);
}
// transmittance LUT parameterisation (Bruneton / Hillaire)
float2 t_uv(float r, float mu) {
    float H = sqrt(RT * RT - RG * RG);
    float rho = sqrt(max(0.0, r * r - RG * RG));
    float disc = r * r * (mu * mu - 1.0) + RT * RT;
    float d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
    float dmin = RT - r, dmax = rho + H;
    return float2((d - dmin) / max(dmax - dmin, 1e-6), rho / H);
}
float4 lut2(uint off, uint w, uint h, float2 uv) {
    float2 x = clamp(uv, 0.0, 1.0) * float2(w, h) - 0.5;
    float2 f = frac(x);
    int2 i0 = clamp(int2(floor(x)), int2(0, 0), int2(w - 1, h - 1));
    int2 i1 = min(i0 + 1, int2(w - 1, h - 1));
    float4* a = pc.atmo;
    float4 v00 = a[off + i0.y * w + i0.x], v10 = a[off + i0.y * w + i1.x];
    float4 v01 = a[off + i1.y * w + i0.x], v11 = a[off + i1.y * w + i1.x];
    return lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
}
float3 trans_lut(float r, float mu) {
    // below the horizon the sun is blocked by the planet
    float3 ro = float3(0.0, r, 0.0), rd = float3(sqrt(max(0.0, 1.0 - mu * mu)), mu, 0.0);
    if (ray_sphere(ro, rd, RG) > 0.0) return float3(0.0, 0.0, 0.0);
    return lut2(OFF_T, T_W, T_H, t_uv(r, mu)).rgb;
}
float3 ms_lut(float r, float mu_s) {
    return lut2(OFF_MS, MS_N, MS_N, float2(mu_s * 0.5 + 0.5, (r - RG) / (RT - RG))).rgb;
}
// in-scattering of one step [t, t + dt] along rd from ro (planet-centred km); updates T and L
void scatter_step(float3 ro, float3 rd, float t, float dt, float3 sun, inout float3 T, inout float3 L) {
    float3 pos = ro + rd * (t + 0.5 * dt);
    float r = length(pos);
    float3 up = pos / r;
    float mus = dot(up, sun);
    Med m = medium(r - RG);
    float c = dot(rd, sun);
    float3 sc = m.rs + m.ms;
    float3 S = (m.rs * phase_r(c) + m.ms * phase_m(c)) * trans_lut(r, mus) + sc * ms_lut(r, mus);
    float3 st = exp(-m.ext * dt);
    L += T * (S - S * st) / max(m.ext, float3(1e-7, 1e-7, 1e-7));
    T *= st;
}
float cam_r() { return RG + max(pc.p.cam_world.y * 0.001, 0.002); }
// sky-view LUT mapping: v < 0.5 above the horizon (dense near it), azimuth from the sun
float2 sv_uv(float3 dir) {
    float r = cam_r();
    float vh = sqrt(r * r - RG * RG);
    float beta = acos(clamp(vh / r, -1.0, 1.0));
    float zha = PI - beta;
    float vza = acos(clamp(dir.y, -1.0, 1.0));
    float v;
    if (vza < zha) { float c = sqrt(max(0.0, 1.0 - vza / zha)); v = (1.0 - c) * 0.5; }
    else { v = sqrt(saturate((vza - zha) / beta)) * 0.5 + 0.5; }
    float3 sun = pc.p.sun_dir.xyz;
    float phi = acos(clamp(dot(normalize(float3(dir.x, 0.0, dir.z) + float3(1e-6, 0, 0)), normalize(float3(sun.x, 0.0, sun.z) + float3(1e-6, 0, 0))), -1.0, 1.0));
    return float2(phi / PI, v);
}
float3 sv_dir(float2 uv, float r) {
    float vh = sqrt(r * r - RG * RG);
    float beta = acos(clamp(vh / r, -1.0, 1.0));
    float zha = PI - beta;
    float vza;
    if (uv.y < 0.5) { float c = 1.0 - 2.0 * uv.y; c = 1.0 - c * c; vza = zha * c; }
    else { float c = uv.y * 2.0 - 1.0; vza = zha + beta * c * c; }
    float phi = uv.x * PI; // symmetric about the sun's vertical plane
    float st = sin(vza);
    return float3(st * cos(phi), cos(vza), st * sin(phi));
}
// sun direction in the LUT frame (sun azimuth = 0)
float3 sun_local() { float mus = pc.p.sun_dir.y; return float3(sqrt(max(0.0, 1.0 - mus * mus)), mus, 0.0); }

[shader("compute")]
[numthreads(8, 8, 1)]
void atmoTransmittanceCS(uint3 id : SV_DispatchThreadID) {
    if (id.x >= T_W || id.y >= T_H) return;
    float2 uv = (float2(id.xy) + 0.5) / float2(T_W, T_H);
    float H = sqrt(RT * RT - RG * RG);
    float rho = H * uv.y;
    float r = sqrt(rho * rho + RG * RG);
    float dmin = RT - r, dmax = rho + H;
    float d = dmin + uv.x * (dmax - dmin);
    float mu = d == 0.0 ? 1.0 : clamp((H * H - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
    float3 ro = float3(0.0, r, 0.0), rd = float3(sqrt(max(0.0, 1.0 - mu * mu)), mu, 0.0);
    float tmax = max(ray_sphere(ro, rd, RT), 0.0);
    float3 od = float3(0.0, 0.0, 0.0);
    float dt = tmax / 40.0;
    for (int i = 0; i < 40; i++) od += medium(length(ro + rd * (i + 0.5) * dt) - RG).ext * dt;
    pc.atmo[OFF_T + id.y * T_W + id.x] = float4(exp(-od), 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void atmoMultiScatterCS(uint3 id : SV_DispatchThreadID) {
    if (id.x >= MS_N || id.y >= MS_N) return;
    float2 uv = (float2(id.xy) + 0.5) / float(MS_N);
    float mus = uv.x * 2.0 - 1.0;
    float r = RG + max(uv.y * (RT - RG), 0.002);
    float3 ro = float3(0.0, r, 0.0), sd = float3(sqrt(max(0.0, 1.0 - mus * mus)), mus, 0.0);
    float3 L2 = float3(0, 0, 0), F = float3(0, 0, 0);
    float albedo = pc.p.atmo.y;
    for (int a = 0; a < 8; a++) {
        for (int b = 0; b < 8; b++) {
            float ct = 1.0 - 2.0 * (a + 0.5) / 8.0;
            float phi = 2.0 * PI * (b + 0.5) / 8.0;
            float st = sqrt(max(0.0, 1.0 - ct * ct));
            float3 rd = float3(st * cos(phi), ct, st * sin(phi));
            float tg = ray_sphere(ro, rd, RG);
            float tmax = tg > 0.0 ? tg : max(ray_sphere(ro, rd, RT), 0.0);
            float dt = tmax / 20.0;
            float3 T = float3(1, 1, 1), Lp = float3(0, 0, 0), Fp = float3(0, 0, 0);
            for (int i = 0; i < 20; i++) {
                float3 pos = ro + rd * (i + 0.5) * dt;
                float rp = length(pos);
                Med m = medium(rp - RG);
                float3 sc = m.rs + m.ms;
                float3 stp = exp(-m.ext * dt);
                float3 inv = 1.0 / max(m.ext, float3(1e-7, 1e-7, 1e-7));
                float3 S = sc * trans_lut(rp, dot(pos / rp, sd)) / (4.0 * PI);
                Lp += T * (S - S * stp) * inv;
                Fp += T * (sc - sc * stp) * inv;
                T *= stp;
            }
            if (tg > 0.0) {
                float3 up = normalize(ro + rd * tg);
                float mu_g = dot(up, sd);
                Lp += T * trans_lut(RG + 1e-3, mu_g) * saturate(mu_g) * albedo / PI;
            }
            L2 += Lp;
            F += Fp / (4.0 * PI);
        }
    }
    L2 /= 64.0;
    F = F / 64.0 * (4.0 * PI);
    pc.atmo[OFF_MS + id.y * MS_N + id.x] = float4(L2 / max(1.0 - F, float3(1e-3, 1e-3, 1e-3)), 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void atmoSkyViewCS(uint3 id : SV_DispatchThreadID) {
    if (id.x >= SV_W || id.y >= SV_H) return;
    float2 uv = (float2(id.xy) + 0.5) / float2(SV_W, SV_H);
    float r = cam_r();
    float3 rd = sv_dir(uv, r);
    float3 ro = float3(0.0, r, 0.0);
    float tg = ray_sphere(ro, rd, RG);
    float tmax = tg > 0.0 ? tg : max(ray_sphere(ro, rd, RT), 0.0);
    float3 T = float3(1, 1, 1), L = float3(0, 0, 0);
    float3 sun = sun_local();
    // denser steps near the camera (quadratic distribution)
    float tprev = 0.0;
    for (int i = 0; i < 40; i++) {
        float tn = tmax * pow((i + 1) / 40.0, 2.0);
        scatter_step(ro, rd, tprev, tn - tprev, sun, T, L);
        tprev = tn;
    }
    pc.atmo[OFF_SV + id.y * SV_W + id.x] = float4(L, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void atmoAerialCS(uint3 id : SV_DispatchThreadID) {
    if (id.x >= AP_N || id.y >= AP_N) return;
    QParams* p = pc.p;
    float2 ndc = (float2(id.xy) + 0.5) / float(AP_N) * 2.0 - 1.0;
    float4 d4 = mulc(p.inv_dir[0], p.inv_dir[1], p.inv_dir[2], p.inv_dir[3], float4(ndc, 1.0, 1.0));
    float3 rd = normalize(d4.xyz);
    float3 ro = float3(0.0, cam_r(), 0.0);
    float3 sun = p.sun_dir.xyz;
    float3 T = float3(1, 1, 1), L = float3(0, 0, 0);
    float tprev = 0.0;
    for (uint k = 0; k < AP_N; k++) {
        float tk = p.atmo.z * pow((k + 1.0) / AP_N, 2.0);
        float dt = (tk - tprev) * 0.5;
        scatter_step(ro, rd, tprev, dt, sun, T, L);
        scatter_step(ro, rd, tprev + dt, dt, sun, T, L);
        tprev = tk;
        pc.atmo[OFF_AP + (k * AP_N + id.y) * AP_N + id.x] = float4(L, dot(T, float3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)));
    }
}

groupshared float3 g_up[64];
groupshared float3 g_side[64];
groupshared float3 g_sh[64][9];

// real SH basis to l = 2 (y up; any fixed axis order works as long as eval matches)
void sh9(float3 d, out float b[9]) {
    b[0] = 0.282095;
    b[1] = 0.488603 * d.y; b[2] = 0.488603 * d.z; b[3] = 0.488603 * d.x;
    b[4] = 1.092548 * d.x * d.y; b[5] = 1.092548 * d.y * d.z; b[6] = 0.315392 * (3.0 * d.z * d.z - 1.0);
    b[7] = 1.092548 * d.x * d.z; b[8] = 0.546274 * (d.x * d.x - d.y * d.y);
}
// valley environment seen from the camera (per unit sun illuminance): mean terrain albedo of
// the valley floor (meadow / forest mix) and of the surrounding mountainsides; the band of
// the "sky" below ~11 deg elevation is mountainside, not sky
static const float CANOPY_SKY_FLOOR = 0.08;
static const float CANOPY_SCATTER = 0.05;
static const float CANOPY_SHADOW_KEEP = 0.88;
static const float ENV_SKY_SAT = 0.35;
static const float3 ENV_GROUND_ALB = float3(0.15, 0.17, 0.07);
static const float3 ENV_SLOPE_ALB = float3(0.17, 0.175, 0.09);

// multi-bounce ambient occlusion (Jimenez et al. 2016, "Practical Real-Time Strategies for
// Accurate Indirect Occlusion"): light blocked by nearby geometry is partly re-scattered by it,
// so occlusion of a green surface by green surroundings darkens less and turns greener
float3 ao_multibounce(float v, float3 alb) {
    float3 a = 2.0404 * alb - 0.3324;
    float3 b = -4.7951 * alb + 0.6417;
    float3 c = 2.7552 * alb + 0.6903;
    return max(float3(v, v, v), ((v * a + b) * v + c) * v);
}
// Sky irradiance on an up-facing and an (azimuth-averaged) vertical surface, from the sky-view LUT.
[shader("compute")]
[numthreads(64, 1, 1)]
void atmoIrradianceCS(uint3 id : SV_DispatchThreadID) {
    float r = cam_r();
    float3 eu = float3(0, 0, 0), es = float3(0, 0, 0);
    for (uint k = id.x; k < SV_W * SV_H; k += 64) {
        uint x = k % SV_W, y = k / SV_W;
        float2 uv = (float2(x, y) + 0.5) / float2(SV_W, SV_H);
        float3 d = sv_dir(uv, r);
        if (d.y <= 0.0) continue;
        float3 d0 = sv_dir(float2(uv.x, uv.y - 0.5 / SV_H), r), d1 = sv_dir(float2(uv.x, uv.y + 0.5 / SV_H), r);
        float dth = abs(acos(clamp(d1.y, -1.0, 1.0)) - acos(clamp(d0.y, -1.0, 1.0)));
        // the LUT covers azimuth [0, pi]; the other half is its mirror image
        float dom = 2.0 * (PI / SV_W) * dth * sqrt(max(0.0, 1.0 - d.y * d.y));
        float3 Lr = pc.atmo[OFF_SV + y * SV_W + x].rgb;
        eu += Lr * d.y * dom;
        es += Lr * sqrt(max(0.0, 1.0 - d.y * d.y)) / PI * dom;
    }
    g_up[id.x] = eu;
    g_side[id.x] = es;
    GroupMemoryBarrierWithGroupSync();
    // irradiance on the ground / on slopes from the sun (transmitted, mostly unshadowed) + sky
    float3 e_sky_up = float3(0, 0, 0);
    for (int i = 0; i < 64; i++) e_sky_up += g_up[i];
    // the sky-view LUT is art-tuned (boosted Rayleigh, no aerosol / cloud) and far too
    // saturated to light with directly: keep 35% of its chroma (zenith irradiance B/R ~2, clear-sky measurements are ~1.6-2)
    e_sky_up = lerp(dot(e_sky_up, float3(0.2126, 0.7152, 0.0722)).xxx, e_sky_up, ENV_SKY_SAT);
    float mus = pc.p.sun_dir.y;
    float3 e_sun = pc.p.sun_color.rgb * trans_lut(r, mus) * max(mus, 0.0);
    float3 l_ground = ENV_GROUND_ALB / PI * (e_sun + e_sky_up);
    float3 l_slope = ENV_SLOPE_ALB / PI * (0.9 * e_sun + 0.7 * e_sky_up);
    // L2 SH of the radiance over the whole sphere (LUT frame: sun at azimuth 0, mirror-symmetric
    // in z): sky above the mountain horizon, mountainside / forest wall below ~10-20 deg, valley floor below 0
    float c[9];
    float3 acc[9];
    for (int j = 0; j < 9; j++) acc[j] = float3(0, 0, 0);
    for (uint k = id.x; k < SV_W * SV_H; k += 64) {
        uint x = k % SV_W, y = k / SV_W;
        float2 uv = (float2(x, y) + 0.5) / float2(SV_W, SV_H);
        float3 d = sv_dir(uv, r);
        float3 d0 = sv_dir(float2(uv.x, uv.y - 0.5 / SV_H), r), d1 = sv_dir(float2(uv.x, uv.y + 0.5 / SV_H), r);
        float dth = abs(acos(clamp(d1.y, -1.0, 1.0)) - acos(clamp(d0.y, -1.0, 1.0)));
        float dom = (PI / SV_W) * dth * sqrt(max(0.0, 1.0 - d.y * d.y)); // one mirror half
        float3 L = pc.atmo[OFF_SV + y * SV_W + x].rgb;
        L = lerp(dot(L, float3(0.2126, 0.7152, 0.0722)).xxx, L, ENV_SKY_SAT);
        L = lerp(l_slope, L, smoothstep(0.16, 0.34, d.y));
        if (d.y < 0.0) L = lerp(l_slope, l_ground, smoothstep(0.0, -0.12, d.y));
        for (int m = 0; m < 2; m++) {
            sh9(float3(d.x, d.y, m == 0 ? d.z : -d.z), c);
            for (int j = 0; j < 9; j++) acc[j] += L * (c[j] * dom);
        }
    }
    for (int j = 0; j < 9; j++) g_sh[id.x][j] = acc[j];
    GroupMemoryBarrierWithGroupSync();
    if (id.x == 0) {
        float3 a = float3(0, 0, 0), b = float3(0, 0, 0);
        for (int i = 0; i < 64; i++) { a += g_up[i]; b += g_side[i]; }
        pc.atmo[OFF_IRR] = float4(a, 1.0);
        pc.atmo[OFF_IRR + 1] = float4(b, 1.0);
        for (int j = 0; j < 9; j++) {
            float3 t = float3(0, 0, 0);
            for (int i = 0; i < 64; i++) t += g_sh[i][j];
            pc.atmo[OFF_IRR + 2 + j] = float4(t, 1.0);
        }
    }
}

// irradiance (per unit sun illuminance) of the valley environment on a surface with world normal
// n, from the L2 SH above (Ramamoorthi & Hanrahan 2001: A0 = pi, A1 = 2pi/3, A2 = pi/4)
float3 env_irradiance(float3 n) {
    float2 sh = normalize(pc.p.sun_dir.xz + float2(1e-6, 0.0));
    float3 d = float3(dot(n.xz, sh), n.y, dot(n.xz, float2(-sh.y, sh.x)));
    float c[9];
    sh9(d, c);
    const float A[9] = { PI, 2.0943951, 2.0943951, 2.0943951, 0.7853982, 0.7853982, 0.7853982, 0.7853982, 0.7853982 };
    float3 e = float3(0, 0, 0);
    for (int j = 0; j < 9; j++) e += pc.atmo[OFF_IRR + 2 + j].rgb * (A[j] * c[j]);
    return max(e, float3(0, 0, 0));
}

float3 sky_lut(float3 dir) { return lut2(OFF_SV, SV_W, SV_H, sv_uv(dir)).rgb; }
// aerial perspective at camera-relative `rel` (m): (inscatter per unit sun illuminance, T)
float4 aerial(float3 rel, float dist) {
    float4 c = vp_main(float4(rel, 1.0));
    float2 ndc = c.xy / c.w;
    float2 uv = ndc * 0.5 + 0.5;
    float sz = sqrt(saturate(dist * 0.001 / pc.p.atmo.z)) * AP_N - 1.0;
    float4 a = float4(0, 0, 0, 1);
    float2 x = clamp(uv, 0.0, 1.0) * AP_N - 0.5;
    float2 f = frac(x);
    int2 i0 = clamp(int2(floor(x)), int2(0, 0), int2(AP_N - 1, AP_N - 1));
    int2 i1 = min(i0 + 1, int2(AP_N - 1, AP_N - 1));
    float4* A = pc.atmo;
    float4 s[2];
    for (int k = 0; k < 2; k++) {
        int z = clamp(int(floor(sz)) + k, 0, int(AP_N) - 1);
        uint base = OFF_AP + z * AP_N * AP_N;
        float4 v00 = A[base + i0.y * AP_N + i0.x], v10 = A[base + i0.y * AP_N + i1.x];
        float4 v01 = A[base + i1.y * AP_N + i0.x], v11 = A[base + i1.y * AP_N + i1.x];
        s[k] = lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
    }
    // in front of the first slice: fade from "no air" to slice 0
    if (sz < 0.0) return lerp(float4(0, 0, 0, 1), s[0], saturate(sz + 1.0));
    return lerp(s[0], s[1], frac(sz));
}

// ---------------------------------------------------------------- terrain
struct VSOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 rel;
    [[vk::location(1)]] float3 nrm;
    [[vk::location(2)]] float4 attr;
};

[shader("vertex")]
VSOut terrainVS(uint vid : SV_VulkanVertexID) {
    QVertex v = pc.v[vid];
    VSOut o;
    o.position = vp_view(v.pos);
    o.rel = v.pos;
    o.nrm = v.nrm;
    o.attr = v.attr;
    return o;
}

[shader("vertex")]
float4 shadowVS(uint vid : SV_VulkanVertexID) : SV_Position {
    QVertex v = pc.v[vid];
    return vp_casc(pc.cascade, float4(v.pos, 1.0));
}

// ---------------------------------------------------------------- instanced geometry
// Drawn with vkCmdDrawIndexedIndirectCount; SV_VulkanInstanceID includes firstInstance (the
// mesh's region start in `visible`), exactly like mesh.slang.
// `visible` entries: instance index | LOD dither code << 24 (qcull.slang)
// Static wind pose of the blade grass (deterministic: a frozen instant of the reference's
// layered wind): a steady lean down-wind plus rolling gust fronts (a noise envelope sliding
// along the wind direction) and a per-blade flutter; displacement grows with height^2 so the
// roots stay put, and the blade is shortened to keep its length.
float3 grass_wind(float3 world, float hf, float y, float rnd) {
    float2 wd = float2(0.8, 0.6);
    float along = dot(world.xz, wd);
    float gust = smoothstep(0.25, 0.75, 0.5 + 0.5 * vnoised(world.xz / 7.0 + float2(along * 0.05, 0.0)).x);
    float wave = sin(along * 0.9 + 2.0 * vnoised(world.xz / 3.0).x);
    float flutter = rnd - 0.5;
    float amt = (0.10 + 0.22 * gust * (0.6 + 0.4 * wave) + 0.05 * flutter) * hf * hf * max(y, 0.0) / 0.35;
    return float3(wd.x * amt, -0.35 * amt * amt / max(y, 0.05), wd.y * amt);
}

float3 inst_rel(uint vid, uint iid, out float3 n, out QVertex v, out uint mat) {
    QInstance I = pc.inst[pc.visible[iid] & 0xFFFFFFu];
    v = pc.v[vid];
    n = I.model[0].xyz * v.nrm.x + I.model[1].xyz * v.nrm.y + I.model[2].xyz * v.nrm.z;
    mat = I.material_id;
    float3 rel = mulc(I.model[0], I.model[1], I.model[2], I.model[3], float4(v.pos, 1.0)).xyz;
    if (v.attr.w > 4.5 && v.attr.w < 5.5) {
        // blade grass tiles (quarter-turn yaw, scale (1, h, 1)): shear onto the ground plane
        // (dH/dx, dH/dz quantised to 6 bits each in material bits 4..15), blades stay upright
        float2 g = float2(float((mat >> 4) & 63u), float((mat >> 10) & 63u)) / 31.5 - 1.0;
        float3 off = mulc(I.model[0], I.model[1], I.model[2], I.model[3], float4(v.pos.x, 0.0, v.pos.z, 0.0)).xyz;
        rel.y += g.x * off.x + g.y * off.z;
        rel += grass_wind(rel + pc.p.cam_world.xyz, v.attr.y, v.pos.y, v.attr.z);
    }
    return rel;
}

struct InstOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 rel;
    [[vk::location(1)]] float3 nrm;
    [[vk::location(2)]] float4 attr;
    [[vk::location(3)]] nointerpolation uint lod; // dither code (qcull.slang)
    [[vk::location(4)]] float2 uv;                // needle / leaf card coordinates (code 6)
};

float ign(float2 px) { return frac(52.9829189 * frac(dot(px, float2(0.06711056, 0.00583715)))); }

// Needle / leaf cards (material code 6): procedural alpha from the card coordinates (u across,
// twig at 0.5; v along the shoot, 0 at the trunk end). u >= 2 selects broadleaf (alder /
// willow) leaves. Mirrored exactly by impostor::card_alpha on the CPU (atlas bake).
// `cover` = 1 when the pattern is sub-pixel: then keep the envelope at the pattern's mean
// coverage with a stable dither instead of aliasing.
// Spruce shoot clusters (card u in [4, 5]): the branch axis and alternate side shoots every
// 0.08 of the branch length, angled ~55 deg forward and shortening toward the tip, each a
// tapering bottle-brush of needles (ragged edge from a fine needle-grain hash); between them
// open gaps (sky / interior shows through). Coordinates in branch-length units: across
// x = (u - 0.5) * 0.62 (the card is 0.62 of the branch length wide), along v. Mirrored by
// `impostor::spruce_alpha` (the bake). Returns the coverage (0 / 1) before dithering.
// .x coverage, .y tone: per-shoot brightness (0.7-1.3) x lighter young growth toward each shoot
// tip, the bare axis darker (the clusters read as lit shoots over a shaded interior)
float2 spruce_card(float u, float v) {
    float x = (u - 0.5) * 0.62;
    float env = 0.31 * (1.0 - pow(saturate(v), 3.0)) * saturate(v / 0.04);
    if (abs(x) > env) return float2(0.0, 1.0);
    float best = abs(x) / (0.06 * (1.0 - 0.5 * saturate(v)));
    float tone = 0.8;
    float fi = floor(v / 0.08);
    for (int k = -3; k <= 0; k++) {
        float i = fi + float(k);
        float side = (frac(i * 0.5) < 0.25) ? 1.0 : -1.0;
        float v0 = i * 0.08 + 0.02;
        float len = 0.38 * (1.0 - 0.62 * saturate(v0));
        float2 dir = float2(side * 0.82, 0.572);
        float2 pa = float2(x, v - v0);
        float h = saturate(dot(pa, dir) / len);
        float d = length(pa - dir * (len * h));
        float q = d / (0.055 * (1.0 - 0.45 * h));
        if (q < best) {
            best = q;
            tone = (0.7 + 0.6 * hash2(float2(i, 3.0))) * (0.75 + 0.5 * h);
        }
    }
    float n = hash2(floor(float2(u * 64.0, v * 96.0)) + 7.0);
    return float2(best < 0.85 + 0.55 * n ? 1.0 : 0.0, tone * (1.12 - 0.3 * saturate(best)));
}
float spruce_keep(float u, float v) { return spruce_card(u, v).x; }

// Conifer cards fade out as they turn edge-on to the view (geometric normal from screen
// derivatives of the camera-relative position): crossed cards seen end-on otherwise show as
// bright X / line patterns through a crown. Same expression in the prepass and the shading FS.
bool card_facing_keep(float3 rel, float2 px) {
    float3 gn = normalize(cross(ddx(rel), ddy(rel)));
    float f = abs(dot(gn, normalize(rel)));
    return ign(px + float2(41.0, 7.0)) < smoothstep(0.06, 0.30, f);
}

bool card_alpha(float2 uv, float cover, float2 px) {
    if (uv.x > 3.5) {
        float keep = spruce_keep(uv.x - 4.0, uv.y);
        if (cover <= 0.0) return keep > 0.5;
        // sub-pixel shoots: a crown is nearly opaque between its tiers (0.9; the resolved pattern
        // covers 0.75; `cover` ramps smoothly with the footprint, so no band), so crossed cards do
        // not show as see-through X sheets
        return ign(px + 17.0) < lerp(keep, 0.9, cover);
    }
    bool leaf = uv.x > 1.5;
    float u = leaf ? uv.x - 2.0 : uv.x;
    float v = uv.y;
    float a = abs(u - 0.5) * 2.0;
    float env = 0.95 * (1.0 - pow(saturate(v), 2.5)) * saturate(v / 0.05);
    if (a > env) return false;
    if (a < 0.07 && v < 0.97) return true;
    // cover: 0 = needle rows / leaves resolved (exact pattern), 1 = sub-pixel (dithered mean
    // coverage 0.78: several card layers overlap in a crown); in between the keep probability blends smoothly, so there is no band
    // where the pattern switches and mid-distance needles do not alias into speckle
    bool pat;
    if (leaf) {
        float2 q = float2(a * 2.2, frac(v * 7.0 + (u < 0.5 ? 0.0 : 0.5)));
        pat = length(q - float2(1.1, 0.5) * float2(0.8, 1.0)) < 0.62;
    } else {
        pat = frac(v * 22.0 - a * 1.6) < 0.58;
    }
    if (cover <= 0.0) return pat;
    return ign(px + 17.0) < lerp(pat ? 1.0 : 0.0, 0.78, cover);
}

[shader("vertex")]
InstOut instVS(uint vid : SV_VulkanVertexID, uint iid : SV_VulkanInstanceID) {
    float3 n; QVertex v; uint mat;
    float3 rel = inst_rel(vid, iid, n, v, mat);
    InstOut o;
    o.position = vp_view(rel);
    o.rel = rel;
    o.nrm = n;
    o.lod = pc.visible[iid] >> 24;
    o.uv = float2(0.0, 0.0);
    // x ao, y height fraction in the plant, z = tag + random (integer part: species | crowding
    // << 3 from material_id bits 10.., fraction: the 10 random bits), w material code (>= 2)
    o.attr = float4(v.attr.x, v.attr.y, float(mat >> 10) + float(mat & 1023u) / 1024.0, v.attr.w);
    // blade grass: the mesh carries a per-blade random (clump + blade) in attr.z
    if (v.attr.w > 4.5 && v.attr.w < 5.5) o.attr.z = frac(v.attr.z + float(mat & 15u) / 16.0) * 0.999;
    // cards: (u, v) from attr.x / attr.z; AO toward the trunk (inner shoot) and the lower crown
    if (v.attr.w > 5.5) {
        o.uv = float2(v.attr.x, v.attr.z);
        o.attr.x = (0.32 + 0.68 * smoothstep(0.0, 0.85, v.attr.z)) * (0.6 + 0.4 * v.attr.y);
    }
    return o;
}

struct InstShadowOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv;
    [[vk::location(1)]] nointerpolation float code;
};

// shadow casters with alpha (needle / leaf cards): cut-out shadows
[shader("vertex")]
InstShadowOut instShadowAlphaVS(uint vid : SV_VulkanVertexID, uint iid : SV_VulkanInstanceID) {
    float3 n; QVertex v; uint mat;
    float3 rel = inst_rel(vid, iid, n, v, mat);
    InstShadowOut o;
    o.position = vp_casc(pc.cascade, float4(rel, 1.0));
    o.uv = float2(v.attr.x, v.attr.z);
    o.code = v.attr.w;
    return o;
}

[shader("vertex")]
float4 instShadowVS(uint vid : SV_VulkanVertexID, uint iid : SV_VulkanInstanceID) : SV_Position {
    float3 n; QVertex v; uint mat;
    float3 rel = inst_rel(vid, iid, n, v, mat);
    return vp_casc(pc.cascade, float4(rel, 1.0));
}

// sheen: view-dependent forward/grazing scatter of a grass sward (meadow terrain only), and
// lean: the sward's lean direction (xz) for its with-grain / against-grain brightness
struct Mat { float3 albedo; float rough; float3 n; float trans; float occ; float sheen; float2 lean; };

// Sub-mesh relief: derivative-damped ("eroded") fbm over one 2D projection, octaves rotated.
// Only octaves the mesh cannot represent (lambda < ~4 mesh cells) and the pixel can resolve
// (lambda > 3 footprints) contribute: the far ring gets the 2-60 m relief it lacks, the near
// voxel mesh only the fine rest. Returns (height m, d/du, d/dv) of the projected height field.
float3 eroded(float2 x, float lam0, int oct, float foot, float cell, float k) {
    float h = 0.0;
    float2 g = float2(0.0, 0.0), dsum = float2(0.0, 0.0);
    float2x2 M = float2x2(1.0, 0.0, 0.0, 1.0);
    float2x2 R = float2x2(0.8, -0.6, 0.6, 0.8);
    float lam = lam0;
    for (int o = 0; o < oct; o++) {
        if (lam < foot * 3.0) break;
        float w = saturate(lam / (foot * 3.0) - 1.0) * (1.0 - smoothstep(2.0 * cell, 5.0 * cell, lam));
        float2 q = mul(M, x) / lam + float2(o * 7.31, o * 3.17);
        float3 nv = vnoised(q);
        float2 d = mul(transpose(M), nv.yz); // d n / d (x / lam)
        dsum += d;
        float e = 1.0 / (1.0 + 0.6 * dot(dsum, dsum)); // valleys smooth, flanks keep detail
        float a = k * lam * w * e;
        h += a * nv.x;
        g += a * d / lam;
        M = mul(R, M);
        lam *= 0.5;
    }
    return float3(h, g);
}

float3 relief_tri(float3 p, float3 w, float lam0, int oct, float foot, float cell, float k) {
    float3 g = float3(0.0, 0.0, 0.0);
    if (w.y > 0.02) { float3 r = eroded(p.xz, lam0, oct, foot, cell, k); g += float3(r.y, 0.0, r.z) * w.y; }
    if (w.x > 0.02) { float3 r = eroded(p.zy + 41.0, lam0, oct, foot, cell, k); g += float3(0.0, r.z, r.y) * w.x; }
    if (w.z > 0.02) { float3 r = eroded(p.xy + 83.0, lam0, oct, foot, cell, k); g += float3(r.y, r.z, 0.0) * w.z; }
    return g;
}


// Steep-rock structure: ridged, fall-line-stretched relief (buttresses, arêtes, couloirs,
// fracture grooves) at 110 / 48 / 21 / 9 / 4 m on one vertical projection. Faded only by the
// pixel footprint: the far-ring mesh has no structure at these scales on faces, and the near
// voxel mesh only from ~4 m down. Returns (crest-ness in [-1, 1], d/du, d/dy).
float3 buttress(float2 x, float foot) {
    float c = 0.0, wsum = 0.0;
    float2 g = float2(0.0, 0.0);
    float lam = 110.0, amp = 1.0;
    for (int o = 0; o < 5; o++) {
        float w = saturate(lam / (foot * 3.0) - 1.0);
        if (w <= 0.0) break;
        // warp so grooves wander and branch instead of running in straight parallel lines
        float3 wv = vnoised(x / (lam * 2.3) + float2(o * 5.3, o * 2.9));
        float2 q = float2(x.x / lam + 0.55 * wv.x, x.y / (lam * 2.4)) + float2(o * 13.7, o * 4.1);
        float3 nv = vnoised(q);
        float r = 1.0 - abs(nv.x);
        float sg = nv.x >= 0.0 ? -1.0 : 1.0;
        // d(r^2) / d(x): chain through q (warp gradient ignored: it is a low-frequency shift)
        float2 dr = 2.0 * r * sg * float2(nv.y / lam, nv.z / (lam * 2.4));
        float a = amp * w;
        c += a * (r * r * 2.0 - 1.0);
        wsum += a;
        g += a * lam * dr;
        lam *= 0.44;
        amp *= 0.62;
    }
    return float3(c / max(wsum, 1e-3), g);
}
float4 buttress_tri(float3 p, float3 w, float foot) {
    float wx = w.x / max(w.x + w.z, 1e-4), wz = 1.0 - wx;
    float c = 0.0;
    float3 g = float3(0.0, 0.0, 0.0);
    if (wx > 0.02) { float3 r = buttress(p.zy + 57.0, foot); c += r.x * wx; g += float3(0.0, r.z, r.y) * wx; }
    if (wz > 0.02) { float3 r = buttress(p.xy + 19.0, foot); c += r.x * wz; g += float3(r.y, r.z, 0.0) * wz; }
    return float4(c, g);
}

// `boulder` = 1: loose blocks (instanced boulders): the same bedrock / lichen / moss model, but
// no meadow, scree or snow-on-gentle-slope blending onto the block faces (snow only up high).
Mat mountain_material(float3 world, float3 n, float4 attr, float foot, float dist, float nlen, float boulder) {
    float3 w = pow(abs(n), float3(4.0));
    w /= (w.x + w.y + w.z);
    float slope0 = 1.0 - n.y;
    float h = world.y;
    float ao = attr.x, cav = attr.y, cvx = attr.z;
    // approximate mesh cell here: far rings use 0.004 x radius, near voxels 0.5-8 m
    float cell = max(0.5, 0.004 * dist);

    // macro fields (lambda >= 50 m): hold up at any distance, never fade
    float m1 = vnoised(world.xz / 610.0 + 3.0).x;
    float m2 = vnoised(world.xz / 190.0 + 11.0).x;
    float m3 = vnoised(world.xz / 63.0 + 5.0).x;
    float m4 = vnoised(world.xz / 21.0 + 17.0).x;
    float macro = 0.55 * m1 + 0.3 * m2 + 0.15 * m3;

    // sub-mesh relief normal: rougher on rock than on meadow, by geometric slope + macro
    float krock = lerp(0.045, 0.12, smoothstep(0.12, 0.45, slope0 + 0.1 * m2));
    float3 g_rel = relief_tri(world, w, 128.0, 7, foot, cell, krock);
    // large steep faces: buttresses / fracture grooves (strong tilt) + cavity darkening
    float rocky = max(smoothstep(0.20, 0.36, slope0), smoothstep(1150.0, 1350.0, h + 60.0 * m2));
    float steep_face = smoothstep(0.12, 0.40, slope0 + 0.08 * m2) * smoothstep(0.05, 0.25, 1.0 - w.y) * rocky;
    float4 bt = steep_face > 0.0 ? buttress_tri(world, w, foot) : float4(0.0, 0.0, 0.0, 0.0);
    float3 g_bt = bt.yzw * 0.42 * steep_face;
    g_rel += g_bt;
    float3 nr = bump(n, g_rel, 1.0);
    // effective slope: materials follow the relief (outcrops poke out of meadows)
    float slope = 1.0 - nr.y;

    float4 r1 = triplanar(world, w, 6.0, 5, foot);
    float4 r2 = triplanar(world, w, 0.9, 3, foot);
    // rock types vary over hundreds of metres (granite grey / darker gneiss / warm oxidised)
    float region = vnoised(world.xz / 420.0 + 5.0).x + 0.5 * vnoised(world.xz / 130.0).x;
    float3 rock = lerp(float3(0.17, 0.165, 0.155), float3(0.32, 0.29, 0.25), saturate(0.5 + 0.7 * region));
    rock *= 1.0 + 0.22 * macro + 0.25 * r1.x + 0.10 * r2.x;
    // lichen / weathering: darker, slightly green-brown blotches on less steep rock
    float lichen = smoothstep(0.15, 0.55, m3 + 0.4 * m2) * (1.0 - smoothstep(0.55, 0.9, slope));
    rock = lerp(rock, rock * float3(0.72, 0.74, 0.66), 0.55 * lichen);
    // strata: dipping, warped bedding planes at three scales (80 / 24 / 6.5 m); bands modulate
    // albedo and ledges give a detail normal. The 80 m set keeps cliffs structured far away.
    float3 wp = vnoised(world.xz / 650.0 + 2.0);
    float3 dsc = float3(0.11 + 25.0 * wp.y / 650.0, 1.0, 0.07 + 25.0 * wp.z / 650.0);
    float sc = world.y + 0.11 * world.x + 0.07 * world.z + 25.0 * wp.x;
    float3 st0 = vnoised(float2(sc / 80.0, 1.3 + world.z * 0.0002));
    float3 st1 = vnoised(float2(sc / 24.0, 3.7 + world.x * 0.0004));
    float3 st2 = vnoised(float2(sc / 6.5, 9.1 + world.z * 0.0011));
    float w0 = saturate(80.0 / (foot * 4.0) - 1.0);
    float w1 = saturate(24.0 / (foot * 4.0) - 1.0), w2 = saturate(6.5 / (foot * 4.0) - 1.0);
    float steep = smoothstep(0.18, 0.5, slope);
    float bandv = (0.45 * st0.x * w0 + 0.35 * st1.x * w1 + 0.2 * st2.x * w2) * steep;
    rock *= lerp(float3(0.80, 0.83, 0.88), float3(1.22, 1.06, 0.88), saturate(0.5 + 1.0 * bandv));
    rock *= 1.0 - 0.18 * steep * saturate(-st2.x * 2.0) * w2; // dark recessed shaly beds
    float3 g_strata = (12.0 * st0.y / 80.0 * w0 + 5.0 * st1.y / 24.0 * w1 + 1.4 * st2.y / 6.5 * w2) * steep * dsc;
    rock = lerp(rock, rock * float3(0.75, 0.72, 0.70), saturate(cav * 2.0)); // darker in cracks
    // grooves collect water staining and debris (darker), crests are fresher and paler
    rock *= 1.0 + steep_face * (0.30 * bt.x - 0.06);
    // water / varnish streaks down the fall line (narrow, long), the dominant cue on real faces
    float wxs = w.x / max(w.x + w.z, 1e-4);
    float su = lerp(world.x, world.z, wxs);
    float sw = saturate(18.0 / (foot * 4.0) - 1.0);
    float sk = vnoised(float2(su / 17.0, world.y / 150.0) + 41.0).x * sw + 0.5 * vnoised(float2(su / 6.5, world.y / 60.0) + 7.0).x * saturate(6.5 / (foot * 4.0) - 1.0);
    rock *= 1.0 - steep_face * 0.22 * smoothstep(0.1, 0.8, sk) * (0.6 + 0.4 * m3);

    // forest-floor outcrops (below ~1000 m): moss on the upward-facing parts, darker weathered
    // and lichen-grey rock elsewhere; bare, pale rock is an above-treeline look
    float lowland = smoothstep(1060.0, 880.0, h + 40.0 * m2);
    float moss = lowland * smoothstep(0.30, 0.75, nr.y + 0.25 * m4 + 0.15 * r1.x) * smoothstep(-0.4, 0.3, m3 + 0.3 * m4);
    rock *= lerp(1.0, 0.72, lowland);
    rock = lerp(rock, float3(0.055, 0.075, 0.035) * (1.0 + 0.4 * r1.x), 0.85 * moss);

    // jointed outcrops (CPU: attr.w = -rock weight, |n| - 1 = joint-crack proximity from the
    // field's exact joint lattice): fresh grey-brown blocks with lichen, dark open joints,
    // block-scale tone variation; moss only on the flat block tops
    float tor_w = smoothstep(0.1, 0.55, -attr.w);
    float crack = saturate(nlen - 1.0);
    if (tor_w > 0.0) {
        float bt6 = vnoised(world.xz / 6.5 + world.y / 9.0 + 51.0).x;
        float3 tr = lerp(float3(0.12, 0.115, 0.10), float3(0.235, 0.22, 0.19), saturate(0.5 + 0.9 * bt6 + 0.25 * region));
        tr *= 1.0 + 0.22 * r1.x + 0.12 * r2.x;
        float lich = smoothstep(0.2, 0.7, m4 + 0.4 * r1.x) * (0.4 + 0.6 * smoothstep(0.3, 0.8, nr.y));
        tr = lerp(tr, tr * float3(0.72, 0.78, 0.58) + float3(0.02, 0.025, 0.008), 0.7 * lich);
        tr *= 1.0 - 0.7 * crack;
        float top_moss = smoothstep(0.82, 0.96, nr.y) * smoothstep(-0.2, 0.4, m3 + 0.3 * m4);
        tr = lerp(tr, float3(0.06, 0.08, 0.035) * (1.0 + 0.4 * r1.x), 0.7 * top_moss);
        rock = lerp(rock, tr, tor_w);
    }

    // scree / talus: lighter, granular, in fans below cliffs (concave, mid slopes)
    float4 s1 = triplanar(world, w, 0.35, 3, foot);
    float3 scree = float3(0.33, 0.315, 0.29) * (1.0 + 0.25 * s1.x + 0.2 * r1.x + 0.12 * m3) * lerp(1.0, 0.6, lowland);

    // meadow: moisture (hollows greener), altitude/convexity (drier, browner), shrub patches
    float4 g1 = triplanar(world, w, 2.5, 4, foot);
    float moist = saturate(0.5 + 1.6 * cav - 1.0 * cvx + 0.5 * m2);
    float3 grass = lerp(float3(0.120, 0.158, 0.078), float3(0.068, 0.125, 0.040), moist);
    // mid-scale clumps / dry patches (12 m, 4 m) keep the meadow from reading as a flat carpet
    float c12 = vnoised(world.xz / 12.0 + 23.0).x * saturate(12.0 / (foot * 4.0) - 1.0);
    float c4 = vnoised(world.xz / 4.3 + 29.0).x * saturate(4.3 / (foot * 4.0) - 1.0);
    grass *= 1.0 + 0.30 * g1.x + 0.18 * m3 + 0.22 * c12 + 0.15 * c4;
    grass = lerp(grass, float3(0.17, 0.175, 0.112), 0.28 * smoothstep(0.35, 0.8, c12 + 0.5 * m4));
    grass = lerp(grass, float3(0.27, 0.23, 0.15), saturate((h - 1000.0) / 450.0 + 0.5 * cvx - 0.35 + 0.3 * m1));
    // sward texture (carries the blade field's structure out to ~250 m): the sward leans
    // downslope (wind direction on the flat), so its pattern is streaked along the fall line;
    // tufts / flattened patches (0.25 m), clumps (0.8 m) and patches (2.5 m) vary tone and hue
    // (green <-> straw), each octave fading by pixel footprint (no shimmer, no fade band)
    // lean direction: the prevailing wind, turning slowly over ~200 m (a smooth field: following
    // the local fall line made the streaks swirl into brush-stroke patches on gentle ground)
    float la = 0.28 + 1.2 * m2 + 0.5 * m3;
    float2 fl = float2(cos(la), sin(la));
    // texture frame: a FIXED rotation (world coords are ~8 km, so a spatially varying rotation
    // spins the pattern into whorls); the slowly turning `fl` only drives sheen and grain
    const float2 fw = float2(0.96, 0.28);
    float2 uv = float2(dot(world.xz, fw), dot(world.xz, float2(-fw.y, fw.x)));
    // footprint on the ground plane: at grazing view a pixel covers foot / |n.v| along the view
    // direction; the geometric mean (area-preserving) of the two axes keeps the octaves from
    // under-filtering into horizontal smears without blurring them away
    float gfoot = foot / sqrt(max(abs(dot(n, normalize(pc.p.cam_world.xyz - world))), 0.04));
    float o25 = saturate(0.25 / (gfoot * 4.0) - 1.0), o8 = saturate(0.8 / (gfoot * 4.0) - 1.0), o25b = saturate(2.5 / (gfoot * 4.0) - 1.0);
    // (round 10) no single grain direction: one fixed 2.5:1 streak frame read as a combed /
    // brushed carpet at 20-60 m. Each octave is now two mildly anisotropic (1.35:1) noises in
    // frames ~65 deg apart or an isotropic one, plus isotropic tussock clumps, dry patches
    // and sparse late flowers (all fading by pixel footprint)
    float2 uv2 = float2(dot(world.xz, float2(-0.50, 0.87)), dot(world.xz, float2(-0.87, -0.50)));
    float2 uv3 = float2(dot(world.xz, float2(0.42, -0.91)), dot(world.xz, float2(0.91, 0.42)));
    // each octave is evaluated only where its footprint weight is non-zero (identical result;
    // far terrain, most of the frame, skips them)
    float t25 = 0.0, t8 = 0.0, t25b = 0.0, clump = 0.0, dry = 0.0;
    if (o25 > 0.0) t25 = 0.65 * (vnoised(uv * float2(2.6, 3.5) + 61.0).x + vnoised(uv2 * float2(2.6, 3.5) + 13.0).x) * o25;
    if (o8 > 0.0) {
        t8 = (0.6 * vnoised(world.xz * 1.05 + 67.0).x + 0.55 * vnoised(uv3 * float2(0.85, 1.15) + 29.0).x) * o8;
        clump = smoothstep(0.15, 0.6, vnoised(world.xz * 0.72 + 81.0).x) * o8;
    }
    if (o25b > 0.0) t25b = vnoised(world.xz * 0.30 + 71.0).x * o25b;
    float dry_w = saturate(0.5 / (gfoot * 4.0) - 0.2);
    if (dry_w > 0.0) dry = smoothstep(0.3, 0.75, vnoised(world.xz * 0.11 + 91.0).x + 0.45 * t8) * dry_w;
    float sward = 0.22 * t25 + 0.30 * t8 + 0.32 * t25b - 0.10 * clump;
    float strawy = smoothstep(0.15, 0.7, t8 + 0.6 * t25b + 0.3 * m4) * 0.22 + 0.35 * dry;
    // shaded gaps between tufts / flattened patches: the sward's self-shadowing reads as dark
    // speckle at mid range (a smooth carpet without it)
    float gaps = smoothstep(0.05, 0.55, -(t25 + 0.6 * t8)) * o8;
    float sw_w = smoothstep(18.0, 40.0, dist); // the blades carry this structure inside ~40 m
    grass *= (1.0 + sward * sw_w) * (1.0 - 0.22 * gaps * sw_w);
    grass = lerp(grass, grass * float3(0.86, 1.02, 0.84), 0.25 * clump * sw_w);
    grass = lerp(grass, grass * float3(1.18, 1.12, 0.94), strawy * sw_w);
    // late flowers (yarrow white / hawkbit yellow) as a sparse speckle while a 0.3 m cell is
    // still >~1.5 px, matching the blade field's flower heads inside 40 m
    float fo = saturate(0.3 / (gfoot * 4.0) - 1.0) * sw_w;
    float fh = fo > 0.0 ? hash2(floor(world.xz * 3.3) + 401.0) : 0.0;
    if (fh > 0.975) fo *= smoothstep(-0.2, 0.4, vnoised(world.xz * 0.05 + 7.0).x);
    if (fh > 0.975) grass = lerp(grass, fh > 0.988 ? float3(0.55, 0.53, 0.47) : float3(0.50, 0.40, 0.08), 0.4 * fo);
    // shrub / dwarf-pine patches: soft-edged, broken up by a finer field (no camo blobs)
    float shrub = smoothstep(-0.1, 0.7, 0.5 * m3 + 0.3 * m2 + 0.35 * m4 + 0.4 * cav) * smoothstep(1300.0, 1050.0, h);
    grass = lerp(grass, float3(0.07, 0.09, 0.045), 0.45 * shrub);
    // near the camera the meadow is drawn as blades (instances to ~46 m): the terrain under
    // them is the thatch / soil layer between the stems, darker and browner; it fades back to
    // the full meadow colour as the blades dither out
    float under = 1.0 - smoothstep(27.0, 47.0, dist);
    grass = lerp(grass, grass * float3(0.72, 0.66, 0.58), under);
    float3 snow = float3(0.86, 0.88, 0.92) * (1.0 + 0.03 * r2.x);

    float jitter = 60.0 * vnoised(world.xz / 230.0).x + 25.0 * r1.x + 40.0 * m2;
    // snow holds on gentle slopes above a ragged snowline, and in hollows a bit lower
    float snow_w = smoothstep(1450.0, 1600.0, h + jitter + 120.0 * cav) * smoothstep(0.40, 0.26, slope - 0.08 * cav);
    snow_w = max(snow_w, smoothstep(1800.0, 2000.0, h + jitter) * smoothstep(0.62, 0.46, slope));
    float grass_w = smoothstep(1350.0, 1150.0, h + jitter) * smoothstep(0.34, 0.15, slope + 0.12 * r1.x - 0.05 * m3) * (1.0 - snow_w);
    float fan = saturate(0.3 + 1.6 * cav - cvx + 0.6 * m2);
    float scree_w = smoothstep(0.16, 0.30, slope) * smoothstep(0.66, 0.42, slope) * fan * (1.0 - snow_w);
    scree_w *= (1.0 - grass_w) * (1.0 - 0.8 * lowland); // no bright talus on the forest floor
    if (boulder > 0.5) {
        grass_w = 0.0;
        scree_w = 0.0;
        snow_w *= smoothstep(1500.0, 1700.0, h);
    }
    float rock_w = saturate(1.0 - snow_w - grass_w - scree_w);

    Mat m;
    m.albedo = rock * rock_w + scree * scree_w + grass * grass_w + snow * snow_w;
    m.rough = 0.78 * rock_w + 0.88 * scree_w + 0.95 * grass_w + 0.5 * snow_w;
    // detail normals: gradient of height fields with amplitudes in metres; snow smooths relief
    float3 g = (r1.yzw * 0.9 + r2.yzw * 0.10 + g_strata) * rock_w + s1.yzw * 0.04 * scree_w + g1.yzw * 0.10 * grass_w + r2.yzw * 0.02 * snow_w;
    g += g_rel * (1.0 - 0.6 * snow_w);
    m.n = bump(n, g, 1.0);
    m.trans = 0.0;
    m.sheen = grass_w * smoothstep(15.0, 45.0, dist) * (1.0 - 0.6 * shrub);
    m.lean = fl;
    // cavity occlusion of the sky / bounce light in the grooves of steep faces
    m.occ = 1.0 + steep_face * rock_w * (0.12 * saturate(bt.x * 1.5) - 0.6 * saturate(-bt.x * 2.0));
    m.occ *= 1.0 - 0.55 * crack * tor_w; // open joints see little sky
    return m;
}

// Instanced materials: 2 conifer foliage, 3 bark, 4 boulder, 5 grass tuft.
Mat inst_material(float3 world, float3 n, float4 attr, float foot, float dist) {
    float code = attr.w, hf = attr.y;
    float tg = floor(attr.z), rnd = attr.z - tg;
    uint tag = uint(tg), spc = tag & 7u;
    float crowd = float((tag >> 3) & 3u) / 3.0;
    float3 w = pow(abs(n), float3(4.0));
    w /= (w.x + w.y + w.z);
    Mat m;
    m.trans = 0.0;
    m.occ = 1.0;
    m.sheen = 0.0;
    m.lean = float2(0.0, 0.0);
    if (code < 2.5 || code > 5.5) {
        // spruce/fir foliage (2: crown meshes, 6: needle / leaf cards): dark blue-green, lighter young growth at the tips/top, needle
        // clumps as fine triplanar noise; strong self-shadowing toward the trunk/bottom
        // species: 0 spruce, 1 fir (darker, bluer), 2 larch (October gold, a few still
        // green-yellow), 3 snag (dead, grey), 4 alder / willow thicket (autumn olive-yellow),
        // 5 mugo pine (dark)
        float4 c1 = triplanar(world, w, 0.45, 3, foot);
        float3 a = lerp(float3(0.045, 0.066, 0.030), float3(0.082, 0.098, 0.038), rnd);
        m.trans = 0.35;
        if (spc == 1u) a = lerp(float3(0.034, 0.054, 0.030), float3(0.056, 0.078, 0.040), rnd);
        if (spc == 2u) {
            a = rnd < 0.22 ? float3(0.12, 0.13, 0.05) : lerp(float3(0.18, 0.155, 0.06), float3(0.27, 0.19, 0.065), (rnd - 0.22) / 0.78);
            m.trans = 0.5;
        }
        if (spc == 3u) {
            a = float3(0.15, 0.12, 0.07) * (0.75 + 0.5 * rnd);
            m.trans = 0.0;
        }
        // shrubs: green alder in late season (dull olive-green, leaf patches turning
        // yellow-brown, a few rusty); mugo pine dark green, a third blue-grey juniper
        if (spc == 4u) {
            float turn = smoothstep(0.15, 0.55, c1.x + 0.6 * (rnd - 0.5) + 0.25 * vnoised(world.xz * 2.7).x);
            a = lerp(lerp(float3(0.072, 0.090, 0.040), float3(0.105, 0.115, 0.050), rnd), lerp(float3(0.17, 0.14, 0.06), float3(0.15, 0.095, 0.045), frac(rnd * 3.1)), 0.5 * turn);
            m.trans = 0.3;
        }
        if (spc == 5u) {
            a = lerp(float3(0.046, 0.068, 0.036), float3(0.070, 0.090, 0.046), rnd);
            if (frac(rnd * 2.7) < 0.33) a = float3(0.058, 0.075, 0.068) * (0.9 + 0.3 * rnd);
            m.trans = 0.2;
        }
        a *= (0.70 + 0.55 * hf) * (1.0 + 0.35 * c1.x);
        // whorls: each tier's underside is shaded by the tier above (fades out with distance,
        // where a crown is a few pixels)
        float tier = frac(hf * 7.0 + rnd * 3.0 + 0.9 * c1.x);
        if (spc < 3u) a *= lerp(1.0, 0.7 + 0.3 * smoothstep(0.0, 0.6, tier), saturate(1.8 - dist / 1400.0));
        m.albedo = a;
        m.rough = 0.85;
        m.n = bump(n, c1.yzw * 0.25, 1.0);
        // crown-scale occlusion: inside a closed stand the lower crown sees little sky
        // plus self-occlusion of the crown itself (needle clumps shade the interior and the
        // lower whorls); snags are open
        float self_occ = spc == 3u ? 0.9 : 0.5 + 0.5 * hf;
        // spruce needle cards: the inner part of each branch (card coordinate 0 at the trunk)
        // sits in the crown's shade: a dark interior with lit outer shoots
        if (spc == 0u && code > 5.5) self_occ *= 0.45 + 0.55 * smoothstep(0.05, 0.85, rnd);
        m.occ = self_occ * lerp(1.0, 0.5, crowd * pow(saturate(1.0 - hf), 1.3));
    } else if (code < 3.5) {
        float4 b1 = triplanar(world * float3(3.0, 0.4, 3.0), w, 0.3, 3, foot);
        m.albedo = (spc == 3u ? float3(0.19, 0.18, 0.17) : float3(0.13, 0.095, 0.07)) * (1.0 + 0.3 * b1.x);
        m.occ = lerp(1.0, 0.5, crowd);
        m.rough = 0.9;
        m.n = n;
    } else if (code < 4.5) {
        // boulders: loose blocks of the local bedrock in the cliff palette (warm grey granite /
        // gneiss to brown-grey schist, rock type varying over hundreds of metres, per-block strata
        // tint warm / cool like the cliff bands), rust staining, crustose lichen (grey-green,
        // ochre, a few black), soil-stained base. Facets read through a per-plane tone (each
        // fracture plane weathered differently) and lighter worn chamfer edges (mesh flag:
        // attr.y + 2 on chamfer triangles).
        bool edge = hf > 1.5;
        hf = edge ? hf - 2.0 : hf;
        float region = vnoised(world.xz / 420.0 + 5.0).x + 0.5 * vnoised(world.xz / 130.0).x;
        float4 g1 = triplanar(world, w, 0.22, 3, foot);
        float4 g2 = triplanar(world, w, 1.3, 2, foot);
        float3 a = lerp(float3(0.12, 0.115, 0.105), float3(0.23, 0.215, 0.19), saturate(0.5 + 0.6 * region + 0.6 * (rnd - 0.5)));
        a *= lerp(float3(0.90, 0.92, 0.95), float3(1.10, 1.03, 0.92), frac(rnd * 5.7));
        // per fracture plane: the face normal is constant over a facet (flat-shaded mesh)
        float ft = hash2(floor(n.xz * 6.0 + n.y * float2(3.1, 7.7)) + rnd * 37.0);
        a *= 0.86 + 0.28 * ft;
        if (edge) a = lerp(a, a * float3(1.25, 1.22, 1.18) + 0.015, 0.8);
        a *= 1.0 + 0.16 * g1.x + 0.12 * g2.x;
        // iron staining: rusty streaks on some blocks, running down from the top
        float rust = smoothstep(0.35, 0.75, vnoised(world.xz * 0.9 + world.y * float2(0.3, 2.4) + rnd * 19.0).x) * step(0.75, frac(rnd * 3.3));
        a = lerp(a, a * float3(1.2, 0.97, 0.78), 0.45 * rust);
        // weathering toward the ground line: damp soil stain, warm brown, darker lower faces
        a = lerp(a * float3(0.58, 0.55, 0.50) + float3(0.012, 0.009, 0.005), a, smoothstep(0.0, 0.4, hf));
        // lichen: ~10 % in small patches on faces seeing the sky, never on worn edges; below
        // ~2 px a patch fades into its average tint
        float lp = vnoised(world.xz * 3.3 + world.y * 2.1 + 13.0).x + 0.5 * vnoised(world.zx * 8.1 - world.y * 5.3 + 3.0).x;
        float lv = saturate(0.06 / max(foot * 4.0, 1e-4) - 0.5);
        float lw = lerp(0.08, smoothstep(0.5, 0.66, lp), lv) * smoothstep(-0.3, 0.5, n.y) * (edge ? 0.3 : 1.0);
        float lk = frac(rnd * 7.3 + 0.4 * lp);
        float3 lc = lk < 0.55 ? float3(0.20, 0.215, 0.16) : (lk < 0.85 ? float3(0.30, 0.22, 0.09) : float3(0.05, 0.05, 0.045));
        a = lerp(a, lc, 0.5 * lw);
        // moss: only on near-horizontal tops, patchy, dull; more in moist / forest ground
        float mw = smoothstep(0.86, 0.97, n.y) * smoothstep(-0.1, 0.5, vnoised(world.xz * 1.3 + 3.0).x) * smoothstep(1300.0, 1050.0, world.y) * (0.35 + 0.35 * crowd);
        a = lerp(a, float3(0.075, 0.085, 0.045), mw);
        m.albedo = a;
        m.rough = lerp(0.8, 0.95, mw);
        m.n = bump(n, g1.yzw * 0.004, 1.0);
    } else {
        // blade grass (5): the terrain's own meadow palette from the same macro fields (cavity /
        // convexity unknown: their mean), so the dithered fade into the terrain's grass shading
        // at ~40 m shows no band; per-blade + clump hue (attr.z), lighter toward the tip, dry
        // straw tips (October), translucent (back-lit glow through the shade() term)
        float m1 = vnoised(world.xz / 610.0 + 3.0).x;
        float m2 = vnoised(world.xz / 190.0 + 11.0).x;
        float m3 = vnoised(world.xz / 63.0 + 5.0).x;
        float m4 = vnoised(world.xz / 21.0 + 17.0).x;
        float c12 = vnoised(world.xz / 12.0 + 23.0).x;
        float3 g = lerp(float3(0.120, 0.158, 0.078), float3(0.068, 0.125, 0.040), saturate(0.5 + 0.5 * m2));
        g *= 1.0 + 0.18 * m3 + 0.22 * c12;
        g = lerp(g, float3(0.17, 0.175, 0.112), 0.28 * smoothstep(0.35, 0.8, c12 + 0.5 * m4));
        g = lerp(g, float3(0.27, 0.23, 0.15), saturate((world.y - 1000.0) / 450.0 - 0.35 + 0.3 * m1));
        float3 base = g * lerp(0.75, 1.22, rnd) * float3(lerp(0.92, 1.08, frac(rnd * 5.3)), 1.0, 1.0);
        float dryw = smoothstep(0.55, 1.0, hf) * (0.2 + 0.6 * smoothstep(0.45, 0.95, frac(rnd * 3.7)));
        m.albedo = lerp(base * (0.82 + 0.32 * hf), g * float3(1.22, 1.24, 1.05), 0.40 * dryw);
        // sub-kinds (grass.rs): 5.1 coarse tussock blades (bluer, darker, dry-tipped), 5.2 straw
        // stems with seed heads (pale tan, brown heads), 5.3 flower heads (late-season white
        // yarrow / yellow hawkbit / blue gentian, fading into the blade tone with distance)
        float sub = code - 5.0;
        if (sub > 0.05 && sub < 0.15) m.albedo = lerp(base * float3(0.86, 0.93, 0.84) * (0.72 + 0.35 * hf), g * float3(1.45, 1.3, 1.0), 0.45 * dryw + 0.2 * smoothstep(0.7, 1.0, hf));
        if (sub > 0.15 && sub < 0.25) m.albedo = lerp(g * float3(1.5, 1.36, 1.0) * (0.85 + 0.25 * frac(rnd * 4.1)), float3(0.17, 0.12, 0.07), smoothstep(0.62, 0.8, hf) * 0.7);
        if (sub > 0.25) {
            float3 fl = rnd < 0.45 ? float3(0.62, 0.60, 0.54) : (rnd < 0.8 ? float3(0.62, 0.47, 0.08) : float3(0.12, 0.17, 0.48));
            m.albedo = lerp(fl, base, smoothstep(10.0, 30.0, dist));
        }
        m.rough = 0.9;
        // far blades shade like the ground they dither into (terrain normal = up, no glow):
        // the blade field and the terrain meadow beyond it match in tone at the fade band
        float far = smoothstep(12.0, 38.0, dist);
        m.n = normalize(lerp(n, float3(0.0, 1.0, 0.0), far));
        m.trans = 0.55 * (1.0 - far);
    }
    return m;
}

Mat cave_material(float3 world, float3 n, float4 attr, float foot) {
    float3 w = pow(abs(n), float3(4.0));
    w /= (w.x + w.y + w.z);
    float4 r1 = triplanar(world, w, 1.4, 5, foot);
    float4 r2 = triplanar(world, w, 0.16, 3, foot);
    float band = vnoised(float2(world.y / 0.37 + 2.0 * vnoised(world.xz / 6.0).x, 0.5)).x;
    // limestone: pale warm grey-beige, subtle banding, faint iron staining running down walls
    float3 lime = float3(0.53, 0.48, 0.41) * (1.0 + 0.14 * r1.x + 0.06 * r2.x);
    lime *= lerp(float3(0.93, 0.93, 0.92), float3(1.04, 1.02, 0.97), 0.5 + 0.5 * band);
    float streak = saturate(vnoised(float2(world.x * 2.1 + world.z * 1.3, world.y * 0.12)).x * 1.4);
    lime = lerp(lime, lime * float3(0.80, 0.68, 0.55), 0.6 * streak * saturate(1.0 - abs(n.y) * 1.6));
    // calcite on convex speleothems / flowstone: lighter, smoother
    float calc = saturate(attr.z * 1.4 - 0.2);
    lime = lerp(lime, float3(0.64, 0.58, 0.49) * (1.0 + 0.10 * r2.x + 0.08 * r1.x), calc * 0.8);
    // floor sediment: darker wet mud / gravel
    float floor_w = smoothstep(0.55, 0.85, n.y) * (1.0 - calc);
    float3 mud = float3(0.25, 0.21, 0.17) * (1.0 + 0.2 * r2.x);
    Mat m;
    m.albedo = lerp(lime, mud, floor_w);
    m.albedo *= lerp(1.0, 0.82, saturate(attr.y * 2.0));
    m.rough = lerp(lerp(0.70, 0.45, calc), 0.32, floor_w); // wet floor, waxy calcite
    // detail normals: gradient of a height field with amplitudes in metres
    m.n = bump(n, r1.yzw * 0.10 + r2.yzw * 0.02, 1.0);
    m.trans = 0.0;
    m.occ = 1.0;
    m.sheen = 0.0;
    m.lean = float2(0.0, 0.0);
    return m;
}

// Earth round 5: the grounded field's real imagery (the graft bake's albedo, so the mountains
// layer and the planet's graft show the same ground). Bilinear taps on two mips (footprint
// LOD), the inner level blended in over 86-97 % of its half extent like the planet's graft.
float4 gmap_tap(uint off, uint n, int2 ij, inout float water) {
    QParams* p = pc.p;
    int2 c = clamp(ij, int2(0, 0), int2(int(n) - 1, int(n) - 1));
    uint t = p.gmap[off + uint(c.y) * n + uint(c.x)];
    float3 s = float3(float(t & 255u), float((t >> 8) & 255u), float((t >> 16) & 255u)) / 255.0;
    float3 lin = select(s <= 0.04045, s / 12.92, pow((s + 0.055) / 1.055, float3(2.4, 2.4, 2.4)));
    uint a = t >> 24;
    water += (a & 128u) != 0u ? 1.0 : 0.0;
    return float4(lin, float(a & 127u) / 127.0);
}

float4 gmap_mip(uint4 lv, float half, uint mip, float2 dxz, inout float water) {
    uint n = lv.x >> mip;
    uint off = lv.z;
    for (uint k = 0u; k < mip; k++) off += (lv.x >> k) * (lv.x >> k);
    float t = 2.0 * half / float(n);
    float2 uv = (dxz + half) / t - 0.5;
    float2 f0 = floor(uv);
    float2 f = uv - f0;
    int2 i0 = int2(f0);
    float w0 = 0.0, w1 = 0.0, w2 = 0.0, w3 = 0.0;
    float4 a = gmap_tap(off, n, i0, w0), b = gmap_tap(off, n, i0 + int2(1, 0), w1);
    float4 c = gmap_tap(off, n, i0 + int2(0, 1), w2), d = gmap_tap(off, n, i0 + int2(1, 1), w3);
    water += lerp(lerp(w0, w1, f.x), lerp(w2, w3, f.x), f.y);
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

float4 gmap_level(uint4 lv, float half, float2 dxz, float foot, inout float water) {
    float lod = clamp(log2(max(foot, 1e-3) / (2.0 * half / float(lv.x))), 0.0, float(lv.y - 1u));
    uint m0 = uint(floor(lod));
    float fm = lod - float(m0);
    float wa = 0.0, wb = 0.0;
    float4 a = gmap_mip(lv, half, m0, dxz, wa);
    if (fm > 1e-4 && m0 + 1u < lv.y) {
        float4 b = gmap_mip(lv, half, m0 + 1u, dxz, wb);
        water += lerp(wa, wb, fm);
        return lerp(a, b, fm);
    }
    water += wa;
    return a;
}

void ground_albedo(inout Mat m, float3 world, float foot) {
    QParams* p = pc.p;
    if (p.gmap_lv[0].x == 0u) return;
    float2 dxz = world.xz - p.gmap_f.xy;
    float water = 0.0;
    float4 g = gmap_level(p.gmap_lv[0], p.gmap_f.z, dxz, foot, water);
    float wi = 1.0 - smoothstep(0.86, 0.97, max(abs(dxz.x), abs(dxz.y)) / p.gmap_f.w);
    if (wi > 0.0) {
        float wat = 0.0;
        float4 gi = gmap_level(p.gmap_lv[1], p.gmap_f.w, dxz, foot, wat);
        g = lerp(g, gi, wi);
        water = lerp(water, wat, wi);
    }
    if (g.a <= 0.0) return;
    m.albedo = lerp(m.albedo, g.rgb, g.a);
    m.sheen *= 1.0 - g.a; // the imagery already shows the sward
    // lakes: calm water reads smooth and flat from altitude
    float wk = saturate(water) * g.a;
    m.rough = lerp(m.rough, 0.08, wk);
    m.lean = lerp(m.lean, float2(0.0, 0.0), wk);
}

float4 shade(VSOut i) {
    QParams* p = pc.p;
    float3 rel = i.rel;
    float dist = length(rel);
    float3 v = -rel / max(dist, 1e-4);
    float3 world = rel + p.cam_world.xyz;
    float3 ng = normalize(i.nrm);
    float foot = dist * p.cam_world.w;
    float ao = i.attr.x;
    float3 col;
    if (p.sky_zenith.w < 0.5) {
        bool inst = i.attr.w > 1.5;
        // two-sided foliage / blades; needle cards keep their outward crown-volume normals
        if (inst && i.attr.w < 5.5 && dot(ng, v) < 0.0) ng = -ng;
        Mat m = inst ? inst_material(world, ng, i.attr, foot, dist) : mountain_material(world, ng, i.attr, foot, dist, length(i.nrm), 0.0);
        if (!inst) ground_albedo(m, world, foot);
        float3 l = p.sun_dir.xyz;
        float nl = dot(ng, l);
        float sh = (nl > 0.0 || m.trans > 0.0) ? sun_shadow(rel, ng, abs(nl), dist) : 0.0;
        bool atm = p.atmo.w > 0.5;
        float3 sun = p.sun_color.rgb * p.sun_dir.w;
        // physically based sun at the surface: illuminance x transmittance through the atmosphere
        if (atm) sun *= trans_lut(RG + max(world.y, 0.0) * 0.001, l.y);
        // cloud shadows: the same cirrus field the sky draws (6 km above the camera), sampled
        // where the sun ray from this point crosses it; thin cloud, soft partial shadow
        if (atm && l.y > 0.05) {
            float2 q = world.xz + l.xz / l.y * (6000.0 + cirrus_base() - world.y);
            float cc = fbmd(q * float2(1.0, 0.35), 2200.0, 4, 1.0).x;
            sun *= 1.0 - 0.3 * smoothstep(0.05, 0.45, cc);
        }
        col = brdf(m.n, v, l, m.albedo, m.rough, 0.04) * sun * sh;
        // forest interior light (live foliage: crown meshes, needle / leaf cards; not snags):
        // a crown is not a closed box. Sky light percolates through gaps between shoots, so the
        // sky visibility keeps a floor (CANOPY_SKY_FLOOR; measured stands transmit ~5-20 % of
        // open-sky diffuse light). Sun light that the shadow map stops is mostly scattered
        // inside the crown (needles reflect / transmit ~10 % each, green-tinted), not absorbed,
        // so the shadowed foliage gets a dim green in-crown sun term.
        float vis = ao * m.occ;
        bool foliage = inst && (i.attr.w < 2.5 || i.attr.w > 5.5) && (uint(floor(i.attr.z)) & 7u) != 3u;
        if (foliage) {
            // the floor is a closed-stand effect (crowding bits of the tag): a solitary tree keeps
            // its own dark interior / tier shading
            float crowd = float((uint(floor(i.attr.z)) >> 3) & 3u) / 3.0;
            vis = lerp(CANOPY_SKY_FLOOR * (0.3 + 0.7 * crowd), 1.0, vis);
            float3 tint = lerp(float3(0.42, 1.0, 0.40), m.albedo / max(max(m.albedo.r, m.albedo.g), 1e-4), 0.5);
            col += m.albedo * tint * sun * (max(l.y, 0.0) * CANOPY_SCATTER * (1.0 - sh) * (0.5 + 0.5 * i.attr.y));
        }
        if (m.sheen > 0.0) {
            // sward sheen: blades seen edge-on at grazing view angles add a soft, slightly
            // desaturated highlight toward the sun; looking with the grain (lean away from the
            // viewer) the sward reads lighter, against the grain darker (exposed stems / shade)
            float nv = saturate(dot(ng, v));
            float graze = pow(1.0 - nv, 3.0);
            float fwd = 0.35 + 0.65 * pow(saturate(dot(-v, l) * 0.5 + 0.5), 2.0);
            float3 sh_c = lerp(m.albedo, float3(dot(m.albedo, float3(0.33, 0.34, 0.33))), 0.4) * 1.6;
            col += sh_c * sun * sh * saturate(nl) * graze * fwd * 0.35 * m.sheen;
            float grain = dot(normalize(-v.xz + 1e-5), m.lean);
            col *= 1.0 + 0.10 * grain * m.sheen * (1.0 - nv);
        }
        // foliage translucency: light through needles/blades, strongest looking toward the sun
        col += m.albedo * sun * sh * m.trans * (0.25 + 0.75 * pow(saturate(dot(-v, l)), 3.0)) * saturate(0.4 - 0.6 * nl);
        // sky ambient (hemisphere) + bounce, occluded by multi-scale AO
        // sky irradiance integrates the whole hemisphere (bright, less saturated horizon), so it
        // is far less blue than the zenith colour; plus bounce from the sunlit valley floor
        // (green-brown) on faces that see the ground. Without the bounce, shadowed scree and
        // rock lit only by a zenith-blue ambient rendered as blue-white streaks.
        float up = saturate(m.n.y * 0.5 + 0.5);
        // (mountains: lantern_pos.w > 0.5 selects the legacy two-term ambient, for A/B)
        if (atm && p.lantern_pos.w < 0.5) {
            // environment SH (sky + mountain horizon + valley-floor bounce, directional)
            col += m.albedo * env_irradiance(m.n) * ao_multibounce(vis, m.albedo) * (p.sun_dir.w * p.fog_color.w);
        } else {
        float3 e_sky = lerp(float3(0.25, 0.28, 0.32), float3(0.19, 0.26, 0.40), up);
        // with the atmosphere: irradiance integrated from the sky-view LUT (up / vertical)
        if (atm) e_sky = lerp(pc.atmo[OFF_IRR + 1].rgb, pc.atmo[OFF_IRR].rgb, up) * p.sun_dir.w;
        // from a valley a large part of the "sky" dome is mountainsides and forest (grey-green,
        // partly sunlit), not blue sky: keep the irradiance level, cut its saturation, and add
        // the light reflected by the sunlit terrain around the horizon to upward-facing surfaces.
        // Without this, green grass in shade under a blue dome rendered teal.
        float3 e_hor = float3(0.0, 0.0, 0.0);
        if (atm) {
            float el = dot(e_sky, float3(0.2126, 0.7152, 0.0722));
            e_sky = lerp(float3(el, el, el), e_sky, 0.62) * 0.85;
            e_hor = float3(0.16, 0.165, 0.14) * sun * max(p.sun_dir.y, 0.0) * 0.12 * up;
        }
        // ground bounce: valley-floor albedo x irradiance on the ground (sun + sky) x view factor
        // of the ground ((1 - n.y) / 2 = 1 - up)
        float3 e_up = atm ? pc.atmo[OFF_IRR].rgb * p.sun_dir.w : float3(0.19, 0.26, 0.40);
        float3 e_bounce = float3(0.16, 0.17, 0.10) * (sun * max(p.sun_dir.y, 0.0) + e_up) * (1.0 - up);
        col += m.albedo * ((e_sky + e_hor) * p.fog_color.w + e_bounce) * vis;
        }
        // debug: lantern_color.w == 1 tints far-ring geometry red (attr.w = 1);
        // == 2 shows the cascade index (r, g, b, white) scaled by the shadow term
        if (p.lantern_color.w > 0.5 && p.lantern_color.w < 1.5) col = lerp(col, float3(1.0, 0.0, 0.0), 0.5 * saturate(i.attr.w) * (inst ? 0.0 : 1.0));
        // debug 4: environment irradiance (up, horizontal toward / away from the sun, down) in
        // four screen quarters, per unit sun illuminance x 4
        if (p.lantern_color.w > 3.5 && atm) {
            float2 q = i.position.xy * 0.5; // SSAA target
            float3 dn = q.x < 960.0 ? (q.y < 540.0 ? float3(0, 1, 0) : float3(normalize(p.sun_dir.xz).x, 0, normalize(p.sun_dir.xz).y)) : (q.y < 540.0 ? float3(-normalize(p.sun_dir.xz).x, 0, -normalize(p.sun_dir.xz).y) : float3(0, -1, 0));
            return float4(env_irradiance(dn) * 4.0, 1.0);
        }
        if (p.lantern_color.w > 1.5 && p.lantern_color.w < 2.5) {
            int ci = 3;
            for (int c = 3; c >= 0; c--) if (dist < p.splits[c]) ci = c;
            float3 cc = ci == 0 ? float3(1, 0.2, 0.2) : (ci == 1 ? float3(0.2, 1, 0.2) : (ci == 2 ? float3(0.3, 0.3, 1) : float3(1, 1, 1)));
            return float4(cc * (0.15 + 0.85 * sh) * saturate(dot(ng, l) * 4.0), 1.0);
        }
        if (atm) {
            float4 ap = aerial(rel, dist);
            // debug 3: aerial-perspective transmittance (r) and in-scatter luminance (g)
            if (p.lantern_color.w > 2.5) return float4(ap.w, dot(ap.rgb, float3(0.3, 0.5, 0.2)) * p.sun_dir.w, 0.0, 1.0);
            col = col * ap.w + ap.rgb * p.sun_dir.w;
        } else {
            float4 f = fog_mountains(rel, -v, dist);
            col = col * f.w + f.rgb * (1.0 - f.w);
        }
    } else {
        Mat m = cave_material(world, ng, i.attr, foot);
        // headlamp spot (shadowed by its own map) + warm lantern deeper in
        float3 lv = p.lamp_pos.xyz - rel;
        float ld = length(lv);
        float3 l = lv / ld;
        float cone = smoothstep(p.lamp_dir.w, p.lamp_color.w, dot(-l, p.lamp_dir.xyz));
        float win = pow(saturate(1.0 - pow(ld / p.lamp_pos.w, 4.0)), 2.0);
        float nl = dot(ng, l);
        float sh = nl > 0.0 ? shadow_pcf(0u, rel, ng, saturate(nl)) : 0.0;
        col = brdf(m.n, v, l, m.albedo, m.rough, 0.04) * p.lamp_color.rgb * (cone * win * sh / max(ld * ld, 0.05));
        float3 kv = p.lantern_pos.xyz - rel;
        float kd = length(kv);
        float3 kl = kv / kd;
        float kwin = pow(saturate(1.0 - pow(kd / p.lantern_pos.w, 4.0)), 2.0);
        // lantern: CPU sphere-traced soft visibility per vertex (attr.w), no light through rock
        float kvis = i.attr.w;
        col += brdf(m.n, v, kl, m.albedo, m.rough, 0.04) * p.lantern_color.rgb * (kwin * kvis / max(kd * kd, 0.1));
        col += m.albedo * p.lantern_color.rgb * (0.02 * kwin * (0.3 + 0.7 * kvis) / max(kd * kd, 1.0)) * ao;
        // one-bounce fill: lamp light scattered off the nearby walls (inverse-square from the lamp)
        col += m.albedo * p.lamp_color.rgb * (p.fog_color.w / max(ld * ld, 1.5)) * ao;
        col += m.albedo * p.fog_color.rgb * ao;
        // fog lit by the lamp (cheap single-scatter estimate along the view ray)
        float T = exp(-dist * p.sky_horizon.w);
        float glow = 0.5 * smoothstep(p.lamp_dir.w, 1.0, dot(-v, p.lamp_dir.xyz)) + 0.5;
        float3 ins = p.fog_color.rgb + p.lamp_color.rgb * 0.0018 * glow;
        col = col * T + ins * (1.0 - T);
    }
    col = aces(grade(col * p.sun_color.w));
    return float4(to_srgb(col), layer_alpha(world));
}

[shader("fragment")]
float4 terrainFS(VSOut i) : SV_Target0 { return shade(i); }

// ---------------------------------------------------------------- LOD crossfade + impostors
// Interleaved gradient noise (Jimenez): a stable per-pixel threshold for the LOD crossfade
// dither; the second variant (shifted pattern) drives the impostors' dithered alpha so the
// two masks are uncorrelated.
// code: bit 7 = keep the upper part (h >= t), bits 0..6 = t * 127; 127 = keep everything
bool lod_keep(uint code, float2 px) {
    if (code == 127u) return true;
    float t = float(code & 127u) / 127.0;
    float h = ign(px);
    return (code & 128u) != 0u ? h >= t : h < t;
}

[shader("fragment")]
float4 instFS(InstOut i) : SV_Target0 {
    if (!lod_keep(i.lod, i.position.xy)) discard;
    if (i.attr.w > 5.5 && !card_alpha(i.uv, smoothstep(0.25, 0.75, fwidth(i.uv.y) * 22.0), i.position.xy)) discard;
    if (i.attr.w > 5.5 && i.uv.x > 3.5 && !card_facing_keep(i.rel, i.position.xy)) discard;
    VSOut o;
    o.position = i.position;
    o.rel = i.rel;
    o.nrm = i.nrm;
    o.attr = i.attr;
    // conifer shoot clusters: per-shoot tone, fading to the mean once the shoots are sub-pixel
    if (i.attr.w > 5.5 && i.uv.x > 3.5) o.attr.x *= lerp(spruce_card(i.uv.x - 4.0, i.uv.y).y, 1.0, smoothstep(0.25, 0.75, fwidth(i.uv.y) * 22.0));
    return shade(o);
}

// depth prepass of the crossfading / alpha-card instance slots: the same discards as instFS
// (so the prepass depth is exactly the surviving surface), no colour
[shader("fragment")]
void instDepthFS(InstOut i) {
    if (!lod_keep(i.lod, i.position.xy)) discard;
    if (i.attr.w > 5.5 && !card_alpha(i.uv, smoothstep(0.25, 0.75, fwidth(i.uv.y) * 22.0), i.position.xy)) discard;
    if (i.attr.w > 5.5 && i.uv.x > 3.5 && !card_facing_keep(i.rel, i.position.xy)) discard;
}

[shader("fragment")]
void instShadowAlphaFS(InstShadowOut i) {
    if (i.code > 5.5 && !card_alpha(i.uv, smoothstep(0.25, 0.75, fwidth(i.uv.y) * 22.0), i.position.xy)) discard;
    // conifer shoots are sub-texel in the shadow map: a card layer stops ~88 % of the sun
    // (sun flecks through the crown edge, dappled light under it); several layers still close
    if (i.code > 5.5 && i.uv.x > 3.5 && ign(i.position.xy + float2(5.0, 29.0)) > CANOPY_SHADOW_KEEP) discard;
}

static const uint IMP_N = 9;     // == impostor::IMP_FRAMES
static const uint IMP_R = 128;   // == impostor::IMP_RES

float3 hemi_oct_dir(float2 ab) {
    float x = (ab.x + ab.y) * 0.5, z = (ab.x - ab.y) * 0.5;
    return normalize(float3(x, max(1.0 - abs(x) - abs(z), 0.0), z));
}
float2 hemi_oct_encode(float3 d) {
    d.y = max(d.y, 0.0);
    float s = abs(d.x) + d.y + abs(d.z);
    float2 p = s > 0.0 ? d.xz / s : float2(1.0, 0.0);
    return float2(p.x + p.y, p.x - p.y);
}
float3 imp_frame_dir(uint2 ij) { return hemi_oct_dir(float2(ij) / float(IMP_N - 1) * 2.0 - 1.0); }
void imp_basis(float3 d, out float3 r, out float3 u) {
    r = d.y > 0.9999 ? float3(1, 0, 0) : normalize(float3(d.z, 0.0, -d.x));
    u = cross(d, r);
}
float3 oct_decode8(uint a, uint b) {
    float2 p = float2(a, b) / 255.0 * 2.0 - 1.0;
    float y = 1.0 - abs(p.x) - abs(p.y);
    float2 xz = y >= 0.0 ? p : float2((1.0 - abs(p.y)) * (p.x >= 0.0 ? 1.0 : -1.0), (1.0 - abs(p.x)) * (p.y >= 0.0 ? 1.0 : -1.0));
    return normalize(float3(xz.x, y, xz.y));
}

// Local (mesh-space) <-> camera-relative transforms of an instance whose model columns are
// orthogonal (scale x rotation x translation): M^-1 v = sum_k e_k dot(m_k, v) / |m_k|^2.
float3 to_local_dir(QInstance I, float3 v) {
    return float3(dot(I.model[0].xyz, v) / dot(I.model[0].xyz, I.model[0].xyz), dot(I.model[1].xyz, v) / dot(I.model[1].xyz, I.model[1].xyz), dot(I.model[2].xyz, v) / dot(I.model[2].xyz, I.model[2].xyz));
}
float3 to_rel(QInstance I, float3 p) { return (I.model[0] * p.x + I.model[1] * p.y + I.model[2] * p.z + I.model[3]).xyz; }
// normal: M^-T n = sum_k m_k n_k / |m_k|^2
float3 nrm_to_rel(QInstance I, float3 n) {
    return I.model[0].xyz * (n.x / dot(I.model[0].xyz, I.model[0].xyz)) + I.model[1].xyz * (n.y / dot(I.model[1].xyz, I.model[1].xyz)) + I.model[2].xyz * (n.z / dot(I.model[2].xyz, I.model[2].xyz));
}

// The 3 frame nodes around view direction `vd` (toward the viewer, mesh space) and their
// barycentric weights in the vertex-centred hemi-octahedral grid.
void imp_frames(float3 vd, out uint2 f[3], out float3 w) {
    float2 g = (hemi_oct_encode(vd) * 0.5 + 0.5) * float(IMP_N - 1);
    float2 c = clamp(floor(g), 0.0, float(IMP_N - 2));
    float2 t = saturate(g - c);
    uint2 b = uint2(c);
    if (t.x + t.y < 1.0) {
        f[0] = b; f[1] = b + uint2(1, 0); f[2] = b + uint2(0, 1);
        w = float3(1.0 - t.x - t.y, t.x, t.y);
    } else {
        f[0] = b + uint2(1, 1); f[1] = b + uint2(0, 1); f[2] = b + uint2(1, 0);
        w = float3(t.x + t.y - 1.0, 1.0 - t.x, 1.0 - t.y);
    }
}

uint2 imp_fetch(uint var, uint frame, float2 uv) {
    if (any(uv < 0.0) || any(uv >= 1.0)) return uint2(0, 0);
    uint2 px = min(uint2(uv * float(IMP_R)), uint2(IMP_R - 1));
    return pc.imp_tex[(var * IMP_N * IMP_N + frame) * IMP_R * IMP_R + px.y * IMP_R + px.x];
}

struct ImpHit { float alpha; float3 p; float3 n; float ao; float hf; float bark; };

// Ray (mesh space, origin o, direction rd) against the 3 blended frames: per frame intersect
// the frame's card plane, read the depth there and re-intersect once at that depth (parallax),
// then blend the frames' samples weighted by frame weight x coverage.
ImpHit imp_trace(uint var, float3 o, float3 rd, float3 vd, bool nearest, bool refine) {
    float4 cr = pc.imp[var * 2];
    float3 c = cr.xyz;
    float r = cr.w;
    uint2 f[3];
    float3 w;
    imp_frames(vd, f, w);
    if (nearest) {
        // one frame (shadow maps): the heaviest of the three
        uint m = w.x >= w.y && w.x >= w.z ? 0u : (w.y >= w.z ? 1u : 2u);
        f[0] = f[m];
        w = float3(1.0, 0.0, 0.0);
    }
    ImpHit h;
    h.alpha = 0.0; h.p = 0.0; h.n = 0.0; h.ao = 0.0; h.hf = 0.0; h.bark = 0.0;
    float wsum = 0.0;
    for (int k = 0; k < 3; ++k) {
        if (w[k] <= 0.0) continue;
        float3 d = imp_frame_dir(f[k]);
        float3 rt, up;
        imp_basis(d, rt, up);
        float den = dot(rd, d);
        if (abs(den) < 1e-5) continue;
        uint fi = f[k].y * IMP_N + f[k].x;
        float s = dot(c - o, d) / den;
        float3 q = o + rd * s - c;
        float2 uv = float2(dot(q, rt) / r * 0.5 + 0.5, 0.5 - dot(q, up) / r * 0.5);
        uint2 t = imp_fetch(var, fi, uv);
        if (refine && (t.x & 255u) != 0u) {
            float dep = (float(t.x >> 16) / 65535.0 * 2.0 - 1.0) * r;
            s = dot(c + d * dep - o, d) / den;
            q = o + rd * s - c;
            uv = float2(dot(q, rt) / r * 0.5 + 0.5, 0.5 - dot(q, up) / r * 0.5);
            uint2 t2 = imp_fetch(var, fi, uv);
            if ((t2.x & 255u) != 0u) t = t2;
        }
        float a = float(t.x & 255u) / 255.0;
        h.alpha += w[k] * a;
        if (a <= 0.0) continue;
        float ww = w[k] * a;
        float dep = (float(t.x >> 16) / 65535.0 * 2.0 - 1.0) * r;
        // the surface point: on the ray where it crosses this frame's depth layer
        float sd = dot(c + d * dep - o, d) / den;
        h.p += (o + rd * sd) * ww;
        h.n += oct_decode8(t.y & 255u, (t.y >> 8) & 255u) * ww;
        h.ao += float((t.y >> 16) & 255u) / 255.0 * ww;
        h.hf += float((t.x >> 8) & 127u) / 127.0 * ww;
        h.bark += float((t.x >> 15) & 1u) * ww;
        wsum += ww;
    }
    if (wsum > 0.0) {
        h.p /= wsum; h.n = normalize(h.n); h.ao /= wsum; h.hf /= wsum; h.bark /= wsum;
    }
    return h;
}

struct ImpOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 lp;                          // mesh-space point on the card
    [[vk::location(1)]] nointerpolation uint4 info;          // instance, variant, lod code, -
    [[vk::location(2)]] nointerpolation float3 lo;           // mesh-space ray origin (camera) / -
    [[vk::location(3)]] nointerpolation float3 vd;           // mesh-space view direction for frame selection
};

// Card (`v.pos.xy` in +-1, `attr.x` = variant) perpendicular to the mesh-space view direction,
// fitted to the projection of the variant's AABB (+6 % for perspective in the camera view).
ImpOut imp_card(uint vid, uint iid, bool shadow) {
    uint e = pc.visible[iid];
    QInstance I = pc.inst[e & 0xFFFFFFu];
    QVertex v = pc.v[vid];
    uint var = uint(v.attr.x + 0.5);
    float4 cr = pc.imp[var * 2];
    float3 he = pc.imp[var * 2 + 1].xyz;
    ImpOut o;
    float3 lo = to_local_dir(I, -I.model[3].xyz);           // camera (rel origin) in mesh space
    float3 vd = shadow ? normalize(to_local_dir(I, pc.p.sun_dir.xyz)) : normalize(lo - cr.xyz);
    float3 rt, up;
    imp_basis(vd, rt, up);
    float k = shadow ? 1.01 : 1.06;
    float2 ext = float2(dot(abs(rt), he), dot(abs(up), he)) * k;
    float3 lp = cr.xyz + rt * (v.pos.x * ext.x) + up * (v.pos.y * ext.y);
    // shadow cards are pushed 0.35 r away from the sun (along the same light ray, so the exact
    // FS is unaffected): the card-depth caster then sits behind the sunlit half of the crown
    float3 rel = to_rel(I, shadow ? lp - vd * (0.35 * cr.w) : lp);
    o.position = shadow ? vp_casc(pc.cascade, float4(rel, 1.0)) : vp_view(rel);
    o.lp = lp;
    o.info = uint4(e & 0xFFFFFFu, var, e >> 24, I.material_id);
    o.lo = lo;
    o.vd = vd;
    return o;
}

[shader("vertex")]
ImpOut impVS(uint vid : SV_VulkanVertexID, uint iid : SV_VulkanInstanceID) { return imp_card(vid, iid, false); }

[shader("vertex")]
ImpOut impShadowVS(uint vid : SV_VulkanVertexID, uint iid : SV_VulkanInstanceID) { return imp_card(vid, iid, true); }

struct ImpFrag { float4 color : SV_Target0; float depth : SV_Depth; };

[shader("fragment")]
ImpFrag impFS(ImpOut i) {
    if (!lod_keep(i.info.z, i.position.xy)) discard;
    QInstance I = pc.inst[i.info.x];
    float3 rd = normalize(i.lp - i.lo);
    ImpHit h = imp_trace(i.info.y, i.lo, rd, i.vd, false, true);
    // dithered alpha: coverage against a second, decorrelated noise pattern (no MSAA here, so
    // no alpha-to-coverage); centred so half-covered texels keep about half their pixels
    if (h.alpha < 0.25 + 0.5 * ign(i.position.xy + float2(17.0, 59.0))) discard;
    float3 rel = to_rel(I, h.p);
    float4 clip = vp_view(rel);
    VSOut o;
    o.position = i.position;
    o.rel = rel;
    o.nrm = normalize(nrm_to_rel(I, h.n));
    uint mat = i.info.w;
    o.attr = float4(h.ao, h.hf, float(mat >> 10) + float(mat & 1023u) / 1024.0, h.bark > 0.5 ? 3.0 : 2.0);
    ImpFrag f;
    f.color = shade(o);
    f.depth = clip.z / clip.w;
    return f;
}

// Far impostors: identical, except the depth is the card's (rasterised) depth instead of the
// reconstructed surface's. Writing SV_Depth forces late depth testing (every overlapping
// fragment of a dense far forest is shaded); with the card depth the early test rejects
// occluded fragments before shading. The card sits on the variant's centre plane, so the
// depth error is at most the crown radius - a few metres at > 1.6 km. No parallax refine
// either (sub-texel at that range).
// Also the card-depth path for crossfading impostors (`impostor_card_depth`): the LOD dither
// (code 127 = keep all for whole-slot instances).
[shader("fragment")]
float4 impFarFS(ImpOut i) : SV_Target0 {
    if (!lod_keep(i.info.z, i.position.xy)) discard;
    QInstance I = pc.inst[i.info.x];
    float3 rd = normalize(i.lp - i.lo);
    ImpHit h = imp_trace(i.info.y, i.lo, rd, i.vd, false, false);
    if (h.alpha < 0.25 + 0.5 * ign(i.position.xy + float2(17.0, 59.0))) discard;
    return imp_card_shade(i, I, h);
}

float4 imp_card_shade(ImpOut i, QInstance I, ImpHit h) {
    VSOut o;
    o.position = i.position;
    o.rel = to_rel(I, h.p);
    o.nrm = normalize(nrm_to_rel(I, h.n));
    uint mat = i.info.w;
    o.attr = float4(h.ao, h.hf, float(mat >> 10) + float(mat & 1023u) / 1024.0, h.bark > 0.5 ? 3.0 : 2.0);
    return shade(o);
}

// Card-depth impostors in the foliage depth prepass: the same dither and coverage test as
// impFarFS, no colour, the card's depth (early depth testing).
[shader("fragment")]
void impDepthFS(ImpOut i) {
    if (!lod_keep(i.info.z, i.position.xy)) discard;
    ImpHit h = imp_trace(i.info.y, i.lo, normalize(i.lp - i.lo), i.vd, false, false);
    if (h.alpha < 0.25 + 0.5 * ign(i.position.xy + float2(17.0, 59.0))) discard;
}

// Card-depth impostors after the prepass (depth EQUAL, no writes): only the front surviving
// fragment of each pixel is shaded. No discards: coverage was decided by impDepthFS, and EQUAL
// only passes for the card that wrote that depth, so a last-bit difference between the two
// shaders' alpha can never open a hole.
[shader("fragment")]
float4 impEqFS(ImpOut i) : SV_Target0 {
    QInstance I = pc.inst[i.info.x];
    ImpHit h = imp_trace(i.info.y, i.lo, normalize(i.lp - i.lo), i.vd, false, false);
    return imp_card_shade(i, I, h);
}

// Far cascades: coverage only. One fetch of the nearest frame where the light ray through the
// card crosses that frame's plane; depth = the card's (no SV_Depth, so early depth testing).
[shader("fragment")]
void impShadowFarFS(ImpOut i) {
    float4 cr = pc.imp[i.info.y * 2];
    uint2 f[3];
    float3 w;
    imp_frames(i.vd, f, w);
    uint m = w.x >= w.y && w.x >= w.z ? 0u : (w.y >= w.z ? 1u : 2u);
    float3 d = imp_frame_dir(f[m]);
    float3 rt, up;
    imp_basis(d, rt, up);
    float den = dot(-i.vd, d);
    float s = dot(cr.xyz - i.lp, d) / (abs(den) < 1e-5 ? -1e-5 : den);
    float3 q = i.lp - i.vd * s - cr.xyz;
    float2 uv = float2(dot(q, rt) / cr.w * 0.5 + 0.5, 0.5 - dot(q, up) / cr.w * 0.5);
    if ((imp_fetch(i.info.y, f[m].y * IMP_N + f[m].x, uv).x & 255u) < 128u) discard;
}

// Shadow casting: orthographic along the sun, so the ray direction is the (mesh-space) sun
// direction reversed; the frames are picked by the sun direction. Written depth bypasses the
// pipeline depth bias, so the point is pushed 1.5 shadow texels away from the sun instead.
[shader("fragment")]
float impShadowFS(ImpOut i) : SV_Depth {
    QInstance I = pc.inst[i.info.x];
    float3 rd = -i.vd;
    float3 o = i.lp - rd * (2.0 * pc.imp[i.info.y * 2].w);
    ImpHit h = imp_trace(i.info.y, o, rd, i.vd, true, true);
    if (h.alpha < 0.5) discard;
    float3 rel = to_rel(I, h.p) - pc.p.sun_dir.xyz * (1.5 * pc.p.texel[pc.cascade]);
    float4 clip = vp_casc(pc.cascade, float4(rel, 1.0));
    return clip.z / clip.w;
}

// ---------------------------------------------------------------- sky
struct SkyOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 ndc;
};

[shader("vertex")]
SkyOut skyVS(uint vid : SV_VulkanVertexID) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    SkyOut o;
    o.ndc = uv * 2.0 - 1.0;
    o.position = float4(o.ndc, 0.0, 1.0); // reverse-Z: depth 0 = infinitely far
    return o;
}

[shader("fragment")]
float4 skyFS(SkyOut i) : SV_Target0 {
    QParams* p = pc.p;
    float4 d4 = mulc(p.inv_dir[0], p.inv_dir[1], p.inv_dir[2], p.inv_dir[3], float4(i.ndc, 1.0, 1.0));
    float3 dir = normalize(d4.xyz);
    float3 col;
    if (p.sky_zenith.w < 0.5 && p.atmo.w > 0.5) {
        // physically based sky: sky-view LUT + sun disk (limb-darkened, transmitted)
        float E = p.sun_dir.w;
        col = sky_lut(dir) * E;
        float mu = dot(dir, p.sun_dir.xyz);
        float cr = cos(p.atmo2.y);
        if (mu > cr) {
            float x = saturate((1.0 - mu) / (1.0 - cr));
            float limb = 1.0 - 0.6 * (1.0 - sqrt(1.0 - x));
            col += E * trans_lut(cam_r(), dir.y) * 60.0 * limb;
        }
        // thin cirrus at ~6 km: lit by transmitted sun (forward-scattering) + sky irradiance,
        // then fogged by the air in front of it (sky-view radiance mixed in by optical depth)
        // layer mode: the cirrus deck is fixed at cirrus_base() + 6 km (none from above it)
        float ch = layer_mode() ? cirrus_base() + 6000.0 - p.cam_world.y : 6000.0;
        if (dir.y > 0.01 && ch > 100.0) {
            float2 q = dir.xz / dir.y * ch + p.cam_world.xz;
            float c = fbmd(q * float2(1.0, 0.35), 2200.0, 5, 1.0).x;
            float cov = smoothstep(0.05, 0.45, c) * saturate(dir.y * 6.0);
            float3 tsun = trans_lut(RG + 6.0, p.sun_dir.y);
            float3 cc = E * (tsun * (0.06 + 0.35 * phase_m(mu)) + pc.atmo[OFF_IRR].rgb * 0.25);
            float far = 1.0 - exp(-(6.0 / dir.y) / 60.0);
            cc = lerp(cc, col, far);
            col = lerp(col, cc, cov * 0.4);
        }
    } else if (p.sky_zenith.w < 0.5) {
        col = sky_color(dir);
        float mu = dot(dir, p.sun_dir.xyz);
        col += p.sun_color.rgb * 30.0 * smoothstep(0.99995, 0.99999, mu);
        // thin cirrus layer at ~4 km
        if (dir.y > 0.01) {
            float2 q = dir.xz / dir.y * 4000.0 + p.cam_world.xz;
            float c = fbmd(q * float2(1.0, 0.35), 2200.0, 5, 1.0).x;
            float cov = smoothstep(0.05, 0.45, c) * saturate(dir.y * 6.0);
            float3 cc = lerp(p.sky_horizon.rgb, float3(1.0, 0.98, 0.95), 0.6) * (1.0 + 0.6 * pow(saturate(mu), 4.0));
            col = lerp(col, cc, cov * 0.55);
        }
        // the horizon dissolves into the same fog the terrain uses
        float4 f = fog_mountains(dir * 30000.0, dir, 30000.0);
        col = lerp(col, f.rgb, (1.0 - f.w) * saturate(1.0 - dir.y * 3.0) * 0.6);
    } else {
        col = p.fog_color.rgb + p.lamp_color.rgb * 0.0018 * (0.5 + 0.5 * smoothstep(p.lamp_dir.w, 1.0, dot(dir, p.lamp_dir.xyz)));
    }
    col = aces(grade(col * p.sun_color.w));
    return float4(to_srgb(col), layer_mode() ? 0.0 : 1.0);
}
