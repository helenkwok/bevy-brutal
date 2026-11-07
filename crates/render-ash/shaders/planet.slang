// Planet from orbit (`earth` scene). Patches of the cube-sphere quadtree are pulled through BDA
// pointers (camera-relative f32 = f64 patch origin - f64 camera, formed on the CPU, + f32 local
// offset); the Earth textures are BC cube maps sampled through one descriptor set (hardware
// filtering + mips). Atmosphere: Hillaire 2020 transmittance + multiple-scattering LUTs, a
// sky-at-ground LUT (azimuth averaged, for sky light and the sky reflected by the sea) and its
// irradiance; from space every pixel ray-marches the atmosphere segment it crosses (no sky-view
// LUT), with the planet shadow (sun below a sample's horizon -> no single scattering, R-E4),
// ozone, an airglow layer (95 km) and a 2D cloud layer (lit by the sun transmittance at cloud
// height, forward scattering at the limb, shadows on the ground) or a ray-marched cloud volume
// (cloud.w = 1: dated VIIRS coverage + cloud-top height, Perlin-Worley shape, self-shadowing).
// Units: km for the atmosphere, radiance per unit solar illuminance.
//   planetVS/planetFS : surface (land, ocean glint, ice, night lights, clouds, aerial perspective)
//   skyVS/skyFS       : rays that miss the surface: atmosphere limb, clouds at the limb, sun disc
//   *CS               : LUTs

struct PVertex {               // == planet::patch::PVertex (16 bytes)
    float3 pos;                // metres from the patch origin
    float height;              // metres above the radius
};
struct PPatch {                // == render_ash::planet::PatchGpu (16 bytes)
    float4 origin;             // xyz patch origin - camera (m, from f64), w level
};
struct PParams {               // == render_ash::planet::PlanetParamsGpu
    float4 view_proj[4];       // camera-relative, reverse-Z infinite (columns)
    float4 cam_km;             // xyz planet-frame camera (km, from f64), w altitude (km, f64)
    float4 sun;                // xyz toward the sun, w angular radius (rad)
    float4 right;              // xyz, w tan(fov_x / 2)
    float4 up;                 // xyz, w tan(fov_y / 2)
    float4 fwd;                // xyz, w debug mode
    float4 planet;             // x ground radius (km), y top (km), z cloud layer radius (km), w cloud optical depth scale
    float4 ocean;              // x Cox-Munk slope variance, y F0, z sea-ice threshold, w relief scale
    float4 light;              // x night-light gain, y airglow gain, z albedo scale, w cloud shadow strength
    float4 atmo;               // x Mie scale, y Rayleigh scale, z ozone scale, w ground albedo (multiple scattering)
    float4 view;               // x width, y height, z pre-exposure (radiance x z is stored: keeps
                               // night radiances ~1e-6 out of the RGBA16F target's subnormals)
    float4 cloud;              // x volume base (km), y default top at full cover (km), z extinction (1/km), w mode (0 = 2D layer, 1 = volume)
    float4 cloud2;             // x view steps, y sun steps, z shape-noise period (km), w multiple-scattering gain
    float4 aero;               // x free-tropospheric aerosol scattering at sea level (1/km), y its scale height (km),
                               // z stratospheric (Junge layer, 20 +- 7 km) aerosol scattering at the peak (1/km),
                               // w procedural sub-texel land-detail gain (0 = off)
    float4 hole;               // clear-sky window: xyz centre direction, w full-clear radius (rad), 0 = off
    float4 meso;               // far-footprint cloud structure: x cloud-top, y fBm log2 strengths, z brightness bias, w fine structure (round 4)
    float4 thaw;               // snow-free disc (descent's FLATTEN companion): xyz centre x strength, w outer radius (rad), 0 = off
    float4 graft;              // shared Alps (round 4): xyz vista nadir direction x strength, w full-graft radius (rad), 0 = off
    float4 graft_e;            // xyz vista east, w graft outer radius (rad)
    float4 graft_n;            // xyz vista north, w outer level half extent (km)
    float4 graft_l;            // x inner level half extent (km), y albedo gain, z AO strength, w texels per level side
    float4 meso2;              // round 4 far-footprint clouds: x coverage gamma at multi-km footprints (0 = planet.w everywhere); y = DEM close graft half extent (km), 0 = off (layer 2); zw = close east/south offset km
    float4 graft_t;            // large-scale tint toward the real Earth: x strength, y inner radius (rad; full by graft.w), z real albedo LOD, w graft LOD
    float4 haze;               // round 6 marine haze over open ocean: x vertical AOD at 550 nm (0 = off), y Angstrom exponent, z single-scattering albedo, w water-mask texel (km)
    float4 cloud3;             // round 7: x cloud-edge feather scale (km, 0 = off), y feather mix, z clear-window edge noise (0 = round circle), w haze dust (blue absorption, 0 = off)
};
struct PPush {
    PParams* p;
    PVertex* v;
    PPatch* patches;
    float4* atmo;
};
[[vk::push_constant]] ConstantBuffer<PPush> pc;

[[vk::binding(0, 0)]] SamplerCube tAlbedo;   // BC7 sRGB: rgb reflectance, a water fraction
[[vk::binding(1, 0)]] SamplerCube tSlope;    // BC5: east / north slope, sign(s) sqrt(|s| / SLOPE_MAX) * 0.5 + 0.5
[[vk::binding(2, 0)]] SamplerCube tNight;    // BC4: Black Marble 2016
[[vk::binding(3, 0)]] SamplerCube tClouds;   // BC5 (KTX2): r cloud map, g cloud-top km x 20 / 255 (0 = unknown); PLN1: aux .r, g = 0
[[vk::binding(4, 0)]] SamplerCube tMasks;    // BC5 (KTX2) / BC7 aux .gb swizzled (PLN1): r ice, g ocean depth / 8 km
[[vk::binding(5, 0)]] Sampler3D tNoise;      // RGBA8 128^3, repeat: Perlin-Worley + 3 Worley fBm octaves
[[vk::binding(6, 0)]] Sampler2DArray tGraftAlb; // shared Alps / DEM: layer 0 outer, 1 mid, 2 optional close; sRGB albedo
[[vk::binding(7, 0)]] Sampler2DArray tGraftAux; // r, g world normal x, z; b sun visibility; a AO

static const float PI = 3.14159265;
static const uint PATCH_VERTS = 1221;   // == planet::patch::VERTS (33^2 + 4 * 33)
static const float SLOPE_MAX = 0.6;
static const uint T_W = 256;
static const uint T_H = 64;
static const uint MS_N = 32;
static const uint SG_N = 32;
static const uint IRR_N = 64;
static const uint OFF_T = 0;
static const uint OFF_MS = 16384;
static const uint OFF_SG = 17408;
static const uint OFF_IRR = 18432;

float RG() { return pc.p.planet.x; }
float RT() { return pc.p.planet.y; }
float4 mulc(float4 c0, float4 c1, float4 c2, float4 c3, float4 v) { return c0 * v.x + c1 * v.y + c2 * v.z + c3 * v.w; }

// ------------------------------------------------------------------ medium (Hillaire / Bruneton)
struct Med { float3 rs; float ms; float3 ext; };
Med medium(float h) {
    PParams* p = pc.p;
    h = max(h, 0.0);
    float dr = exp(-h / 8.0) * p.atmo.y;
    float dm = exp(-h / 1.2) * p.atmo.x;
    float doz = max(0.0, 1.0 - abs(h - 25.0) / 15.0) * p.atmo.z;
    Med m;
    // Rayleigh at the sRGB primaries' effective wavelengths (615 / 550 / 465 nm, beta ~ lambda^-4
    // from Hillaire's 680 / 550 / 440 nm values), so the RGB output matches camera / eye bands
    m.rs = float3(8.67e-3, 13.558e-3, 26.6e-3) * dr;
    // aerosols beyond Hillaire's clean 1.2 km boundary layer (vertical AOD ~0.005, ~25x below the
    // global mean): a free-tropospheric mode and the stratospheric Junge layer. They scatter the
    // reddened grazing sunlight forward, which is what makes the low limb red-orange at twilight
    // seen from orbit, and add the slight day-side haze of EPIC / ISS views. Same phase function,
    // single-scattering albedo ~0.95.
    float dft = p.aero.x * exp(-h / max(p.aero.y, 0.1));
    float dst = p.aero.z * max(0.0, 1.0 - abs(h - 20.0) / 7.0);
    m.ms = 3.996e-3 * dm + dft + dst;
    m.ext = m.rs + m.ms + 0.444e-3 * dm + 0.05 * (dft + dst)
          + float3(2.74e-3, 1.881e-3, 0.17e-3) * doz;   // ozone (Chappuis) at 615 / 550 / 465 nm
    return m;
}
// both roots of |ro + t rd| = R (stable form, R-E2); x > y means no hit
float2 sphere2(float3 ro, float3 rd, float R) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - R * R;
    float disc = b * b - c;
    if (disc < 0.0) return float2(1.0, -1.0);
    float sq = sqrt(disc);
    float q = b >= 0.0 ? -b - sq : -b + sq;   // avoids cancellation
    float t0 = c / q, t1 = q;
    return float2(min(t0, t1), max(t0, t1));
}
float phase_r(float c) { return 3.0 / (16.0 * PI) * (1.0 + c * c); }
float phase_hg(float c, float g) {
    float k = (1.0 - g * g) / (4.0 * PI);
    return k / pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5);
}
float phase_m(float c) {
    float g = 0.75;   // continental/maritime mix; a little more backscatter than the sky's 0.8
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + c * c) / pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5);
}
float2 t_uv(float r, float mu) {
    float rg = RG(), rt = RT();
    float H = sqrt(rt * rt - rg * rg);
    float rho = sqrt(max(0.0, r * r - rg * rg));
    float disc = r * r * (mu * mu - 1.0) + rt * rt;
    float d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
    float dmin = rt - r, dmax = rho + H;
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
// Transmittance to the sun from radius r at sun cosine mu, with the planet shadow: the sun disc
// (angular radius sun.w) sinks below the sample's geometric horizon -> fades to 0 (penumbra).
float3 sun_trans(float r, float mu) {
    float rg = RG();
    r = clamp(r, rg + 1e-4, RT());
    float mu_h = -sqrt(max(0.0, 1.0 - (rg / r) * (rg / r)));
    float s = pc.p.sun.w * sqrt(max(0.0, 1.0 - mu_h * mu_h)) + 1e-4;
    float vis = smoothstep(mu_h - s, mu_h + s, mu);
    if (vis <= 0.0) return float3(0.0, 0.0, 0.0);
    return lut2(OFF_T, T_W, T_H, t_uv(r, max(mu, mu_h + s))).rgb * vis;
}
float3 ms_lut(float r, float mu_s) {
    return lut2(OFF_MS, MS_N, MS_N, float2(mu_s * 0.5 + 0.5, (r - RG()) / (RT() - RG()))).rgb;
}
// Airglow. OI 557.7 nm green line: a thin layer peaking at 95 km with a sharp underside
// (sigma 2.2 km below, 3.5 km above, FWHM ~7 km, as rocket/limb profiles show), so from orbit
// it reads as a faint thin line ~95 km above the dark limb rather than a band. OI 630 nm red
// line: a much fainter, broad layer at ~250 km (sigma 45 km) that gives the dim reddish fringe
// above the green line in long ISS exposures. Emission per km, scaled by light.y.
float3 airglow(float h) {
    float x = (h - 95.0) / (h < 95.0 ? 2.2 : 3.5);
    return float3(0.30, 1.0, 0.42) * exp(-x * x) * pc.p.light.y;
}
float3 airglow_red(float h) {
    float x = (h - 250.0) / 45.0;
    return float3(1.0, 0.07, 0.05) * (0.012 * exp(-x * x)) * pc.p.light.y;
}
// One march step [t, t + dt] along rd from ro (planet-centred km); updates T and L.
void scatter_step(float3 ro, float3 rd, float t, float dt, float3 sun, inout float3 T, inout float3 L) {
    float3 pos = ro + rd * (t + 0.5 * dt);
    float r = length(pos);
    float mus = dot(pos / r, sun);
    Med m = medium(r - RG());
    float c = dot(rd, sun);
    float3 sc = m.rs + m.ms;
    float3 S = (m.rs * phase_r(c) + m.ms * phase_m(c)) * sun_trans(r, mus) + sc * ms_lut(r, mus);
    float3 st = exp(-m.ext * dt);
    float3 e = max(m.ext, float3(1e-7, 1e-7, 1e-7));
    L += T * (S - S * st) / e;
    T *= st;
}
// March [t0, t1] with n steps, denser toward the end (dens = 1) or the start (dens = 0).
void march(float3 ro, float3 rd, float t0, float t1, int n, int dens, inout float3 T, inout float3 L) {
    float3 sun = pc.p.sun.xyz;
    float prev = 0.0;
    for (int i = 1; i <= n; i++) {
        float s = float(i) / float(n);
        float u = dens == 1 ? 1.0 - (1.0 - s) * (1.0 - s) : s * s;
        float a = t0 + (t1 - t0) * prev, b = t0 + (t1 - t0) * u;
        scatter_step(ro, rd, a, b - a, sun, T, L);
        prev = u;
    }
}
// atmosphere between t0 and t1 on the ray (clipped to the atmosphere shell), split at the
// perigee so the dense part near the closest approach gets the small steps
// Airglow emission integrated over [a, b] with its own uniform steps (the layer is ~10 km thick,
// far thinner than the scattering march's steps from orbit). No extinction: above ~80 km T ~ 1.
float3 airglow_seg(float3 ro, float3 rd, float a, float b, int red) {
    if (b <= a || pc.p.light.y <= 0.0) return float3(0, 0, 0);
    float3 acc = float3(0, 0, 0);
    float dt = (b - a) / 24.0;
    for (int i = 0; i < 24; i++) {
        float h = length(ro + rd * (a + (i + 0.5) * dt)) - RG();
        acc += red == 1 ? airglow_red(h) : airglow(h);
    }
    return acc * dt;
}
// Emission of one airglow shell [RG + lo, RG + hi] on [t0, t1]: the inbound part (before the
// perigee tp / the inner sphere) and the outbound part, each with its own steps.
void airglow_shell(float3 ro, float3 rd, float t0, float t1, float tp, float lo, float hi, int red,
                   inout float3 Lin, inout float3 Lout) {
    float2 ga = sphere2(ro, rd, RG() + hi), gi = sphere2(ro, rd, RG() + lo);
    if (ga.x > ga.y) return;
    bool inner = gi.x <= gi.y;
    float a0 = max(ga.x, t0), a1 = min(min(inner ? gi.x : tp, t1), tp);
    float b0 = max(max(inner ? gi.y : tp, t0), tp), b1 = min(ga.y, t1);
    Lin += airglow_seg(ro, rd, a0, a1, red);
    Lout += airglow_seg(ro, rd, b0, b1, red);
}
void atmosphere(float3 ro, float3 rd, float t0, float t1, int n, inout float3 T, inout float3 L) {
    // airglow shells reach above the scattering top (RT = RG + 100 km): integrate them on the
    // unclipped ray, then the scattering march on the part inside RT. Above ~80 km T ~ 1, so
    // the inbound emission is seen through the current T, the outbound through the segment's.
    float tp0 = clamp(-dot(ro, rd), max(t0, 0.0), t1);
    float3 gin = float3(0, 0, 0), gout = float3(0, 0, 0);
    airglow_shell(ro, rd, max(t0, 0.0), t1, tp0, 80.0, 112.0, 0, gin, gout);
    airglow_shell(ro, rd, max(t0, 0.0), t1, tp0, 130.0, 400.0, 1, gin, gout);
    L += T * gin;
    float2 top = sphere2(ro, rd, RT());
    if (top.x <= top.y) {
        float s0 = max(t0, max(top.x, 0.0)), s1 = min(t1, top.y);
        if (s1 > s0) {
            float tp = clamp(-dot(ro, rd), s0, s1);
            if (tp > s0 + 1e-3) march(ro, rd, s0, tp, n, 1, T, L);
            if (s1 > tp + 1e-3) march(ro, rd, tp, s1, n, 0, T, L);
        }
    }
    L += T * gout;
}

// ------------------------------------------------------------------ LUT kernels
[shader("compute")]
[numthreads(8, 8, 1)]
void transmittanceCS(uint3 id : SV_DispatchThreadID) {
    if (id.x >= T_W || id.y >= T_H) return;
    float rg = RG(), rt = RT();
    float2 uv = (float2(id.xy) + 0.5) / float2(T_W, T_H);
    float H = sqrt(rt * rt - rg * rg);
    float rho = H * uv.y;
    float r = sqrt(rho * rho + rg * rg);
    float dmin = rt - r, dmax = rho + H;
    float d = dmin + uv.x * (dmax - dmin);
    float mu = d == 0.0 ? 1.0 : clamp((H * H - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
    float3 ro = float3(0.0, r, 0.0), rd = float3(sqrt(max(0.0, 1.0 - mu * mu)), mu, 0.0);
    float tmax = max(sphere2(ro, rd, rt).y, 0.0);
    float3 od = float3(0.0, 0.0, 0.0);
    float dt = tmax / 64.0;
    for (int i = 0; i < 64; i++) od += medium(length(ro + rd * (i + 0.5) * dt) - rg).ext * dt;
    pc.atmo[OFF_T + id.y * T_W + id.x] = float4(exp(-od), 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void multiScatterCS(uint3 id : SV_DispatchThreadID) {
    if (id.x >= MS_N || id.y >= MS_N) return;
    float rg = RG(), rt = RT();
    float2 uv = (float2(id.xy) + 0.5) / float(MS_N);
    float mus = uv.x * 2.0 - 1.0;
    float r = rg + max(uv.y * (rt - rg), 0.002);
    float3 ro = float3(0.0, r, 0.0), sd = float3(sqrt(max(0.0, 1.0 - mus * mus)), mus, 0.0);
    float3 L2 = float3(0, 0, 0), F = float3(0, 0, 0);
    float albedo = pc.p.atmo.w;
    for (int a = 0; a < 8; a++) {
        for (int b = 0; b < 8; b++) {
            float ct = 1.0 - 2.0 * (a + 0.5) / 8.0;
            float phi = 2.0 * PI * (b + 0.5) / 8.0;
            float st = sqrt(max(0.0, 1.0 - ct * ct));
            float3 rd = float3(st * cos(phi), ct, st * sin(phi));
            float2 g = sphere2(ro, rd, rg);
            bool hit = g.x <= g.y && g.x > 0.0;
            float tmax = hit ? g.x : max(sphere2(ro, rd, rt).y, 0.0);
            float dt = tmax / 20.0;
            float3 T = float3(1, 1, 1), Lp = float3(0, 0, 0), Fp = float3(0, 0, 0);
            for (int i = 0; i < 20; i++) {
                float3 pos = ro + rd * (i + 0.5) * dt;
                float rp = length(pos);
                Med m = medium(rp - rg);
                float3 sc = m.rs + m.ms;
                float3 stp = exp(-m.ext * dt);
                float3 inv = 1.0 / max(m.ext, float3(1e-7, 1e-7, 1e-7));
                float3 S = sc * sun_trans(rp, dot(pos / rp, sd)) / (4.0 * PI);
                Lp += T * (S - S * stp) * inv;
                Fp += T * (sc - sc * stp) * inv;
                T *= stp;
            }
            if (hit) {
                float3 upg = normalize(ro + rd * g.x);
                float mu_g = dot(upg, sd);
                Lp += T * sun_trans(rg + 1e-3, mu_g) * saturate(mu_g) * albedo / PI;
            }
            L2 += Lp;
            F += Fp / (4.0 * PI);
        }
    }
    L2 /= 64.0;
    F = F / 64.0 * (4.0 * PI);
    pc.atmo[OFF_MS + id.y * MS_N + id.x] = float4(L2 / max(1.0 - F, float3(1e-3, 1e-3, 1e-3)), 1.0);
}

// sky radiance seen from sea level, averaged over azimuth: x = sun cosine, y = view cosine
[shader("compute")]
[numthreads(8, 8, 1)]
void skyGroundCS(uint3 id : SV_DispatchThreadID) {
    if (id.x >= SG_N || id.y >= SG_N) return;
    float2 uv = (float2(id.xy) + 0.5) / float(SG_N);
    float mus = uv.x * 2.0 - 1.0;
    float muv = uv.y;                       // upper hemisphere only
    float r = RG() + 0.01;
    float3 ro = float3(0.0, r, 0.0);
    float3 sun = float3(sqrt(max(0.0, 1.0 - mus * mus)), mus, 0.0);
    float3 acc = float3(0, 0, 0);
    for (int k = 0; k < 8; k++) {
        float phi = PI * (k + 0.5) / 8.0;
        float st = sqrt(max(0.0, 1.0 - muv * muv));
        float3 rd = float3(st * cos(phi), muv, st * sin(phi));
        float tmax = sphere2(ro, rd, RT()).y;
        float3 T = float3(1, 1, 1), L = float3(0, 0, 0);
        float prev = 0.0;
        for (int i = 1; i <= 24; i++) {
            float u = pow(float(i) / 24.0, 2.0);
            scatter_step(ro, rd, tmax * prev, tmax * (u - prev), sun, T, L);
            prev = u;
        }
        acc += L;
    }
    pc.atmo[OFF_SG + id.y * SG_N + id.x] = float4(acc / 8.0, 1.0);
}

float3 sky_ground(float mus, float muv) {
    return lut2(OFF_SG, SG_N, SG_N, float2(mus * 0.5 + 0.5, saturate(muv))).rgb;
}

// sky irradiance on a horizontal surface at sea level vs sun cosine
[shader("compute")]
[numthreads(64, 1, 1)]
void irradianceCS(uint3 id : SV_DispatchThreadID) {
    if (id.x >= IRR_N) return;
    float mus = (id.x + 0.5) / IRR_N * 2.0 - 1.0;
    float3 e = float3(0, 0, 0);
    for (int j = 0; j < 32; j++) {
        float mu = (j + 0.5) / 32.0;
        e += sky_ground(mus, mu) * mu / 32.0;
    }
    pc.atmo[OFF_IRR + id.x] = float4(e * 2.0 * PI, 1.0);
}

float3 sky_irr(float mus) {
    float x = clamp((mus * 0.5 + 0.5) * IRR_N - 0.5, 0.0, IRR_N - 1.0);
    uint i0 = uint(x);
    uint i1 = min(i0 + 1, IRR_N - 1);
    return lerp(pc.atmo[OFF_IRR + i0].rgb, pc.atmo[OFF_IRR + i1].rgb, frac(x));
}

// ------------------------------------------------------------------ surface
struct VSOut {
    float4 position : SV_Position;
    float3 rel : REL;          // camera-relative (m)
    float height : HEIGHT;
};

[shader("vertex")]
VSOut planetVS(uint vid : SV_VulkanVertexID) {
    PParams* p = pc.p;
    PVertex v = pc.v[vid];
    // vertexOffset = patch * VERTS, so the vertex id carries the patch (no firstInstance needed)
    float3 rel = pc.patches[vid / PATCH_VERTS].origin.xyz + v.pos;
    VSOut o;
    o.position = mulc(p.view_proj[0], p.view_proj[1], p.view_proj[2], p.view_proj[3], float4(rel, 1.0));
    o.rel = rel;
    o.height = v.height;
    return o;
}

float lum(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// Cube textures are sampled with explicit direction gradients: implicit cube derivatives are
// formed from per-lane face (s, t) and blow up for 2x2 quads straddling a cube-face edge
// (lavapipe picks a far-too-coarse mip there -> a dotted seam along the face edges). The
// direction itself is continuous across faces, so its screen derivatives are the right footprint.
static float3 g_dx = float3(0, 0, 0);
static float3 g_dy = float3(0, 0, 0);
float4 cube(SamplerCube t, float3 d) { return t.SampleGrad(d, g_dx, g_dy); }

// cloud opacity along a view (or sun) direction crossing the layer at cosine mu to the vertical
// The Blue Marble cloud map is a display-encoded brightness over a dark background, i.e. roughly
// nadir coverage x cloud albedo: map it to the nadir opacity a_n (exponent planet.w), then to an
// optical depth so oblique paths (limb, low sun) get more opaque as 1 - exp(-tau / |mu|).
float cloud_alpha(float c, float mu) {
    float an = pow(saturate((c - 0.04) / 0.92), pc.p.planet.w) * 0.995;
    float tau = -log(1.0 - an);
    return 1.0 - exp(-tau / max(abs(mu), 0.06));
}
// Radiance of the cloud layer at direction `up` (unit), seen along `rd`; a = opacity seen along rd.
float3 cloud_radiance(float3 up, float3 rd, float c, float a, float3 n) {
    PParams* p = pc.p;
    float3 sun = p.sun.xyz;
    float rc = p.planet.z;
    float mus = dot(up, sun);
    float mun = dot(n, sun);
    float3 Ts = sun_trans(rc, mus);
    // thick clouds are bright and close to Lambertian seen from above (albedo up to ~0.8 with the
    // tops' multiple scattering); thin clouds forward-scatter the sun toward a viewer looking at it
    float alb = 0.62 + 0.2 * a;   // thin cloud is dimmer only through its opacity, thick tops ~0.8
    float diffuse = alb / PI * saturate(lerp(mus, mun, 0.7) * 0.85 + 0.15) * smoothstep(-0.12, 0.05, mus);
    float fwd = phase_hg(dot(rd, sun), 0.65) * 0.35 * a * (1.0 - a * 0.5);
    float3 amb = sky_irr(mus) * alb / PI * 0.8 * (0.75 + 0.25 * dot(n, up));
    return Ts * (diffuse + fwd) + amb;
}


float3 east_of(float3 up) {
    float3 e = cross(float3(0.0, 0.0, 1.0), up);
    float l = length(e);
    return l < 1e-6 ? float3(0.0, 1.0, 0.0) : e / l;
}
// ---- cloud detail below the cloud map's ~5 km texels (only where the pixel footprint resolves it)
static float g_fp = 1e9;   // pixel footprint at the surface (km); 1e9 = no detail (sky pass, shadows)
float hash3(float3 p) {
    p = frac(p * 0.3183099 + 0.1);
    p *= 17.0;
    return frac(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float vnoise(float3 x) {
    float3 i = floor(x), f = frac(x);
    f = f * f * (3.0 - 2.0 * f);
    return lerp(lerp(lerp(hash3(i), hash3(i + float3(1, 0, 0)), f.x), lerp(hash3(i + float3(0, 1, 0)), hash3(i + float3(1, 1, 0)), f.x), f.y),
                lerp(lerp(hash3(i + float3(0, 0, 1)), hash3(i + float3(1, 0, 1)), f.x), lerp(hash3(i + float3(0, 1, 1)), hash3(i + float3(1, 1, 1)), f.x), f.y), f.z);
}
// zero-mean fBm at 16, 8, 4, 2 km, each octave faded in once it spans >= ~3 pixel footprints
float cloud_detail(float3 dir) {
    float3 x = dir * (pc.p.planet.z / 16.0);
    float acc = 0.0, amp = 0.5, scale = 16.0;
    for (int o = 0; o < 4; o++) {
        float w = saturate(scale / (3.0 * g_fp) - 1.0);
        if (w <= 0.0) break;
        acc += (vnoise(x) - 0.5) * amp * w;
        x = x * 2.03 + float3(1.7, 9.2, 3.1);
        amp *= 0.55;
        scale *= 0.5;
    }
    return acc;
}
// the cloud map (r = nadir opacity, g = top height) as every cloud path reads it, with the
// clear-sky window (hole.w > 0: the descent's fair-weather area) applied once, here: the
// volume, the 2D sheet, the cloud-top relief and the ground's cloud shadow all agree
float hole_keep(float3 dir) {
    PParams* p = pc.p;
    if (p.hole.w <= 0.0) return 1.0;
    float3 d = normalize(dir);
    float ang = acos(clamp(dot(d, p.hole.xyz), -1.0, 1.0));
    // round 7 (cloud3.z): wobble the window's radius with two octaves of noise at ~0.6 / 0.3 of
    // the radius, so its edge follows no circle and reads as a fair-weather area, not a cut-out
    if (p.cloud3.z > 0.0) {
        float3 x = d / (0.6 * p.hole.w);
        float n = vnoise(x + 3.7) * 0.65 + vnoise(x * 2.03 + 8.1) * 0.35;
        ang *= 1.0 + p.cloud3.z * (n - 0.5) * 2.0;
    }
    return smoothstep(p.hole.w, 2.2 * p.hole.w, ang);
}
float2 cloud_map(float3 dir) {
    float2 m = cube(tClouds, dir).rg;
    // Earth round 7 (cloud3.x, y): feather the cloud edges at fine footprints. The VIIRS map goes
    // from clear to solid within one ~5 km texel, so at 400-150 km decks ended in cliffs. Mix in
    // the coverage at a coarse mip (~cloud3.x km), with the mix itself modulated by ~9 km noise:
    // edges get a broken, thinning skirt several texels wide (which cloud_cov's detail and the
    // volume's cells then break into fragments), interiors (coarse = fine) are unchanged. Only
    // below ~2.5-5 km footprints, so the full-disk (EPIC) views keep their cloud fraction.
    PParams* p = pc.p;
    float we = p.cloud3.x > 0.0 ? p.cloud3.y * (1.0 - smoothstep(2.5, 5.0, g_fp)) : 0.0;
    if (we > 0.0) {
        uint fw, fh;
        tClouds.GetDimensions(fw, fh);
        float lod = max(log2(float(fw) * p.cloud3.x / (0.5 * PI * RG())), 0.0);
        float cb = tClouds.SampleLevel(dir, lod).r;
        float nz = vnoise(normalize(dir) * (RG() / 9.0) + 2.3);
        m.x = lerp(m.x, cb, saturate(we * (0.3 + 1.4 * nz)));
    }
    // Earth round 5: erode rather than scale. Inside the window's ramp, thin cover clears first
    // and thick cores shrink from their own edges (threshold 1 - keep), so the window reads as
    // fair weather with natural cloud edges instead of a circular cut-out of uniformly
    // thinned cloud. keep == 1 (no window / outside it) leaves the coverage unchanged.
    float k = hole_keep(dir);
    if (k < 1.0) {
        // round 7 (cloud3.z > 0): break the cover up before eroding it. Solid decks (m = 1) survive
        // the erosion unchanged until keep reaches 0, so an overcast window had a crisp rim; scaling
        // the cover by ~12 km noise inside the ramp lets the deck break into cells first
        // (factor mean ~1: thin cover is not cleared any faster than in round 6).
        if (p.cloud3.z > 0.0) {
            float3 xn = normalize(dir) * (RG() / 12.0);
            float n = vnoise(xn + 5.1) * 0.6 + vnoise(xn * 2.07 + 1.3) * 0.4;
            m.x = saturate(m.x * (1.0 + (1.0 - k) * (2.0 * n - 1.0) * 0.8));
        }
        m.x = saturate((m.x - (1.0 - k)) / max(k, 1e-3));
    }
    return m;
}
float cloud_tex(float3 dir) { return cloud_map(dir).x; }
// ---- mesoscale structure for multi-km footprints (Earth round 3). The VIIRS map is near-clipped
// inside cloud masses (its interior brightness spread is ~0.1 of the range), and at full-disk
// footprints (~10-15 km) its mips average away what is left, so thick decks rendered as one flat
// white sheet. EPIC shows bright convective cores, greyer thinner decks and cellular / streaky
// texture there. Two optical-depth factors for the far (uniform-column) regime, mean ~1:
// - cloud-top height (the VIIRS CTH map: real, independent data): deep tops are optically
//   thicker, low decks thinner;
// - zero-mean fBm at 256 .. 32 km, each octave faded in only once it spans >= 3 pixel footprints
//   (as cloud_detail: nothing below the display's resolution, so no aliasing), stretched 2:1
//   along a slowly turning axis (streets / bands rather than isotropic blobs).
float cloud_meso(float3 dir) {
    float R = pc.p.planet.z;
    float3 x = dir * (R / 256.0);
    // anisotropy: squash one axis that rotates over ~3000 km
    float a = vnoise(dir * (R / 3000.0) + 4.1) * 6.2832;
    float3 E = east_of(dir), N = cross(dir, E);
    float3 ax = E * cos(a) + N * sin(a);
    x += ax * dot(x, ax);
    // meso.w (Earth round 4): fine structure. 0 = round 3 (4 octaves, persistence 0.6, octaves
    // from 3 footprints); > 0 adds a 16 km octave, raises the persistence toward 0.6 + 0.25 w and
    // fades octaves in from 3 - w footprints (w <= 1: still >= 2 footprints, no aliasing)
    float fw = pc.p.meso.w;
    float acc = 0.0, amp = 0.5, scale = 256.0;
    int no = fw > 0.0 ? 5 : 4;
    for (int o = 0; o < no; o++) {
        float w = saturate(scale / ((3.0 - fw) * g_fp) - 1.0);
        acc += (vnoise(x) - 0.5) * amp * w;
        x = x * 2.03 + float3(5.3, 1.9, 7.7);
        amp *= 0.6 + 0.25 * fw;
        scale *= 0.5;
    }
    return acc;
}

// coverage with sub-texel structure: the detail erodes / thickens mid-coverage most (edges, thin
// sheets), leaves clear sky clear and solid decks solid
float cloud_cov(float3 dir, float c) {
    if (g_fp > 6.0) return c;
    return saturate(c + cloud_detail(dir) * 2.2 * sqrt(saturate(c * (1.0 - c) * 4.0)) * smoothstep(0.02, 0.12, c));
}
float cloud_at(float3 dir) { return cloud_cov(dir, cloud_tex(dir)); }
// relief normal of the cloud tops: coverage as a height field (~2.5 km per unit), forward differences
float3 cloud_normal(float3 uq, float c) {
    if (g_fp > 4.0) return uq;
    float e_km = max(2.0 * g_fp, 1.0);
    float e = e_km / pc.p.planet.z;
    float3 E = east_of(uq), Nn = cross(uq, E);
    float cx = cloud_at(normalize(uq + E * e)), cy = cloud_at(normalize(uq + Nn * e));
    float k = 2.5 / e_km * saturate(1.5 - g_fp / 4.0);
    return normalize(uq - E * ((cx - c) * k) - Nn * ((cy - c) * k));
}

// ---- cloud volume (cloud.w = 1). A column per map direction: coverage c = the map's nadir
// opacity, top = the VIIRS cloud-top height (or base + (cloud.y - base) c^1.5 where unknown);
// thick cover fills base..top, thin cover (cirrus over clear sky, the map's dim high tops) is a
// sheet just under its measured top. Inside: Perlin-Worley shape noise remapped by coverage,
// a domed vertical profile (flat bases, rounded turrets), Worley erosion below ~4 km footprints.
static const float CV_LO = 0.3;    // shell (km above RG): lowest base, highest top
static const float CV_HI = 14.5;
bool cloud_volume() { return pc.p.cloud.w > 0.5; }
struct CCol { float c; float hb; float ht; float ce; float eh; };
CCol cloud_col(float3 up) {
    PParams* p = pc.p;
    float2 m = cloud_map(up);
    CCol k;
    // round 4: at multi-km footprints (the meso regime) the coverage gamma eases toward
    // meso2.x: the thin decks the map's mips spread out keep their opacity (EPIC's cloud fraction)
    float gam = p.meso2.x > 0.0 ? lerp(p.planet.w, p.meso2.x, smoothstep(0.6, 2.5, g_fp)) : p.planet.w;
    k.c = pow(saturate((m.x - 0.04) / 0.92), gam);
    // round 7 (cloud3.x > 0, fine footprints): the column's *shape* (deck thickness, how far the
    // tops sink) follows the coverage at a coarse mip (2 x cloud3.x km, thinned toward the clear
    // window), not the 5 km texel: decks taper over ~10-30 km toward their edges and toward the
    // window instead of ending in walls. Opacity still follows k.c.
    k.ce = k.c;
    k.eh = 0.6;
    float wsh = p.cloud3.x > 0.0 ? saturate(p.cloud3.y * (1.0 - smoothstep(2.5, 5.0, g_fp))) : 0.0;
    if (wsh > 0.0) {
        uint fw, fh;
        tClouds.GetDimensions(fw, fh);
        float lod = max(log2(float(fw) * 2.0 * p.cloud3.x / (0.5 * PI * RG())), 0.0);
        float cs = tClouds.SampleLevel(up, lod).r * hole_keep(up);
        k.ce = lerp(k.c, min(k.c, pow(saturate((cs - 0.04) / 0.92), gam)), wsh);
        k.eh = lerp(0.6, 0.95, wsh);
    }
    float base = p.cloud.x;
    float top = m.y > 0.5 / 255.0 ? m.y * (255.0 / 20.0) : base + 0.5 + (p.cloud.y - base) * pow(k.c, 1.5);
    top = clamp(top, CV_LO + 0.4, CV_HI - 0.1);
    float span = max(top - base, 0.4);
    float thick = max(span * saturate(0.12 + 0.88 * smoothstep(0.15, 0.85, k.ce)), 0.35);
    k.ht = top;
    k.hb = max(min(base, top - 0.4), top - thick);
    return k;
}
// far-regime optical-depth factor (see cloud_meso): meso.x = log2 range of the cloud-top
// factor (low decks .. deep tops), meso.y = log2 amplitude of the fBm factor; 0, 0 = off.
// (meso.z, the brightness bias, applies in the march: see meso_refl_ratio's use)
float meso_tau_factor(float3 dir, CCol k) {
    float4 m = pc.p.meso;
    float tf = exp2(m.x * (smoothstep(2.5, 12.5, k.ht) - 0.45));
    return tf * exp2(m.y * cloud_meso(dir));
}
float remap01(float v, float lo, float hi) { return saturate((v - lo) / max(hi - lo, 1e-4)); }
// column optical depth of a map column (nadir opacity c, solid decks +3)
float col_tau(CCol k) { return -log(1.0 - 0.995 * k.c) + 3.0 * smoothstep(0.7, 1.0, k.c); }
// far-regime weight of the meso structure: 0 at ISS footprints (< 0.6 km), 1 beyond 2.5 km
float meso_w() { return smoothstep(0.6, 2.5, g_fp); }
// Brightness ratio of a column whose optical depth is scaled by f: the two-stream reflectance
// R(x) = x / (2 + x), x = (1 - g) tau, g = 0.85, against the unscaled column. The march's
// radiance saturates within the top optical depth or two, so without this a thicker column is
// no brighter; real thick decks keep brightening with depth (multiple scattering from below).
float meso_refl_ratio(float tau, float f) {
    float x = 0.15 * tau, xf = x * f;
    return (xf / (2.0 + xf)) / max(x / (2.0 + x), 1e-4);
}
// density (x cloud.z = extinction) at p (planet frame, km), height h above RG. Two regimes with
// the same mean column opacity 1 - exp(-tau_c), tau_c = -ln(1 - c) (the map's nadir opacity):
// near (pixel footprint < the cells): solid cells covering an area fraction ~c, where the 3D
// Perlin-Worley noise picks which part of the texel is cloud and how tall each turret grows,
// eroded by Worley detail; far: a uniform-density column (no noise to alias into dots).
float cloud_density(float3 p, float h, CCol k, bool detail) {
    if (k.c < 0.01 || h <= k.hb || h >= k.ht) return 0.0;
    float H = k.ht - k.hb;
    float hn = (h - k.hb) / H;
    float sig = pc.p.cloud.z;
    float tau_c = col_tau(k);
    float T = pc.p.cloud2.z;
    // large-scale (~15 km) top relief for every regime: anvil plateaus, overshooting domes
    float nb = tNoise.SampleLevel(p / (T * 6.0) + 0.21, 0.0).r;
    // tops sink toward cloud edges (thickness follows the map's opacity): domes, not cliffs
    hn /= lerp(0.72, 1.0, nb) * lerp(0.3, 1.0, smoothstep(0.04, k.eh, k.ce));
    if (hn >= 1.0) return 0.0;
    float dome = smoothstep(0.0, 0.05, hn) * (1.0 - smoothstep(0.5, 1.0, hn));
    float du = dome * tau_c / (max(sig, 1e-3) * H * 0.55);
    float w = 1.0 - smoothstep(0.6, 2.5, g_fp);
    // multi-km footprints: mesoscale optical-depth structure (none at ISS footprints, w = 1)
    if (w < 1.0) du *= lerp(1.0, meso_tau_factor(p / length(p), k), meso_w());
    if (w <= 0.0) return du;
    // two shape octaves at incommensurate periods, the second rotated: no visible 128^3 tiling
    float4 n = tNoise.SampleLevel(p / T, 0.0);
    float3 p2 = float3(0.8 * p.x - 0.6 * p.y, 0.6 * p.x + 0.8 * p.y, p.z) / (T * 2.71) + 0.37;
    float4 n2 = tNoise.SampleLevel(p2, 0.0);
    float lowf = n.g * 0.625 + n.b * 0.25 + n.a * 0.125;
    float B = remap01(n.r * 0.6 + n2.r * 0.4, lowf * 0.6 + (n2.g * 0.7 + n2.b * 0.3) * 0.4 - 1.0, 1.0);
    // turret height: cells with more noise reach the column top, others stop lower
    float hc = hn / lerp(0.45, 1.0, saturate(B * 1.6 - 0.3));
    float cdome = smoothstep(0.0, 0.04, hc) * (1.0 - smoothstep(0.55, 1.0, hc));
    // round 7 (cloud3.y with cloud3.x > 0): widen the cells' coverage threshold on both sides,
    // so cell edges (and with them the deck edges) fade over a band instead of switching from
    // clear to solid; 0 = round 6
    float soft = pc.p.cloud3.x > 0.0 ? pc.p.cloud3.y : 0.0;
    // and near deck edges (coarse coverage k.ce low) let fewer cells through: broken fringes
    float ec = soft > 0.0 ? lerp(1.0, 0.3 + 0.7 * smoothstep(0.0, 0.8, k.ce), soft) : 1.0;
    float cell = saturate((k.c * ec - (1.0 - B) * 0.92 + 0.15 * soft) / (0.14 + 0.5 * soft + g_fp / T));
    float dp = cell * cdome;
    if (detail && dp > 0.0) {
        float4 e = tNoise.SampleLevel(p / (T * 0.17), 0.0);
        float hf = e.g * 0.625 + e.b * 0.25 + e.a * 0.125;
        hf = lerp(hf, 1.0 - hf, saturate(hn * 3.0));   // wispy bases, billowy tops
        dp = remap01(dp, hf * 0.5 * (1.0 - smoothstep(0.15, 0.35, g_fp)), 1.0);
    }
    // a solid cell's optical depth over its height, but never below the uniform column
    float dps = dp * max(1.0, tau_c / (max(sig, 1e-3) * H * 0.45 * max(k.c, 0.05)));
    // translucent fringes where the coarse coverage is low (round 7 shaping only)
    if (soft > 0.0) dps *= lerp(1.0, 0.35 + 0.65 * smoothstep(0.0, 0.6, k.ce), soft);
    return lerp(du, dps, w);
}
float cloud_density_at(float3 p, bool detail) {
    float r = length(p);
    float h = r - RG();
    if (h <= CV_LO || h >= CV_HI) return 0.0;
    return cloud_density(p, h, cloud_col(p / r), detail) ;
}
// optical depth toward the sun from p: steps of 0.25, 0.5, ... km (self-shadowing)
float cloud_tau_sun(float3 p, float3 sun, int n) {
    float tau = 0.0, t = 0.0, dt = 0.25;
    float sig = pc.p.cloud.z;
    for (int i = 0; i < n; i++) {
        float3 q = p + sun * (t + 0.5 * dt);
        tau += cloud_density_at(q, false) * sig * dt;
        t += dt;
        dt *= 2.0;
        if (length(q) - RG() > CV_HI) break;
    }
    return tau;
}
float ign(float2 px) { return frac(52.9829189 * frac(dot(px, float2(0.06711056, 0.00583715)))); }
// March the cloud volume over [t0, t1] of the ray; returns radiance (Lc) and grey transmittance
// (Tc), and the depth of the first cloud sample (tc, -1 if none) for the atmosphere split.
void cloud_march(float3 ro, float3 rd, float t0, float t1, int n, float jit, out float3 Lc, out float Tc, out float tc) {
    PParams* p = pc.p;
    float3 sun = p.sun.xyz;
    Lc = float3(0, 0, 0);
    Tc = 1.0;
    tc = -1.0;
    if (t1 <= t0) return;
    float dt = max((t1 - t0) / float(n), 0.12);
    int steps = min(int(ceil((t1 - t0) / dt)), n);
    float sig = p.cloud.z;
    float cs = dot(rd, sun);
    // dual-lobe phase (forward silver lining + backscatter glory side), octaves halve g
    float ph[3];
    for (int o = 0; o < 3; o++) {
        float s = pow(0.5, float(o));
        ph[o] = lerp(phase_hg(cs, -0.25 * s), phase_hg(cs, 0.75 * s), 0.72);
    }
    bool detail = g_fp < 0.35;
    int nsun = int(p.cloud2.y);
    // coarse steps over the shell; inside each, only the part of the ray within that column's
    // [base, top] is sampled (<= 4 substeps of <= 0.15 km) so thin decks are always resolved
    for (int i = 0; i <= steps && Tc > 0.0; i++) {
        // coarse grid jittered per pixel too: no contour terraces where columns change
        float ta = max(t0 + (float(i) + jit - 1.0) * dt, t0), tb = min(t0 + (float(i) + jit) * dt, t1);
        if (i == steps) tb = t1;
        if (tb <= ta) continue;
        float3 qa = ro + rd * ta, qb = ro + rd * tb;
        float h_a = length(qa) - RG(), h_b = length(qb) - RG();
        float3 qm = ro + rd * (0.5 * (ta + tb));
        CCol k = cloud_col(normalize(qm));
        if (k.c < 0.01) continue;
        float hlo = min(h_a, h_b), hhi = max(h_a, h_b);
        // the ray's perigee may lie inside the step (limb rays): h is not monotone there
        float tp = clamp(-dot(ro, rd), ta, tb);
        hlo = min(hlo, length(ro + rd * tp) - RG());
        if (hhi <= k.hb || hlo >= k.ht) continue;
        float sa = ta, sb = tb;
        if (abs(h_b - h_a) > 1e-4 && (tp <= ta + 1e-4 || tp >= tb - 1e-4)) {
            // monotone in the step: solve h(t) linear for the overlap with [hb, ht]
            float u0 = (k.hb - h_a) / (h_b - h_a), u1 = (k.ht - h_a) / (h_b - h_a);
            float lo = saturate(min(u0, u1)), hi = saturate(max(u0, u1));
            sa = ta + (tb - ta) * lo;
            sb = ta + (tb - ta) * hi;
        }
        float len = sb - sa;
        if (len <= 0.0) continue;
        // multi-km footprints: the column's brightness follows its (meso-scaled) optical depth
        float fr = 1.0;
        // meso.z: log2 brightness bias (the ratio's mean is > 1; opacity stays unbiased so the
        // cloud fraction is the map's)
        if (meso_w() > 0.0) fr = meso_refl_ratio(col_tau(k), lerp(1.0, meso_tau_factor(normalize(qm), k), meso_w())) * exp2(p.meso.z * meso_w());
        int m = clamp(int(ceil(len / 0.15)), 1, 4);
        float sd = len / float(m);
        for (int j = 0; j < m; j++) {
            float t = sa + (float(j) + jit) * sd;
            float3 q = ro + rd * t;
            float r = length(q);
            float h = r - RG();
            float d = cloud_density(q, h, j == 0 ? k : cloud_col(q / r), detail);
            if (d <= 0.0) continue;
            if (tc < 0.0) tc = t;
            float ext = d * sig;
            float mus = dot(q / r, sun);
            float3 Ts = sun_trans(r, mus);
            float3 S = float3(0, 0, 0);
            if (Ts.x + Ts.y + Ts.z > 1e-5) {
                float tau = cloud_tau_sun(q, sun, nsun);
                float a = 1.0, e = 1.0, ms = 0.0;
                for (int o = 0; o < 3; o++) { ms += a * ph[o] * exp(-tau * e); a *= 0.55; e *= 0.4; }
                S = Ts * ms * p.cloud2.w;
            }
            // ambient: sky dome (more toward the tops) + light bounced off the ground / lower deck
            float hn = saturate((h - k.hb) / (k.ht - k.hb));
            S += sky_irr(mus) / (2.0 * PI) * (0.45 + 0.55 * hn)
               + Ts * saturate(mus) * (0.08 + 0.35 * k.c) / (2.0 * PI) * (1.0 - hn);
            float st = exp(-ext * sd);
            Lc += Tc * S * fr * (1.0 - st);   // albedo ~1: (S sig - S sig st) / sig
            Tc *= st;
            if (Tc < 0.01) { Tc = 0.0; break; }
        }
    }
}
// cloud transmittance of the sun ray from a surface point (ground shadows, same density)
float cloud_shadow_vol(float3 P, float3 sun) {
    float2 a = sphere2(P, sun, RG() + CV_LO), b = sphere2(P, sun, RG() + CV_HI);
    float t0 = max(a.y, 0.0), t1 = b.y;
    if (t1 <= t0) return 1.0;
    t1 = min(t1, t0 + 60.0);   // grazing sun: the first 60 km of the slant path
    float tau = 0.0, dt = (t1 - t0) / 8.0;
    for (int i = 0; i < 8; i++) tau += cloud_density_at(P + sun * (t0 + (i + 0.5) * dt), false) * pc.p.cloud.z * dt;
    return exp(-tau);
}

// Beckmann NDF (Cox-Munk isotropic, slope variance m2)
float D_beckmann(float nh, float m2) {
    float c2 = nh * nh;
    return exp((c2 - 1.0) / (m2 * c2)) / (PI * m2 * c2 * c2);
}
float lambda_beckmann(float c, float m2) {
    float a = c / sqrt(m2 * max(1.0 - c * c, 1e-6));
    return a < 1.6 ? (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a) : 0.0;
}

// Earth round 6: the marine boundary layer over open ocean (sea salt / sulphate, AOD ~0.1 at
// 550 nm, nearly grey: Angstrom ~0.4). A thin slab at the surface, below everything the
// aerial-perspective LUT integrates: the surface radiance is attenuated along the view slant and
// the slab adds its single scattering of the transmitted sun (plane-parallel closed form) and of
// the sky light. Phase: two-term HG (forward 0.7, back -0.35, 15 %): backscatter ~0.045 / sr,
// lidar ratio ~25 sr as measured for maritime aerosol, so the haze also shows in the
// sun-behind-the-camera EPIC geometry. Weight: the albedo's water fraction at a coarse mip
// (~haze.w km texels), so it fades softly across coasts instead of following the shoreline.
float3 ocean_haze(float3 L, float3 up, float3 V, float3 rd, float3 sun, float3 Es, float3 Esky) {
    PParams* p = pc.p;
    if (p.haze.x <= 0.0) return L;
    uint fw, fh;
    tAlbedo.GetDimensions(fw, fh);
    float face_km = 0.5 * PI * RG();
    float lod = max(log2(float(fw) * p.haze.w / face_km), 0.0);
    float wm = smoothstep(0.55, 0.95, tAlbedo.SampleLevel(up, lod).a);
    if (wm <= 0.0) return L;
    float3 tau = p.haze.x * wm * pow(float3(615.0, 550.0, 465.0) / 550.0, float3(-p.haze.y, -p.haze.y, -p.haze.y));
    float mv = max(dot(up, V), 0.03);
    float ms = dot(up, sun);
    float3 Tv = exp(-tau / mv);
    float c = dot(sun, rd);
    float ph = lerp(phase_hg(c, 0.7), phase_hg(c, -0.35), 0.15);
    // round 7 (cloud3.w): dust-like absorption, stronger in the blue (Saharan dust: w0 ~0.98 / 0.95 /
    // 0.89 at 615 / 550 / 465 nm for cloud3.w = 1 with haze.z = 0.98)
    float3 w0 = p.haze.z - p.cloud3.w * float3(0.0, 0.03, 0.09);
    float3 Lss = ms > 0.0 ? Es * (w0 * ph * ms / (ms + mv)) * (1.0 - exp(-tau * (1.0 / max(ms, 1e-3) + 1.0 / mv))) : float3(0.0, 0.0, 0.0);
    // sky light: roughly isotropic radiance Esky / pi, half of it scattered upward
    float3 Lsk = Esky / PI * (0.5 * w0) * (1.0 - Tv);
    return L * Tv + Lss + Lsk;
}

float3 surface_radiance(float3 P, float3 rd, float3 up, float h) {
    PParams* p = pc.p;
    float3 sun = p.sun.xyz;
    float r = length(P);
    float4 a4 = cube(tAlbedo, up);
    float2 masks = cube(tMasks, up).rg;
    // Snow-free disc (Earth round 3 descent): where the terrain is pressed flat under the mountains
    // layer, the Blue Marble glaciers lose their relief and read as soft low clouds; inside the
    // disc the reflectance is capped at a valley forest's (snow, ice, bare rock) and the ice mask
    // to 0, and the slope map's normals (the real relief the geometry no longer has: lit, they draw
    // the old ridges as bright rims) flatten (full inside 0.7 w, none beyond w; FLATTEN's ramp is
    // r0 = 0.71 r1).
    float tw = 0.0;
    if (p.thaw.w > 0.0) {
        float k = length(p.thaw.xyz);
        tw = k > 0.0 ? k * (1.0 - smoothstep(0.7 * p.thaw.w, p.thaw.w, acos(clamp(dot(up, p.thaw.xyz / k), -1.0, 1.0)))) : 0.0;
        // (a per-channel cap at valley-forest reflectance, not a snow threshold: the glaciers'
        // mip-blended rims are only part snow and a threshold leaves them as bright outlines)
        a4.rgb = lerp(a4.rgb, min(a4.rgb, float3(0.05, 0.065, 0.04)), tw);
        masks.r *= 1.0 - tw;
    }
    float3 alb = a4.rgb * p.light.z;
    float water = saturate(a4.a);
    float2 s = cube(tSlope, up).rg * 2.0 - 1.0;
    float2 g = sign(s) * s * s * SLOPE_MAX * p.ocean.w * (1.0 - tw);
    float3 e = east_of(up);
    float3 nn = cross(up, e);
    float3 n_land = normalize(up - g.x * e - g.y * nn);
    // Shared Alps (Earth round 4): inside the graft disc the surface is the mountains scene's own
    // terrain (the geometry already is: patch heights from the same field). Albedo, normal (the
    // vista frame's world normal, exactly what the mountains layer shades with), sun visibility
    // and AO from the baked map; the real Earth returns over graft.w..graft_e.w.
    float sun_vis = 1.0, gao = 1.0;
    float gw = 0.0;
    if (p.graft.w > 0.0) {
        float gk = length(p.graft.xyz);
        float3 gc = p.graft.xyz / max(gk, 1e-20);
        float ge = dot(up, p.graft_e.xyz), gn = dot(up, p.graft_n.xyz);
        float gs = sqrt(ge * ge + gn * gn);
        float ang = atan2(gs, dot(up, gc));
        gw = gk * (1.0 - smoothstep(p.graft.w, p.graft_e.w, ang));
        if (gw > 0.0) {
            // world (x - cx, z - cz) in km: azimuthal equidistant about the vista (planet::graft)
            float2 dxz = gs > 0.0 ? ang * RG() * float2(ge, -gn) / gs : float2(0.0, 0.0);
            float2 uo = dxz / (2.0 * p.graft_n.w) + 0.5, ui = dxz / (2.0 * p.graft_l.x) + 0.5;
            // explicit LOD from the pixel footprint (this branch is non-uniform: no implicit
            // derivatives); graft_l.w = texels per level side
            float lo = log2(max(g_fp * p.graft_l.w / (2.0 * p.graft_n.w), 1e-4));
            float li = log2(max(g_fp * p.graft_l.w / (2.0 * p.graft_l.x), 1e-4));
            float4 ga = tGraftAlb.SampleLevel(float3(uo, 0.0), lo), gb = tGraftAux.SampleLevel(float3(uo, 0.0), lo);
            float ei = max(abs(ui.x - 0.5), abs(ui.y - 0.5)) * 2.0;
            float wi = 1.0 - smoothstep(0.86, 0.97, ei);
            if (wi > 0.0) {
                ga = lerp(ga, tGraftAlb.SampleLevel(float3(ui, 1.0), li), wi);
                gb = lerp(gb, tGraftAux.SampleLevel(float3(ui, 1.0), li), wi);
            }
            // DEM close streamed tile (layer 2): finer concentric level so OSM BuildingLod
            // objects (≤ 6.8 m texels) resolve under the camera. meso2.y = half extent km.
            float hc = p.meso2.y;
            if (hc > 0.0) {
                float2 uc = (dxz - p.meso2.zw) / (2.0 * hc) + 0.5;
                float lc = log2(max(g_fp * p.graft_l.w / (2.0 * hc), 1e-4));
                float ec = max(abs(uc.x - 0.5), abs(uc.y - 0.5)) * 2.0;
                float wc = 1.0 - smoothstep(0.86, 0.97, ec);
                if (wc > 0.0) {
                    ga = lerp(ga, tGraftAlb.SampleLevel(float3(uc, 2.0), lc), wc);
                    gb = lerp(gb, tGraftAux.SampleLevel(float3(uc, 2.0), lc), wc);
                }
            }
            float2 nxz = gb.rg * 2.0 - 1.0;
            float3 gnw = normalize(p.graft_e.xyz * nxz.x + gc * sqrt(saturate(1.0 - dot(nxz, nxz))) - p.graft_n.xyz * nxz.y);
            float3 galb = ga.rgb * p.graft_l.y;
            // toward the graft's edge its large-scale (~20 km) colour follows the real Earth's
            // (Blue Marble low-pass / graft low-pass): the disc does not show as a colour region
            // where the real terrain returns; the Alps around the vista keep the mountains' own
            float wt = p.graft_t.x * smoothstep(p.graft_t.y, p.graft.w, ang);
            if (wt > 0.0) {
                float3 rl = tAlbedo.SampleLevel(up, p.graft_t.z).rgb * p.light.z;
                float3 gl = tGraftAlb.SampleLevel(float3(uo, 0.0), p.graft_t.w).rgb * p.graft_l.y;
                galb *= lerp(float3(1.0), clamp((rl + 0.005) / (gl + 0.005), 0.25, 4.0), wt);
            }
            alb = lerp(alb, galb, gw);
            n_land = normalize(lerp(n_land, gnw, gw));
            // graft albedo alpha = 1 - water (Earth round 5: real lakes; 1 in procedural bakes)
            water = lerp(water, 1.0 - ga.a, gw);
            masks.r *= 1.0 - gw;
            sun_vis = lerp(1.0, gb.b, gw);
            gao = lerp(1.0, lerp(1.0, gb.a, p.graft_l.z), gw);
        }
    }
    // Procedural sub-texel land detail (aero.w gain): the cube maps resolve ~2.4 km, so from the
    // ISS (0.3-2 km pixels) the ground is magnified and soft. Below a 3.5 km footprint, four
    // octaves of value noise (2.4 / 1.0 / 0.42 / 0.18 km), each faded out before it can alias, modulate
    // the albedo in log space (~+-15%, colour-preserving) and act as a height field whose
    // amplitude follows the texture's own relief (flat plains stay flat, ranges get crinkled).
    float dw = (1.0 - smoothstep(1.2, 3.5, g_fp)) * (1.0 - water) * p.aero.w * (1.0 - gw);
    if (dw > 0.0) {
        float3 xk = up * RG();
        float f = 0.0, amp = 0.5, per = 2.4;
        float2 gr = float2(0.0, 0.0);
        for (int o = 0; o < 4; o++) {
            float fade = 1.0 - smoothstep(per * 0.35, per * 0.7, g_fp);
            if (fade > 0.0) {
                float3 q = xk / per + float(o) * 7.31;
                float n0 = vnoise(q), n1 = vnoise(q + e * 0.35), n2 = vnoise(q + nn * 0.35);
                f += amp * fade * (n0 - 0.5);
                gr += amp * fade * float2(n1 - n0, n2 - n0) / (0.35 * per);
            }
            amp *= 0.55;
            per *= 0.42;
        }
        alb *= exp(0.6 * f * dw);
        float relief = saturate(length(g) * 6.0 + 0.12);
        n_land = normalize(n_land - (gr.x * e + gr.y * nn) * (0.45 * relief * dw));
    }
    float mus = dot(up, sun);
    float3 Ts = sun_trans(r, mus);
    // cloud shadow: the layer's opacity where the sun ray from P crosses it
    float rc = p.planet.z;
    float2 cs = sphere2(P, sun, rc);
    float shadow = 1.0;
    if (cloud_volume()) {
        shadow = lerp(1.0, cloud_shadow_vol(P, sun), p.light.w);
    } else if (cs.y > 0.0) {
        float3 q = normalize(P + sun * cs.y);
        shadow = 1.0 - p.light.w * cloud_alpha(cloud_at(q), mus);
    }
    float3 Es = Ts * shadow;
    float3 Esky = sky_irr(mus);
    float3 V = -rd;
    // land / land ice: Lambert on the relief normal
    float nl = saturate(dot(n_land, sun));
    float3 Lland = alb / PI * (Es * nl * sun_vis + Esky * (0.55 + 0.45 * dot(n_land, up)) * gao);
    // ocean: Cox-Munk glint (Cook-Torrance, Beckmann, Smith G), Fresnel, sky reflection, and the
    // water body from the Blue Marble ocean reflectance; sea ice (bright water texels) is diffuse
    // mesoscale wind variability (40-150 km): the Cox-Munk slope variance is linear in wind
    // speed, so calm / rough patches modulate the glint the way real glint images show
    float3 wx = up * (RG() / 150.0);
    float wv = vnoise(wx) * 0.65 + vnoise(wx * 2.7 + 3.1) * 0.35;
    float m2 = p.ocean.x * exp((wv - 0.5) * 1.6);
    float nv = max(saturate(dot(up, V)), 1e-4);
    float nlw = saturate(dot(up, sun));
    float3 H = normalize(V + sun);
    float nh = saturate(dot(up, H));
    float vh = saturate(dot(V, H));
    float F0 = p.ocean.y;
    float F = F0 + (1.0 - F0) * pow(1.0 - vh, 5.0);
    float G = 1.0 / (1.0 + lambda_beckmann(nv, m2) + lambda_beckmann(max(nlw, 1e-4), m2));
    float spec = nlw > 0.0 ? F * D_beckmann(max(nh, 1e-3), m2) * G / (4.0 * nv) : 0.0;
    float Fv = F0 + (1.0 - F0) * pow(1.0 - nv, 5.0);
    float3 refl = reflect(rd, up);
    float3 Lsky = sky_ground(mus, saturate(dot(refl, up)));
    // water-leaving reflectance: open ocean ~(0.002, 0.007, 0.026) (case-1 water, blue band
    // ~0.02-0.04); the Blue Marble ocean colour is kept where it is brighter (shallows, blooms)
    float3 rho_w = max(alb, float3(0.0022, 0.0075, 0.026));
    float3 Lsea = Es * spec + Fv * Lsky + (1.0 - Fv) * rho_w / PI * (Es * nlw + Esky) * 0.75;
    float ice = max(masks.r * water, water * smoothstep(p.ocean.z, p.ocean.z * 2.0, lum(alb)));
    float3 Lice = alb / PI * (Es * nlw + Esky);
    float3 Lw = lerp(Lsea, Lice, ice);
    float3 L = lerp(Lland, Lw, water);
    L = ocean_haze(L, up, V, rd, sun, Es, Esky);
    // night lights (always on; drowned by daylight), dimmed by cloud cover above
    float nv0 = cube(tNight, up).r;
    float3 city = float3(1.0, 0.72, 0.42) * pow(nv0, 2.2) * p.light.x;   // sodium + LED mix
    float cov_city;
    if (cloud_volume()) {
        CCol k = cloud_col(up);
        cov_city = 1.0 - exp(-k.c * p.cloud.z * (k.ht - k.hb) * 0.15);
    } else {
        cov_city = cloud_alpha(cloud_at(up), 1.0);
    }
    L += city * (1.0 - 0.75 * cov_city);
    return L;
}

struct FSOut { float4 color : SV_Target0; };

float3 debug_color(float3 up, int mode) {
    if (mode == 1) return cube(tAlbedo, up).rgb;
    if (mode == 2) return float3(cube(tAlbedo, up).a, cube(tMasks, up).r, 0.0);
    if (mode == 3) return float3(cube(tSlope, up).rg, 0.0);
    if (mode == 4) return float3(cube(tClouds, up).r, cube(tMasks, up).rg);
    if (mode == 5) return cube(tNight, up).rrr;
    return float3(0, 0, 0);
}

[shader("fragment")]
FSOut planetFS(VSOut i) {
    PParams* p = pc.p;
    float3 relk = i.rel * 0.001;
    float dist = length(relk);
    float3 rd = relk / dist;
    float3 ro = p.cam_km.xyz;
    // Far cameras (EPIC at 1.5e6 km): sphere2 from |ro| ~ 1e6 km cancels catastrophically in f32
    // (b^2 - c with both ~1e12: ~20 km root error, more than the cloud shell is thick). Rebase the
    // ray origin to 2500 km before the surface point, which still precedes every atmosphere /
    // cloud / airglow crossing of a ray that ends on the ground; all t below are from the new ro.
    float t_off = max(dist - 2500.0, 0.0);
    ro += rd * t_off;
    dist -= t_off;
    // The raster point lies on a chord (up to ~s^2/8R below the sphere for coarse patches):
    // re-intersect the view ray with the sphere through the interpolated height so the
    // atmosphere path and the shading point do not depend on the tessellation.
    float2 hs = sphere2(ro, rd, RG() + i.height * 0.001);
    if (hs.x <= hs.y && hs.x > 0.0) dist = hs.x;
    float3 P = ro + rd * dist;
    float3 up = normalize(P);
    g_dx = ddx(up);
    g_dy = ddy(up);
    g_fp = max(length(g_dx), length(g_dy)) * RG();
    FSOut o;
    int mode = int(p.fwd.w);
    if (mode > 0) { o.color = float4(debug_color(up, mode), 0.0); return o; }
    float3 Ls = mode == -1 ? float3(0, 0, 0) : surface_radiance(P, rd, up, i.height);
    // cloud layer between camera and ground: first crossing of the cloud sphere
    float rc = p.planet.z;
    float2 cq = sphere2(ro, rd, rc);
    float3 T = float3(1, 1, 1), L = float3(0, 0, 0);
    bool through = !cloud_volume() && length(ro) > rc && cq.x <= cq.y && cq.x > 0.0 && cq.x < dist;
    float3 Lc = float3(0, 0, 0);
    float Tc = 1.0, tc = -1.0;
    if (cloud_volume() && mode != -1) {
        // the view ray's part inside the cloud shell, before the ground / the shell base
        float2 hi = sphere2(ro, rd, RG() + CV_HI), lo = sphere2(ro, rd, RG() + CV_LO);
        float ta = max(hi.x, 0.0);
        float tb = min(dist, (lo.x <= lo.y && lo.x > 0.0) ? lo.x : dist);
        cloud_march(ro, rd, ta, tb, int(p.cloud2.x), ign(i.position.xy), Lc, Tc, tc);
    }
    if (tc > 0.0) {
        atmosphere(ro, rd, 0.0, tc, 16, T, L);
        float3 T2 = float3(1, 1, 1), L2 = float3(0, 0, 0);
        atmosphere(ro, rd, tc, dist, 8, T2, L2);
        L += T * (Lc + Tc * (L2 + T2 * Ls));
    } else if (through) {
        float3 Q = ro + rd * cq.x;
        float3 uq = normalize(Q);
        float c = cloud_at(uq);
        float a = cloud_alpha(c, dot(rd, uq));
        atmosphere(ro, rd, 0.0, cq.x, 16, T, L);
        float3 Lc = cloud_radiance(uq, rd, c, a, cloud_normal(uq, c));
        float3 T2 = float3(1, 1, 1), L2 = float3(0, 0, 0);
        atmosphere(ro, rd, cq.x, dist, 8, T2, L2);
        L += T * (a * Lc + (1.0 - a) * (L2 + T2 * Ls));
    } else {
        atmosphere(ro, rd, 0.0, dist, 16, T, L);
        L += T * Ls;
    }
    o.color = float4(min(L * p.view.z, 6.0e4), 0.0);
    return o;
}

// ------------------------------------------------------------------ sky / limb
struct SkyOut { float4 position : SV_Position; float2 ndc : NDC; };

[shader("vertex")]
SkyOut skyVS(uint vid : SV_VulkanVertexID) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    SkyOut o;
    o.ndc = uv * 2.0 - 1.0;
    o.position = float4(o.ndc, 0.0, 1.0);   // reverse-Z: depth 0 = infinitely far
    return o;
}

[shader("fragment")]
FSOut skyFS(SkyOut i) {
    PParams* p = pc.p;
    float3 rd = normalize(p.fwd.xyz + p.right.xyz * (i.ndc.x * p.right.w) - p.up.xyz * (i.ndc.y * p.up.w));
    float3 ro = p.cam_km.xyz;
    // far cameras: rebase to 3000 km before the ray's closest approach (the 400 km airglow shell's
    // half chord is < 2300 km), as in planetFS, so the shell intersections stay precise in f32
    float t_off = max(-dot(ro, rd) - 3000.0, 0.0);
    ro += rd * t_off;
    float3 sun = p.sun.xyz;
    float3 T = float3(1, 1, 1), L = float3(0, 0, 0);
    float2 g = sphere2(ro, rd, RG());
    bool ground = g.x <= g.y && g.y > 0.0;
    float tend = ground ? max(g.x, 0.0) : 1e9;
    float rc = p.planet.z;
    float2 cq = sphere2(ro, rd, rc);
    float3 Lc = float3(0, 0, 0);
    float Tc = 1.0, tc = -1.0;
    if (cloud_volume()) {
        // limb: the ray's chord through the cloud shell (it misses the ground), capped steps
        float2 hi = sphere2(ro, rd, RG() + CV_HI);
        if (hi.x <= hi.y && hi.y > 0.0) {
            float ta = max(hi.x, 0.0), tb = min(hi.y, tend);
            int n = clamp(int((tb - ta) / 3.0), 16, 96);
            cloud_march(ro, rd, ta, tb, n, ign(i.position.xy), Lc, Tc, tc);
        }
    }
    if (tc > 0.0) {
        atmosphere(ro, rd, 0.0, tc, 24, T, L);
        float3 T2 = float3(1, 1, 1), L2 = float3(0, 0, 0);
        atmosphere(ro, rd, tc, tend, 24, T2, L2);
        L += T * (Lc + Tc * L2);
        T *= Tc * T2;
    } else if (!cloud_volume() && cq.x <= cq.y && cq.y > 0.0 && cq.x > 0.0 && cq.x < tend && length(ro) > rc) {
        // grazing rays through the cloud layer (limb clouds): composite at the entry crossing
        float3 uq = normalize(ro + rd * cq.x);
        float c = cloud_at(uq);
        float a = cloud_alpha(c, dot(rd, uq));
        atmosphere(ro, rd, 0.0, cq.x, 24, T, L);
        float3 Lc = cloud_radiance(uq, rd, c, a, uq);
        float3 T2 = float3(1, 1, 1), L2 = float3(0, 0, 0);
        atmosphere(ro, rd, cq.x, tend, 24, T2, L2);
        L += T * (a * Lc + (1.0 - a) * L2);
        T *= (1.0 - a) * T2;
    } else {
        atmosphere(ro, rd, 0.0, tend, 24, T, L);
    }
    // sun disc (limb darkened), only where the ray leaves the planet's shadow
    float cs = dot(rd, sun);
    float cr = cos(p.sun.w);
    if (!ground && cs > cr) {
        float x = saturate((1.0 - cs) / (1.0 - cr));
        float ld = 1.0 - 0.6 * (1.0 - sqrt(1.0 - x));
        L += T * ld / (PI * p.sun.w * p.sun.w);
    }
    FSOut o;
    // alpha = transmittance toward space (the CPU adds stars there); 0 behind the planet
    o.color = float4(min(L * p.view.z, 6.0e4), ground ? 0.0 : dot(T, float3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)));
    return o;
}
