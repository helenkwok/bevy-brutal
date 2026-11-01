// Solar system (round 1): every body of the system frame in one full-screen pass.
//
// The CPU forms each body's centre minus the camera in f64 (system frame, ~1 mm) and hands the GPU
// f32 camera-relative centres + the ecliptic -> body-fixed rotation. Per pixel: the ray meets each
// body's ellipsoid in the body's own frame (scaled to a unit-ish sphere, closest-approach form so
// the f32 cancellation error is ~|centre| x 6e-8, << a pixel at any distance a body covers one),
// the nearest hit is shaded with that body's scattering model, Saturn's rings are composited in
// front / behind, and the Sun's photosphere / corona fill what is left.
//
// Units: radiance in solar illuminance at 1 AU per steradian x exposure (as planet.slang), so a
// white Lambert plate facing the Sun at 1 AU has radiance 1/pi.
//
// Surfaces (shade.w = kind):
//   0 star      photosphere, limb darkening I(mu) = I0 mu^alpha (alpha per channel), granulation map
//   1 regolith  lunar-Lambert (McEwen 1991): A [(1 - L) mu0 + 2 L mu0 / (mu0 + mu)]
//   2 dusty     regolith + a thin dust-haze limb (Mars)
//   3 cloud     Minnaert (k) + a forward-scattering haze shell (Venus)
//   4 giant     Minnaert (k) + a thin haze shell
//   5 earth     far LOD: Lambert land / ocean (Blue Marble July) under the VIIRS cloud map, a Rayleigh
//               veil and shell (the planet renderer takes over above ~24 px)
// Shadows: every other body's disc against the Sun's disc (exact circle-overlap penumbra), the rings
// on Saturn, Saturn on its rings.
struct SBody {
    float4 pos;     // xyz centre - camera (m, ecliptic J2000 axes), w equatorial radius (m)
    float4 r0;      // rows of the ecliptic -> body-fixed rotation; r0.w polar / equatorial radius
    float4 r1;      // r1.w texture layer (-1 = flat tint)
    float4 r2;      // r2.w cloud layer (-1 = none)
    float4 shade;   // x albedo scale (on the linear texture), y Minnaert k / lunar-Lambert L, z haze strength, w kind
    float4 tint;    // rgb colour multiplier, w haze scale height (m)
    float4 ring;    // x inner radius (m), y outer radius (m), z ring albedo, w ring layer (-1 none)
    float4 extra;   // (z: haze colour mix toward the local albedo colour on the disc; the limb shell is white)
                    // x 1 = skip (sub-pixel: the CPU splats its flux), 2 = occluder only (drawn by the planet
                    // renderer: black here, alpha -1 where a nearer body hides it), y haze HG g, z haze colour mix,
                    // w lunar-Lambert phase decay (rad)
};
struct SParams {
    float4 right;   // xyz, w tan(fov_x / 2)
    float4 up;      // xyz, w tan(fov_y / 2)
    float4 fwd;     // xyz, w exposure
    float4 sun;     // xyz Sun centre - camera (m), w Sun radius (m)
    float4 view;    // x bodies, y flags (1 corona, 2 fixed light (line-up), 4 no shadows), z AU (m), w unused
    float4 corona;  // x occulting disc radius (rad, 0 = none), y corona gain, z streamer contrast, w F-corona gain
    float4 sunpole; // xyz Sun north pole (ecliptic), w granulation contrast
    float4 light;   // xyz fixed light direction (flag 2), w fixed irradiance
    float4 limb;    // xyz photosphere limb-darkening exponents (r, g, b), w pixel angle (rad)
    SBody bodies[16];
};
struct SPush { SParams* p; };
[[vk::push_constant]] ConstantBuffer<SPush> pc;

[[vk::binding(0, 0)]] Sampler2DArray tBody;   // 2048 x 1024 equirect layers, R8G8B8A8_SRGB (the sampler returns linear), lon 0 at u = 0.5
[[vk::binding(1, 0)]] Sampler2DArray tRing;   // radial ring profile, UNORM: rgb colour (sRGB-encoded, decoded here), a opacity

static const float PI = 3.14159265;

struct VOut { float4 position : SV_Position; float2 ndc : NDC; };

[shader("vertex")]
VOut fullVS(uint vid : SV_VulkanVertexID) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.ndc = uv * 2.0 - 1.0;
    o.position = float4(o.ndc, 0.0, 1.0);
    return o;
}

float3 srgb_to_linear(float3 c) { return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4)); }

float3x3 rot(SBody b) { return float3x3(b.r0.xyz, b.r1.xyz, b.r2.xyz); }

// Ray (origin o, unit d, both body-fixed, metres) vs the ellipsoid (a, a, a * q): (t, foot, disc)
// with the closest-approach foot f in scaled space (z / q) and disc = r^2 - |f|^2 (scaled, / |d'|^2).
struct Hit { float t; float3 p; float b; };  // t = world distance (inf = miss), p body-fixed point, b closest-approach / a
Hit ellipsoid(float3 o, float3 d, float a, float q) {
    float3 os = float3(o.xy, o.z / q) / a;
    float3 ds = float3(d.xy, d.z / q) / a;
    float dd = dot(ds, ds);
    float tca = -dot(os, ds) / dd;
    float3 f = os + ds * tca;            // closest approach (units of a)
    float ff = dot(f, f);
    Hit h;
    h.b = sqrt(ff);
    h.t = 1e30;
    h.p = float3(0.0);
    if (ff < 1.0 && tca > 0.0) {
        float s = sqrt((1.0 - ff) / dd);
        h.t = tca - s;
        float3 ps = f - ds * s;          // the near intersection (scaled)
        h.p = float3(ps.xy, ps.z * q) * a;
        if (h.t < 0.0) h.t = 1e30;       // inside / behind
    }
    return h;
}

// Area of the intersection of two discs (radii r1, r2, centre distance d), all angles.
float lens(float r1, float r2, float d) {
    if (d >= r1 + r2) return 0.0;
    if (d <= abs(r1 - r2)) return PI * min(r1, r2) * min(r1, r2);
    float a = r1 * r1 * acos(clamp((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1), -1.0, 1.0));
    float b = r2 * r2 * acos(clamp((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2), -1.0, 1.0));
    float c = 0.5 * sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)));
    return a + b - c;
}

// Fraction of the Sun's disc visible from world point w (camera-relative), skipping body `self`.
float sun_visible(SParams* p, float3 w, int self) {
    if ((uint(p.view.y) & 6u) != 0u) return 1.0;
    float3 ls = p.sun.xyz - w;
    float dsun = length(ls);
    float3 l = ls / dsun;
    float as_ = asin(min(p.sun.w / dsun, 1.0));
    float vis = 1.0;
    int n = int(p.view.x);
    for (int j = 1; j < n; j++) {
        if (j == self) continue;
        SBody b = p.bodies[j];
        float3 v = b.pos.xyz - w;
        float dv = length(v);
        if (dv > dsun || dot(v, l) <= 0.0) continue;
        float r = b.pos.w * (0.5 + 0.5 * b.r0.w);
        float ab = asin(min(r / dv, 1.0));
        float sep = acos(clamp(dot(v / dv, l), -1.0, 1.0));
        vis *= 1.0 - lens(as_, ab, sep) / (PI * as_ * as_);
    }
    return max(vis, 0.0);
}

float3 sun_dir(SParams* p, float3 w, out float E) {
    if ((uint(p.view.y) & 2u) != 0u) { E = p.light.w; return p.light.xyz; }
    float3 ls = p.sun.xyz - w;
    float d = length(ls);
    E = (p.view.z / d) * (p.view.z / d);
    return ls / d;
}

float hg(float g, float c) { return (1.0 - g * g) / (4.0 * PI * pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5)); }

// Ring opacity / colour at radius r (m); u along the profile.
float4 ring_sample(SBody b, float r) {
    float u = (r - b.ring.x) / (b.ring.y - b.ring.x);
    if (u <= 0.0 || u >= 1.0) return float4(0.0);
    float4 s = tRing.SampleLevel(float3(u, 0.5, b.ring.w), 0.0);
    return float4(srgb_to_linear(s.rgb), s.a);
}

// Shade a body hit. w = world point (camera-relative), pb body-fixed point, v toward the camera.
float3 shade_body(SParams* p, int i, SBody b, float3 pb, float3 w, float3 d, float t, float px_rad) {
    float3x3 R = rot(b);
    float q = b.r0.w;
    float3 nb = normalize(float3(pb.xy, pb.z / (q * q)));
    float3 n = mul(transpose(R), nb);
    float3 v = -d;
    float mu = max(dot(n, v), 0.0);
    int kind = int(b.shade.w);
    // texture
    float lon = atan2(pb.y, pb.x);
    float lat = asin(clamp(pb.z / length(pb), -1.0, 1.0));
    float2 uv = float2(0.5 + lon / (2.0 * PI), 0.5 - lat / PI);
    float foot = t * px_rad / max(mu, 0.05);                  // world footprint of a pixel (m)
    float lod = log2(max(foot * 2048.0 / (2.0 * PI * b.pos.w), 1e-6));
    // the Earth's tint is an output gain (SS2 calibration to the planet renderer), applied at the end
    float3 alb = kind == 5 ? float3(1.0) : b.tint.rgb;
    if (b.r1.w >= 0.0) alb *= tBody.SampleLevel(float3(uv, b.r1.w), lod).rgb;   // R8G8B8A8_SRGB: decoded by the sampler
    alb *= b.shade.x;
    if (kind == 0) {
        // photosphere: limb darkening, granulation (luminance only, low contrast)
        float3 I0 = (p.limb.xyz + 2.0) / (2.0 * PI * pow(p.sun.w / p.view.z, 2.0));
        float g = dot(tBody.SampleLevel(float3(uv, b.r1.w), lod).rgb, float3(0.2126, 0.7152, 0.0722));
        float gran = 1.0 + p.sunpole.w * (g / 0.35 - 1.0);
        return I0 * pow(float3(max(mu, 1e-3)), p.limb.xyz) * gran;
    }
    float E;
    float3 l = sun_dir(p, w, E);
    float mu0 = dot(n, l);
    float vis = sun_visible(p, w, i);
    // Saturn: the rings' shadow on the globe
    if (b.ring.y > 0.0 && (uint(p.view.y) & 6u) == 0u) {
        float3 lb = mul(R, l);
        float s = -pb.z / lb.z;
        if (s > 0.0) {
            float4 rs = ring_sample(b, length((pb + lb * s).xy));
            vis *= 1.0 - rs.a;
        }
    }
    float3 col = float3(0.0);
    float phase = acos(clamp(dot(l, v), -1.0, 1.0));
    if (mu0 > 0.0) {
        float k = b.shade.y;
        float f;
        if (kind == 1 || kind == 2) {
            float L = k * exp(-phase / max(b.extra.w, 1e-3));
            f = (1.0 - L) * mu0 + 2.0 * L * mu0 / (mu0 + max(mu, 1e-3));
        } else if (kind == 5) {
            f = mu0;
        } else {
            f = pow(mu0, k) * pow(max(mu, 1e-3), k - 1.0);
        }
        float3 a = alb;
        if (kind == 5 && b.r2.w >= 0.0) {
            float c = pow(tBody.SampleLevel(float3(uv, b.r2.w), lod).r, 1.0 / 2.2);   // coverage: back to the map's encoding
            c = clamp((c - 0.05) / 0.9, 0.0, 1.0);
            a = lerp(a, float3(0.62), c * 0.85);
            // Rayleigh veil: a blue airlight growing toward the limb
            float m = 1.0 / max(mu, 0.05) + 1.0 / max(mu0, 0.05);
            float3 tauR = float3(0.06, 0.12, 0.26);
            float3 T = exp(-tauR * m);
            col = E / PI * vis * (a * f * T + (1.0 - T) * mu0 / (mu0 + max(mu, 1e-3)) * 0.75 * float3(0.85, 0.95, 1.0));
        } else {
            col = E / PI * vis * a * f;
        }
    }
    // haze on the disc toward the limb (Venus / Mars / giants): forward-scattering veil
    if (b.shade.z > 0.0) {
        float lit = smoothstep(-0.25, 0.15, mu0);
        float path = min(1.0 / max(mu, 0.02), 30.0);
        float veil = b.shade.z * (1.0 - exp(-0.02 * path)) * hg(b.extra.y, dot(l, -v)) * 4.0 * PI;
        float3 hcol = lerp(float3(1.0), alb / max(dot(alb, float3(0.2126, 0.7152, 0.0722)), 1e-4), b.extra.z);
        col += E / PI * vis * lit * veil * hcol * 0.25;
    }
    return kind == 5 ? col * b.tint.rgb : col;
}

// Haze / atmosphere shell for rays that pass just outside a body's limb (b = closest approach / a).
float3 shell(SParams* p, SBody b, int i, float3 o, float3 d, float hb, float tca_world) {
    if (b.shade.z <= 0.0 || b.tint.w <= 0.0 || hb <= 1.0) return float3(0.0);
    float a = b.pos.w;
    float h = (hb - 1.0) * a;                          // ray's closest altitude (m)
    float H = b.tint.w;
    if (h > 12.0 * H || tca_world <= 0.0) return float3(0.0);
    // column through the shell ~ exp(-h / H) sqrt(2 pi a H), relative to the vertical column H
    float col = exp(-h / H) * sqrt(2.0 * PI * a / H);
    float3 w = d * tca_world;                          // the closest point (camera-relative)
    float E;
    float3 l = sun_dir(p, w, E);
    float3 nq = normalize(w - b.pos.xyz);
    float lit = smoothstep(-0.12, 0.08, dot(nq, l)) * sun_visible(p, w, i);
    float tau = 0.02 * b.shade.z;
    float3 hcol = float3(1.0);
    if (int(b.shade.w) == 5) hcol = float3(0.30, 0.55, 1.0);
    float ph = hg(b.extra.y, dot(l, d)) * 4.0 * PI;
    return E / PI * lit * (1.0 - exp(-tau * col)) * ph * hcol * 0.25;
}

float hash(float2 p) { return frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453); }
float vnoise(float2 x) {
    float2 i = floor(x), f = frac(x);
    f = f * f * (3.0 - 2.0 * f);
    return lerp(lerp(hash(i), hash(i + float2(1, 0)), f.x), lerp(hash(i + float2(0, 1)), hash(i + float2(1, 1)), f.x), f.y);
}

// K + F corona (Baumbach 1937, in units of the mean photospheric radiance), streamers along the
// solar equator modulated in position angle; rho = elongation / solar radius.
float3 corona(SParams* p, float3 d) {
    float3 sc = p.sun.xyz;
    float ds = length(sc);
    float3 s = sc / ds;
    float th = acos(clamp(dot(d, s), -1.0, 1.0));
    float rs = asin(min(p.sun.w / ds, 1.0));
    float rho = th / rs;
    if (rho <= 1.0) return float3(0.0);
    float K = 1e-6 * (0.0532 / pow(rho, 2.5) + 1.425 / pow(rho, 7.0) + 2.565 / pow(rho, 17.0));
    float F = p.corona.w * 1e-6 * 0.0532 / pow(rho, 2.5) * 0.4;
    // position angle about the Sun's pole (streamers at low heliolatitude near solar maximum:
    // 2026 is just past cycle 25's maximum, so streamers at all latitudes, stronger near the equator)
    float3 e1 = normalize(p.sunpole.xyz - s * dot(p.sunpole.xyz, s));
    float3 e2 = cross(s, e1);
    float3 r = d - s * dot(d, s);
    float pa = atan2(dot(r, e2), dot(r, e1));
    float lat = abs(cos(pa));                                       // ~ |sin heliolatitude| of the plane-of-sky point
    float st = 0.0;
    float amp = 1.0;
    float fr = 6.0;
    for (int k = 0; k < 4; k++) {
        st += amp * vnoise(float2(pa * fr / PI + 17.0 * k, log(rho) * 0.6 * (k + 1)));
        amp *= 0.55;
        fr *= 2.1;
    }
    float streamer = (1.0 - 0.75 * lat * lat) * (0.5 + st * 0.6);
    float Kmod = K * max(0.15, 1.0 + p.corona.z * (streamer - 0.6));
    float Imean = 1.0 / (PI * pow(p.sun.w / p.view.z, 2.0));       // mean photospheric radiance
    return Imean * p.corona.y * (Kmod * float3(1.0, 0.98, 0.95) + F * float3(1.0, 0.92, 0.80));
}

struct FSOut { float4 color : SV_Target0; };

[shader("fragment")]
FSOut systemFS(VOut i) {
    SParams* p = pc.p;
    float3 d = normalize(p.fwd.xyz + p.right.xyz * (i.ndc.x * p.right.w) - p.up.xyz * (i.ndc.y * p.up.w));
    float px_rad = p.limb.w;
    FSOut o;
    // occulting disc (coronagraph shot) in front of everything
    if (p.corona.x > 0.0 && dot(d, normalize(p.sun.xyz)) > cos(p.corona.x)) { o.color = float4(0.0); return o; }
    int n = int(p.view.x);
    float tbest = 1e30;
    int ib = -1;
    float3 pbest = float3(0.0);
    float3 add = float3(0.0);
    float t_occ = 1e30;                  // occluder (planet-renderer body) incl. its atmosphere shell
    for (int k = 0; k < n; k++) {
        SBody b = p.bodies[k];
        if (b.extra.x > 0.5 && b.extra.x < 1.5) continue;
        float3x3 R = rot(b);
        float3 o = mul(R, -b.pos.xyz);
        float3 db = mul(R, d);
        Hit h = ellipsoid(o, db, b.pos.w, b.r0.w);
        if (b.extra.x > 1.5) {
            Hit ha = ellipsoid(o, db, b.pos.w + 120e3, b.r0.w);
            t_occ = min(t_occ, ha.t);
        } else if (h.t >= 1e29) {
            add += shell(p, b, k, o, d, h.b, dot(b.pos.xyz, d));
        }
        if (h.t < tbest) { tbest = h.t; ib = k; pbest = h.p; }
    }
    float3 col = float3(0.0);
    float alpha = 1.0;
    if (ib >= 0) {
        SBody b = p.bodies[ib];
        alpha = 0.0;
        if (b.extra.x < 1.5) col = shade_body(p, ib, b, pbest, d * tbest, d, tbest, px_rad);
    } else {
        col = add;
        if ((uint(p.view.y) & 1u) != 0u) col += corona(p, d);
    }
    // rings, back to front is not needed (one ringed body): over the hit or over space
    for (int k = 0; k < n; k++) {
        SBody b = p.bodies[k];
        if (b.ring.y <= 0.0 || b.extra.x > 0.5) continue;
        float3x3 R = rot(b);
        float3 o = mul(R, -b.pos.xyz);
        float3 db = mul(R, d);
        if (abs(db.z) < 1e-9) continue;
        float t = -o.z / db.z;
        if (t <= 0.0 || t >= tbest) continue;
        float3 pr = o + db * t;
        float4 rs = ring_sample(b, length(pr.xy));
        if (rs.a <= 0.0) continue;
        float3 w = d * t;
        float E;
        float3 l = sun_dir(p, w, E);
        float3 lb = mul(R, l);
        float mu0 = max(abs(lb.z), 1e-4);
        float mu = max(abs(db.z), 1e-4);
        bool lit_side = (lb.z > 0.0) == (o.z > 0.0);
        // normal optical depth from the profile's opacity (taken as the normal-incidence opacity)
        float tau = -log(max(1.0 - rs.a * 0.98, 1e-3));
        float vis = 1.0;
        if ((uint(p.view.y) & 6u) == 0u) {
            Hit hs = ellipsoid(pr + lb * 1.0, lb, b.pos.w, b.r0.w);
            vis = hs.t < 1e29 ? 0.0 : sun_visible(p, w, k);
        }
        float phase = acos(clamp(dot(l, -d), -1.0, 1.0));
        // ring-particle phase function: strongly backscattering (opposition surge), forward lobe small
        float P = 1.6 * exp(-phase / 0.55) + 0.25;
        float om = b.ring.z;
        // single scattering in a slab (Chandrasekhar): reflection on the lit face, diffuse
        // transmission through to the unlit face
        float I;
        if (lit_side) I = om * P / 4.0 * mu0 / (mu0 + mu) * (1.0 - exp(-tau * (1.0 / mu + 1.0 / mu0)));
        else I = om * P / 4.0 * mu0 / max(abs(mu - mu0), 1e-3) * abs(exp(-tau / mu) - exp(-tau / mu0));
        float3 rc = rs.rgb / max(dot(rs.rgb, float3(0.2126, 0.7152, 0.0722)), 1e-3);
        // I is the slab's I/F (pi F = the solar irradiance E): radiance = E / pi * I
        float3 Lr = E / PI * vis * I * rc;
        float opa = 1.0 - exp(-tau / mu);
        col = Lr + col * (1.0 - opa);
        alpha *= 1.0 - opa;
    }
    // radiance x exposure; alpha = transmittance toward space (stars), -1 = a body hides the
    // occluder (the planet renderer's Earth): the composite keeps this pass alone there
    float a = alpha;
    if (t_occ < 1e29 && tbest < t_occ && ib >= 0 && p.bodies[ib].extra.x < 1.5) a = -1.0;
    o.color = float4(col * p.fwd.w, a);
    return o;
}
