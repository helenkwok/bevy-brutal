//! Mesh a `geo-fetch terrain-grid` height grid into LOD'd, skirted chunks and write CPU previews.
//!
//! cargo run --release -p terrain --example dem_preview -- \
//!     [--grid /workspace/data/geo/terrain/grids/adelaide_z12] [--out out/quality/dem] [--name adelaide_z12] \
//!     [--tau 2.0] [--size 1280x720] [--fov 50] [--eye E,N,H] [--target E,N,H] [--no-reference]
//!
//! `--eye/--target` are local ENU metres (east, north, height) from the grid origin. Writes:
//! * `<name>_view.png`      perspective render of the selected LOD set (skirts drawn red if visible)
//! * `<name>_view_lod.png`  same view, tinted by chunk LOD
//! * `<name>_reference.png` every chunk at LOD0 (full resolution), and `<name>_diff.png` (x4)
//! * `<name>_lod_map.png`   top-down hillshade with chunk borders/LOD tint and the eye
//! * `<name>.json`          build/mesh timings, per-LOD errors, counts, image diff stats
//!
//! A pure CPU rasterizer (z-buffer, back-face culling like the GPU surface pass), so it runs
//! anywhere; heights are coloured in real metres. The GPU counterpart is render-ash's
//! `dem_terrain` example (terrain slots + `SurfacePalette::HeightRange`).
use std::path::{Path, PathBuf};
use std::time::Instant;

use glam::{Mat4, Vec3, Vec4};
use terrain::dem::{DEM_CHUNK_QUADS, DEM_SURFACE_INDICES, DemChunkId, DemGrid, DemTerrain, LodView, dem_chunk_indices};
use terrain::{CHUNK_SAMPLES, TerrainVertex};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn vec3_arg(args: &[String], name: &str, default: [f32; 3]) -> [f32; 3] {
    arg(args, name).map_or(default, |s| {
        let v: Vec<f32> = s.split(',').map(|x| x.trim().parse().expect("number")).collect();
        [v[0], v[1], v[2]]
    })
}

/// ENU (east, north, up) -> mesh space (x = east, y = up, z = -north).
fn enu(v: [f32; 3]) -> Vec3 {
    Vec3::new(v[0], v[2], -v[1])
}

const LOD_TINT: [[f32; 3]; 8] = [
    [0.95, 0.35, 0.30],
    [0.95, 0.70, 0.25],
    [0.85, 0.90, 0.30],
    [0.35, 0.85, 0.40],
    [0.30, 0.75, 0.90],
    [0.45, 0.45, 0.95],
    [0.80, 0.45, 0.90],
    [0.90, 0.90, 0.90],
];

fn hypsometric(h: f32) -> [f32; 3] {
    let stops: [(f32, [f32; 3]); 6] = [
        (0.6, [0.16, 0.30, 0.42]),
        (0.61, [0.74, 0.71, 0.58]),
        (40.0, [0.55, 0.60, 0.40]),
        (200.0, [0.36, 0.48, 0.28]),
        (450.0, [0.46, 0.42, 0.30]),
        (650.0, [0.62, 0.58, 0.52]),
    ];
    if h <= stops[0].0 {
        return stops[0].1;
    }
    for w in stops.windows(2) {
        if h <= w[1].0 {
            let t = (h - w[0].0) / (w[1].0 - w[0].0);
            return [0, 1, 2].map(|k| w[0].1[k] + (w[1].1[k] - w[0].1[k]) * t);
        }
    }
    stops[5].1
}

struct Raster {
    w: usize,
    h: usize,
    color: Vec<[f32; 3]>,
    depth: Vec<f32>,
    /// 0 = sky, 1 = surface, 2 = interior skirt, 3 = outer-border skirt
    kind: Vec<u8>,
}

impl Raster {
    fn new(w: usize, h: usize) -> Self {
        let sky = (0..w * h)
            .map(|k| {
                let t = (k / w) as f32 / h as f32;
                [0.55 + 0.25 * t, 0.68 + 0.17 * t, 0.86 + 0.06 * t]
            })
            .collect();
        Self { w, h, color: sky, depth: vec![f32::INFINITY; w * h], kind: vec![0; w * h] }
    }

    /// Draw one triangle (positions already in clip space). Back faces and anything crossing the
    /// near plane are skipped (the preview cameras sit well above the terrain).
    fn tri(&mut self, clip: [Vec4; 3], col: [[f32; 3]; 3], kind: u8) {
        if clip.iter().any(|c| c.w < 1e-3) {
            return;
        }
        let s = clip.map(|c| {
            let n = c / c.w;
            Vec3::new((n.x * 0.5 + 0.5) * self.w as f32, (0.5 - n.y * 0.5) * self.h as f32, n.z)
        });
        let area = (s[1].x - s[0].x) * (s[2].y - s[0].y) - (s[2].x - s[0].x) * (s[1].y - s[0].y);
        // y is flipped to image rows, so CCW-from-outside becomes negative area here
        if area >= 0.0 {
            return;
        }
        let (x0, x1) = (s.iter().map(|p| p.x).fold(f32::MAX, f32::min), s.iter().map(|p| p.x).fold(f32::MIN, f32::max));
        let (y0, y1) = (s.iter().map(|p| p.y).fold(f32::MAX, f32::min), s.iter().map(|p| p.y).fold(f32::MIN, f32::max));
        let (x0, x1) = ((x0.floor().max(0.0)) as usize, (x1.ceil().min(self.w as f32 - 1.0)).max(0.0) as usize);
        let (y0, y1) = ((y0.floor().max(0.0)) as usize, (y1.ceil().min(self.h as f32 - 1.0)).max(0.0) as usize);
        let inv = 1.0 / area;
        for py in y0..=y1 {
            for px in x0..=x1 {
                let (fx, fy) = (px as f32 + 0.5, py as f32 + 0.5);
                let e = |a: Vec3, b: Vec3| (b.x - a.x) * (fy - a.y) - (fx - a.x) * (b.y - a.y);
                let (b0, b1, b2) = (e(s[1], s[2]) * inv, e(s[2], s[0]) * inv, e(s[0], s[1]) * inv);
                if b0 < 0.0 || b1 < 0.0 || b2 < 0.0 {
                    continue;
                }
                let z = b0 * s[0].z + b1 * s[1].z + b2 * s[2].z;
                let k = py * self.w + px;
                if z < self.depth[k] {
                    self.depth[k] = z;
                    self.color[k] = [0, 1, 2].map(|c| b0 * col[0][c] + b1 * col[1][c] + b2 * col[2][c]);
                    self.kind[k] = kind;
                }
            }
        }
    }

    fn rgba(&self) -> Vec<u8> {
        self.color
            .iter()
            .flat_map(|c| {
                // colours are already display-referred (sRGB-ish), no transfer curve
                let g = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                [g(c[0]), g(c[1]), g(c[2]), 255]
            })
            .collect()
    }
}

fn write_png(path: &Path, w: usize, h: usize, rgba: &[u8]) {
    let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(path).unwrap()), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(rgba).unwrap();
    eprintln!("wrote {}", path.display());
}

struct Scene<'a> {
    t: &'a DemTerrain,
    indices: Vec<u32>,
    view_proj: Mat4,
    eye: Vec3,
    sun: Vec3,
}

impl Scene<'_> {
    /// Rasterize `chunks`; returns (raster, triangles submitted, mesh ms).
    fn render(&self, chunks: &[DemChunkId], w: usize, h: usize, tint_lod: bool) -> (Raster, usize, f64) {
        let g = self.t.grid();
        let mut r = Raster::new(w, h);
        let (mut tris, mut mesh_ms) = (0, 0.0);
        for &id in chunks {
            let m0 = Instant::now();
            let v = self.t.mesh_chunk(id);
            mesh_ms += m0.elapsed().as_secs_f64() * 1e3;
            let (c0, r0) = id.first_sample();
            let span = DEM_CHUNK_QUADS * id.stride();
            // skirt rings: 0 south, 1 east, 2 north, 3 west
            let border = [r0 == 0, c0 + span >= g.width - 1, r0 + span >= g.height - 1, c0 == 0];
            let shade = |tv: &TerrainVertex, skirt: bool| -> [f32; 3] {
                let n = Vec3::from(tv.normal);
                let p = Vec3::from(tv.pos);
                let base = if skirt {
                    [0.85, 0.12, 0.10]
                } else if tint_lod {
                    let a = hypsometric(tv.pos[1]);
                    let t = LOD_TINT[id.lod.min(7) as usize];
                    [0, 1, 2].map(|k| a[k] * 0.45 + t[k] * 0.55)
                } else {
                    hypsometric(tv.pos[1])
                };
                let lambert = n.dot(self.sun).max(0.0);
                let sky = 0.5 + 0.5 * n.y;
                let lit = [0, 1, 2].map(|k| base[k] * (0.25 * sky + 0.85 * lambert));
                // aerial perspective toward the horizon colour
                let fog = 1.0 - (-(p - self.eye).length() / 30_000.0).exp();
                let haze = [0.62, 0.72, 0.86];
                [0, 1, 2].map(|k| lit[k] * (1.0 - fog) + haze[k] * fog)
            };
            let clip: Vec<Vec4> =
                v.iter().map(|tv| self.view_proj * Vec4::new(tv.pos[0], tv.pos[1], tv.pos[2], 1.0)).collect();
            let skirt_start = CHUNK_SAMPLES * CHUNK_SAMPLES;
            for (k, t) in self.indices.chunks(3).enumerate() {
                let skirt = k * 3 >= DEM_SURFACE_INDICES;
                let kind = if skirt {
                    let ring = (t.iter().map(|&i| i as usize).max().unwrap() - skirt_start) / CHUNK_SAMPLES;
                    if border[ring] { 3 } else { 2 }
                } else {
                    1
                };
                let idx = [t[0] as usize, t[1] as usize, t[2] as usize];
                r.tri(idx.map(|i| clip[i]), idx.map(|i| shade(&v[i], skirt)), kind);
                tris += 1;
            }
        }
        (r, tris, mesh_ms)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grid_dir =
        PathBuf::from(arg(&args, "--grid").unwrap_or_else(|| "/workspace/data/geo/terrain/grids/adelaide_z12".into()));
    let out = PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "out/quality/dem".into()));
    let name = arg(&args, "--name")
        .unwrap_or_else(|| grid_dir.file_name().map_or("dem".into(), |s| s.to_string_lossy().into_owned()));
    let tau: f32 = arg(&args, "--tau").map_or(2.0, |s| s.parse().unwrap());
    let fov_deg: f32 = arg(&args, "--fov").map_or(50.0, |s| s.parse().unwrap());
    let (w, h) = arg(&args, "--size")
        .map(|s| {
            let (a, b) = s.split_once('x').unwrap();
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .unwrap_or((1280usize, 720usize));
    // default: above the south-western suburbs, looking east-north-east at the Adelaide Hills face
    let eye_enu = vec3_arg(&args, "--eye", [-3000.0, -9000.0, 900.0]);
    let target_enu = vec3_arg(&args, "--target", [8000.0, -4000.0, 250.0]);
    let reference = !args.iter().any(|a| a == "--no-reference");
    std::fs::create_dir_all(&out).unwrap();

    let t_load = Instant::now();
    let hg = geo_fetch::grid::read_grid(&grid_dir).unwrap_or_else(|e| panic!("{}: {e}", grid_dir.display()));
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    let (gw, gh, spacing) = (hg.width, hg.height, hg.spacing_m);
    let grid = DemGrid::new(gw, gh, spacing as f32, hg.x0 as f32, hg.y0 as f32, hg.heights).unwrap();
    let (hmin, hmax) = grid.height_range();
    let t_build = Instant::now();
    let t = DemTerrain::new(grid);
    let build_ms = t_build.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "{name}: {gw}x{gh} @ {spacing} m, h {hmin:.1}..{hmax:.1} m | root LOD {} | load {load_ms:.0} ms, build {build_ms:.0} ms",
        t.root_lod()
    );

    let mut levels = Vec::new();
    for l in 0..=t.root_lod() {
        let (cx, cy) = t.chunks_at(l);
        let ids = (0..cy).flat_map(|y| (0..cx).map(move |x| DemChunkId { lod: l, x, y }));
        let errs: Vec<f32> = ids.map(|id| t.geometric_error(id)).collect();
        let max = errs.iter().copied().fold(0.0, f32::max);
        let mean = errs.iter().sum::<f32>() / errs.len() as f32;
        levels.push(format!(
            "{{\"lod\":{l},\"spacing_m\":{},\"chunks\":[{cx},{cy}],\"error_max_m\":{max:.2},\"error_mean_m\":{mean:.2}}}",
            spacing * (1u32 << l) as f64
        ));
    }

    let eye = enu(eye_enu);
    let fov = fov_deg.to_radians();
    let view = LodView { eye: eye.to_array(), fov_y: fov, viewport_height_px: h as f32, max_error_px: tau };
    let t_sel = Instant::now();
    let selected = t.select(&view);
    let select_ms = t_sel.elapsed().as_secs_f64() * 1e3;
    let mut per_lod = vec![0usize; t.root_lod() as usize + 1];
    for id in &selected {
        per_lod[id.lod as usize] += 1;
    }
    eprintln!("selected {} chunks (tau {tau} px), per LOD {per_lod:?}, select {select_ms:.2} ms", selected.len());

    // sun from the north-west, 35° up (southern-hemisphere afternoon)
    let sun = (Vec3::new(-0.55, 0.0, -0.8).normalize() * 35f32.to_radians().cos() + Vec3::Y * 35f32.to_radians().sin())
        .normalize();
    let view_proj = glam::camera::rh::proj::opengl::perspective(fov, w as f32 / h as f32, 5.0, 100_000.0)
        * glam::camera::rh::view::look_at_mat4(eye, enu(target_enu), Vec3::Y);
    let scene = Scene { t: &t, indices: dem_chunk_indices(), view_proj, eye, sun };

    let t_r = Instant::now();
    let (img, tris, mesh_ms) = scene.render(&selected, w, h, false);
    let render_ms = t_r.elapsed().as_secs_f64() * 1e3;
    write_png(&out.join(format!("{name}_view.png")), w, h, &img.rgba());
    let count = |r: &Raster, k: u8| r.kind.iter().filter(|&&x| x == k).count();
    let (skirt_inner_px, skirt_border_px) = (count(&img, 2), count(&img, 3));
    let (lod_img, _, _) = scene.render(&selected, w, h, true);
    write_png(&out.join(format!("{name}_view_lod.png")), w, h, &lod_img.rgba());

    let mut ref_json = String::from("null");
    if reference {
        let (cx, cy) = t.chunks_at(0);
        let all0: Vec<DemChunkId> = (0..cy).flat_map(|y| (0..cx).map(move |x| DemChunkId { lod: 0, x, y })).collect();
        let (rimg, rtris, _) = scene.render(&all0, w, h, false);
        write_png(&out.join(format!("{name}_reference.png")), w, h, &rimg.rgba());
        let (a, b) = (img.rgba(), rimg.rgba());
        let mut diff = vec![0u8; a.len()];
        let (mut sum, mut n, mut over8) = (0.0f64, 0usize, 0usize);
        for p in 0..w * h {
            let d = (0..3).map(|c| (a[p * 4 + c] as i32 - b[p * 4 + c] as i32).unsigned_abs()).max().unwrap();
            if img.kind[p] != 0 || rimg.kind[p] != 0 {
                sum += d as f64;
                n += 1;
                over8 += usize::from(d > 8);
            }
            let v = (d * 4).min(255) as u8;
            diff[p * 4..p * 4 + 4].copy_from_slice(&[v, v, v, 255]);
        }
        write_png(&out.join(format!("{name}_diff.png")), w, h, &diff);
        ref_json = format!(
            "{{\"chunks\":{},\"triangles\":{rtris},\"mean_abs_diff_8bit\":{:.3},\"px_diff_gt8_frac\":{:.5}}}",
            all0.len(),
            sum / n.max(1) as f64,
            over8 as f64 / n.max(1) as f64
        );
        eprintln!("reference: {} LOD0 chunks, {rtris} tris | {ref_json}", all0.len());
    }

    lod_map(&t, &selected, eye, &out.join(format!("{name}_lod_map.png")));

    let json = format!(
        "{{\"name\":\"{name}\",\"grid\":{{\"dir\":\"{}\",\"cols\":{gw},\"rows\":{gh},\"spacing_m\":{spacing},\"height_min_m\":{hmin:.2},\"height_max_m\":{hmax:.2}}},\
\"load_ms\":{load_ms:.1},\"build_ms\":{build_ms:.1},\"root_lod\":{},\"levels\":[{}],\
\"view\":{{\"eye_enu\":{eye_enu:?},\"target_enu\":{target_enu:?},\"fov_deg\":{fov_deg},\"size\":[{w},{h}],\"tau_px\":{tau}}},\
\"selection\":{{\"chunks\":{},\"per_lod\":{per_lod:?},\"select_ms\":{select_ms:.3},\"vertices\":{},\"triangles\":{tris},\"mesh_ms_total\":{mesh_ms:.2},\"mesh_ms_per_chunk\":{:.3}}},\
\"full_res_triangles\":{},\"cpu_render_ms\":{render_ms:.0},\"skirt_px_interior\":{skirt_inner_px},\"skirt_px_border\":{skirt_border_px},\"reference\":{ref_json}}}",
        grid_dir.display(),
        t.root_lod(),
        levels.join(","),
        selected.len(),
        selected.len() * terrain::dem::DEM_CHUNK_VERTS,
        mesh_ms / selected.len() as f64,
        (gw - 1) * (gh - 1) * 2,
    );
    std::fs::write(out.join(format!("{name}.json")), &json).unwrap();
    println!("{json}");
}

/// Top-down map: hillshade, LOD tint per selected chunk, chunk borders, eye marker. North up.
fn lod_map(t: &DemTerrain, selected: &[DemChunkId], eye: Vec3, path: &Path) {
    let g = t.grid();
    let scale = (1024.0 / g.width.max(g.height) as f32).min(1.0);
    let (w, h) = (((g.width as f32) * scale) as usize, ((g.height as f32) * scale) as usize);
    let mut img = vec![0u8; w * h * 4];
    let mut lod_of = vec![0u8; (g.width - 1) * (g.height - 1)];
    for id in selected {
        let (c0, r0) = id.first_sample();
        let span = DEM_CHUNK_QUADS * id.stride();
        for r in r0..(r0 + span).min(g.height - 1) {
            for c in c0..(c0 + span).min(g.width - 1) {
                lod_of[r * (g.width - 1) + c] = id.lod;
            }
        }
    }
    for py in 0..h {
        for px in 0..w {
            let c = ((px as f32 / scale) as usize).min(g.width - 2);
            let r = (((h - 1 - py) as f32 / scale) as usize).min(g.height - 2);
            let dx = (g.get(c + 1, r) - g.get(c, r)) / g.spacing_m;
            let dy = (g.get(c, r + 1) - g.get(c, r)) / g.spacing_m;
            let n = Vec3::new(-dx, 1.0, dy).normalize();
            let shade = n.dot(Vec3::new(-0.5, 0.7, -0.5).normalize()).max(0.0);
            let tint = LOD_TINT[lod_of[r * (g.width - 1) + c].min(7) as usize];
            let base = if g.get(c, r) <= 0.6 { [0.2, 0.3, 0.45] } else { [0.8, 0.8, 0.8] };
            let o = (py * w + px) * 4;
            for k in 0..3 {
                img[o + k] = ((base[k] * (0.25 + 0.75 * shade) * 0.55 + tint[k] * 0.45).min(1.0) * 255.0) as u8;
            }
            img[o + 3] = 255;
        }
    }
    let mut put = |c: usize, r: usize, col: [u8; 3]| {
        let px = ((c as f32 * scale) as usize).min(w - 1);
        let py = h - 1 - ((r as f32 * scale) as usize).min(h - 1);
        img[(py * w + px) * 4..(py * w + px) * 4 + 3].copy_from_slice(&col);
    };
    for id in selected {
        let (c0, r0) = id.first_sample();
        let span = DEM_CHUNK_QUADS * id.stride();
        let (c1, r1) = ((c0 + span).min(g.width - 1), (r0 + span).min(g.height - 1));
        for c in c0..=c1 {
            put(c, r0, [20, 20, 20]);
            put(c, r1, [20, 20, 20]);
        }
        for r in r0..=r1 {
            put(c0, r, [20, 20, 20]);
            put(c1, r, [20, 20, 20]);
        }
    }
    let (ec, er) = ((eye.x - g.x0_m) / g.spacing_m, (-eye.z - g.y0_m) / g.spacing_m);
    for d in -6i32..=6 {
        for (a, b) in [(d, 0), (0, d)] {
            let (c, r) = (ec + a as f32 / scale, er + b as f32 / scale);
            if c >= 0.0 && r >= 0.0 && (c as usize) < g.width && (r as usize) < g.height {
                put(c as usize, r as usize, [255, 255, 255]);
            }
        }
    }
    write_png(path, w, h, &img);
}
