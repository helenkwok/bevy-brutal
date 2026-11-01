//! `cargo run --release -p planet --example earth_bake -- [data dir] [out]`
//! Bakes the sources fetched by `tools/fetch_earth_data.sh`:
//! - `out` a directory (default `<data dir>/ktx2`): KTX2 cube maps, BCn + zstd (the renderer's
//!   default `EARTH_TEX=ktx2`). `BAKE_RAW=1` also writes `<out>_raw` (uncompressed RGBA8 / RG8 /
//!   R8 + zstd: the quality reference, `EARTH_TEX=raw`); `BAKE_NOSC=1` also writes `<out>_nosc`
//!   (BCn without supercompression, for the disk / load-time comparison). `BAKE_ZSTD` level (19).
//! - `out` ending in `.pln`: the round-1 `PLN1` pack (`EARTH_TEX=pln`).
//!
//! Env: `BAKE_ALBEDO`, `BAKE_SLOPE`, `BAKE_NIGHT`, `BAKE_AUX`, `BAKE_CLOUDS`, `BAKE_HEIGHT` (face
//! sizes), `BAKE_SS`, `BAKE_BC7` (very_fast | fast | basic).
use planet::bake::{BakeConfig, Ktx2Set, Sources, bake, bake_ktx2};
use planet::ktx2::Supercompression;
use std::path::PathBuf;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(a.get(1).cloned().unwrap_or("/workspace/data/earth".into()));
    let out = a.get(2).map(PathBuf::from).unwrap_or(dir.join("ktx2"));
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let d = BakeConfig::default();
    let cfg = BakeConfig {
        albedo: env("BAKE_ALBEDO", d.albedo),
        slope: env("BAKE_SLOPE", d.slope),
        night: env("BAKE_NIGHT", d.night),
        aux: env("BAKE_AUX", d.aux),
        clouds: env("BAKE_CLOUDS", d.clouds),
        height: env("BAKE_HEIGHT", d.height),
        ss: env("BAKE_SS", d.ss),
    };
    let t = std::time::Instant::now();
    let src = Sources::load(&dir);
    println!("sources loaded in {:.1} s", t.elapsed().as_secs_f64());
    if out.extension().is_some_and(|e| e == "pln") {
        let pack = bake(&src, &cfg, &mut |s| println!("{s}"));
        drop(src);
        let bytes = pack.to_bytes();
        std::fs::write(&out, &bytes).expect("write pack");
        let rt = planet::pack::PlanetPack::from_bytes(&bytes).expect("verify pack");
        println!(
            "wrote {} ({} MB, GPU textures {} MB, {} textures), verified, total {:.1} s",
            out.display(),
            bytes.len() >> 20,
            rt.gpu_bytes() >> 20,
            rt.textures.len(),
            t.elapsed().as_secs_f64()
        );
        return;
    }
    let z = Supercompression::Zstd(env("BAKE_ZSTD", 19) as i32);
    let mut sets = vec![Ktx2Set { dir: out.clone(), raw: false, sc: z }];
    let sib = |suf: &str| out.with_file_name(format!("{}_{suf}", out.file_name().unwrap().to_string_lossy()));
    if env("BAKE_NOSC", 0) == 1 {
        sets.push(Ktx2Set { dir: sib("nosc"), raw: false, sc: Supercompression::None });
    }
    if env("BAKE_RAW", 0) == 1 {
        sets.push(Ktx2Set { dir: sib("raw"), raw: true, sc: z });
    }
    let rep = bake_ktx2(&src, &cfg, &sets, &mut |s| println!("{s}")).expect("write KTX2");
    for (i, s) in sets.iter().enumerate() {
        let (f, g) = rep.iter().filter(|r| r.1 == i).fold((0, 0), |a, r| (a.0 + r.2, a.1 + r.3));
        println!(
            "set {}: files {:.1} MB, GPU (incl. CPU height) {:.1} MB",
            s.dir.display(),
            f as f64 / 1048576.0,
            g as f64 / 1048576.0
        );
    }
    println!("total {:.1} s", t.elapsed().as_secs_f64());
}
