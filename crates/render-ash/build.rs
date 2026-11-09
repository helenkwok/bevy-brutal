//! Compile the Slang shaders to SPIR-V 1.6 (Vulkan 1.3) in OUT_DIR with `slangc`.
//! Lookup: $SLANGC, then `slangc` on PATH. If $SLANGC is set it must run (build fails otherwise);
//! if it is unset and no `slangc` is on PATH, fall back to the checked-in `shaders/spv/*.spv`
//! (regenerate with `tools/compile_shaders.sh`).
use std::{env, fs, path::PathBuf, process::Command};

/// (source, entry point, stage, output name)
const SHADERS: &[(&str, &str, &str, &str)] = &[
    ("cull.slang", "main", "compute", "cull"),
    ("scan.slang", "scanBlocks", "compute", "scan_blocks"),
    ("scan.slang", "addOffsets", "compute", "scan_add"),
    ("scan.slang", "scanBlocksWave", "compute", "scan_blocks_wave"),
    ("scatter.slang", "main", "compute", "scatter"),
    ("compact.slang", "main", "compute", "compact"),
    ("mesh.slang", "vertexMain", "vertex", "mesh_vs"),
    ("mesh.slang", "fragmentMain", "fragment", "mesh_fs"),
    ("surface.slang", "vertexMain", "vertex", "surface_vs"),
    ("surface.slang", "fragmentMain", "fragment", "surface_fs"),
    ("voxel.slang", "countMain", "compute", "voxel_count"),
    ("voxel.slang", "emitMain", "compute", "voxel_emit"),
    ("tq.slang", "decodeMain", "compute", "tq_decode"),
    ("quality.slang", "terrainVS", "vertex", "quality_terrain_vs"),
    ("quality.slang", "terrainFS", "fragment", "quality_terrain_fs"),
    ("quality.slang", "shadowVS", "vertex", "quality_shadow_vs"),
    ("quality.slang", "skyVS", "vertex", "quality_sky_vs"),
    ("quality.slang", "skyFS", "fragment", "quality_sky_fs"),
    ("quality.slang", "instVS", "vertex", "quality_inst_vs"),
    ("quality.slang", "instShadowVS", "vertex", "quality_inst_shadow_vs"),
    ("quality.slang", "instShadowAlphaVS", "vertex", "quality_inst_shadow_alpha_vs"),
    ("quality.slang", "instShadowAlphaFS", "fragment", "quality_inst_shadow_alpha_fs"),
    ("quality.slang", "instFS", "fragment", "quality_inst_fs"),
    ("quality.slang", "instDepthFS", "fragment", "quality_inst_depth_fs"),
    ("quality.slang", "impVS", "vertex", "quality_imp_vs"),
    ("quality.slang", "impFS", "fragment", "quality_imp_fs"),
    ("quality.slang", "impFarFS", "fragment", "quality_imp_far_fs"),
    ("quality.slang", "impDepthFS", "fragment", "quality_imp_depth_fs"),
    ("quality.slang", "impEqFS", "fragment", "quality_imp_eq_fs"),
    ("quality.slang", "impShadowFarFS", "fragment", "quality_imp_shadow_far_fs"),
    ("quality.slang", "impShadowVS", "vertex", "quality_imp_shadow_vs"),
    ("quality.slang", "impShadowFS", "fragment", "quality_imp_shadow_fs"),
    ("qcull.slang", "cullMain", "compute", "qcull_cull"),
    ("qcull.slang", "scatterMain", "compute", "qcull_scatter"),
    ("qcull.slang", "drawsMain", "compute", "qcull_draws"),
    ("quality.slang", "atmoTransmittanceCS", "compute", "quality_atmo_transmittance"),
    ("quality.slang", "atmoMultiScatterCS", "compute", "quality_atmo_multiscatter"),
    ("quality.slang", "atmoSkyViewCS", "compute", "quality_atmo_skyview"),
    ("quality.slang", "atmoAerialCS", "compute", "quality_atmo_aerial"),
    ("quality.slang", "atmoIrradianceCS", "compute", "quality_atmo_irradiance"),
    ("planet.slang", "planetVS", "vertex", "planet_vs"),
    ("planet.slang", "planetFS", "fragment", "planet_fs"),
    ("planet.slang", "skyVS", "vertex", "planet_sky_vs"),
    ("planet.slang", "skyFS", "fragment", "planet_sky_fs"),
    ("planet.slang", "transmittanceCS", "compute", "planet_transmittance"),
    ("planet.slang", "multiScatterCS", "compute", "planet_multiscatter"),
    ("planet.slang", "skyGroundCS", "compute", "planet_skyground"),
    ("planet.slang", "irradianceCS", "compute", "planet_irradiance"),
    ("universe.slang", "vertexMain", "vertex", "universe_vs"),
    ("universe.slang", "fragmentMain", "fragment", "universe_fs"),
    ("universe_tonemap.slang", "tonemapVS", "vertex", "universe_tonemap_vs"),
    ("universe_tonemap.slang", "tonemapFS", "fragment", "universe_tonemap_fs"),
    ("system.slang", "fullVS", "vertex", "system_vs"),
    ("system.slang", "systemFS", "fragment", "system_fs"),
];

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("shaders");
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=SLANGC");
    // An explicit $SLANGC (as CI sets) must work: never fall back to stale SPIR-V silently then.
    let explicit = env::var("SLANGC").ok().filter(|s| !s.is_empty());
    let slangc = explicit.clone().unwrap_or_else(|| "slangc".into());
    let have = Command::new(&slangc).arg("-v").output().is_ok_and(|o| o.status.success());
    assert!(have || explicit.is_none(), "SLANGC={slangc} is set but does not run (`{slangc} -v` failed)");
    for (src, entry, stage, name) in SHADERS {
        let dst = out.join(format!("{name}.spv"));
        if have {
            let o = Command::new(&slangc)
                .arg(dir.join(src))
                .args(["-entry", entry, "-stage", stage])
                .args(["-target", "spirv", "-profile", "spirv_1_6", "-fvk-use-scalar-layout", "-O2"])
                .arg("-I")
                .arg(&dir)
                .arg("-o")
                .arg(&dst)
                .output()
                .expect("run slangc");
            assert!(
                o.status.success(),
                "slangc failed for {src}:{entry}\n{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            );
        } else {
            println!("cargo:warning=slangc not found; using checked-in SPIR-V for {name}");
            fs::copy(dir.join("spv").join(format!("{name}.spv")), &dst).expect("prebuilt SPIR-V missing");
        }
    }
}
