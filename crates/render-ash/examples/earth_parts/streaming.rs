//! Earth example glue: static far coverage plus an incrementally baked, movable close layer.
use glam::DVec3;
use planet::{dem::DemPlacement, stream::CloseStream};
use render_ash::{
    AshBackend,
    planet::{PlanetTextures, rgba8_mips},
};
use std::time::Duration;

type Mips = Vec<Vec<Vec<u8>>>;

pub struct DemStream {
    pub sun: DVec3,
    pub params: [[f32; 4]; 5],
    size: u32,
    half: f64,
    far: (Mips, Mips),
    cache: CloseStream,
    rows: u32,
    budget: Duration,
    uploaded: Option<[f64; 2]>,
    close: [f32; 3],
}
impl DemStream {
    pub fn new(sun: DVec3, params: [[f32; 4]; 5], size: u32, half: f64, far: (Mips, Mips)) -> Self {
        let number = |key: &str, default: f64, max: f64| {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| v.is_finite())
                .unwrap_or(default)
                .clamp(0.0, max)
        };
        let capacity = number("DEM_CACHE_TILES", 3.0, 16.0) as usize;
        let rows = number("DEM_BAKE_ROWS", 32.0, 4096.0) as u32;
        let budget = Duration::from_secs_f64(number("DEM_BAKE_MS", 4.0, 1000.0) / 1000.0);
        eprintln!(
            "dem stream: half={half:.0}m size={size} cache={capacity} complete + 1 partial; rows/frame={rows} time/frame={:.1}ms (one row indivisible)",
            budget.as_secs_f64() * 1000.0
        );
        Self {
            sun,
            params,
            size,
            half,
            far,
            cache: CloseStream::new(half, size, capacity),
            rows,
            budget,
            uploaded: None,
            close: [0.0; 3],
        }
    }
    /// Called once immediately before EVERY render (including REPEAT), never just at startup.
    pub fn frame(
        &mut self,
        be: &mut AshBackend,
        tex: &mut PlanetTextures,
        pl: &DemPlacement,
        cam: DVec3,
        forward: DVec3,
        ph: &(dyn Fn(DVec3) -> f64 + Sync),
    ) -> [f32; 3] {
        let target = pl.look_at_world(cam, forward);
        self.cache.request(target);
        let stats = self
            .cache
            .advance(self.rows, self.budget, |o, y| pl.bake_close_row(self.sun, self.size, self.half, o, y, ph));
        if let Some(tile) = self.cache.active() {
            if self.uploaded != Some(tile.offset) {
                let upload = std::time::Instant::now();
                let mut alb = self.far.0.clone();
                let mut aux = self.far.1.clone();
                alb.push(rgba8_mips(self.size, &tile.albedo, true));
                aux.push(rgba8_mips(self.size, &tile.aux, false));
                // planet_set_graft waits for GPU idle before replacing descriptors and retiring
                // the previous arrays. No in-flight frame can read an evicted view/allocation.
                be.planet_set_graft(tex, self.size, &alb, &aux).expect("stream close graft upload");
                self.uploaded = Some(tile.offset);
                eprintln!(
                    "dem stream activate: {:?}m upload+mips {:.1}ms (outside bake budget)",
                    tile.offset,
                    upload.elapsed().as_secs_f64() * 1000.0
                );
            }
            self.close =
                [(self.half / 1000.0) as f32, (tile.offset[0] / 1000.0) as f32, (tile.offset[1] / 1000.0) as f32];
        } else {
            // Invalid look-at / disabled stream: far/mid remains bound. Retain an uploaded tile
            // but do not sample it until it is useful again.
            self.close = [0.0; 3];
        }
        eprintln!(
            "dem stream frame: target={target:?} desired={:?} rows={} pending={} cached={} bake_ms={:.3} active={:?}",
            self.cache.desired_offset(),
            stats.rows,
            self.cache.pending_rows(),
            self.cache.cached_len(),
            stats.elapsed.as_secs_f64() * 1000.0,
            self.close
        );
        self.close
    }
}
