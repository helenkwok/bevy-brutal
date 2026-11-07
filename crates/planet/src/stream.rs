//! Row-budgeted close graft cache. Far/mid coverage is owned by the renderer.
use std::time::Duration;

/// One complete CPU tile. Its offset is east/south in graft-world metres, not ENU north.
#[derive(Debug)]
pub struct CloseTile {
    pub offset: [f64; 2],
    pub albedo: Vec<u8>,
    pub aux: Vec<u8>,
}

#[derive(Debug, Default)]
pub struct BakeStats {
    pub rows: u32,
    pub elapsed: Duration,
    pub activated: bool,
}

struct Entry {
    key: [i64; 2],
    tile: CloseTile,
    used: u64,
}
struct Pending {
    key: [i64; 2],
    tile: CloseTile,
    row: u32,
}

/// Latest-visible-first scheduler: at most `capacity` complete tiles plus one partial tile.
/// There is no backlog or asynchronous result. One row is indivisible; elapsed time is checked
/// before starting each row, so a slow row may exceed the time budget. Mips/uploads are separate.
/// The active completed tile stays pinned until replacement is complete (even with capacity 1).
/// Capacity 0, size 0, or invalid half extents disable streaming; zero budgets pause baking.
pub struct CloseStream {
    half: f64,
    size: u32,
    capacity: usize,
    desired: Option<[i64; 2]>,
    active: Option<[i64; 2]>,
    cache: Vec<Entry>,
    pending: Option<Pending>,
    clock: u64,
    changed: bool,
}
impl CloseStream {
    pub fn new(half: f64, size: u32, capacity: usize) -> Self {
        let enabled = half.is_finite() && half >= 1.0 && size > 0;
        Self {
            half,
            size,
            capacity: if enabled { capacity } else { 0 },
            desired: None,
            active: None,
            cache: Vec::new(),
            pending: None,
            clock: 0,
            changed: false,
        }
    }
    fn offset(&self, key: [i64; 2]) -> [f64; 2] {
        key.map(|k| k as f64 * self.half * 0.5)
    }
    pub fn desired_offset(&self) -> Option<[f64; 2]> {
        self.desired.map(|k| self.offset(k))
    }
    pub fn cached_len(&self) -> usize {
        self.cache.len()
    }
    pub fn pending_rows(&self) -> u32 {
        self.pending.as_ref().map_or(0, |p| p.row)
    }
    pub fn active(&self) -> Option<&CloseTile> {
        self.active.and_then(|k| self.cache.iter().find(|e| e.key == k).map(|e| &e.tile))
    }
    /// Call every frame with the current usable look-at, or None for sky/outside DEM.
    pub fn request(&mut self, target: Option<[f64; 2]>) {
        self.clock = self.clock.saturating_add(1);
        let target = target.filter(|p| self.capacity > 0 && p.iter().all(|x| x.is_finite() && x.abs() < 1e12));
        let desired = target.map(|p| {
            let step = self.half * 0.5;
            if let Some(k) = self.desired {
                let o = self.offset(k);
                if (p[0] - o[0]).abs() <= step * 0.75 && (p[1] - o[1]).abs() <= step * 0.75 {
                    return k;
                }
            }
            p.map(|x| (x / step).round() as i64)
        });
        self.desired = desired;
        if self.pending.as_ref().is_some_and(|p| Some(p.key) != desired) {
            self.pending = None;
        }
        if let Some(k) = desired {
            if let Some(e) = self.cache.iter_mut().find(|e| e.key == k) {
                e.used = self.clock;
                self.changed |= self.active != Some(k);
                self.active = Some(k);
            }
        } else {
            self.changed |= self.active.is_some();
            self.active = None;
        }
        if let Some(k) = self.active
            && let Some(e) = self.cache.iter_mut().find(|e| e.key == k)
        {
            e.used = self.clock;
        }
    }
    /// The callback bakes exactly one row (size * 4 bytes per channel) at the supplied offset.
    /// Activation is atomic and only for the latest request; partial/cancelled data is never shown.
    pub fn advance(
        &mut self,
        max_rows: u32,
        budget: Duration,
        mut bake: impl FnMut([f64; 2], u32) -> (Vec<u8>, Vec<u8>),
    ) -> BakeStats {
        let start = std::time::Instant::now();
        let mut stats = BakeStats { activated: std::mem::take(&mut self.changed), ..Default::default() };
        let Some(key) = self.desired else {
            return stats;
        };
        if self.active == Some(key) || max_rows == 0 || budget.is_zero() {
            return stats;
        }
        if self.pending.is_none() {
            self.pending = Some(Pending {
                key,
                tile: CloseTile { offset: self.offset(key), albedo: Vec::new(), aux: Vec::new() },
                row: 0,
            });
        }
        let p = self.pending.as_mut().unwrap();
        while stats.rows < max_rows && p.row < self.size && start.elapsed() < budget {
            let (a, b) = bake(p.tile.offset, p.row);
            assert_eq!(a.len(), self.size as usize * 4);
            assert_eq!(b.len(), a.len());
            p.tile.albedo.extend(a);
            p.tile.aux.extend(b);
            p.row += 1;
            stats.rows += 1;
        }
        if p.row == self.size {
            let p = self.pending.take().unwrap();
            // New tile becomes active before eviction; the previous active GPU resource remains
            // owned by the renderer until it performs a lifetime-safe upload/swap.
            self.active = Some(key);
            while self.cache.len() >= self.capacity {
                let victim = self.cache.iter().enumerate().min_by_key(|(_, e)| e.used).unwrap().0;
                self.cache.swap_remove(victim);
            }
            self.cache.push(Entry { key, tile: p.tile, used: self.clock });
            stats.activated = true;
        }
        stats.elapsed = start.elapsed();
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(_: [f64; 2], y: u32) -> (Vec<u8>, Vec<u8>) {
        (vec![y as u8; 16], vec![0; 16])
    }
    #[test]
    fn duplicate_requests_preserve_progress_and_cancelled_rows_never_activate() {
        let mut s = CloseStream::new(3000.0, 4, 3);
        s.request(Some([0.0, 0.0]));
        s.advance(2, Duration::from_secs(1), row);
        s.request(Some([10.0, 10.0]));
        assert_eq!(s.pending_rows(), 2);
        s.request(Some([9000.0, 0.0]));
        assert_eq!(s.pending_rows(), 0);
        s.advance(3, Duration::from_secs(1), row);
        assert!(s.active().is_none());
        s.advance(1, Duration::from_secs(1), row);
        assert_eq!(s.active().unwrap().offset, [9000.0, 0.0]);
        assert_eq!(s.active().unwrap().albedo.len(), 64);
        assert_eq!(s.cached_len(), 1);
    }
    #[test]
    fn follows_look_at_with_hysteresis() {
        let mut s = CloseStream::new(3000.0, 4, 3);
        s.request(Some([5000.0, -2500.0]));
        assert_eq!(s.desired_offset(), Some([4500.0, -3000.0]));
        s.request(Some([5600.0, -2500.0]));
        assert_eq!(s.desired_offset(), Some([4500.0, -3000.0]));
        s.request(Some([5630.0, -2500.0]));
        assert_eq!(s.desired_offset(), Some([6000.0, -3000.0]));
    }
    #[test]
    fn rows_are_budgeted_across_frames_and_coverage_retained() {
        let mut s = CloseStream::new(3000.0, 4, 2);
        s.request(Some([0.0, 0.0]));
        assert_eq!(s.advance(2, Duration::from_secs(1), row).rows, 2);
        assert!(s.active().is_none());
        s.advance(2, Duration::from_secs(1), row);
        assert_eq!(s.active().unwrap().offset, [0.0, 0.0]);
        s.request(Some([6000.0, 0.0]));
        s.advance(1, Duration::from_secs(1), row);
        assert_eq!(s.active().unwrap().offset, [0.0, 0.0]);
        s.advance(4, Duration::from_secs(1), row);
        assert_eq!(s.active().unwrap().offset, [6000.0, 0.0]);
    }
    #[test]
    fn teleport_discards_stale_pending_and_cache_is_lru() {
        let mut s = CloseStream::new(3000.0, 4, 2);
        for x in [0.0, 6000.0] {
            s.request(Some([x, 0.0]));
            s.advance(4, Duration::from_secs(1), row);
        }
        s.request(Some([0.0, 0.0]));
        assert_eq!(s.advance(4, Duration::from_secs(1), row).rows, 0, "cached revisit");
        s.request(Some([12000.0, 0.0]));
        s.advance(1, Duration::from_secs(1), row);
        for x in 20..1000 {
            s.request(Some([x as f64 * 6000.0, 0.0]));
            assert!(s.pending_rows() <= 1);
        }
        s.advance(4, Duration::from_secs(1), row);
        assert_eq!(s.cached_len(), 2);
        s.request(Some([0.0, 0.0]));
        assert_eq!(s.advance(4, Duration::from_secs(1), row).rows, 0);
        s.request(Some([6000.0, 0.0]));
        assert_eq!(s.advance(4, Duration::from_secs(1), row).rows, 4, "old LRU evicted");
    }
    #[test]
    fn zero_budget_invalid_target_and_single_slot() {
        let mut s = CloseStream::new(3000.0, 4, 1);
        s.request(Some([0.0, 0.0]));
        assert_eq!(s.advance(0, Duration::from_secs(1), row).rows, 0);
        assert_eq!(s.advance(4, Duration::ZERO, row).rows, 0);
        s.advance(4, Duration::from_secs(1), row);
        s.request(Some([6000.0, 0.0]));
        s.advance(4, Duration::from_secs(1), row);
        assert_eq!(s.cached_len(), 1);
        assert_eq!(s.active().unwrap().offset, [6000.0, 0.0]);
        s.request(Some([f64::NAN, 0.0]));
        assert!(s.desired_offset().is_none());
        assert!(s.active().is_none());
        assert_eq!(s.advance(4, Duration::from_secs(1), row).rows, 0);
        let mut disabled = CloseStream::new(3000.0, 4, 0);
        disabled.request(Some([0.0, 0.0]));
        assert_eq!(disabled.advance(4, Duration::from_secs(1), row).rows, 0);
    }
    #[test]
    fn time_budget_stops_after_one_indivisible_row() {
        let mut s = CloseStream::new(3000.0, 4, 2);
        s.request(Some([0.0, 0.0]));
        let stats = s.advance(4, Duration::from_millis(1), |o, y| {
            std::thread::sleep(Duration::from_millis(3));
            row(o, y)
        });
        assert_eq!(stats.rows, 1);
        assert!(stats.elapsed >= Duration::from_millis(3));
    }
}
