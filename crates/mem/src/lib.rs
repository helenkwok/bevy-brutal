//! Memory utilities for hot paths: a per-frame bump arena for `Copy` data and a
//! fixed-capacity index pool. Unsafe is allowed here; tests must pass under Miri.

use core::cell::{Cell, UnsafeCell};
use core::mem::{align_of, size_of, MaybeUninit};

/// Bump arena over a fixed, 64-byte-aligned backing buffer. Allocations live until
/// [`FrameArena::reset`], which requires `&mut self` and so cannot run while any slice is borrowed.
pub struct FrameArena {
    // UnsafeCell: we hand out `&mut` into disjoint ranges from `&self` (bump allocation).
    buf: Box<[UnsafeCell<MaybeUninit<Block>>]>,
    used: Cell<usize>,
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Block([u8; 64]);

impl FrameArena {
    /// Create an arena with at least `bytes` capacity.
    pub fn with_capacity(bytes: usize) -> Self {
        let blocks = bytes.div_ceil(64).max(1);
        let buf = (0..blocks).map(|_| UnsafeCell::new(MaybeUninit::uninit())).collect();
        Self { buf, used: Cell::new(0) }
    }

    pub fn capacity(&self) -> usize {
        self.buf.len() * 64
    }

    pub fn used(&self) -> usize {
        self.used.get()
    }

    /// Free everything. Needs `&mut self`, so no outstanding borrows can exist.
    pub fn reset(&mut self) {
        self.used.set(0);
    }

    /// Allocate a slice of `n` copies of `value`. Returns `None` if out of space.
    // `&self -> &mut [T]` is the bump-allocator pattern (cf. bumpalo): each call returns a range
    // disjoint from all live ones, and storage is UnsafeCell, so this is sound.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_slice_fill<T: Copy>(&self, n: usize, value: T) -> Option<&mut [T]> {
        let align = align_of::<T>();
        assert!(align <= 64, "FrameArena supports alignment up to 64");
        let start = self.used.get().next_multiple_of(align);
        let bytes = size_of::<T>().checked_mul(n)?;
        let end = start.checked_add(bytes)?;
        if end > self.capacity() {
            return None;
        }
        self.used.set(end);
        let base = UnsafeCell::raw_get(self.buf.as_ptr()) as *mut u8;
        // SAFETY: `start..end` is within the buffer (checked above) and disjoint from every
        // previously returned range because `used` only grows until `reset(&mut self)`.
        // `start` is a multiple of align_of::<T>() and the base is 64-aligned, so the pointer is
        // aligned. All n elements are initialised by the loop before the slice is formed.
        // Writing through a pointer derived from `&self` is allowed because the storage is
        // `UnsafeCell` and the range is never handed out twice. Miri-checked in tests.
        unsafe {
            let ptr = base.add(start) as *mut T;
            for i in 0..n {
                ptr.add(i).write(value);
            }
            Some(core::slice::from_raw_parts_mut(ptr, n))
        }
    }
}

/// Fixed-capacity free-list of `u32` slot indices (e.g. GPU buffer slots). No allocation after `new`.
pub struct SlotPool {
    free: Vec<u32>,
    capacity: u32,
}

impl SlotPool {
    pub fn new(capacity: u32) -> Self {
        Self { free: (0..capacity).rev().collect(), capacity }
    }
    pub fn alloc(&mut self) -> Option<u32> {
        self.free.pop()
    }
    pub fn free(&mut self, slot: u32) {
        debug_assert!(slot < self.capacity);
        debug_assert!(!self.free.contains(&slot), "double free of slot {slot}");
        self.free.push(slot);
    }
    pub fn available(&self) -> usize {
        self.free.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_alloc_and_reset() {
        let mut a = FrameArena::with_capacity(1024);
        {
            let x = a.alloc_slice_fill(10, 1.5f32).unwrap();
            let y = a.alloc_slice_fill(3, 7u64).unwrap();
            x[0] = 2.0;
            assert_eq!(x[0], 2.0);
            assert_eq!(y, &[7, 7, 7]);
            assert_eq!(y.as_ptr() as usize % align_of::<u64>(), 0);
        }
        assert!(a.used() >= 64);
        a.reset();
        assert_eq!(a.used(), 0);
        assert!(a.alloc_slice_fill(2000, 0u8).is_none());
    }

    #[test]
    fn pool() {
        let mut p = SlotPool::new(3);
        let a = p.alloc().unwrap();
        let _b = p.alloc().unwrap();
        p.free(a);
        assert_eq!(p.available(), 2);
    }
}
