//! Memory-bounded cache of archive bytes, archive metadata and decoded slabs.
//!
//! Restoring one object from an archive needs the archive, its decompressed
//! metadata and the decoded slab of slices that holds the object's pixels.
//! Viewers read a series object after object, so caching the slab turns a
//! series read into one decode per slab instead of one per object.

use std::sync::{Arc, Mutex};

use lru::LruCache;
use tomoz_codec::Volume;

/// What a cache entry holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheKey {
    /// An archive file and its metadata.
    Archive(i64),
    /// A decoded slab: (archive, stack, slab index).
    Slab(i64, usize, usize),
}

/// An archive held in memory.
pub struct ArchiveEntry {
    /// The archive file.
    pub bytes: Vec<u8>,
    /// Its decompressed metadata.
    pub metadata: Vec<u8>,
}

/// A cached value.
#[derive(Clone)]
pub enum Cached {
    /// An archive.
    Archive(Arc<ArchiveEntry>),
    /// Decoded slices.
    Slab(Arc<Volume>),
}

impl Cached {
    fn bytes(&self) -> u64 {
        match self {
            Self::Archive(a) => (a.bytes.len() + a.metadata.len()) as u64,
            Self::Slab(v) => (v.samples().len() * 4) as u64,
        }
    }
}

struct Inner {
    lru: LruCache<CacheKey, Cached>,
    used: u64,
}

/// LRU cache with a byte budget.
pub struct Cache {
    inner: Mutex<Inner>,
    budget: u64,
}

impl Cache {
    /// An empty cache of at most `budget` bytes.
    #[must_use]
    pub fn new(budget: u64) -> Self {
        Self { inner: Mutex::new(Inner { lru: LruCache::unbounded(), used: 0 }), budget }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The entry for `key`, marked as recently used.
    pub fn get(&self, key: &CacheKey) -> Option<Cached> {
        self.lock().lru.get(key).cloned()
    }

    /// Inserts an entry, evicting the least recently used ones beyond the
    /// budget. Entries larger than the whole budget are not kept.
    pub fn insert(&self, key: CacheKey, value: Cached) {
        let size = value.bytes();
        if size > self.budget {
            return;
        }
        let mut inner = self.lock();
        if let Some(old) = inner.lru.put(key, value) {
            inner.used -= old.bytes();
        }
        inner.used += size;
        while inner.used > self.budget {
            match inner.lru.pop_lru() {
                Some((_, v)) => inner.used -= v.bytes(),
                None => break,
            }
        }
    }

    /// Drops every entry of an archive.
    pub fn forget_archive(&self, id: i64) {
        let mut inner = self.lock();
        let keys: Vec<CacheKey> = inner
            .lru
            .iter()
            .map(|(k, _)| *k)
            .filter(|k| matches!(k, CacheKey::Archive(a) | CacheKey::Slab(a, _, _) if *a == id))
            .collect();
        for k in keys {
            if let Some(v) = inner.lru.pop(&k) {
                inner.used -= v.bytes();
            }
        }
    }

    /// Bytes in use.
    pub fn used(&self) -> u64 {
        self.lock().used
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slab(n: usize) -> Cached {
        Cached::Slab(Arc::new(Volume::new(1, 1, n, 8, false, vec![0; n]).unwrap()))
    }

    #[test]
    fn evicts_least_recently_used_within_budget() {
        let c = Cache::new(100);
        c.insert(CacheKey::Slab(1, 0, 0), slab(10));
        c.insert(CacheKey::Slab(1, 0, 1), slab(10));
        assert!(c.get(&CacheKey::Slab(1, 0, 0)).is_some());
        c.insert(CacheKey::Slab(2, 0, 0), slab(10));
        assert_eq!(c.used(), 80);
        assert!(c.get(&CacheKey::Slab(1, 0, 1)).is_none(), "least recently used goes first");
        assert!(c.get(&CacheKey::Slab(1, 0, 0)).is_some());
        c.insert(CacheKey::Slab(3, 0, 0), slab(30));
        assert!(c.used() <= 100);
        c.forget_archive(1);
        assert!(c.get(&CacheKey::Slab(1, 0, 0)).is_none());
    }
}
