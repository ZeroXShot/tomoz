//! Object store: SQLite index, raw objects, compacted archives.
//!
//! The index is the source of truth. Files are written to a temporary name,
//! synced and renamed before the index refers to them, and deleted only
//! after the index stops referring to them; files left over by a crash
//! between those steps are unreferenced and removed at startup.
//!
//! A DICOM object with codable pixel data is first stored raw and recorded
//! under its series. Once a series has been quiet for a while, the compactor
//! packs its raw objects into a Tomoz archive, verifies that every object
//! restores byte for byte, and switches the index entries from the raw files
//! to the archive in one transaction. Objects overwritten or deleted while
//! the archive was being built keep their newer state.

use std::collections::hash_map::RandomState;
use std::fs;
use std::hash::{BuildHasher, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use md5::{Digest as _, Md5};
use rusqlite::{Connection, OptionalExtension, params};
use sha2::Sha256;
use tomoz_codec::{DecodeOptions, EncodeOptions, Model, ModelSet, Volume};

use crate::cache::{Cache, CacheKey, Cached};
use crate::metrics::Metrics;

/// Errors of the store, mapped to S3 errors by the HTTP layer.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// The bucket does not exist.
    #[error("no such bucket")]
    NoSuchBucket,
    /// The key does not exist.
    #[error("no such key")]
    NoSuchKey,
    /// The multipart upload does not exist.
    #[error("no such upload")]
    NoSuchUpload,
    /// The bucket name is invalid.
    #[error("invalid bucket name")]
    InvalidBucketName,
    /// The bucket is not empty.
    #[error("bucket not empty")]
    BucketNotEmpty,
    /// A request argument is invalid.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// A multipart part is missing or does not match.
    #[error("invalid part: {0}")]
    InvalidPart(String),
    /// Stored data failed verification.
    #[error("data corruption: {0}")]
    Corrupt(String),
    /// I/O failure.
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    /// Index failure.
    #[error("index: {0}")]
    Index(#[from] rusqlite::Error),
}

/// Where the current version of an object is, as the index says.
struct Located {
    meta: ObjectMeta,
    blob: Option<String>,
    archive: Option<(i64, String)>,
    instance: Option<i64>,
    sha256: Vec<u8>,
}

/// Metadata of a stored object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectMeta {
    /// Key.
    pub key: String,
    /// Size in bytes.
    pub size: u64,
    /// Entity tag, quoted, as S3 returns it.
    pub etag: String,
    /// Content type given at upload.
    pub content_type: Option<String>,
    /// Last modification, seconds since the epoch.
    pub modified: i64,
    /// Whether the object lives in a compacted archive.
    pub archived: bool,
}

/// One page of a listing.
#[derive(Debug, Default)]
pub struct Listing {
    /// Objects.
    pub objects: Vec<ObjectMeta>,
    /// Common prefixes when a delimiter was given.
    pub prefixes: Vec<String>,
    /// Position to continue from (exclusive), if truncated.
    pub next: Option<String>,
}

/// Result of one compaction.
#[derive(Debug, Default, Clone)]
pub struct Compaction {
    /// Objects moved into the archive.
    pub objects: usize,
    /// Their total size.
    pub input_bytes: u64,
    /// Size of the archive.
    pub archive_bytes: u64,
}

/// Valid S3 bucket names: 3 to 63 lowercase letters, digits, dots and
/// hyphens, starting and ending with a letter or digit.
#[must_use]
pub fn valid_bucket(name: &str) -> bool {
    let b = name.as_bytes();
    (3..=63).contains(&b.len())
        && b.iter().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'.' || *c == b'-')
        && b[0].is_ascii_alphanumeric()
        && b[b.len() - 1].is_ascii_alphanumeric()
        && !name.contains("..")
}

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

/// Random 128-bit identifiers from the operating system's hash seeds.
struct Ids {
    a: RandomState,
    b: RandomState,
    counter: AtomicU64,
}

impl Ids {
    fn new() -> Self {
        Self { a: RandomState::new(), b: RandomState::new(), counter: AtomicU64::new(0) }
    }

    fn next(&self) -> String {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let t = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
        let mut h1 = self.a.build_hasher();
        h1.write_u64(n);
        h1.write_u64(t);
        let mut h2 = self.b.build_hasher();
        h2.write_u64(t);
        h2.write_u64(n);
        format!("{:016x}{:016x}", h1.finish(), h2.finish())
    }
}

/// Settings of the store taken from the gateway configuration.
#[derive(Clone, Debug)]
pub struct StoreOptions {
    /// Data directory.
    pub dir: PathBuf,
    /// zstd level of archive metadata.
    pub zstd_level: i32,
    /// Slices per tile in archives.
    pub slab: u16,
    /// Cache budget in bytes.
    pub cache_bytes: u64,
}

/// The store.
pub struct Store {
    dir: PathBuf,
    db: Mutex<Connection>,
    ids: Ids,
    models: ModelSet,
    registry: Vec<Model>,
    zstd_level: i32,
    slab: u16,
    cache: Cache,
    metrics: Arc<Metrics>,
}

const SCHEMA: &str = "
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS buckets (name TEXT PRIMARY KEY, created INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS archives (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL UNIQUE,
    series TEXT NOT NULL,
    live INTEGER NOT NULL,
    bytes INTEGER NOT NULL,
    input_bytes INTEGER NOT NULL,
    created INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS objects (
    bucket TEXT NOT NULL REFERENCES buckets(name),
    key TEXT NOT NULL,
    size INTEGER NOT NULL,
    etag TEXT NOT NULL,
    content_type TEXT,
    modified INTEGER NOT NULL,
    sha256 BLOB NOT NULL,
    blob TEXT UNIQUE,
    archive INTEGER REFERENCES archives(id),
    instance INTEGER,
    series TEXT,
    PRIMARY KEY (bucket, key),
    CHECK ((blob IS NULL) != (archive IS NULL))
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS objects_pending ON objects(series) WHERE blob IS NOT NULL AND series IS NOT NULL;
CREATE INDEX IF NOT EXISTS objects_archive ON objects(archive) WHERE archive IS NOT NULL;
CREATE TABLE IF NOT EXISTS series (key TEXT PRIMARY KEY, last_put INTEGER NOT NULL) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS uploads (
    id TEXT PRIMARY KEY,
    bucket TEXT NOT NULL REFERENCES buckets(name),
    key TEXT NOT NULL,
    content_type TEXT,
    created INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS parts (
    upload TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    number INTEGER NOT NULL,
    size INTEGER NOT NULL,
    etag TEXT NOT NULL,
    md5 BLOB NOT NULL,
    blob TEXT NOT NULL UNIQUE,
    PRIMARY KEY (upload, number)
) WITHOUT ROWID;
";

/// Writes `data` to `path` atomically: temporary file, sync, rename.
fn write_atomic(tmp_dir: &Path, path: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = tmp_dir.join(format!("{}.tmp", path.file_name().map_or_else(|| "x".into(), |n| n.to_string_lossy())));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    // Make the rename durable (POSIX needs the directory synced; Windows
    // cannot open directories as files and journals the rename itself).
    #[cfg(unix)]
    if let Some(dir) = path.parent() {
        fs::File::open(dir)?.sync_all()?;
    }
    Ok(())
}

/// The series an object would be compacted with, if its pixel data can be
/// coded.
fn series_of(bucket: &str, body: &[u8]) -> Option<String> {
    let parsed = tomoz_dicom::parse(body).ok()?;
    let span = parsed.pixel?;
    if span.encapsulated || parsed.has_float_pixels || parsed.encoding == tomoz_dicom::DatasetEncoding::Deflated {
        return None;
    }
    tomoz_dicom::PixelLayout::from_attributes(&parsed.attributes, parsed.encoding).ok()?;
    let uid = parsed.attributes.series_instance_uid?;
    Some(format!("{bucket}\u{0}{uid}"))
}

fn etag_of(md5: &[u8]) -> String {
    format!("\"{}\"", md5.iter().map(|b| format!("{b:02x}")).collect::<String>())
}

impl Store {
    /// Opens (or creates) a store and removes files left by a crash.
    ///
    /// # Errors
    ///
    /// I/O and index errors.
    pub fn open(options: &StoreOptions, models: ModelSet, metrics: Arc<Metrics>) -> Result<Self, StoreError> {
        let dir = options.dir.clone();
        for sub in ["raw", "archives", "parts", "tmp"] {
            fs::create_dir_all(dir.join(sub))?;
        }
        for e in fs::read_dir(dir.join("tmp"))? {
            let _ = fs::remove_file(e?.path());
        }
        let db = Connection::open(dir.join("index.sqlite"))?;
        db.execute_batch(SCHEMA)?;
        db.busy_timeout(std::time::Duration::from_secs(30))?;
        let registry = vec![models.two_d.clone(), models.three_d.clone()];
        let store = Self {
            dir,
            db: Mutex::new(db),
            ids: Ids::new(),
            models,
            registry,
            zstd_level: options.zstd_level,
            slab: options.slab,
            cache: Cache::new(options.cache_bytes),
            metrics,
        };
        store.remove_orphans()?;
        store.refresh_gauges()?;
        Ok(store)
    }

    /// The metrics the store updates.
    #[must_use]
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// Bytes held by the cache.
    #[must_use]
    pub fn cache_bytes(&self) -> u64 {
        self.cache.used()
    }

    fn db(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn raw_path(&self, blob: &str) -> PathBuf {
        self.dir.join("raw").join(&blob[..2]).join(blob)
    }

    fn archive_path(&self, file: &str) -> PathBuf {
        self.dir.join("archives").join(&file[..2]).join(file)
    }

    fn part_path(&self, blob: &str) -> PathBuf {
        self.dir.join("parts").join(&blob[..2]).join(blob)
    }

    /// Deletes files the index does not refer to.
    fn remove_orphans(&self) -> Result<usize, StoreError> {
        let db = self.db();
        let mut removed = 0;
        let referenced = |sql: &str| -> Result<std::collections::HashSet<String>, StoreError> {
            let mut stmt = db.prepare(sql)?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            Ok(rows.collect::<Result<_, _>>()?)
        };
        let blobs = referenced("SELECT blob FROM objects WHERE blob IS NOT NULL")?;
        let archives = referenced("SELECT file FROM archives")?;
        let parts = referenced("SELECT blob FROM parts")?;
        for (sub, keep) in [("raw", &blobs), ("archives", &archives), ("parts", &parts)] {
            for shard in fs::read_dir(self.dir.join(sub))? {
                let shard = shard?.path();
                if !shard.is_dir() {
                    continue;
                }
                for f in fs::read_dir(&shard)? {
                    let path = f?.path();
                    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    if !keep.contains(&name) {
                        fs::remove_file(&path)?;
                        removed += 1;
                    }
                }
            }
        }
        if removed > 0 {
            tracing::warn!(removed, "removed files left over by an interrupted operation");
        }
        Ok(removed)
    }

    /// Updates the gauges from the index.
    ///
    /// # Errors
    ///
    /// Index errors.
    pub fn refresh_gauges(&self) -> Result<(), StoreError> {
        let db = self.db();
        let (raw_n, raw_b): (i64, i64) =
            db.query_row("SELECT count(*), coalesce(sum(size), 0) FROM objects WHERE blob IS NOT NULL", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })?;
        let (arc_n, arc_b): (i64, i64) =
            db.query_row("SELECT count(*), coalesce(sum(size), 0) FROM objects WHERE archive IS NOT NULL", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })?;
        let (files, stored): (i64, i64) =
            db.query_row("SELECT count(*), coalesce(sum(bytes), 0) FROM archives", [], |r| Ok((r.get(0)?, r.get(1)?)))?;
        let pending: i64 = db.query_row(
            "SELECT count(DISTINCT series) FROM objects WHERE blob IS NOT NULL AND series IS NOT NULL",
            [],
            |r| r.get(0),
        )?;
        self.metrics.set_store(
            raw_n as u64,
            raw_b as u64,
            arc_n as u64,
            arc_b as u64,
            files as u64,
            stored as u64,
            pending as u64,
        );
        Ok(())
    }

    /// Creates a bucket (idempotent).
    ///
    /// # Errors
    ///
    /// [`StoreError::InvalidBucketName`] and index errors.
    pub fn create_bucket(&self, name: &str) -> Result<(), StoreError> {
        if !valid_bucket(name) {
            return Err(StoreError::InvalidBucketName);
        }
        self.db().execute("INSERT OR IGNORE INTO buckets (name, created) VALUES (?1, ?2)", params![name, now()])?;
        Ok(())
    }

    /// Deletes an empty bucket.
    ///
    /// # Errors
    ///
    /// [`StoreError::NoSuchBucket`], [`StoreError::BucketNotEmpty`].
    pub fn delete_bucket(&self, name: &str) -> Result<(), StoreError> {
        let db = self.db();
        let n: i64 = db.query_row("SELECT count(*) FROM objects WHERE bucket = ?1", [name], |r| r.get(0))?;
        if n > 0 {
            return Err(StoreError::BucketNotEmpty);
        }
        if db.execute("DELETE FROM buckets WHERE name = ?1", [name])? == 0 {
            return Err(StoreError::NoSuchBucket);
        }
        Ok(())
    }

    /// Bucket names and creation times.
    ///
    /// # Errors
    ///
    /// Index errors.
    pub fn buckets(&self) -> Result<Vec<(String, i64)>, StoreError> {
        let db = self.db();
        let mut stmt = db.prepare("SELECT name, created FROM buckets ORDER BY name")?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Whether a bucket exists.
    ///
    /// # Errors
    ///
    /// Index errors.
    pub fn has_bucket(&self, name: &str) -> Result<bool, StoreError> {
        Ok(self.db().query_row("SELECT 1 FROM buckets WHERE name = ?1", [name], |_| Ok(())).optional()?.is_some())
    }

    fn require_bucket(&self, db: &Connection, bucket: &str) -> Result<(), StoreError> {
        db.query_row("SELECT 1 FROM buckets WHERE name = ?1", [bucket], |_| Ok(()))
            .optional()?
            .ok_or(StoreError::NoSuchBucket)
    }

    /// Stores an object.
    ///
    /// # Errors
    ///
    /// [`StoreError::NoSuchBucket`], I/O and index errors.
    pub fn put(
        &self,
        bucket: &str,
        key: &str,
        body: &[u8],
        content_type: Option<&str>,
    ) -> Result<ObjectMeta, StoreError> {
        let md5 = Md5::digest(body);
        self.put_with_etag(bucket, key, body, content_type, etag_of(&md5))
    }

    fn put_with_etag(
        &self,
        bucket: &str,
        key: &str,
        body: &[u8],
        content_type: Option<&str>,
        etag: String,
    ) -> Result<ObjectMeta, StoreError> {
        if key.is_empty() || key.len() > 1024 {
            return Err(StoreError::InvalidArgument("keys are 1 to 1024 bytes".into()));
        }
        if !self.has_bucket(bucket)? {
            return Err(StoreError::NoSuchBucket);
        }
        let sha256: [u8; 32] = Sha256::digest(body).into();
        let series = series_of(bucket, body);
        let blob = self.ids.next();
        write_atomic(&self.dir.join("tmp"), &self.raw_path(&blob), body)?;
        let modified = now();
        let released = {
            let mut db = self.db();
            let tx = db.transaction()?;
            self.require_bucket(&tx, bucket)?;
            let released = Self::release(&tx, bucket, key)?;
            tx.execute(
                "INSERT INTO objects (bucket, key, size, etag, content_type, modified, sha256, blob, series)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![bucket, key, body.len() as i64, etag, content_type, modified, &sha256[..], blob, series],
            )?;
            if let Some(s) = &series {
                tx.execute(
                    "INSERT INTO series (key, last_put) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET last_put = ?2",
                    params![s, modified],
                )?;
            }
            tx.commit()?;
            released
        };
        self.cleanup(released);
        self.metrics.objects_put.inc();
        self.metrics.bytes_in.add(body.len() as u64);
        Ok(ObjectMeta {
            key: key.to_owned(),
            size: body.len() as u64,
            etag,
            content_type: content_type.map(str::to_owned),
            modified,
            archived: false,
        })
    }

    /// Removes the current version of `key` from the index; returns the files
    /// to delete once the transaction commits.
    fn release(tx: &rusqlite::Transaction<'_>, bucket: &str, key: &str) -> Result<Released, StoreError> {
        let old: Option<(Option<String>, Option<i64>)> = tx
            .query_row("SELECT blob, archive FROM objects WHERE bucket = ?1 AND key = ?2", params![bucket, key], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .optional()?;
        let mut released = Released::default();
        if let Some((blob, archive)) = old {
            tx.execute("DELETE FROM objects WHERE bucket = ?1 AND key = ?2", params![bucket, key])?;
            released.blob = blob;
            if let Some(id) = archive {
                tx.execute("UPDATE archives SET live = live - 1 WHERE id = ?1", [id])?;
                let (live, file): (i64, String) =
                    tx.query_row("SELECT live, file FROM archives WHERE id = ?1", [id], |r| {
                        Ok((r.get(0)?, r.get(1)?))
                    })?;
                if live <= 0 {
                    tx.execute("DELETE FROM archives WHERE id = ?1", [id])?;
                    released.archive = Some((id, file));
                }
            }
        }
        Ok(released)
    }

    fn cleanup(&self, released: Released) {
        if let Some(blob) = released.blob {
            let _ = fs::remove_file(self.raw_path(&blob));
        }
        if let Some((id, file)) = released.archive {
            self.cache.forget_archive(id);
            let _ = fs::remove_file(self.archive_path(&file));
        }
    }

    /// Metadata of an object.
    ///
    /// # Errors
    ///
    /// [`StoreError::NoSuchBucket`], [`StoreError::NoSuchKey`].
    pub fn head(&self, bucket: &str, key: &str) -> Result<ObjectMeta, StoreError> {
        let db = self.db();
        self.require_bucket(&db, bucket)?;
        db.query_row(
            "SELECT size, etag, content_type, modified, archive IS NOT NULL FROM objects WHERE bucket = ?1 AND key = ?2",
            params![bucket, key],
            |r| {
                Ok(ObjectMeta {
                    key: key.to_owned(),
                    size: r.get::<_, i64>(0)? as u64,
                    etag: r.get(1)?,
                    content_type: r.get(2)?,
                    modified: r.get(3)?,
                    archived: r.get(4)?,
                })
            },
        )
        .optional()?
        .ok_or(StoreError::NoSuchKey)
    }

    /// Reads an object.
    ///
    /// # Errors
    ///
    /// [`StoreError::NoSuchKey`], [`StoreError::Corrupt`] if restored data
    /// does not match its digest, I/O and index errors.
    pub fn get(&self, bucket: &str, key: &str) -> Result<(ObjectMeta, Arc<Vec<u8>>), StoreError> {
        // The index is read, then the file. A compaction, overwrite or delete
        // that commits in between removes the file the index pointed to; the
        // index has already moved on (to the archive, or to nothing), so
        // looking again gives the current state.
        let mut attempt = 0;
        loop {
            let location = self.locate(bucket, key)?;
            match self.read_located(bucket, key, location) {
                Err(StoreError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound && attempt < 4 => attempt += 1,
                result => return result,
            }
        }
    }

    fn locate(&self, bucket: &str, key: &str) -> Result<Located, StoreError> {
        let db = self.db();
        self.require_bucket(&db, bucket)?;
        db.query_row(
            "SELECT o.size, o.etag, o.content_type, o.modified, o.blob, o.archive, o.instance, o.sha256, a.file
             FROM objects o LEFT JOIN archives a ON a.id = o.archive WHERE o.bucket = ?1 AND o.key = ?2",
            params![bucket, key],
            |r| {
                let meta = ObjectMeta {
                    key: key.to_owned(),
                    size: r.get::<_, i64>(0)? as u64,
                    etag: r.get(1)?,
                    content_type: r.get(2)?,
                    modified: r.get(3)?,
                    archived: r.get::<_, Option<i64>>(5)?.is_some(),
                };
                let archive = match (r.get::<_, Option<i64>>(5)?, r.get::<_, Option<String>>(8)?) {
                    (Some(id), Some(file)) => Some((id, file)),
                    _ => None,
                };
                Ok(Located { meta, blob: r.get(4)?, archive, instance: r.get(6)?, sha256: r.get(7)? })
            },
        )
        .optional()?
        .ok_or(StoreError::NoSuchKey)
    }

    fn read_located(
        &self,
        bucket: &str,
        key: &str,
        location: Located,
    ) -> Result<(ObjectMeta, Arc<Vec<u8>>), StoreError> {
        let data = match (location.blob, location.archive, location.instance) {
            (Some(blob), _, _) => {
                let data = fs::read(self.raw_path(&blob))?;
                if <[u8; 32]>::from(Sha256::digest(&data))[..] != location.sha256[..] {
                    return Err(StoreError::Corrupt(format!("raw object {bucket}/{key}")));
                }
                Arc::new(data)
            }
            (None, Some((id, file)), Some(instance)) => Arc::new(self.restore(id, &file, instance as usize)?),
            _ => return Err(StoreError::Corrupt(format!("index entry of {bucket}/{key}"))),
        };
        self.metrics.bytes_out.add(data.len() as u64);
        Ok((location.meta, data))
    }

    /// Restores one instance of an archive through the cache.
    fn restore(&self, id: i64, file: &str, instance: usize) -> Result<Vec<u8>, StoreError> {
        let corrupt = |e: tomoz_archive::Error| StoreError::Corrupt(format!("archive {file}: {e}"));
        let entry = match self.cache.get(&CacheKey::Archive(id)) {
            Some(Cached::Archive(a)) => {
                self.metrics.cache_hits.inc();
                a
            }
            _ => {
                self.metrics.cache_misses.inc();
                let bytes = fs::read(self.archive_path(file))?;
                let archive = tomoz_archive::Archive::open(&bytes).map_err(corrupt)?;
                let metadata = archive.metadata().map_err(corrupt)?;
                drop(archive);
                let a = Arc::new(crate::cache::ArchiveEntry { bytes, metadata });
                self.cache.insert(CacheKey::Archive(id), Cached::Archive(a.clone()));
                a
            }
        };
        let archive = tomoz_archive::Archive::open(&entry.bytes).map_err(corrupt)?;
        if instance >= archive.instances().len() {
            return Err(StoreError::Corrupt(format!("archive {file} has no instance {instance}")));
        }
        let Some((stack, range)) = archive.instance_slices(instance) else {
            return archive.rebuild(instance, None, &entry.metadata).map_err(corrupt);
        };
        let options = DecodeOptions::with_registry(&self.registry);
        let stack_bytes = archive.stack(stack);
        let slab = usize::from(tomoz_codec::inspect(stack_bytes).map_err(|e| StoreError::Corrupt(e.to_string()))?.slab);
        let first_slab = range.start / slab;
        let slices = if range.end <= (first_slab + 1) * slab {
            // The common case: one slab, decoded once and shared by the
            // instances it holds.
            let key = CacheKey::Slab(id, stack, first_slab);
            let volume = match self.cache.get(&key) {
                Some(Cached::Slab(v)) => {
                    self.metrics.cache_hits.inc();
                    v
                }
                _ => {
                    self.metrics.cache_misses.inc();
                    let depth = tomoz_codec::inspect(stack_bytes).map_err(|e| StoreError::Corrupt(e.to_string()))?.depth
                        as usize;
                    let z = first_slab * slab..((first_slab + 1) * slab).min(depth);
                    let v = Arc::new(
                        tomoz_codec::decode_slices(stack_bytes, &options, z)
                            .map_err(|e| StoreError::Corrupt(e.to_string()))?,
                    );
                    self.cache.insert(key, Cached::Slab(v.clone()));
                    v
                }
            };
            let plane = volume.height() * volume.width();
            let lo = (range.start - first_slab * slab) * plane;
            let hi = (range.end - first_slab * slab) * plane;
            Volume::new(
                range.len(),
                volume.height(),
                volume.width(),
                volume.bits(),
                volume.signed(),
                volume.samples()[lo..hi].to_vec(),
            )
            .map_err(|e| StoreError::Corrupt(e.to_string()))?
        } else {
            tomoz_codec::decode_slices(stack_bytes, &options, range).map_err(|e| StoreError::Corrupt(e.to_string()))?
        };
        archive.rebuild(instance, Some(&slices), &entry.metadata).map_err(corrupt)
    }

    /// Deletes an object; deleting a missing key succeeds, as in S3.
    ///
    /// # Errors
    ///
    /// [`StoreError::NoSuchBucket`], index errors.
    pub fn delete(&self, bucket: &str, key: &str) -> Result<(), StoreError> {
        let released = {
            let mut db = self.db();
            let tx = db.transaction()?;
            self.require_bucket(&tx, bucket)?;
            let r = Self::release(&tx, bucket, key)?;
            tx.commit()?;
            r
        };
        self.cleanup(released);
        Ok(())
    }

    /// Lists keys after `after` that start with `prefix`, grouping keys that
    /// contain `delimiter` after the prefix.
    ///
    /// # Errors
    ///
    /// [`StoreError::NoSuchBucket`], index errors.
    pub fn list(
        &self,
        bucket: &str,
        prefix: &str,
        delimiter: Option<&str>,
        after: Option<&str>,
        max: usize,
    ) -> Result<Listing, StoreError> {
        let db = self.db();
        self.require_bucket(&db, bucket)?;
        let mut listing = Listing::default();
        let mut cursor = after.map_or_else(|| prefix.to_owned(), str::to_owned);
        let mut inclusive = after.is_none();
        let mut stmt_incl = db.prepare(
            "SELECT key, size, etag, content_type, modified, archive IS NOT NULL FROM objects
             WHERE bucket = ?1 AND key >= ?2 ORDER BY key LIMIT ?3",
        )?;
        let mut stmt_excl = db.prepare(
            "SELECT key, size, etag, content_type, modified, archive IS NOT NULL FROM objects
             WHERE bucket = ?1 AND key > ?2 ORDER BY key LIMIT ?3",
        )?;
        let total = |l: &Listing| l.objects.len() + l.prefixes.len();
        'scan: loop {
            let batch = 256;
            let stmt = if inclusive { &mut stmt_incl } else { &mut stmt_excl };
            let rows: Vec<ObjectMeta> = stmt
                .query_map(params![bucket, cursor, batch], |r| {
                    Ok(ObjectMeta {
                        key: r.get(0)?,
                        size: r.get::<_, i64>(1)? as u64,
                        etag: r.get(2)?,
                        content_type: r.get(3)?,
                        modified: r.get(4)?,
                        archived: r.get(5)?,
                    })
                })?
                .collect::<Result<_, _>>()?;
            if rows.is_empty() {
                break;
            }
            let n = rows.len();
            for o in rows {
                if !o.key.starts_with(prefix) {
                    break 'scan;
                }
                if total(&listing) == max {
                    // Resume after the last key or common prefix returned.
                    listing.next = Some(cursor.clone());
                    break 'scan;
                }
                cursor.clone_from(&o.key);
                inclusive = false;
                if let Some(d) = delimiter.filter(|d| !d.is_empty())
                    && let Some(pos) = o.key[prefix.len()..].find(d)
                {
                    let common = o.key[..prefix.len() + pos + d.len()].to_owned();
                    if listing.prefixes.last() != Some(&common) {
                        listing.prefixes.push(common.clone());
                    }
                    // Skip every key under this prefix.
                    cursor = format!("{common}\u{10FFFF}");
                    continue;
                }
                listing.objects.push(o);
            }
            if n < batch {
                break;
            }
        }
        Ok(listing)
    }

    /// Series due for compaction: quiet for `quiet` seconds with at least
    /// `min_objects` raw objects.
    ///
    /// # Errors
    ///
    /// Index errors.
    pub fn due(&self, quiet: u64, min_objects: u32) -> Result<Vec<String>, StoreError> {
        let db = self.db();
        let mut stmt = db.prepare(
            "SELECT o.series FROM objects o JOIN series s ON s.key = o.series
             WHERE o.blob IS NOT NULL AND o.series IS NOT NULL AND s.last_put <= ?1
             GROUP BY o.series HAVING count(*) >= ?2 ORDER BY min(s.last_put)",
        )?;
        let rows = stmt.query_map(params![now() - quiet as i64, i64::from(min_objects)], |r| r.get(0))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Packs the raw objects of a series into an archive.
    ///
    /// # Errors
    ///
    /// I/O, index and verification errors; on error nothing changes.
    pub fn compact(&self, series: &str) -> Result<Compaction, StoreError> {
        let members: Vec<(String, String, String, Vec<u8>)> = {
            let db = self.db();
            let mut stmt = db.prepare(
                "SELECT bucket, key, blob, sha256 FROM objects WHERE series = ?1 AND blob IS NOT NULL ORDER BY key",
            )?;
            let rows = stmt.query_map([series], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?;
            rows.collect::<Result<_, _>>()?
        };
        if members.is_empty() {
            return Ok(Compaction::default());
        }
        let mut files = Vec::with_capacity(members.len());
        for (_, key, blob, sha) in &members {
            let data = fs::read(self.raw_path(blob))?;
            if <[u8; 32]>::from(Sha256::digest(&data))[..] != sha[..] {
                return Err(StoreError::Corrupt(format!("raw object {key} before compaction")));
            }
            files.push((key.clone(), data));
        }
        let refs: Vec<(String, &[u8])> = files.iter().map(|(k, d)| (k.clone(), d.as_slice())).collect();
        let options = tomoz_archive::PackOptions {
            zstd_level: self.zstd_level,
            ..tomoz_archive::PackOptions::new(EncodeOptions {
                slab: self.slab,
                ..EncodeOptions::with_models(self.models.clone())
            })
        };
        let (bytes, report) = tomoz_archive::pack(&refs, &options).map_err(|e| StoreError::Corrupt(e.to_string()))?;
        // Every object must come back byte for byte before the raw copies go.
        let archive = tomoz_archive::Archive::open(&bytes).map_err(|e| StoreError::Corrupt(e.to_string()))?;
        archive.restore_all(&self.registry).map_err(|e| StoreError::Corrupt(format!("verification failed: {e}")))?;
        let file = format!("{}.tmzd", self.ids.next());
        write_atomic(&self.dir.join("tmp"), &self.archive_path(&file), &bytes)?;
        let moved = {
            let mut db = self.db();
            let tx = db.transaction()?;
            tx.execute(
                "INSERT INTO archives (file, series, live, bytes, input_bytes, created) VALUES (?1, ?2, 0, ?3, ?4, ?5)",
                params![file, series, bytes.len() as i64, report.input_bytes as i64, now()],
            )?;
            let id = tx.last_insert_rowid();
            let mut moved = Vec::new();
            for (i, (bucket, key, blob, _)) in members.iter().enumerate() {
                // Only objects unchanged since they were read move.
                let n = tx.execute(
                    "UPDATE objects SET blob = NULL, archive = ?1, instance = ?2 WHERE bucket = ?3 AND key = ?4 AND blob = ?5",
                    params![id, i as i64, bucket, key, blob],
                )?;
                if n == 1 {
                    moved.push(blob.clone());
                }
            }
            if moved.is_empty() {
                tx.execute("DELETE FROM archives WHERE id = ?1", [id])?;
            } else {
                tx.execute("UPDATE archives SET live = ?1 WHERE id = ?2", params![moved.len() as i64, id])?;
            }
            tx.commit()?;
            moved
        };
        if moved.is_empty() {
            let _ = fs::remove_file(self.archive_path(&file));
            return Ok(Compaction::default());
        }
        for blob in &moved {
            let _ = fs::remove_file(self.raw_path(blob));
        }
        self.metrics.compactions.inc();
        self.metrics.compaction_input_bytes.add(report.input_bytes);
        self.metrics.compaction_output_bytes.add(bytes.len() as u64);
        let _ = self.refresh_gauges();
        Ok(Compaction { objects: moved.len(), input_bytes: report.input_bytes, archive_bytes: bytes.len() as u64 })
    }

    /// Starts a multipart upload; returns its identifier.
    ///
    /// # Errors
    ///
    /// [`StoreError::NoSuchBucket`], index errors.
    pub fn create_upload(&self, bucket: &str, key: &str, content_type: Option<&str>) -> Result<String, StoreError> {
        let id = self.ids.next();
        let db = self.db();
        self.require_bucket(&db, bucket)?;
        db.execute(
            "INSERT INTO uploads (id, bucket, key, content_type, created) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, bucket, key, content_type, now()],
        )?;
        Ok(id)
    }

    /// Stores one part; returns its entity tag.
    ///
    /// # Errors
    ///
    /// [`StoreError::NoSuchUpload`], I/O and index errors.
    pub fn upload_part(
        &self,
        bucket: &str,
        key: &str,
        upload: &str,
        number: u32,
        body: &[u8],
    ) -> Result<String, StoreError> {
        if !(1..=10_000).contains(&number) {
            return Err(StoreError::InvalidArgument("part numbers are 1 to 10000".into()));
        }
        self.upload_exists(bucket, key, upload)?;
        let md5 = Md5::digest(body);
        let etag = etag_of(&md5);
        let blob = self.ids.next();
        write_atomic(&self.dir.join("tmp"), &self.part_path(&blob), body)?;
        let old: Option<String> = {
            let mut db = self.db();
            let tx = db.transaction()?;
            let old = tx
                .query_row("SELECT blob FROM parts WHERE upload = ?1 AND number = ?2", params![upload, number], |r| {
                    r.get(0)
                })
                .optional()?;
            tx.execute(
                "INSERT OR REPLACE INTO parts (upload, number, size, etag, md5, blob) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![upload, number, body.len() as i64, etag, &md5[..], blob],
            )?;
            tx.commit()?;
            old
        };
        if let Some(old) = old {
            let _ = fs::remove_file(self.part_path(&old));
        }
        Ok(etag)
    }

    fn upload_exists(&self, bucket: &str, key: &str, upload: &str) -> Result<Option<String>, StoreError> {
        self.db()
            .query_row(
                "SELECT content_type FROM uploads WHERE id = ?1 AND bucket = ?2 AND key = ?3",
                params![upload, bucket, key],
                |r| r.get(0),
            )
            .optional()?
            .ok_or(StoreError::NoSuchUpload)
    }

    /// Assembles the listed parts into the object.
    ///
    /// # Errors
    ///
    /// [`StoreError::NoSuchUpload`], [`StoreError::InvalidPart`] and the
    /// errors of [`Store::put`].
    pub fn complete_upload(
        &self,
        bucket: &str,
        key: &str,
        upload: &str,
        parts: &[(u32, String)],
    ) -> Result<ObjectMeta, StoreError> {
        let content_type = self.upload_exists(bucket, key, upload)?;
        if parts.is_empty() || parts.windows(2).any(|w| w[0].0 >= w[1].0) {
            return Err(StoreError::InvalidPart("parts must be listed in ascending order".into()));
        }
        let stored: Vec<(u32, String, Vec<u8>, String)> = {
            let db = self.db();
            let mut stmt = db.prepare("SELECT number, etag, md5, blob FROM parts WHERE upload = ?1 ORDER BY number")?;
            let rows = stmt.query_map([upload], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?;
            rows.collect::<Result<_, _>>()?
        };
        let mut body = Vec::new();
        let mut md5s = Vec::new();
        for (number, etag) in parts {
            let found = stored
                .iter()
                .find(|p| p.0 == *number)
                .ok_or_else(|| StoreError::InvalidPart(format!("part {number} was not uploaded")))?;
            if found.1.trim_matches('"') != etag.trim_matches('"') {
                return Err(StoreError::InvalidPart(format!("part {number} has a different ETag")));
            }
            body.extend_from_slice(&fs::read(self.part_path(&found.3))?);
            md5s.extend_from_slice(&found.2);
        }
        let etag = format!(
            "\"{}-{}\"",
            Md5::digest(&md5s).iter().map(|b| format!("{b:02x}")).collect::<String>(),
            parts.len()
        );
        let meta = self.put_with_etag(bucket, key, &body, content_type.as_deref(), etag)?;
        self.abort_upload(bucket, key, upload)?;
        Ok(meta)
    }

    /// Discards a multipart upload and its parts.
    ///
    /// # Errors
    ///
    /// [`StoreError::NoSuchUpload`], index errors.
    pub fn abort_upload(&self, bucket: &str, key: &str, upload: &str) -> Result<(), StoreError> {
        self.upload_exists(bucket, key, upload)?;
        let blobs: Vec<String> = {
            let mut db = self.db();
            let tx = db.transaction()?;
            let blobs = {
                let mut stmt = tx.prepare("SELECT blob FROM parts WHERE upload = ?1")?;
                let rows = stmt.query_map([upload], |r| r.get(0))?;
                rows.collect::<Result<_, _>>()?
            };
            tx.execute("DELETE FROM uploads WHERE id = ?1", [upload])?;
            tx.commit()?;
            blobs
        };
        for b in blobs {
            let _ = fs::remove_file(self.part_path(&b));
        }
        Ok(())
    }
}

#[derive(Default)]
struct Released {
    blob: Option<String>,
    archive: Option<(i64, String)>,
}
