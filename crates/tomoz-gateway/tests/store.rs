//! Store behaviour: compaction, concurrent overwrites, crash leftovers.

#![allow(clippy::unwrap_used)] // Helpers of tests panic on unexpected errors.

use std::sync::Arc;

use tomoz_codec::ModelSet;
use tomoz_gateway::metrics::Metrics;
use tomoz_gateway::store::{Store, StoreError, StoreOptions};

fn models() -> ModelSet {
    ModelSet::untrained()
}

fn open(dir: &std::path::Path) -> Store {
    let options = StoreOptions { dir: dir.to_path_buf(), zstd_level: 3, slab: 4, cache_bytes: 64 << 20 };
    Store::open(&options, models(), Arc::new(Metrics::default())).unwrap()
}

/// A tiny explicit VR little endian CT slice of series `series`.
fn slice(series: &str, n: u16) -> Vec<u8> {
    let mut ds = Vec::new();
    let mut el = |g: u16, e: u16, vr: &[u8; 2], v: &[u8]| {
        let mut v = v.to_vec();
        if v.len() % 2 == 1 {
            v.push(if vr == b"UI" { 0 } else { b' ' });
        }
        ds.extend_from_slice(&g.to_le_bytes());
        ds.extend_from_slice(&e.to_le_bytes());
        ds.extend_from_slice(vr);
        if vr == b"OW" {
            ds.extend_from_slice(&[0, 0]);
            ds.extend_from_slice(&(v.len() as u32).to_le_bytes());
        } else {
            ds.extend_from_slice(&(v.len() as u16).to_le_bytes());
        }
        ds.extend_from_slice(&v);
    };
    el(0x0008, 0x0018, b"UI", format!("1.2.{series}.{n}").as_bytes());
    el(0x0020, 0x000E, b"UI", series.as_bytes());
    el(0x0020, 0x0013, b"IS", n.to_string().as_bytes());
    el(0x0020, 0x0032, b"DS", format!("0\\0\\{n}").as_bytes());
    el(0x0020, 0x0037, b"DS", b"1\\0\\0\\0\\1\\0");
    for (e, v) in [(0x0002u16, 1u16), (0x0010, 16), (0x0011, 16), (0x0100, 16), (0x0101, 12), (0x0102, 11), (0x0103, 0)]
    {
        el(0x0028, e, b"US", &v.to_le_bytes());
    }
    let px: Vec<u8> = (0..256u16).flat_map(|i| ((i * 7 + n * 13) % 4096).to_le_bytes()).collect();
    el(0x7FE0, 0x0010, b"OW", &px);
    let mut out = vec![0u8; 128];
    out.extend_from_slice(b"DICM");
    let ts = b"1.2.840.10008.1.2.1\0";
    out.extend_from_slice(&[0x02, 0x00, 0x10, 0x00, b'U', b'I']);
    out.extend_from_slice(&(ts.len() as u16).to_le_bytes());
    out.extend_from_slice(ts);
    out.extend_from_slice(&ds);
    out
}

#[test]
fn compaction_preserves_every_object() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    store.create_bucket("pacs").unwrap();
    let files: Vec<(String, Vec<u8>)> = (0..10).map(|i| (format!("s/{i:02}.dcm"), slice("9.9", i))).collect();
    for (k, d) in &files {
        store.put("pacs", k, d, Some("application/dicom")).unwrap();
    }
    store.put("pacs", "notes.txt", b"not dicom", None).unwrap();
    let due = store.due(0, 2).unwrap();
    assert_eq!(due.len(), 1, "one DICOM series is due; the text object is never compacted");
    let c = store.compact(&due[0]).unwrap();
    assert_eq!(c.objects, 10);
    assert!(c.archive_bytes < c.input_bytes);
    for (k, d) in &files {
        let (meta, data) = store.get("pacs", k).unwrap();
        assert!(meta.archived);
        assert_eq!(&*data, d, "{k}");
    }
    assert!(store.due(0, 2).unwrap().is_empty());
    // Overwrite and delete archived objects; the archive goes once empty.
    store.put("pacs", "s/00.dcm", b"replaced", None).unwrap();
    assert_eq!(&*store.get("pacs", "s/00.dcm").unwrap().1, b"replaced");
    for (k, _) in &files[1..] {
        store.delete("pacs", k).unwrap();
    }
    let archives = std::fs::read_dir(dir.path().join("archives"))
        .unwrap()
        .flat_map(|d| std::fs::read_dir(d.unwrap().path()).unwrap())
        .count();
    assert_eq!(archives, 0, "an archive without live objects is deleted");
    assert!(matches!(store.get("pacs", "s/05.dcm"), Err(StoreError::NoSuchKey)));
}

#[test]
fn objects_changed_during_compaction_keep_their_new_version() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(open(dir.path()));
    store.create_bucket("pacs").unwrap();
    for i in 0..8 {
        store.put("pacs", &format!("k{i}"), &slice("4.4", i), None).unwrap();
    }
    let series = store.due(0, 2).unwrap().remove(0);
    // Race a writer against the compactor many times; whatever the
    // interleaving, the newest version of every object must win.
    let writer = {
        let s = store.clone();
        std::thread::spawn(move || {
            for round in 0..20u16 {
                s.put("pacs", "k3", &slice("4.4", 100 + round), None).unwrap();
            }
        })
    };
    let _ = store.compact(&series).unwrap();
    writer.join().unwrap();
    assert_eq!(&*store.get("pacs", "k3").unwrap().1, &slice("4.4", 119));
    for i in [0u16, 1, 2, 4, 5, 6, 7] {
        assert_eq!(&*store.get("pacs", &format!("k{i}")).unwrap().1, &slice("4.4", i));
    }
}

#[test]
fn reads_racing_compaction_and_deletes_never_fail() {
    // A read looks up the index, then reads a file that a concurrent
    // compaction, overwrite or delete may remove. Readers must see either
    // version or NoSuchKey, never an I/O error.
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(open(dir.path()));
    store.create_bucket("pacs").unwrap();
    for round in 0..6u16 {
        let series = format!("7.{round}");
        let originals: Vec<Vec<u8>> = (0..8).map(|i| slice(&series, i)).collect();
        for (i, d) in originals.iter().enumerate() {
            store.put("pacs", &format!("r{round}/k{i}"), d, None).unwrap();
        }
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..3)
            .map(|t| {
                let (s, stop, originals) = (store.clone(), stop.clone(), originals.clone());
                std::thread::spawn(move || {
                    let mut reads = 0;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) || reads == 0 {
                        for (i, original) in originals.iter().enumerate() {
                            match s.get("pacs", &format!("r{round}/k{i}")) {
                                Ok((_, data)) => assert!(&*data == original || &*data == b"new", "reader {t}"),
                                Err(StoreError::NoSuchKey) => {}
                                Err(e) => panic!("reader {t}, object {i}: {e}"),
                            }
                            reads += 1;
                        }
                    }
                })
            })
            .collect();
        let due = store.due(0, 2).unwrap();
        let target = due.iter().find(|d| d.ends_with(&series)).unwrap().clone();
        store.compact(&target).unwrap();
        store.put("pacs", &format!("r{round}/k1"), b"new", None).unwrap();
        store.delete("pacs", &format!("r{round}/k2")).unwrap();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
    }
}

#[test]
fn reads_racing_overwrites_see_a_whole_version() {
    // Every overwrite deletes the file the previous version lived in, so a
    // reader that looked the key up just before reads a missing file unless
    // it looks again.
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(open(dir.path()));
    store.create_bucket("pacs").unwrap();
    let versions: Vec<Vec<u8>> = (0..2).map(|i| slice("6.6", i)).collect();
    store.put("pacs", "hot", &versions[0], None).unwrap();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let readers: Vec<_> = (0..3)
        .map(|_| {
            let (s, stop, versions) = (store.clone(), stop.clone(), versions.clone());
            std::thread::spawn(move || {
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let (_, data) = s.get("pacs", "hot").unwrap();
                    assert!(versions.contains(&data));
                }
            })
        })
        .collect();
    for i in 0..1000 {
        store.put("pacs", "hot", &versions[i % 2], None).unwrap();
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for r in readers {
        r.join().unwrap();
    }
}

#[test]
fn leftovers_of_a_crash_are_removed_and_listing_pages() {
    let dir = tempfile::tempdir().unwrap();
    {
        let store = open(dir.path());
        store.create_bucket("b01").unwrap();
        for i in 0..25 {
            store.put("b01", &format!("dir{}/obj{i:02}", i % 3), b"x", None).unwrap();
        }
    }
    // Files no index entry refers to, as a crash would leave them.
    std::fs::create_dir_all(dir.path().join("raw/ff")).unwrap();
    std::fs::write(dir.path().join("raw/ff/ffffffffffffffffffffffffffffffff"), b"orphan").unwrap();
    std::fs::write(dir.path().join("tmp/partial.tmp"), b"partial").unwrap();
    let store = open(dir.path());
    assert!(!dir.path().join("raw/ff/ffffffffffffffffffffffffffffffff").exists());
    assert!(!dir.path().join("tmp/partial.tmp").exists());
    let mut keys = Vec::new();
    let mut after: Option<String> = None;
    loop {
        let page = store.list("b01", "dir1/", None, after.as_deref(), 3).unwrap();
        keys.extend(page.objects.iter().map(|o| o.key.clone()));
        match page.next {
            Some(n) => after = Some(n),
            None => break,
        }
    }
    assert_eq!(keys.len(), 8);
    assert!(keys.windows(2).all(|w| w[0] < w[1]));
    let top = store.list("b01", "", Some("/"), None, 2).unwrap();
    assert_eq!(top.prefixes, vec!["dir0/", "dir1/"]);
    let rest = store.list("b01", "", Some("/"), top.next.as_deref(), 10).unwrap();
    assert_eq!(rest.prefixes, vec!["dir2/"]);
    assert!(rest.next.is_none());
}

#[test]
fn multipart_upload() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    store.create_bucket("mp1").unwrap();
    let id = store.create_upload("mp1", "big", Some("application/octet-stream")).unwrap();
    let e1 = store.upload_part("mp1", "big", &id, 1, &[1; 1000]).unwrap();
    let e2 = store.upload_part("mp1", "big", &id, 2, &[2; 10]).unwrap();
    assert!(store.complete_upload("mp1", "big", &id, &[(2, e2.clone()), (1, e1.clone())]).is_err());
    let meta = store.complete_upload("mp1", "big", &id, &[(1, e1), (2, e2)]).unwrap();
    assert!(meta.etag.ends_with("-2\""));
    let (_, data) = store.get("mp1", "big").unwrap();
    assert_eq!(data.len(), 1010);
    assert!(matches!(store.abort_upload("mp1", "big", &id), Err(StoreError::NoSuchUpload)));
}
