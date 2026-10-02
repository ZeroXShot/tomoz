//! Packing and restoring synthetic series.

#![allow(clippy::unwrap_used)] // Helpers of tests panic on unexpected errors.

use tomoz_archive::{Archive, InstanceKind, PackOptions, StoredReason, pack};
use tomoz_codec::{EncodeOptions, ModelSet};

fn models() -> &'static ModelSet {
    static M: std::sync::OnceLock<ModelSet> = std::sync::OnceLock::new();
    M.get_or_init(ModelSet::untrained)
}

/// Minimal explicit VR little endian writer.
struct Ds(Vec<u8>);

impl Ds {
    fn el(&mut self, g: u16, e: u16, vr: &[u8; 2], v: &[u8]) -> &mut Self {
        let mut v = v.to_vec();
        if v.len() % 2 == 1 {
            v.push(if vr == b"UI" { 0 } else { b' ' });
        }
        self.0.extend_from_slice(&g.to_le_bytes());
        self.0.extend_from_slice(&e.to_le_bytes());
        self.0.extend_from_slice(vr);
        if matches!(vr, b"OB" | b"OW" | b"SQ" | b"UN") {
            self.0.extend_from_slice(&[0, 0]);
            self.0.extend_from_slice(&(v.len() as u32).to_le_bytes());
        } else {
            self.0.extend_from_slice(&(v.len() as u16).to_le_bytes());
        }
        self.0.extend_from_slice(&v);
        self
    }
    fn us(&mut self, g: u16, e: u16, v: u16) -> &mut Self {
        self.el(g, e, b"US", &v.to_le_bytes())
    }
}

fn file(ts: &str, body: impl FnOnce(&mut Ds)) -> Vec<u8> {
    let mut meta = Ds(Vec::new());
    meta.el(0x0002, 0x0010, b"UI", ts.as_bytes());
    let mut out = vec![0u8; 128];
    out.extend_from_slice(b"DICM");
    let mut gl = Ds(Vec::new());
    gl.el(0x0002, 0x0000, b"UL", &(meta.0.len() as u32).to_le_bytes());
    out.extend_from_slice(&gl.0);
    out.extend_from_slice(&meta.0);
    let mut ds = Ds(Vec::new());
    body(&mut ds);
    out.extend_from_slice(&ds.0);
    out
}

fn ct_slice(series: &str, instance: usize, z: f64, frames: u16, high_garbage: bool) -> Vec<u8> {
    file("1.2.840.10008.1.2.1", |d| {
        d.el(0x0008, 0x0018, b"UI", format!("1.2.3.{series}.{instance}").as_bytes())
            .el(0x0008, 0x0060, b"CS", b"CT")
            .el(0x0010, 0x0010, b"PN", format!("PATIENT^{instance}").as_bytes())
            .el(0x0020, 0x000E, b"UI", series.as_bytes())
            .el(0x0020, 0x0013, b"IS", instance.to_string().as_bytes())
            .el(0x0020, 0x0032, b"DS", format!("-100\\-120\\{z}").as_bytes())
            .el(0x0020, 0x0037, b"DS", b"1\\0\\0\\0\\1\\0")
            .us(0x0028, 0x0002, 1)
            .el(0x0028, 0x0004, b"CS", b"MONOCHROME2");
        if frames > 1 {
            d.el(0x0028, 0x0008, b"IS", frames.to_string().as_bytes());
        }
        d.us(0x0028, 0x0010, 20)
            .us(0x0028, 0x0011, 24)
            .us(0x0028, 0x0100, 16)
            .us(0x0028, 0x0101, 12)
            .us(0x0028, 0x0102, 11)
            .us(0x0028, 0x0103, 1);
        let mut px = Vec::new();
        for f in 0..frames as i32 {
            for y in 0..20i32 {
                for x in 0..24i32 {
                    let v = if x < 4 { -2000 } else { (x * 30 + y * 7 + (z as i32) * 11 + f * 3) % 2000 - 1000 };
                    let mut cell = v as i16 as u16 & 0x0FFF;
                    if v < 0 {
                        cell |= 0xF000; // sign extension
                    }
                    if high_garbage && (x + y) % 5 == 0 {
                        cell ^= 0x4000;
                    }
                    px.extend_from_slice(&cell.to_le_bytes());
                }
            }
        }
        d.el(0x7FE0, 0x0010, b"OW", &px);
        d.el(0xFFFC, 0xFFFC, b"OB", &[0; 4]);
    })
}

/// A projection image (no position or orientation), such as one view of a
/// mammography series.
fn projection_image(series: &str, instance: usize) -> Vec<u8> {
    file("1.2.840.10008.1.2.1", |d| {
        d.el(0x0008, 0x0018, b"UI", format!("1.2.3.{series}.{instance}").as_bytes())
            .el(0x0008, 0x0060, b"CS", b"MG")
            .el(0x0020, 0x000E, b"UI", series.as_bytes())
            .el(0x0020, 0x0013, b"IS", instance.to_string().as_bytes())
            .us(0x0028, 0x0002, 1)
            .el(0x0028, 0x0004, b"CS", b"MONOCHROME2")
            .us(0x0028, 0x0010, 16)
            .us(0x0028, 0x0011, 12)
            .us(0x0028, 0x0100, 16)
            .us(0x0028, 0x0101, 14)
            .us(0x0028, 0x0102, 13)
            .us(0x0028, 0x0103, 0);
        let px: Vec<u8> =
            (0..16 * 12).flat_map(|i| (((i * 97 + instance * 4001) % 16384) as u16).to_le_bytes()).collect();
        d.el(0x7FE0, 0x0010, b"OW", &px);
    })
}

fn pack_and_restore(files: &[(String, Vec<u8>)]) -> (Vec<u8>, tomoz_archive::PackReport) {
    let inputs: Vec<(String, &[u8])> = files.iter().map(|(n, b)| (n.clone(), b.as_slice())).collect();
    let (bytes, report) = pack(&inputs, &PackOptions::new(EncodeOptions::with_models(models().clone()))).unwrap();
    let archive = Archive::open(&bytes).unwrap();
    let restored = archive.restore_all(models()).unwrap();
    assert_eq!(restored.len(), files.len());
    for (i, ((name, original), back)) in files.iter().zip(&restored).enumerate() {
        assert_eq!(back, original, "{name}");
        assert_eq!(&archive.restore(i, models()).unwrap(), original, "{name} alone");
    }
    (bytes, report)
}

#[test]
fn series_with_mixed_content() {
    // Slices in scrambled order, a second series, a multi-frame instance, a
    // slice with overlay bits, an encapsulated file and a non-DICOM file.
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    for (i, z) in [3.0, 0.0, 4.0, 1.0, 2.0].into_iter().enumerate() {
        files.push((format!("a/{i}.dcm"), ct_slice("9.1", i, z, 1, i == 2)));
    }
    files.push(("b/0.dcm".into(), ct_slice("9.2", 0, 10.0, 1, false)));
    files.push(("c/mf.dcm".into(), ct_slice("9.3", 0, 0.0, 3, false)));
    files.push((
        "d/jpeg.dcm".into(),
        file("1.2.840.10008.1.2.4.70", |d| {
            d.el(0x0008, 0x0018, b"UI", b"1.2.3.4");
            d.0.extend_from_slice(&[0xE0, 0x7F, 0x10, 0x00, b'O', b'B', 0, 0, 0xFF, 0xFF, 0xFF, 0xFF]);
            d.0.extend_from_slice(&[0xFE, 0xFF, 0x00, 0xE0, 0, 0, 0, 0]);
            d.0.extend_from_slice(&[0xFE, 0xFF, 0x00, 0xE0, 4, 0, 0, 0, 1, 2, 3, 4]);
            d.0.extend_from_slice(&[0xFE, 0xFF, 0xDD, 0xE0, 0, 0, 0, 0]);
        }),
    ));
    files.push(("e/readme.txt".into(), b"not a DICOM file".to_vec()));
    let (bytes, report) = pack_and_restore(&files);
    assert_eq!(report.instances, 9);
    assert_eq!(report.coded, 7);
    assert_eq!(report.stacks, 3);
    assert_eq!(report.stored.get(&StoredReason::Encapsulated), Some(&1));
    assert!(report.output_bytes < report.input_bytes);
    let archive = Archive::open(&bytes).unwrap();
    // The five slices of series 9.1 share a stack, ordered by position.
    let slices: Vec<(u32, u32)> = archive.instances()[..5]
        .iter()
        .map(|e| match e.kind {
            InstanceKind::Coded { stack, first_slice, .. } => (stack, first_slice),
            InstanceKind::Stored { .. } => panic!("slice stored"),
        })
        .collect();
    assert!(slices.iter().all(|s| s.0 == slices[0].0));
    assert_eq!(slices.iter().map(|s| s.1).collect::<Vec<_>>(), vec![3, 0, 4, 1, 2]);
}

#[test]
fn projection_images_are_coded_independently() {
    // Views of one series without positions share a stack, but no slice is
    // predicted from another: one tile per slice.
    let views: Vec<(String, Vec<u8>)> = (0..3).map(|i| (format!("view{i}.dcm"), projection_image("9.9", i))).collect();
    let (bytes, report) = pack_and_restore(&views);
    assert_eq!(report.stacks, 1);
    let header = tomoz_codec::inspect(Archive::open(&bytes).unwrap().stack(0)).unwrap();
    assert_eq!((header.depth, header.slab), (3, 1));
    // Slices with positions keep inter-slice prediction.
    let ct: Vec<(String, Vec<u8>)> =
        (0..3).map(|i| (format!("ct{i}.dcm"), ct_slice("8.8", i, i as f64, 1, false))).collect();
    let (bytes, _) = pack_and_restore(&ct);
    assert!(tomoz_codec::inspect(Archive::open(&bytes).unwrap().stack(0)).unwrap().slab > 1);
}

#[test]
fn corruption_never_restores_wrong_bytes() {
    let files: Vec<(String, Vec<u8>)> =
        (0..4).map(|i| (format!("{i}.dcm"), ct_slice("7.7", i, i as f64, 1, false))).collect();
    let (bytes, _) = pack_and_restore(&files);
    for i in (0..bytes.len()).step_by(11) {
        let mut b = bytes.clone();
        b[i] ^= 0xA5;
        if let Ok(a) = Archive::open(&b)
            && let Ok(restored) = a.restore_all(models())
        {
            for (r, (_, f)) in restored.iter().zip(&files) {
                assert_eq!(r, f, "flip at {i} restored different bytes");
            }
        }
    }
}

#[test]
fn empty_input() {
    let (bytes, report) = pack_and_restore(&[]);
    assert_eq!(report.instances, 0);
    assert_eq!(Archive::open(&bytes).unwrap().instances().len(), 0);
}
