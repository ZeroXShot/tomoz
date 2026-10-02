//! Packing DICOM files into an archive.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};
use tomoz_codec::{EncodeOptions, Volume};
use tomoz_dicom::{DatasetEncoding, ParsedFile, PixelLayout, Samples};

use crate::format::{FIXED, InstanceEntry, InstanceKind};
use crate::{Error, MAGIC, VERSION};

/// Settings of [`pack`].
#[derive(Clone, Debug)]
pub struct PackOptions {
    /// Settings of the volume codec.
    pub codec: EncodeOptions,
    /// zstd level for headers and stored files.
    pub zstd_level: i32,
}

impl PackOptions {
    /// Default settings with the given codec settings.
    #[must_use]
    pub fn new(codec: EncodeOptions) -> Self {
        Self { codec, zstd_level: 19 }
    }
}

/// Why an instance was stored whole instead of coded.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum StoredReason {
    /// The file could not be parsed as DICOM.
    Unparsable(String),
    /// The file has no Pixel Data element.
    NoPixelData,
    /// The pixel data is already compressed.
    Encapsulated,
    /// The data set is deflate-compressed.
    Deflated,
    /// Float or double float pixel data.
    FloatPixels,
    /// A layout Tomoz does not code (colour, 1 or 32 bits, ...).
    Layout(String),
}

/// What [`pack`] did.
#[derive(Clone, Debug, Default)]
pub struct PackReport {
    /// Number of instances.
    pub instances: usize,
    /// Instances whose pixel data was coded.
    pub coded: usize,
    /// Instances stored whole, by reason.
    pub stored: BTreeMap<StoredReason, usize>,
    /// Number of stacks.
    pub stacks: usize,
    /// Total size of the input files.
    pub input_bytes: u64,
    /// Size of the archive.
    pub output_bytes: u64,
    /// Bytes of the coded stacks.
    pub stack_bytes: u64,
    /// Bytes of the compressed metadata.
    pub metadata_bytes: u64,
    /// Bytes of the stored files.
    pub stored_bytes: u64,
}

/// A codable instance waiting for its stack.
struct Codable {
    index: usize,
    layout: PixelLayout,
    samples: Samples,
    prefix: std::ops::Range<usize>,
    suffix: std::ops::Range<usize>,
    order: (i64, i64, String),
}

/// Grouping key: instances of one stack share a series and a geometry.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct StackKey {
    series: String,
    rows: u32,
    columns: u32,
    bits_stored: u8,
    signed: bool,
    orientation: Option<[i64; 6]>,
    /// Multi-frame instances get a stack of their own.
    multiframe: Option<usize>,
}

fn classify(bytes: &[u8]) -> Result<(ParsedFile, PixelLayout, Samples), StoredReason> {
    let parsed = tomoz_dicom::parse(bytes).map_err(|e| StoredReason::Unparsable(e.to_string()))?;
    if parsed.encoding == DatasetEncoding::Deflated {
        return Err(StoredReason::Deflated);
    }
    if parsed.has_float_pixels {
        return Err(StoredReason::FloatPixels);
    }
    let span = parsed.pixel.ok_or(StoredReason::NoPixelData)?;
    if span.encapsulated {
        return Err(StoredReason::Encapsulated);
    }
    let layout = PixelLayout::from_attributes(&parsed.attributes, parsed.encoding)
        .map_err(|e| StoredReason::Layout(e.to_string()))?;
    let samples =
        layout.extract(&bytes[span.value_start..span.value_end]).map_err(|e| StoredReason::Layout(e.to_string()))?;
    Ok((parsed, layout, samples))
}

/// Position of a slice along its normal, in 1/10000 mm, if known.
fn slice_position(p: &ParsedFile) -> Option<i64> {
    let o = p.attributes.image_orientation?;
    let pos = p.attributes.image_position?;
    let n = [o[1] * o[5] - o[2] * o[4], o[2] * o[3] - o[0] * o[5], o[0] * o[4] - o[1] * o[3]];
    let d = n[0] * pos[0] + n[1] * pos[1] + n[2] * pos[2];
    d.is_finite().then(|| (d * 1e4).round() as i64)
}

/// Packs `files` (name, bytes) into an archive.
///
/// # Errors
///
/// Codec and zstd failures, or inputs beyond the limits of the format.
pub fn pack(files: &[(String, &[u8])], options: &PackOptions) -> Result<(Vec<u8>, PackReport), Error> {
    let mut report = PackReport { instances: files.len(), ..PackReport::default() };
    let mut entries: Vec<Option<InstanceEntry>> = vec![None; files.len()];
    let mut stored_blob = Vec::new();
    let mut groups: BTreeMap<StackKey, Vec<Codable>> = BTreeMap::new();

    for (index, (name, bytes)) in files.iter().enumerate() {
        report.input_bytes += bytes.len() as u64;
        let sha256: [u8; 32] = Sha256::digest(bytes).into();
        match classify(bytes) {
            Ok((parsed, layout, samples)) => {
                let span = parsed.pixel.expect("classified files have pixel data");
                let a = &parsed.attributes;
                let key = StackKey {
                    series: a.series_instance_uid.clone().unwrap_or_default(),
                    rows: layout.rows,
                    columns: layout.columns,
                    bits_stored: layout.bits_stored,
                    signed: layout.signed,
                    orientation: a.image_orientation.map(|o| o.map(|v| (v * 1e3).round() as i64)),
                    multiframe: (layout.frames > 1).then_some(index),
                };
                let order = (
                    slice_position(&parsed).unwrap_or(i64::MIN),
                    a.instance_number.unwrap_or(0),
                    a.sop_instance_uid.clone().unwrap_or_default(),
                );
                entries[index] = Some(InstanceEntry {
                    name: name.clone(),
                    size: bytes.len() as u64,
                    sha256,
                    kind: InstanceKind::Stored { offset: 0, length: 0 },
                });
                groups.entry(key).or_default().push(Codable {
                    index,
                    layout,
                    samples,
                    prefix: 0..span.value_start,
                    suffix: span.value_end..bytes.len(),
                    order,
                });
            }
            Err(reason) => {
                let frame = zstd::bulk::compress(bytes, options.zstd_level).map_err(|e| Error::Zstd(e.to_string()))?;
                entries[index] = Some(InstanceEntry {
                    name: name.clone(),
                    size: bytes.len() as u64,
                    sha256,
                    kind: InstanceKind::Stored { offset: stored_blob.len() as u64, length: frame.len() as u64 },
                });
                stored_blob.extend_from_slice(&frame);
                *report.stored.entry(reason).or_default() += 1;
            }
        }
    }

    let mut stacks: Vec<Vec<u8>> = Vec::new();
    let mut metadata = Vec::new();
    for (key, mut members) in groups {
        members.sort_by(|a, b| a.order.cmp(&b.order).then(a.index.cmp(&b.index)));
        let depth: usize = members.iter().map(|m| m.layout.frames as usize).sum();
        let mut values = Vec::with_capacity(depth * key.rows as usize * key.columns as usize);
        let stack = u32::try_from(stacks.len()).map_err(|_| Error::TooLarge("stacks"))?;
        let mut slice = 0u32;
        for m in &members {
            values.extend_from_slice(&m.samples.values);
            let bytes = files[m.index].1;
            let meta_offset = metadata.len() as u64;
            metadata.extend_from_slice(&bytes[m.prefix.clone()]);
            metadata.extend_from_slice(&bytes[m.suffix.clone()]);
            metadata.extend_from_slice(&m.samples.trailing);
            if let Some(plane) = &m.samples.irregular {
                for v in plane {
                    metadata.extend_from_slice(&v.to_le_bytes());
                }
            }
            let entry = entries[m.index].as_mut().expect("entry recorded");
            entry.kind = InstanceKind::Coded {
                stack,
                first_slice: slice,
                layout: m.layout,
                convention: m.samples.convention,
                meta_offset,
                prefix: m.prefix.len() as u64,
                suffix: m.suffix.len() as u64,
                trailing: u32::try_from(m.samples.trailing.len()).map_err(|_| Error::TooLarge("trailing bytes"))?,
            };
            slice += m.layout.frames;
            report.coded += 1;
        }
        let volume = Volume::new(depth, key.rows as usize, key.columns as usize, key.bits_stored, key.signed, values)?;
        // Slices are predicted from the previous ones only when the stack is
        // known to be spatially coherent: one multi-frame instance, or
        // instances that all have a position along the normal. Projection
        // images (radiographs, mammograms: different views of one series)
        // are coded independently, one tile per slice.
        let coherent = members.len() == 1 || members.iter().all(|m| m.order.0 != i64::MIN);
        let codec = if coherent { options.codec.clone() } else { EncodeOptions { slab: 1, ..options.codec.clone() } };
        stacks.push(tomoz_codec::encode(&volume, &codec)?);
    }

    let metadata_frame = if metadata.is_empty() {
        Vec::new()
    } else {
        zstd::bulk::compress(&metadata, options.zstd_level).map_err(|e| Error::Zstd(e.to_string()))?
    };
    let mut table = Vec::new();
    let instance_count = u32::try_from(entries.len()).map_err(|_| Error::TooLarge("instances"))?;
    for e in entries.into_iter().flatten() {
        e.write(&mut table);
    }

    // Layout: fixed header, stack table, instance table, metadata, stored
    // files, stacks.
    let stack_table_len = 16 * stacks.len();
    let table_offset = (FIXED + stack_table_len) as u64;
    let metadata_offset = table_offset + table.len() as u64;
    let stored_offset = metadata_offset + metadata_frame.len() as u64;
    let mut stack_offset = stored_offset + stored_blob.len() as u64;
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&instance_count.to_le_bytes());
    out.extend_from_slice(&(stacks.len() as u32).to_le_bytes());
    for (o, l) in [
        (table_offset, table.len() as u64),
        (metadata_offset, metadata_frame.len() as u64),
        (stored_offset, stored_blob.len() as u64),
    ] {
        out.extend_from_slice(&o.to_le_bytes());
        out.extend_from_slice(&l.to_le_bytes());
    }
    out.extend_from_slice(&crc32c::crc32c(&table).to_le_bytes());
    let header_crc_pos = out.len();
    out.extend_from_slice(&0u32.to_le_bytes());
    for s in &stacks {
        out.extend_from_slice(&stack_offset.to_le_bytes());
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        stack_offset += s.len() as u64;
    }
    let crc = crc32c::crc32c_append(crc32c::crc32c(&out[..header_crc_pos]), &out[FIXED..]);
    out[header_crc_pos..header_crc_pos + 4].copy_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&table);
    out.extend_from_slice(&metadata_frame);
    out.extend_from_slice(&stored_blob);
    report.stacks = stacks.len();
    report.metadata_bytes = metadata_frame.len() as u64;
    report.stored_bytes = stored_blob.len() as u64;
    for s in stacks {
        report.stack_bytes += s.len() as u64;
        out.extend_from_slice(&s);
    }
    report.output_bytes = out.len() as u64;
    Ok((out, report))
}
