//! Learned, deterministic lossless compression of medical image volumes.
//!
//! Tomoz codes each sample of a volume with a prediction from a small
//! integer neural network that sees the causal neighbourhood of the sample
//! in its own slice and in the two previous slices. The network outputs the
//! coefficients of a local linear predictor and a scale; an adaptive range
//! coder codes the residual in a context chosen by that scale. Because the
//! network and every step around it run on integers with specified rounding
//! ([`tomoz_nn`]), the decoder reproduces the encoder's predictions exactly
//! on every platform.
//!
//! Volumes are cut into tiles (a slab of consecutive slices × a stripe of
//! rows) coded independently, which allows parallel coding and decoding a
//! slice without decoding the whole volume. Every tile carries checksums of
//! its stream and of its samples, and the container a SHA-256 of all
//! samples.
//!
//! ```
//! use tomoz_codec::{DecodeOptions, EncodeOptions, ModelSet, Volume, decode, encode};
//!
//! let samples: Vec<i32> = (0..4 * 32 * 32).map(|i| (i % 97) as i32).collect();
//! let volume = Volume::new(4, 32, 32, 12, false, samples)?;
//! let models = ModelSet::untrained();
//! let bytes = encode(&volume, &EncodeOptions::with_models(models.clone()))?;
//! let back = decode(&bytes, &DecodeOptions::with_registry(&models))?;
//! assert_eq!(back, volume);
//! # Ok::<(), tomoz_codec::Error>(())
//! ```

mod container;
mod context;
mod math;
mod model;
mod residual;
mod tile;
mod volume;

pub use container::Header;
pub use model::{
    FLAT_CONTEXTS, INPUTS_2D, INPUTS_3D, Model, ModelId, ModelKind, ModelRegistry, ModelSet, OUTPUTS, Priors,
    SIGN_CONTEXTS, TOKEN_CONTEXTS, TOKENS,
};
pub use volume::{MAX_BITS, Volume, value_range};

use container::{TileEntry, VERSION};
use tile::TileGeom;
use volume::Mapping;

/// Errors of the codec.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The data is not a Tomoz container.
    #[error("not a Tomoz container")]
    NotTomoz,
    /// The data ends before the container does.
    #[error("container is truncated")]
    Truncated,
    /// The container uses a feature this version does not support.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The container is internally inconsistent.
    #[error("corrupt container: {0}")]
    Corrupt(&'static str),
    /// A checksum does not match.
    #[error("checksum mismatch in {0}")]
    Checksum(String),
    /// The container names a model this decoder does not have.
    #[error("model {0} is not available")]
    UnknownModel(ModelId),
    /// A model file is invalid.
    #[error("invalid model: {0}")]
    Model(String),
    /// The volume passed to the encoder is invalid.
    #[error("invalid volume: {0}")]
    InvalidVolume(String),
    /// Memory for the decoded samples cannot be allocated.
    #[error("cannot allocate {0} samples")]
    TooLarge(usize),
}

/// Encoder settings. None of them affects the decoded samples; the tiling
/// trades compression for parallelism and random access.
#[derive(Clone, Debug)]
pub struct EncodeOptions {
    /// Models to code with.
    pub models: ModelSet,
    /// Slices per tile.
    pub slab: u16,
    /// Rows per tile.
    pub stripe: u16,
    /// Allow histogram packing of sparse value sets.
    pub packing: bool,
    /// Worker threads (`None`: one per core; `Some(1)`: the calling thread).
    pub threads: Option<usize>,
}

impl EncodeOptions {
    /// Default tiling (32 slices × 512 rows) with the given models.
    #[must_use]
    pub fn with_models(models: ModelSet) -> Self {
        Self { models, slab: 32, stripe: 512, packing: true, threads: None }
    }
}

/// Decoder settings.
#[derive(Clone, Copy)]
pub struct DecodeOptions<'a> {
    /// Where models named by the container are looked up.
    pub registry: &'a dyn ModelRegistry,
    /// Largest volume, in samples, the decoder accepts (default 2³⁴). The
    /// samples are held as `i32`, so callers decoding untrusted containers
    /// should set a limit that fits their memory: a few bytes of container
    /// can describe a large constant volume.
    pub max_samples: u64,
    /// Check the SHA-256 of the decoded samples (the per-tile checksums are
    /// always checked).
    pub verify_sha256: bool,
    /// Worker threads (`None`: one per core; `Some(1)`: the calling thread).
    pub threads: Option<usize>,
}

impl<'a> DecodeOptions<'a> {
    /// Default limits with the given model registry.
    #[must_use]
    pub fn with_registry(registry: &'a dyn ModelRegistry) -> Self {
        Self { registry, max_samples: container::DEFAULT_MAX_SAMPLES, verify_sha256: true, threads: None }
    }
}

fn tiles(depth: usize, height: usize, slab: usize, stripe: usize) -> Vec<(usize, usize, usize, usize)> {
    let mut out = Vec::new();
    for z0 in (0..depth).step_by(slab) {
        for y0 in (0..height).step_by(stripe) {
            out.push((z0, y0, slab.min(depth - z0), stripe.min(height - y0)));
        }
    }
    out
}

fn extract(samples: &[i32], height: usize, width: usize, g: TileGeom) -> Vec<i32> {
    let mut out = Vec::with_capacity(g.samples());
    for z in g.z0..g.z0 + g.depth {
        for y in g.y0..g.y0 + g.height {
            let start = (z * height + y) * width;
            out.extend_from_slice(&samples[start..start + width]);
        }
    }
    out
}

fn crc_of_samples(samples: &[i32], mapping: &Mapping) -> u32 {
    let mut crc = 0u32;
    for chunk in samples.chunks(4096) {
        let bytes: Vec<u8> = chunk
            .iter()
            .flat_map(|&v| {
                let original = match &mapping.table {
                    None => v + mapping.offset,
                    Some(t) => t[v as usize] + mapping.offset,
                };
                (original as u16).to_le_bytes()
            })
            .collect();
        crc = crc32c::crc32c_append(crc, &bytes);
    }
    crc
}

/// Runs `f` on every tile index, in parallel unless `threads == Some(1)`.
/// The result does not depend on the number of threads.
#[cfg(feature = "parallel")]
fn map_tiles<T: Send, F: Fn(usize) -> T + Sync + Send>(n: usize, threads: Option<usize>, f: F) -> Vec<T> {
    use rayon::prelude::*;
    match threads {
        Some(1) => (0..n).map(f).collect(),
        Some(t) => match rayon::ThreadPoolBuilder::new().num_threads(t).build() {
            Ok(pool) => pool.install(|| (0..n).into_par_iter().map(&f).collect()),
            Err(_) => (0..n).map(f).collect(),
        },
        None => (0..n).into_par_iter().map(f).collect(),
    }
}

#[cfg(not(feature = "parallel"))]
fn map_tiles<T, F: Fn(usize) -> T>(n: usize, _threads: Option<usize>, f: F) -> Vec<T> {
    (0..n).map(f).collect()
}

/// Compresses a volume.
///
/// # Errors
///
/// [`Error::InvalidVolume`] if the tiling is invalid.
pub fn encode(volume: &Volume, options: &EncodeOptions) -> Result<Vec<u8>, Error> {
    encode_with_metadata(volume, &[], options)
}

/// Compresses a volume and stores `metadata` (opaque bytes, at most 16 MiB)
/// in the container header, where [`inspect`] finds it.
///
/// # Errors
///
/// [`Error::InvalidVolume`] if the tiling is invalid or the metadata too
/// large.
pub fn encode_with_metadata(volume: &Volume, metadata: &[u8], options: &EncodeOptions) -> Result<Vec<u8>, Error> {
    if metadata.len() > 1 << 24 {
        return Err(Error::InvalidVolume("metadata larger than 16 MiB".into()));
    }
    if options.slab == 0 || options.stripe == 0 {
        return Err(Error::InvalidVolume("tile dimensions must be positive".into()));
    }
    let (d, h, w) = (volume.depth(), volume.height(), volume.width());
    if u32::try_from(d).is_err() || u32::try_from(h).is_err() || u32::try_from(w).is_err() {
        return Err(Error::InvalidVolume("dimension exceeds 2^32 - 1".into()));
    }
    let mapping = Mapping::analyze(volume.samples(), options.packing);
    let mapped = mapping.map(volume.samples());
    let max_value = i64::from(mapping.max_mapped);
    let layout = tiles(d, h, usize::from(options.slab), usize::from(options.stripe));
    let streams = map_tiles(layout.len(), options.threads, |i| {
        let (z0, y0, depth, height) = layout[i];
        let geom = TileGeom { z0, y0, depth, height, width: w };
        let region = extract(&mapped, h, w, geom);
        let stream = tile::encode(&region, geom, &options.models, max_value);
        let samples_crc = crc_of_samples(&region, &mapping);
        (stream, samples_crc)
    });
    let mut entries = Vec::with_capacity(streams.len());
    let mut offset = 0u64;
    for (stream, samples_crc) in &streams {
        let length = u32::try_from(stream.len()).map_err(|_| Error::InvalidVolume("tile larger than 4 GiB".into()))?;
        entries.push(TileEntry { offset, length, stream_crc: crc32c::crc32c(stream), samples_crc: *samples_crc });
        offset += u64::from(length);
    }
    let header = Header {
        version: VERSION,
        depth: d as u32,
        height: h as u32,
        width: w as u32,
        bits: volume.bits(),
        signed: volume.signed(),
        offset: mapping.offset,
        max_mapped: mapping.max_mapped,
        slab: options.slab,
        stripe: options.stripe,
        model_2d: options.models.two_d.id(),
        model_3d: options.models.three_d.id(),
        sha256: volume.sha256(),
        packing: mapping.encode_table(),
        metadata: metadata.to_vec(),
        tiles: entries,
        header_len: 0,
    };
    let mut out = header.write();
    for (stream, _) in streams {
        out.extend_from_slice(&stream);
    }
    Ok(out)
}

/// Predictions of every sample, tile by tile in raster order: `None` where
/// the flat-region flag alone codes the sample, else the predicted mean in
/// 1/16 units and the log2 scale in 1/8 octaves. For tests against the
/// reference implementation and for diagnostics.
#[doc(hidden)]
#[must_use]
pub fn trace_predictions(volume: &Volume, options: &EncodeOptions) -> Vec<Option<(i64, i64)>> {
    let (d, h, w) = (volume.depth(), volume.height(), volume.width());
    let mapping = Mapping::analyze(volume.samples(), options.packing);
    let mapped = mapping.map(volume.samples());
    let mut out = Vec::with_capacity(mapped.len());
    for (z0, y0, depth, height) in tiles(d, h, usize::from(options.slab), usize::from(options.stripe)) {
        let geom = TileGeom { z0, y0, depth, height, width: w };
        let region = extract(&mapped, h, w, geom);
        out.extend(tile::trace(&region, geom, &options.models, i64::from(mapping.max_mapped)));
    }
    out
}

/// Reads the header of a container without decoding it.
///
/// # Errors
///
/// As [`Header::parse`].
pub fn inspect(bytes: &[u8]) -> Result<Header, Error> {
    Header::parse(bytes, u64::MAX)
}

/// A container whose header has been checked and whose models are resolved.
struct Prepared<'a> {
    header: Header,
    models: ModelSet,
    mapping: Mapping,
    data: &'a [u8],
    layout: Vec<(usize, usize, usize, usize)>,
}

fn prepare<'a>(bytes: &'a [u8], options: &DecodeOptions<'_>) -> Result<Prepared<'a>, Error> {
    let header = Header::parse(bytes, options.max_samples)?;
    let models = ModelSet {
        two_d: options.registry.get(&header.model_2d).ok_or(Error::UnknownModel(header.model_2d))?,
        three_d: options.registry.get(&header.model_3d).ok_or(Error::UnknownModel(header.model_3d))?,
    };
    if models.two_d.kind() != ModelKind::TwoD || models.three_d.kind() != ModelKind::ThreeD {
        return Err(Error::Corrupt("model kinds do not match their roles"));
    }
    let mapping = if header.packed() {
        Mapping::decode_table(&header.packing, header.offset, header.max_mapped)?
    } else {
        Mapping { offset: header.offset, max_mapped: header.max_mapped, table: None }
    };
    let data = &bytes[header.header_len..];
    if (data.len() as u64) < header.data_len() {
        return Err(Error::Truncated);
    }
    let layout =
        tiles(header.depth as usize, header.height as usize, usize::from(header.slab), usize::from(header.stripe));
    Ok(Prepared { header, models, mapping, data, layout })
}

impl Prepared<'_> {
    /// Decodes the tiles of slabs `slabs` into the samples of slices
    /// `slabs.start * slab ..` (mapped values restored).
    fn decode_slabs(&self, slabs: std::ops::Range<usize>, threads: Option<usize>) -> Result<(usize, Vec<i32>), Error> {
        let (h, w) = (self.header.height as usize, self.header.width as usize);
        let stripes = self.header.stripes();
        let slab = usize::from(self.header.slab);
        let first = slabs.start * stripes;
        let count = (slabs.end - slabs.start) * stripes;
        let max_value = i64::from(self.header.max_mapped);
        let decoded = map_tiles(count, threads, |k| -> Result<Vec<i32>, Error> {
            let i = first + k;
            let entry = self.header.tiles[i];
            let stream = &self.data[entry.offset as usize..(entry.offset + u64::from(entry.length)) as usize];
            if crc32c::crc32c(stream) != entry.stream_crc {
                return Err(Error::Checksum(format!("stream of tile {i}")));
            }
            let (z0, y0, depth, height) = self.layout[i];
            let geom = TileGeom { z0, y0, depth, height, width: w };
            let mut out = Vec::new();
            tile::decode(stream, geom, &self.models, max_value, &mut out)?;
            if crc_of_samples(&out, &self.mapping) != entry.samples_crc {
                return Err(Error::Checksum(format!("samples of tile {i}")));
            }
            Ok(out)
        });
        let z_first = slabs.start * slab;
        let z_end = (slabs.end * slab).min(self.header.depth as usize);
        let n = (z_end - z_first) * h * w;
        let mut samples = Vec::new();
        samples.try_reserve_exact(n).map_err(|_| Error::TooLarge(n))?;
        samples.resize(n, 0);
        for (k, tile) in decoded.into_iter().enumerate() {
            let tile = tile?;
            let (z0, y0, _, height) = self.layout[first + k];
            for (j, row) in tile.chunks_exact(w).enumerate() {
                let (z, y) = (z0 + j / height - z_first, y0 + j % height);
                let start = (z * h + y) * w;
                samples[start..start + w].copy_from_slice(row);
            }
        }
        self.mapping.unmap_in_place(&mut samples);
        Ok((z_first, samples))
    }

    fn volume(&self, depth: usize, samples: Vec<i32>) -> Result<Volume, Error> {
        let h = &self.header;
        Volume::new(depth, h.height as usize, h.width as usize, h.bits, h.signed, samples)
            .map_err(|_| Error::Corrupt("decoded values outside the declared format"))
    }
}

/// Decompresses a volume.
///
/// # Errors
///
/// [`Error::UnknownModel`] if the registry lacks a model, [`Error::Checksum`]
/// if a checksum fails, and the errors of [`Header::parse`].
pub fn decode(bytes: &[u8], options: &DecodeOptions<'_>) -> Result<Volume, Error> {
    let p = prepare(bytes, options)?;
    let (_, samples) = p.decode_slabs(0..p.header.slabs(), options.threads)?;
    let volume = p.volume(p.header.depth as usize, samples)?;
    if options.verify_sha256 && volume.sha256() != p.header.sha256 {
        return Err(Error::Checksum("volume SHA-256".into()));
    }
    Ok(volume)
}

/// Decompresses slices `slices` of a volume, decoding only the tiles that
/// contain them. Tile checksums are verified; the SHA-256 of the whole volume
/// cannot be.
///
/// # Errors
///
/// [`Error::InvalidVolume`] for an empty or out-of-range slice range, and the
/// errors of [`decode`].
pub fn decode_slices(
    bytes: &[u8],
    options: &DecodeOptions<'_>,
    slices: std::ops::Range<usize>,
) -> Result<Volume, Error> {
    let p = prepare(bytes, options)?;
    if slices.start >= slices.end || slices.end > p.header.depth as usize {
        return Err(Error::InvalidVolume(format!("slices {slices:?} outside 0..{}", p.header.depth)));
    }
    let slab = usize::from(p.header.slab);
    let (z_first, samples) = p.decode_slabs(slices.start / slab..slices.end.div_ceil(slab), options.threads)?;
    let plane = p.header.height as usize * p.header.width as usize;
    let lo = (slices.start - z_first) * plane;
    let hi = (slices.end - z_first) * plane;
    p.volume(slices.len(), samples[lo..hi].to_vec())
}

impl ModelSet {
    /// The models released with this version of Tomoz (TZ1, in `models/`):
    /// trained on public CT, MR, PET and radiography collections that are
    /// disjoint from the evaluation data (see `models/README.md`).
    #[must_use]
    pub fn builtin() -> Self {
        static BUILTIN: std::sync::OnceLock<ModelSet> = std::sync::OnceLock::new();
        BUILTIN
            .get_or_init(|| Self {
                two_d: Model::from_bytes(include_bytes!("../models/tz1-2d.tzm"))
                    .expect("the built-in 2-D model is valid"),
                three_d: Model::from_bytes(include_bytes!("../models/tz1-3d.tzm"))
                    .expect("the built-in 3-D model is valid"),
            })
            .clone()
    }

    /// Models whose networks output zeros: every sample is predicted by the
    /// reference predictor with the local activity as scale. Useful for
    /// tests and as a baseline; real data should use trained models.
    #[must_use]
    pub fn untrained() -> Self {
        let make = |kind: ModelKind, name: &str| {
            let layers = vec![
                tomoz_nn::Layer {
                    inputs: kind.inputs(),
                    outputs: 16,
                    weights: vec![0; kind.inputs() * 16],
                    bias: vec![0; 16],
                    shift: 0,
                    activation: tomoz_nn::Activation::ClippedRelu,
                },
                tomoz_nn::Layer {
                    inputs: 16,
                    outputs: OUTPUTS,
                    weights: vec![0; 16 * OUTPUTS],
                    bias: vec![0; OUTPUTS],
                    shift: 0,
                    activation: tomoz_nn::Activation::Linear,
                },
            ];
            let net = tomoz_nn::Network::new(layers).expect("valid untrained network");
            Model::new(kind, name, 12, &net, &Priors::default()).expect("valid untrained model")
        };
        Self { two_d: make(ModelKind::TwoD, "untrained-2d"), three_d: make(ModelKind::ThreeD, "untrained-3d") }
    }
}
