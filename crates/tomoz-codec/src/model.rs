//! Predictor models: the network, the head scale and the entropy coder
//! priors, in the TZM1 file format.

use std::sync::Arc;

use sha2::{Digest, Sha256};
use tomoz_entropy::PROB_ONE;
use tomoz_nn::Network;

use crate::Error;

/// Number of contexts of the residual token model.
pub const TOKEN_CONTEXTS: usize = 20;
/// Number of residual tokens.
pub const TOKENS: usize = 16;
/// Number of contexts of the sign model.
pub const SIGN_CONTEXTS: usize = 68;
/// Number of contexts of the flat-region flag.
pub const FLAT_CONTEXTS: usize = 4;

/// Network inputs of the 2-D model.
pub const INPUTS_2D: usize = 12;
/// Network inputs of the 3-D model.
pub const INPUTS_3D: usize = 30;
/// Network outputs of both models.
pub const OUTPUTS: usize = 6;

const MAGIC: &[u8; 4] = b"TZM1";
const MAX_NAME: usize = 64;

/// Which neighbourhood a model predicts from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelKind {
    /// Rows above in the current slice only.
    TwoD,
    /// Rows above and the two previous slices.
    ThreeD,
}

impl ModelKind {
    /// Number of network inputs.
    #[must_use]
    pub fn inputs(self) -> usize {
        match self {
            Self::TwoD => INPUTS_2D,
            Self::ThreeD => INPUTS_3D,
        }
    }

    fn code(self) -> u8 {
        match self {
            Self::TwoD => 2,
            Self::ThreeD => 3,
        }
    }
}

/// Content identifier of a model: the first 16 bytes of the SHA-256 of its
/// file. Bitstreams name their models by identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelId(pub [u8; 16]);

impl std::fmt::Debug for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

/// Initial state of the adaptive entropy models.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Priors {
    /// Cumulative frequencies of the residual tokens per context, in units of
    /// 2<sup>-15</sup>; `token[c][0] == 0`.
    pub token: [[u16; TOKENS]; TOKEN_CONTEXTS],
    /// Probability of a positive sign (bit zero) per context.
    pub sign: [u16; SIGN_CONTEXTS],
    /// Probability that a flat-region sample does not equal its left
    /// neighbour (bit zero means "equal"), per context.
    pub flat: [u16; FLAT_CONTEXTS],
}

impl Default for Priors {
    fn default() -> Self {
        let mut token = [[0u16; TOKENS]; TOKEN_CONTEXTS];
        for row in &mut token {
            for (i, v) in row.iter_mut().enumerate() {
                *v = (i as u32 * PROB_ONE / TOKENS as u32) as u16;
            }
        }
        let half = (PROB_ONE / 2) as u16;
        Self { token, sign: [half; SIGN_CONTEXTS], flat: [half; FLAT_CONTEXTS] }
    }
}

/// A validated predictor model.
#[derive(Clone, Debug)]
pub struct Model {
    id: ModelId,
    kind: ModelKind,
    name: String,
    out_shift: u8,
    network: Network,
    priors: Priors,
    bytes: Arc<[u8]>,
}

impl Model {
    /// Builds a model from its parts.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] if the network does not have the inputs and outputs of
    /// `kind`, the head shift is out of range or the priors are invalid.
    pub fn new(kind: ModelKind, name: &str, out_shift: u8, network: &Network, priors: &Priors) -> Result<Self, Error> {
        let bytes = encode(kind, name, out_shift, network, priors)?;
        Self::from_bytes(&bytes)
    }

    /// Decodes and validates a TZM1 model file.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] for malformed or inconsistent files.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let bad = |what: &str| Error::Model(what.to_owned());
        if bytes.len() < 9 || &bytes[..4] != MAGIC {
            return Err(bad("not a TZM1 model"));
        }
        let kind = match bytes[4] {
            2 => ModelKind::TwoD,
            3 => ModelKind::ThreeD,
            _ => return Err(bad("unknown model kind")),
        };
        let out_shift = bytes[5];
        if !(1..=30).contains(&out_shift) {
            return Err(bad("head shift out of range"));
        }
        if bytes[6] != 0 || bytes[7] != 0 {
            return Err(bad("unknown model flags"));
        }
        let name_len = usize::from(bytes[8]);
        let name_end = 9 + name_len;
        if name_len > MAX_NAME || bytes.len() < name_end {
            return Err(bad("invalid model name"));
        }
        let name = std::str::from_utf8(&bytes[9..name_end]).map_err(|_| bad("model name is not UTF-8"))?.to_owned();
        let (network, used) = Network::from_bytes(&bytes[name_end..]).map_err(|e| Error::Model(e.to_string()))?;
        if network.inputs() != kind.inputs() || network.outputs() != OUTPUTS {
            return Err(bad("network shape does not match the model kind"));
        }
        let mut r = &bytes[name_end + used..];
        let mut next = || -> Result<u16, Error> {
            let (v, rest) = r.split_first_chunk::<2>().ok_or_else(|| bad("truncated priors"))?;
            r = rest;
            Ok(u16::from_le_bytes(*v))
        };
        let mut priors = Priors::default();
        for row in &mut priors.token {
            for v in row.iter_mut() {
                *v = next()?;
            }
        }
        for v in &mut priors.sign {
            *v = next()?;
        }
        for v in &mut priors.flat {
            *v = next()?;
        }
        if !r.is_empty() {
            return Err(bad("trailing bytes after the priors"));
        }
        validate_priors(&priors).map_err(bad)?;
        let digest = Sha256::digest(bytes);
        let mut id = [0u8; 16];
        id.copy_from_slice(&digest[..16]);
        Ok(Self { id: ModelId(id), kind, name, out_shift, network, priors, bytes: bytes.into() })
    }

    /// Content identifier.
    #[must_use]
    pub fn id(&self) -> ModelId {
        self.id
    }

    /// Neighbourhood of the model.
    #[must_use]
    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Shift that converts network outputs to head units.
    #[must_use]
    pub fn out_shift(&self) -> u8 {
        self.out_shift
    }

    /// The network.
    #[must_use]
    pub fn network(&self) -> &Network {
        &self.network
    }

    /// Entropy coder priors.
    #[must_use]
    pub fn priors(&self) -> &Priors {
        &self.priors
    }

    /// The model file.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

fn validate_priors(p: &Priors) -> Result<(), &'static str> {
    for row in &p.token {
        if row[0] != 0 || row.windows(2).any(|w| w[0] > w[1]) || u32::from(row[TOKENS - 1]) > PROB_ONE {
            return Err("token priors are not a cumulative distribution");
        }
    }
    let valid = |v: u16| v >= 1 && u32::from(v) < PROB_ONE;
    if !p.sign.iter().chain(&p.flat).copied().all(valid) {
        return Err("binary priors out of range");
    }
    Ok(())
}

fn encode(kind: ModelKind, name: &str, out_shift: u8, network: &Network, priors: &Priors) -> Result<Vec<u8>, Error> {
    if name.len() > MAX_NAME {
        return Err(Error::Model("model name longer than 64 bytes".into()));
    }
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&[kind.code(), out_shift, 0, 0, name.len() as u8]);
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(&network.to_bytes());
    for v in priors.token.iter().flatten().chain(&priors.sign).chain(&priors.flat) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

/// The pair of models a volume is coded with.
#[derive(Clone, Debug)]
pub struct ModelSet {
    /// Model for the first slice of every tile and for 2-D images.
    pub two_d: Model,
    /// Model for the other slices.
    pub three_d: Model,
}

/// Resolves model identifiers found in bitstreams.
pub trait ModelRegistry: Sync {
    /// The model with identifier `id`, if known.
    fn get(&self, id: &ModelId) -> Option<Model>;
}

impl ModelRegistry for Vec<Model> {
    fn get(&self, id: &ModelId) -> Option<Model> {
        self.iter().find(|m| m.id() == *id).cloned()
    }
}

impl ModelRegistry for ModelSet {
    fn get(&self, id: &ModelId) -> Option<Model> {
        [&self.two_d, &self.three_d].into_iter().find(|m| m.id() == *id).cloned()
    }
}
