//! Python bindings: `tomoz._tomoz`.
//!
//! Coding releases the GIL, so several threads can encode or decode at once.

#![allow(unsafe_code)] // PyO3's generated module glue.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use tomoz_codec::{DecodeOptions, EncodeOptions, Model, ModelSet, Volume};

/// A pair of predictor models (2-D and 3-D).
#[pyclass(name = "Models", module = "tomoz", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyModels {
    set: ModelSet,
}

#[pymethods]
impl PyModels {
    /// Loads models from TZM1 files given as bytes.
    #[new]
    fn new(model_2d: &[u8], model_3d: &[u8]) -> PyResult<Self> {
        let two_d = Model::from_bytes(model_2d).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let three_d = Model::from_bytes(model_3d).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { set: ModelSet { two_d, three_d } })
    }

    /// Identifiers of the 2-D and 3-D models.
    #[getter]
    fn ids(&self) -> (String, String) {
        (self.set.two_d.id().to_string(), self.set.three_d.id().to_string())
    }

    fn __repr__(&self) -> String {
        format!("Models(2d={}, 3d={})", self.set.two_d.name(), self.set.three_d.name())
    }
}

fn models(m: Option<&PyModels>) -> ModelSet {
    m.map_or_else(ModelSet::builtin, |m| m.set.clone())
}

fn registry(m: Option<&PyModels>) -> Vec<Model> {
    let builtin = ModelSet::builtin();
    let mut out = vec![builtin.two_d, builtin.three_d];
    if let Some(m) = m {
        out.push(m.set.two_d.clone());
        out.push(m.set.three_d.clone());
    }
    out
}

#[allow(clippy::needless_pass_by_value)] // Used with `map_err`.
fn codec_err(e: tomoz_codec::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Converts a 2-D or 3-D integer array to a volume.
fn to_volume(array: &Bound<'_, PyAny>, bits: Option<u8>) -> PyResult<Volume> {
    let shape = |a: &[usize]| -> PyResult<(usize, usize, usize)> {
        match *a {
            [h, w] => Ok((1, h, w)),
            [z, h, w] => Ok((z, h, w)),
            _ => Err(PyValueError::new_err("expected a 2-D (rows, columns) or 3-D (slices, rows, columns) array")),
        }
    };
    macro_rules! convert {
        ($t:ty, $width:expr, $signed:expr) => {
            if let Ok(a) = array.extract::<PyReadonlyArrayDyn<'_, $t>>() {
                let (d, h, w) = shape(a.shape())?;
                let samples: Vec<i32> = a.as_array().iter().map(|&v| i32::from(v)).collect();
                let bits = bits.unwrap_or($width);
                return Volume::new(d, h, w, bits, $signed, samples).map_err(codec_err);
            }
        };
    }
    convert!(u8, 8, false);
    convert!(i8, 8, true);
    convert!(u16, 16, false);
    convert!(i16, 16, true);
    Err(PyTypeError::new_err("expected a NumPy array of uint8, int8, uint16 or int16"))
}

fn to_array<'py>(py: Python<'py>, v: &Volume) -> PyResult<Bound<'py, PyAny>> {
    let shape = [v.depth(), v.height(), v.width()];
    macro_rules! make {
        ($t:ty) => {{
            let data: Vec<$t> = v.samples().iter().map(|&s| s as $t).collect();
            Ok(PyArray1::<$t>::from_vec(py, data).reshape(shape)?.into_any())
        }};
    }
    match (v.bits() <= 8, v.signed()) {
        (true, false) => make!(u8),
        (true, true) => make!(i8),
        (false, false) => make!(u16),
        (false, true) => make!(i16),
    }
}

/// Compresses a volume (2-D or 3-D array of uint8, int8, uint16 or int16).
///
/// `bits` declares the significant bits (default: the width of the dtype);
/// `metadata` is stored verbatim in the container.
#[pyfunction]
#[pyo3(signature = (array, *, bits=None, slab=32, stripe=512, packing=true, metadata=None, models=None, threads=None))]
#[allow(clippy::too_many_arguments)]
fn encode<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    bits: Option<u8>,
    slab: u16,
    stripe: u16,
    packing: bool,
    metadata: Option<&[u8]>,
    models: Option<&PyModels>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyBytes>> {
    let volume = to_volume(array, bits)?;
    let options = EncodeOptions { slab, stripe, packing, threads, ..EncodeOptions::with_models(self::models(models)) };
    let metadata = metadata.unwrap_or_default().to_vec();
    let bytes = py.detach(|| tomoz_codec::encode_with_metadata(&volume, &metadata, &options)).map_err(codec_err)?;
    Ok(PyBytes::new(py, &bytes))
}

/// Decompresses a volume; returns a (slices, rows, columns) array.
#[pyfunction]
#[pyo3(signature = (data, *, models=None, verify=true, threads=None))]
fn decode<'py>(
    py: Python<'py>,
    data: &[u8],
    models: Option<&PyModels>,
    verify: bool,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let reg = registry(models);
    let volume = py
        .detach(|| {
            let mut options = DecodeOptions::with_registry(&reg);
            options.verify_sha256 = verify;
            options.threads = threads;
            tomoz_codec::decode(data, &options)
        })
        .map_err(codec_err)?;
    to_array(py, &volume)
}

/// The header of a container as a dict.
#[pyfunction]
fn info<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let h = tomoz_codec::inspect(data).map_err(codec_err)?;
    let d = PyDict::new(py);
    d.set_item("shape", (h.depth, h.height, h.width))?;
    d.set_item("bits", h.bits)?;
    d.set_item("signed", h.signed)?;
    d.set_item("histogram_packing", h.packed())?;
    d.set_item("slab", h.slab)?;
    d.set_item("stripe", h.stripe)?;
    d.set_item("tiles", h.tile_count())?;
    d.set_item("model_2d", h.model_2d.to_string())?;
    d.set_item("model_3d", h.model_3d.to_string())?;
    d.set_item("sha256", h.sha256.iter().map(|b| format!("{b:02x}")).collect::<String>())?;
    d.set_item("metadata", PyBytes::new(py, &h.metadata))?;
    let samples = u64::from(h.depth) * u64::from(h.height) * u64::from(h.width);
    d.set_item("bits_per_sample", data.len() as f64 * 8.0 / samples as f64)?;
    Ok(d)
}

/// Packs DICOM files, given as (name, bytes) pairs, into a byte-exact
/// archive. Returns (archive bytes, report dict).
#[pyfunction]
#[pyo3(signature = (files, *, zstd_level=19, models=None))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts owned arguments.
fn pack_dicom<'py>(
    py: Python<'py>,
    files: Vec<(String, Vec<u8>)>,
    zstd_level: i32,
    models: Option<&PyModels>,
) -> PyResult<Bound<'py, PyTuple>> {
    let options = tomoz_archive::PackOptions {
        zstd_level,
        ..tomoz_archive::PackOptions::new(EncodeOptions::with_models(self::models(models)))
    };
    let (bytes, report) = py
        .detach(|| {
            let refs: Vec<(String, &[u8])> = files.iter().map(|(n, b)| (n.clone(), b.as_slice())).collect();
            tomoz_archive::pack(&refs, &options)
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let r = PyDict::new(py);
    r.set_item("instances", report.instances)?;
    r.set_item("coded", report.coded)?;
    r.set_item("stacks", report.stacks)?;
    r.set_item("input_bytes", report.input_bytes)?;
    r.set_item("output_bytes", report.output_bytes)?;
    r.set_item("stack_bytes", report.stack_bytes)?;
    r.set_item("metadata_bytes", report.metadata_bytes)?;
    r.set_item("stored_bytes", report.stored_bytes)?;
    let stored = PyDict::new(py);
    for (reason, n) in &report.stored {
        stored.set_item(format!("{reason:?}"), n)?;
    }
    r.set_item("stored", stored)?;
    PyTuple::new(py, [PyBytes::new(py, &bytes).into_any(), r.into_any()])
}

/// Restores every file of an archive; returns a list of (name, bytes).
#[pyfunction]
#[pyo3(signature = (data, *, models=None))]
fn unpack_dicom<'py>(py: Python<'py>, data: &[u8], models: Option<&PyModels>) -> PyResult<Bound<'py, PyList>> {
    let reg = registry(models);
    let (names, files) = py
        .detach(|| -> Result<_, tomoz_archive::Error> {
            let a = tomoz_archive::Archive::open(data)?;
            let names: Vec<String> = a.instances().iter().map(|e| e.name.clone()).collect();
            Ok((names, a.restore_all(&reg)?))
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = PyList::empty(py);
    for (n, f) in names.into_iter().zip(files) {
        out.append((n, PyBytes::new(py, &f)))?;
    }
    Ok(out)
}

/// Identifiers and names of the built-in models.
#[pyfunction]
fn builtin_models(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let m = ModelSet::builtin();
    let d = PyDict::new(py);
    d.set_item("2d", (m.two_d.name(), m.two_d.id().to_string()))?;
    d.set_item("3d", (m.three_d.name(), m.three_d.id().to_string()))?;
    Ok(d)
}

#[pymodule]
fn _tomoz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModels>()?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(info, m)?)?;
    m.add_function(wrap_pyfunction!(pack_dicom, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_dicom, m)?)?;
    m.add_function(wrap_pyfunction!(builtin_models, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
