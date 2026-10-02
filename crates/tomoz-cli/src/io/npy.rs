//! NumPy `.npy` arrays of 8- and 16-bit integers.

use anyhow::{Context, Result, bail};
use tomoz_codec::Volume;

/// Reads a 2-D (H, W) or 3-D (Z, H, W) C-ordered array of `u1`, `i1`, `u2` or
/// `i2` samples.
pub fn read(bytes: &[u8]) -> Result<Volume> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not a .npy file");
    }
    let major = bytes[6];
    let (header_len, start) = match major {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        2 | 3 => {
            let b = bytes.get(8..12).context("truncated .npy header")?;
            (u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize, 12)
        }
        v => bail!("unsupported .npy version {v}"),
    };
    let header = std::str::from_utf8(bytes.get(start..start + header_len).context("truncated .npy header")?)?;
    let descr = field(header, "descr").context("missing dtype in .npy header")?;
    if field(header, "fortran_order").is_some_and(|v| v.starts_with("True")) {
        bail!("Fortran-ordered arrays are not supported");
    }
    let shape_text = header.split("'shape':").nth(1).context("missing shape in .npy header")?;
    let shape_text =
        &shape_text[shape_text.find('(').context("bad shape")? + 1..shape_text.find(')').context("bad shape")?];
    let shape: Vec<usize> =
        shape_text.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::parse).collect::<Result<_, _>>()?;
    let (depth, height, width) = match shape[..] {
        [h, w] => (1, h, w),
        [z, h, w] => (z, h, w),
        _ => bail!("expected a 2-D or 3-D array, got shape {shape:?}"),
    };
    let data = &bytes[start + header_len..];
    let n = depth * height * width;
    let descr = descr.trim_matches(|c| c == '\'' || c == '"');
    let (samples, bits, signed): (Vec<i32>, u8, bool) = match descr {
        "|u1" | "<u1" | "u1" => {
            (data.get(..n).context("array data truncated")?.iter().map(|&v| i32::from(v)).collect(), 8, false)
        }
        "|i1" | "<i1" | "i1" => {
            (data.get(..n).context("array data truncated")?.iter().map(|&v| i32::from(v as i8)).collect(), 8, true)
        }
        "<u2" | ">u2" | "<i2" | ">i2" => {
            let raw = data.get(..2 * n).context("array data truncated")?;
            let big = descr.starts_with('>');
            let signed = descr.ends_with("i2");
            let values = raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| {
                    let v = if big { u16::from_be_bytes(c) } else { u16::from_le_bytes(c) };
                    if signed { i32::from(v as i16) } else { i32::from(v) }
                })
                .collect();
            (values, 16, signed)
        }
        other => bail!("unsupported dtype {other}: Tomoz codes 8- and 16-bit integers"),
    };
    Ok(Volume::new(depth, height, width, bits, signed, samples)?)
}

fn field<'a>(header: &'a str, name: &str) -> Option<&'a str> {
    let rest = header.split(&format!("'{name}':")).nth(1)?;
    Some(rest.trim_start().split(',').next()?.trim())
}

/// Writes a volume as a C-ordered 3-D array (2-D if it has one slice).
pub fn write(v: &Volume) -> Vec<u8> {
    let descr = match (v.bits() <= 8, v.signed()) {
        (true, false) => "|u1",
        (true, true) => "|i1",
        (false, false) => "<u2",
        (false, true) => "<i2",
    };
    let shape = if v.depth() == 1 {
        format!("({}, {})", v.height(), v.width())
    } else {
        format!("({}, {}, {})", v.depth(), v.height(), v.width())
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    // Pad so that the data starts on a 64-byte boundary, ending with '\n'.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for &s in v.samples() {
        if v.bits() <= 8 {
            out.push(s as u8);
        } else {
            out.extend_from_slice(&(s as u16).to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for (bits, signed) in [(8, false), (8, true), (16, false), (16, true)] {
            let (lo, hi) = tomoz_codec::value_range(bits, signed);
            let samples: Vec<i32> = (0..2 * 3 * 5).map(|i| lo + (i * 997) % (hi - lo + 1)).collect();
            let v = Volume::new(2, 3, 5, bits, signed, samples).unwrap();
            assert_eq!(read(&write(&v)).unwrap(), v);
        }
    }
}
