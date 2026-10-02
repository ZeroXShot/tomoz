//! AWS Signature Version 4 verification for S3 requests.
//!
//! Supports signatures in the `Authorization` header with a signed payload
//! hash, `UNSIGNED-PAYLOAD`, and the `aws-chunked` streaming encodings
//! (signed chunks, unsigned chunks with trailing checksums, signed chunks
//! with a signed trailer). Every comparison of signatures is constant-time.

use std::collections::HashMap;
use std::path::Path;

use hmac::{Hmac, KeyInit, Mac};
use percent_encoding::percent_decode_str;
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

const ALGORITHM: &str = "AWS4-HMAC-SHA256";
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Why a request was not authenticated.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum AuthError {
    /// No or malformed `Authorization` header.
    #[error("missing or malformed authorization: {0}")]
    Malformed(&'static str),
    /// The access key is unknown.
    #[error("unknown access key")]
    UnknownKey,
    /// The signature does not match.
    #[error("signature does not match")]
    Signature,
    /// The request date is too far from the server clock.
    #[error("request time too skewed")]
    Skewed,
    /// The payload does not match its declared hash or checksum.
    #[error("payload does not match: {0}")]
    Payload(&'static str),
}

/// Access keys and their secrets.
#[derive(Clone, Default)]
pub struct Credentials {
    keys: HashMap<String, String>,
}

impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print secrets.
        f.debug_struct("Credentials").field("keys", &self.keys.len()).finish()
    }
}

impl Credentials {
    /// Reads `ACCESS_KEY_ID:SECRET_ACCESS_KEY` lines (blank lines and `#`
    /// comments ignored).
    ///
    /// # Errors
    ///
    /// I/O errors and malformed lines.
    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        Self::parse(&std::fs::read_to_string(path)?)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Parses the credentials file format.
    ///
    /// # Errors
    ///
    /// A message naming the first malformed line.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut keys = HashMap::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (id, secret) =
                line.split_once(':').ok_or_else(|| format!("line {}: expected ACCESS_KEY_ID:SECRET", n + 1))?;
            if id.is_empty() || secret.len() < 8 {
                return Err(format!("line {}: empty key id or secret shorter than 8 characters", n + 1));
            }
            keys.insert(id.to_owned(), secret.to_owned());
        }
        if keys.is_empty() {
            return Err("no credentials".into());
        }
        Ok(Self { keys })
    }

    /// A single key pair (tests and examples).
    #[must_use]
    pub fn single(id: &str, secret: &str) -> Self {
        Self { keys: HashMap::from([(id.to_owned(), secret.to_owned())]) }
    }
}

/// The parts of a request that are signed.
pub struct SignedRequest<'a> {
    /// HTTP method.
    pub method: &'a str,
    /// Raw (percent-encoded) path.
    pub path: &'a str,
    /// Raw query string, without `?`.
    pub query: &'a str,
    /// Headers, names in lower case.
    pub headers: &'a [(String, String)],
}

impl SignedRequest<'_> {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }
}

/// How the body of an authenticated request must be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Payload {
    /// The body is the payload; its SHA-256 must equal this (hex).
    Hashed(String),
    /// The body is the payload, not covered by the signature.
    Unsigned,
    /// `aws-chunked` with a signature per chunk.
    SignedChunks {
        /// Whether a signed trailer with checksums follows the last chunk.
        trailer: bool,
    },
    /// `aws-chunked` without signatures, with trailing checksums.
    UnsignedChunks,
}

/// A verified signature: what is needed to verify the payload.
pub struct Verified {
    /// Access key that signed the request.
    pub access_key: String,
    /// Payload mode.
    pub payload: Payload,
    signing_key: Vec<u8>,
    seed: String,
    date: String,
    scope: String,
}

impl std::fmt::Debug for Verified {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Verified")
            .field("access_key", &self.access_key)
            .field("payload", &self.payload)
            .finish_non_exhaustive()
    }
}

fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut m = <HmacSha256 as KeyInit>::new_from_slice(key).expect("HMAC accepts keys of any length");
    m.update(data);
    m.finalize().into_bytes().to_vec()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

/// Constant-time equality.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Percent-encodes per SigV4: everything but unreserved characters (and `/`
/// when `slash` is kept).
fn uri_encode(input: &[u8], keep_slash: bool) -> String {
    let mut out = String::with_capacity(input.len());
    for &b in input {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') || (keep_slash && b == b'/') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn canonical_query(query: &str) -> String {
    let mut pairs: Vec<(String, String)> = query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            let k: Vec<u8> = percent_decode_str(k).collect();
            let v: Vec<u8> = percent_decode_str(v).collect();
            (uri_encode(&k, false), uri_encode(&v, false))
        })
        .filter(|(k, _)| k != "X-Amz-Signature")
        .collect();
    pairs.sort();
    pairs.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&")
}

/// Seconds since the epoch of a `YYYYMMDDTHHMMSSZ` timestamp.
fn parse_amz_date(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() != 16 || b[8] != b'T' || b[15] != b'Z' {
        return None;
    }
    let num = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let (y, m, d) = (num(0..4)?, num(4..6)?, num(6..8)?);
    let (hh, mm, ss) = (num(9..11)?, num(11..13)?, num(13..15)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || hh > 23 || mm > 59 || ss > 60 {
        return None;
    }
    // Days from the civil date (H. Hinnant's algorithm).
    let y2 = if m <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + hh * 3600 + mm * 60 + ss)
}

/// Verifies the `Authorization` header of `req`.
///
/// # Errors
///
/// An [`AuthError`] describing the failure.
pub fn verify(
    req: &SignedRequest<'_>,
    creds: &Credentials,
    region: &str,
    now: i64,
    max_skew: i64,
) -> Result<Verified, AuthError> {
    let auth = req.header("authorization").ok_or(AuthError::Malformed("no authorization header"))?;
    let rest = auth.strip_prefix(ALGORITHM).ok_or(AuthError::Malformed("unsupported algorithm"))?;
    let mut credential = None;
    let mut signed_headers = None;
    let mut signature = None;
    for part in rest.split(',') {
        let part = part.trim();
        if let Some(v) = part.strip_prefix("Credential=") {
            credential = Some(v);
        } else if let Some(v) = part.strip_prefix("SignedHeaders=") {
            signed_headers = Some(v);
        } else if let Some(v) = part.strip_prefix("Signature=") {
            signature = Some(v);
        }
    }
    let (Some(credential), Some(signed_headers), Some(signature)) = (credential, signed_headers, signature) else {
        return Err(AuthError::Malformed("incomplete authorization header"));
    };
    let mut cred = credential.splitn(5, '/');
    let (Some(key_id), Some(date), Some(cred_region), Some(service), Some(terminal)) =
        (cred.next(), cred.next(), cred.next(), cred.next(), cred.next())
    else {
        return Err(AuthError::Malformed("credential scope"));
    };
    if service != "s3" || terminal != "aws4_request" || cred_region != region {
        return Err(AuthError::Malformed("credential scope does not match this endpoint"));
    }
    let amz_date = req.header("x-amz-date").ok_or(AuthError::Malformed("no x-amz-date header"))?;
    let t = parse_amz_date(amz_date).ok_or(AuthError::Malformed("bad x-amz-date"))?;
    if !amz_date.starts_with(date) {
        return Err(AuthError::Malformed("credential date differs from x-amz-date"));
    }
    if (t - now).abs() > max_skew {
        return Err(AuthError::Skewed);
    }
    let secret = creds.keys.get(key_id).ok_or(AuthError::UnknownKey)?;
    let names: Vec<&str> = signed_headers.split(';').collect();
    if !names.contains(&"host") {
        return Err(AuthError::Malformed("host must be signed"));
    }
    let mut canonical_headers = String::new();
    for name in &names {
        let values: Vec<String> = req
            .headers
            .iter()
            .filter(|(n, _)| n == name)
            .map(|(_, v)| v.split_whitespace().collect::<Vec<_>>().join(" "))
            .collect();
        if values.is_empty() {
            return Err(AuthError::Malformed("a signed header is missing"));
        }
        canonical_headers.push_str(&format!("{name}:{}\n", values.join(",")));
    }
    let payload_hash =
        req.header("x-amz-content-sha256").ok_or(AuthError::Malformed("no x-amz-content-sha256 header"))?;
    let path: Vec<u8> = percent_decode_str(req.path).collect();
    let canonical = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        req.method,
        uri_encode(&path, true),
        canonical_query(req.query),
        canonical_headers,
        signed_headers,
        payload_hash
    );
    let scope = format!("{date}/{region}/s3/aws4_request");
    let string_to_sign = format!("{ALGORITHM}\n{amz_date}\n{scope}\n{}", sha256_hex(canonical.as_bytes()));
    let k_date = hmac(format!("AWS4{secret}").as_bytes(), date.as_bytes());
    let k_region = hmac(&k_date, region.as_bytes());
    let k_service = hmac(&k_region, b"s3");
    let signing_key = hmac(&k_service, b"aws4_request");
    let expected = hex(&hmac(&signing_key, string_to_sign.as_bytes()));
    if !same(&expected, signature) {
        return Err(AuthError::Signature);
    }
    let payload = match payload_hash {
        "UNSIGNED-PAYLOAD" => Payload::Unsigned,
        "STREAMING-AWS4-HMAC-SHA256-PAYLOAD" => Payload::SignedChunks { trailer: false },
        "STREAMING-AWS4-HMAC-SHA256-PAYLOAD-TRAILER" => Payload::SignedChunks { trailer: true },
        "STREAMING-UNSIGNED-PAYLOAD-TRAILER" => Payload::UnsignedChunks,
        h if h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()) => Payload::Hashed(h.to_ascii_lowercase()),
        _ => return Err(AuthError::Malformed("unsupported x-amz-content-sha256")),
    };
    Ok(Verified {
        access_key: key_id.to_owned(),
        payload,
        signing_key,
        seed: signature.to_owned(),
        date: amz_date.to_owned(),
        scope,
    })
}

/// The payload of an unauthenticated request (authentication disabled):
/// streaming encodings are still decoded.
#[must_use]
pub fn unauthenticated_payload(headers: &[(String, String)]) -> Payload {
    match headers.iter().find(|(n, _)| n == "x-amz-content-sha256").map(|(_, v)| v.as_str()) {
        Some(
            "STREAMING-AWS4-HMAC-SHA256-PAYLOAD"
            | "STREAMING-AWS4-HMAC-SHA256-PAYLOAD-TRAILER"
            | "STREAMING-UNSIGNED-PAYLOAD-TRAILER",
        ) => Payload::UnsignedChunks,
        _ => Payload::Unsigned,
    }
}

/// Extracts and checks the payload of a request body.
///
/// `verified` is `None` when authentication is disabled; chunk signatures
/// are then skipped but chunk framing and checksums are still checked.
///
/// # Errors
///
/// [`AuthError::Payload`] when framing, a hash, a chunk signature or a
/// checksum does not match.
pub fn payload(
    body: &[u8],
    mode: &Payload,
    verified: Option<&Verified>,
    headers: &[(String, String)],
) -> Result<Vec<u8>, AuthError> {
    let data = match mode {
        Payload::Hashed(h) => {
            if !same(&sha256_hex(body), h) {
                return Err(AuthError::Payload("SHA-256 of the body differs from x-amz-content-sha256"));
            }
            body.to_vec()
        }
        Payload::Unsigned => body.to_vec(),
        Payload::SignedChunks { trailer } => dechunk(body, verified, *trailer, headers)?,
        Payload::UnsignedChunks => dechunk(body, None, true, headers)?,
    };
    // Checksums sent as headers (trailers are checked while de-chunking).
    check_checksums(&data, headers.iter().map(|(n, v)| (n.as_str(), v.as_str())))?;
    Ok(data)
}

fn check_checksums<'a>(data: &[u8], fields: impl Iterator<Item = (&'a str, &'a str)>) -> Result<(), AuthError> {
    use base64_lite::encode;
    for (name, value) in fields {
        let expected = match name {
            "x-amz-checksum-crc32" => encode(&crc32fast::hash(data).to_be_bytes()),
            "x-amz-checksum-crc32c" => encode(&crc32c::crc32c(data).to_be_bytes()),
            "x-amz-checksum-sha256" => encode(&Sha256::digest(data)),
            _ => continue,
        };
        if expected != value.trim() {
            return Err(AuthError::Payload("checksum does not match"));
        }
    }
    Ok(())
}

/// Decodes `aws-chunked` data, verifying chunk signatures when `verified` is
/// given and trailing checksums when present.
fn dechunk(
    body: &[u8],
    verified: Option<&Verified>,
    trailer: bool,
    headers: &[(String, String)],
) -> Result<Vec<u8>, AuthError> {
    let bad = AuthError::Payload;
    let mut out = Vec::with_capacity(body.len());
    let mut pos = 0;
    let mut previous = verified.map(|v| v.seed.clone());
    loop {
        let line_end = find_crlf(body, pos).ok_or(bad("chunk header"))?;
        let line = std::str::from_utf8(&body[pos..line_end]).map_err(|_| bad("chunk header"))?;
        let (size_hex, ext) = line.split_once(';').unwrap_or((line, ""));
        let size = usize::from_str_radix(size_hex.trim(), 16).map_err(|_| bad("chunk size"))?;
        let start = line_end + 2;
        let end = start.checked_add(size).ok_or(bad("chunk size"))?;
        let chunk = body.get(start..end).ok_or(bad("chunk shorter than declared"))?;
        if let (Some(v), Some(prev)) = (verified, previous.as_mut()) {
            let sig = ext.strip_prefix("chunk-signature=").ok_or(bad("chunk signature missing"))?;
            let sts = format!(
                "AWS4-HMAC-SHA256-PAYLOAD\n{}\n{}\n{prev}\n{EMPTY_SHA256}\n{}",
                v.date,
                v.scope,
                sha256_hex(chunk)
            );
            let expected = hex(&hmac(&v.signing_key, sts.as_bytes()));
            if !same(&expected, sig) {
                return Err(bad("chunk signature"));
            }
            *prev = expected;
        }
        out.extend_from_slice(chunk);
        if size == 0 {
            pos = start;
            break;
        }
        if body.get(end..end + 2) != Some(b"\r\n") {
            return Err(bad("chunk not terminated by CRLF"));
        }
        pos = end + 2;
    }
    if trailer {
        let mut fields = Vec::new();
        let mut canonical = String::new();
        loop {
            let line_end = find_crlf(body, pos).ok_or(bad("trailer"))?;
            let line = std::str::from_utf8(&body[pos..line_end]).map_err(|_| bad("trailer"))?;
            pos = line_end + 2;
            if line.is_empty() {
                break;
            }
            let (name, value) = line.split_once(':').ok_or(bad("trailer field"))?;
            let name = name.trim().to_ascii_lowercase();
            if name == "x-amz-trailer-signature" {
                if let (Some(v), Some(prev)) = (verified, previous.as_ref()) {
                    let sts = format!(
                        "AWS4-HMAC-SHA256-TRAILER\n{}\n{}\n{prev}\n{}",
                        v.date,
                        v.scope,
                        sha256_hex(canonical.as_bytes())
                    );
                    if !same(&hex(&hmac(&v.signing_key, sts.as_bytes())), value.trim()) {
                        return Err(bad("trailer signature"));
                    }
                }
                continue;
            }
            canonical.push_str(&format!("{name}:{}\n", value.trim()));
            fields.push((name, value.trim().to_owned()));
        }
        check_checksums(&out, fields.iter().map(|(n, v)| (n.as_str(), v.as_str())))?;
    } else if body.get(pos..pos + 2) == Some(b"\r\n") {
        pos += 2;
    }
    if pos != body.len() {
        return Err(bad("data after the last chunk"));
    }
    if let Some(len) =
        headers.iter().find(|(n, _)| n == "x-amz-decoded-content-length").and_then(|(_, v)| v.parse::<usize>().ok())
        && len != out.len()
    {
        return Err(bad("decoded length differs from x-amz-decoded-content-length"));
    }
    Ok(out)
}

fn find_crlf(b: &[u8], from: usize) -> Option<usize> {
    b.get(from..)?.windows(2).position(|w| w == b"\r\n").map(|p| from + p)
}

/// Standard base64 encoding, for checksum headers.
mod base64_lite {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub fn encode(data: &[u8]) -> String {
        let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
        for c in data.chunks(3) {
            let n = (u32::from(c[0]) << 16)
                | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
                | u32::from(*c.get(2).unwrap_or(&0));
            out.push(ALPHABET[(n >> 18) as usize & 63] as char);
            out.push(ALPHABET[(n >> 12) as usize & 63] as char);
            out.push(if c.len() > 1 { ALPHABET[(n >> 6) as usize & 63] as char } else { '=' });
            out.push(if c.len() > 2 { ALPHABET[n as usize & 63] as char } else { '=' });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The "GET Object" example of the AWS SigV4 documentation for S3.
    #[test]
    fn aws_documentation_example() {
        let headers = vec![
            ("host".to_owned(), "examplebucket.s3.amazonaws.com".to_owned()),
            ("range".to_owned(), "bytes=0-9".to_owned()),
            ("x-amz-content-sha256".to_owned(), EMPTY_SHA256.to_owned()),
            ("x-amz-date".to_owned(), "20130524T000000Z".to_owned()),
            (
                "authorization".to_owned(),
                "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,\
                 SignedHeaders=host;range;x-amz-content-sha256;x-amz-date,\
                 Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
                    .to_owned(),
            ),
        ];
        let req = SignedRequest { method: "GET", path: "/test.txt", query: "", headers: &headers };
        let creds = Credentials::single("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
        let t = parse_amz_date("20130524T000000Z").unwrap();
        let v = verify(&req, &creds, "us-east-1", t, 900).unwrap();
        assert_eq!(v.payload, Payload::Hashed(EMPTY_SHA256.into()));
        assert_eq!(verify(&req, &creds, "us-east-1", t + 1000, 900).unwrap_err(), AuthError::Skewed);
        assert_eq!(
            verify(&req, &Credentials::single("AKIAIOSFODNN7EXAMPLE", "wrong-secret"), "us-east-1", t, 900)
                .unwrap_err(),
            AuthError::Signature
        );
        assert!(verify(&req, &creds, "eu-west-1", t, 900).is_err());
    }

    /// The streaming PUT example of the AWS documentation (signed chunks).
    #[test]
    fn aws_streaming_example() {
        let headers = vec![
            ("host".to_owned(), "s3.amazonaws.com".to_owned()),
            ("x-amz-date".to_owned(), "20130524T000000Z".to_owned()),
            ("x-amz-storage-class".to_owned(), "REDUCED_REDUNDANCY".to_owned()),
            ("x-amz-content-sha256".to_owned(), "STREAMING-AWS4-HMAC-SHA256-PAYLOAD".to_owned()),
            ("content-encoding".to_owned(), "aws-chunked".to_owned()),
            ("x-amz-decoded-content-length".to_owned(), "66560".to_owned()),
            ("content-length".to_owned(), "66824".to_owned()),
            (
                "authorization".to_owned(),
                "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,\
                 SignedHeaders=content-encoding;content-length;host;x-amz-content-sha256;x-amz-date;x-amz-decoded-content-length;x-amz-storage-class,\
                 Signature=4f232c4386841ef735655705268965c44a0e4690baa4adea153f7db9fa80a0a9"
                    .to_owned(),
            ),
        ];
        let req = SignedRequest { method: "PUT", path: "/examplebucket/chunkObject.txt", query: "", headers: &headers };
        let creds = Credentials::single("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
        let t = parse_amz_date("20130524T000000Z").unwrap();
        let v = verify(&req, &creds, "us-east-1", t, 900).unwrap();
        assert_eq!(v.payload, Payload::SignedChunks { trailer: false });
        let mut body = Vec::new();
        body.extend_from_slice(
            b"10000;chunk-signature=ad80c730a21e5b8d04586a2213dd63b9a0e99e0e2307b0ade35a65485a288648\r\n",
        );
        body.extend_from_slice(&[b'a'; 65536]);
        body.extend_from_slice(
            b"\r\n400;chunk-signature=0055627c9e194cb4542bae2aa5492e3c1575bbb81b612b7d234b86a503ef5497\r\n",
        );
        body.extend_from_slice(&[b'a'; 1024]);
        body.extend_from_slice(
            b"\r\n0;chunk-signature=b6c6ea8a5354eaf15b3cb7646744f4275b71ea724fed81ceb9323e279d449df9\r\n\r\n",
        );
        let data = payload(&body, &v.payload, Some(&v), &headers).unwrap();
        assert_eq!(data.len(), 66560);
        let mut tampered = body.clone();
        tampered[100] = b'b';
        assert!(payload(&tampered, &v.payload, Some(&v), &headers).is_err());
    }

    #[test]
    fn unsigned_chunks_with_trailing_checksum() {
        let data = b"hello world";
        let crc = base64_lite::encode(&crc32fast::hash(data).to_be_bytes());
        let body = format!("b\r\nhello world\r\n0\r\nx-amz-checksum-crc32:{crc}\r\n\r\n");
        let headers = vec![("x-amz-decoded-content-length".to_owned(), "11".to_owned())];
        assert_eq!(payload(body.as_bytes(), &Payload::UnsignedChunks, None, &headers).unwrap(), data);
        let wrong = body.replace(&crc, "AAAAAA==");
        assert!(payload(wrong.as_bytes(), &Payload::UnsignedChunks, None, &headers).is_err());
    }

    #[test]
    fn query_canonicalisation() {
        assert_eq!(canonical_query("prefix=a%20b&list-type=2&delimiter=%2F"), "delimiter=%2F&list-type=2&prefix=a%20b");
        assert_eq!(canonical_query("uploads"), "uploads=");
        assert_eq!(uri_encode("a b/c~d".as_bytes(), true), "a%20b/c~d");
        assert_eq!(base64_lite::encode(b"ab"), "YWI=");
    }

    #[test]
    fn credentials_file() {
        let c = Credentials::parse("# comment\nAKID:secret-key-1\n\n").unwrap();
        assert!(c.keys.contains_key("AKID"));
        assert!(Credentials::parse("AKID:short").is_err());
        assert!(Credentials::parse("").is_err());
        assert!(!format!("{c:?}").contains("secret"));
    }
}
