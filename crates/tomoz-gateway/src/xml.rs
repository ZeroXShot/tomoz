//! The XML documents of the S3 API that the gateway reads and writes.

use std::fmt::Write as _;

use crate::store::{Listing, ObjectMeta};

/// Escapes text for XML content.
#[must_use]
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c if (c as u32) < 0x20 && !matches!(c, '\t' | '\n' | '\r') => {
                let _ = write!(out, "&#x{:X};", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn unescape(s: &str) -> String {
    s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'").replace("&amp;", "&")
}

/// ISO 8601 timestamp of seconds since the epoch.
#[must_use]
pub fn iso8601(t: i64) -> String {
    let days = t.div_euclid(86_400);
    let secs = t.rem_euclid(86_400);
    // Civil date from days (H. Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.000Z", secs / 3600, secs / 60 % 60, secs % 60)
}

const HEADER: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
const NS: &str = "http://s3.amazonaws.com/doc/2006-03-01/";

/// An `<Error>` document.
#[must_use]
pub fn error(code: &str, message: &str, resource: &str, request_id: &str) -> String {
    format!(
        "{HEADER}<Error><Code>{}</Code><Message>{}</Message><Resource>{}</Resource><RequestId>{}</RequestId></Error>",
        escape(code),
        escape(message),
        escape(resource),
        escape(request_id)
    )
}

/// `ListAllMyBucketsResult`.
#[must_use]
pub fn list_buckets(buckets: &[(String, i64)]) -> String {
    let mut o = format!(
        "{HEADER}<ListAllMyBucketsResult xmlns=\"{NS}\"><Owner><ID>tomoz</ID><DisplayName>tomoz</DisplayName></Owner><Buckets>"
    );
    for (name, created) in buckets {
        let _ = write!(
            o,
            "<Bucket><Name>{}</Name><CreationDate>{}</CreationDate></Bucket>",
            escape(name),
            iso8601(*created)
        );
    }
    o.push_str("</Buckets></ListAllMyBucketsResult>");
    o
}

/// Parameters echoed in a listing.
pub struct ListParams<'a> {
    /// Bucket.
    pub bucket: &'a str,
    /// Prefix.
    pub prefix: &'a str,
    /// Delimiter.
    pub delimiter: Option<&'a str>,
    /// Maximum keys.
    pub max_keys: usize,
    /// Continuation token received.
    pub continuation: Option<&'a str>,
    /// `start-after` received.
    pub start_after: Option<&'a str>,
    /// Whether keys are URL-encoded (`encoding-type=url`).
    pub url_encode: bool,
    /// ListObjects version 1 (marker-based) instead of version 2.
    pub v1: bool,
    /// Marker received (version 1).
    pub marker: Option<&'a str>,
}

fn key_text(key: &str, url_encode: bool) -> String {
    if url_encode {
        percent_encoding::utf8_percent_encode(key, percent_encoding::NON_ALPHANUMERIC).to_string()
    } else {
        escape(key)
    }
}

fn object_xml(o: &mut String, m: &ObjectMeta, url_encode: bool) {
    let _ = write!(
        o,
        "<Contents><Key>{}</Key><LastModified>{}</LastModified><ETag>{}</ETag><Size>{}</Size><StorageClass>STANDARD</StorageClass></Contents>",
        key_text(&m.key, url_encode),
        iso8601(m.modified),
        escape(&m.etag),
        m.size
    );
}

/// `ListBucketResult`, version 1 or 2. The continuation token is the hex
/// encoding of the position to resume from.
#[must_use]
pub fn list_objects(p: &ListParams<'_>, l: &Listing) -> String {
    let mut o = format!(
        "{HEADER}<ListBucketResult xmlns=\"{NS}\"><Name>{}</Name><Prefix>{}</Prefix>",
        escape(p.bucket),
        key_text(p.prefix, p.url_encode)
    );
    if let Some(d) = p.delimiter {
        let _ = write!(o, "<Delimiter>{}</Delimiter>", escape(d));
    }
    let _ = write!(o, "<MaxKeys>{}</MaxKeys><IsTruncated>{}</IsTruncated>", p.max_keys, l.next.is_some());
    if p.url_encode {
        o.push_str("<EncodingType>url</EncodingType>");
    }
    if p.v1 {
        let _ = write!(o, "<Marker>{}</Marker>", key_text(p.marker.unwrap_or(""), p.url_encode));
        if let Some(n) = &l.next {
            let _ = write!(o, "<NextMarker>{}</NextMarker>", key_text(n, p.url_encode));
        }
    } else {
        let count = l.objects.len() + l.prefixes.len();
        let _ = write!(o, "<KeyCount>{count}</KeyCount>");
        if let Some(t) = p.continuation {
            let _ = write!(o, "<ContinuationToken>{}</ContinuationToken>", escape(t));
        }
        if let Some(n) = &l.next {
            let _ = write!(o, "<NextContinuationToken>{}</NextContinuationToken>", hex(n.as_bytes()));
        }
        if let Some(s) = p.start_after {
            let _ = write!(o, "<StartAfter>{}</StartAfter>", key_text(s, p.url_encode));
        }
    }
    for m in &l.objects {
        object_xml(&mut o, m, p.url_encode);
    }
    for prefix in &l.prefixes {
        let _ = write!(o, "<CommonPrefixes><Prefix>{}</Prefix></CommonPrefixes>", key_text(prefix, p.url_encode));
    }
    o.push_str("</ListBucketResult>");
    o
}

/// Hex encoding of bytes.
#[must_use]
pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Decodes a hex continuation token.
#[must_use]
pub fn unhex(s: &str) -> Option<String> {
    // `from_str_radix` alone would accept a sign ("+f").
    if !s.len().is_multiple_of(2) || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let bytes: Option<Vec<u8>> =
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok()).collect();
    String::from_utf8(bytes?).ok()
}

/// `InitiateMultipartUploadResult`.
#[must_use]
pub fn initiate_multipart(bucket: &str, key: &str, upload: &str) -> String {
    format!(
        "{HEADER}<InitiateMultipartUploadResult xmlns=\"{NS}\"><Bucket>{}</Bucket><Key>{}</Key><UploadId>{}</UploadId></InitiateMultipartUploadResult>",
        escape(bucket),
        escape(key),
        escape(upload)
    )
}

/// `CompleteMultipartUploadResult`.
#[must_use]
pub fn complete_multipart(bucket: &str, key: &str, etag: &str) -> String {
    format!(
        "{HEADER}<CompleteMultipartUploadResult xmlns=\"{NS}\"><Location>/{}/{}</Location><Bucket>{}</Bucket><Key>{}</Key><ETag>{}</ETag></CompleteMultipartUploadResult>",
        escape(bucket),
        escape(key),
        escape(bucket),
        escape(key),
        escape(etag)
    )
}

/// `DeleteResult` for a multi-object delete.
#[must_use]
pub fn delete_result(deleted: &[String], errors: &[(String, String, String)], quiet: bool) -> String {
    let mut o = format!("{HEADER}<DeleteResult xmlns=\"{NS}\">");
    if !quiet {
        for k in deleted {
            let _ = write!(o, "<Deleted><Key>{}</Key></Deleted>", escape(k));
        }
    }
    for (k, code, msg) in errors {
        let _ = write!(
            o,
            "<Error><Key>{}</Key><Code>{}</Code><Message>{}</Message></Error>",
            escape(k),
            escape(code),
            escape(msg)
        );
    }
    o.push_str("</DeleteResult>");
    o
}

/// `CopyObjectResult`.
#[must_use]
pub fn copy_result(etag: &str, modified: i64) -> String {
    format!(
        "{HEADER}<CopyObjectResult><LastModified>{}</LastModified><ETag>{}</ETag></CopyObjectResult>",
        iso8601(modified),
        escape(etag)
    )
}

/// Text of every `<tag>...</tag>` element, in order. Enough for the small,
/// flat request documents of S3 (no attributes, no nesting of the same tag).
fn elements<'a>(xml: &'a str, tag: &str) -> Vec<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find(&open) {
        let after = &rest[start + open.len()..];
        let Some(end) = after.find(&close) else { break };
        out.push(&after[..end]);
        rest = &after[end + close.len()..];
    }
    out
}

/// Parts of a `CompleteMultipartUpload` request: (number, etag).
///
/// # Errors
///
/// A message for malformed documents.
pub fn parse_complete(xml: &str) -> Result<Vec<(u32, String)>, String> {
    elements(xml, "Part")
        .into_iter()
        .map(|part| {
            let number =
                elements(part, "PartNumber").first().and_then(|n| n.trim().parse().ok()).ok_or("missing PartNumber")?;
            let etag = elements(part, "ETag").first().map(|e| unescape(e.trim())).ok_or("missing ETag")?;
            Ok((number, etag))
        })
        .collect()
}

/// Keys of a `Delete` (multi-object delete) request and its quiet flag.
#[must_use]
pub fn parse_delete(xml: &str) -> (Vec<String>, bool) {
    let keys =
        elements(xml, "Object").into_iter().filter_map(|o| elements(o, "Key").first().map(|k| unescape(k))).collect();
    let quiet = elements(xml, "Quiet").first().is_some_and(|q| q.trim().eq_ignore_ascii_case("true"));
    (keys, quiet)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates() {
        assert_eq!(iso8601(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(iso8601(1_790_000_000), "2026-09-21T14:13:20.000Z");
    }

    #[test]
    fn request_documents() {
        let xml = "<CompleteMultipartUpload><Part><PartNumber>1</PartNumber><ETag>&quot;a&quot;</ETag></Part><Part><ETag>\"b\"</ETag><PartNumber>2</PartNumber></Part></CompleteMultipartUpload>";
        assert_eq!(parse_complete(xml).unwrap(), vec![(1, "\"a\"".into()), (2, "\"b\"".into())]);
        let xml = "<Delete><Quiet>true</Quiet><Object><Key>a&amp;b</Key></Object><Object><Key>c</Key><VersionId>x</VersionId></Object></Delete>";
        assert_eq!(parse_delete(xml), (vec!["a&b".into(), "c".into()], true));
        assert_eq!(unhex(&hex("key/é".as_bytes())).as_deref(), Some("key/é"));
        assert_eq!(unhex("+1"), None); // found by fuzzing
        assert_eq!(escape("<a&'\u{1}>"), "&lt;a&amp;&apos;&#x1;&gt;");
    }
}
