//! The S3 API: routing, authentication and the operations.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http::{HeaderMap, HeaderValue, Method, Request, Response, StatusCode, header};
use http_body_util::{BodyExt, Full, Limited};
use md5::{Digest as _, Md5};
use percent_encoding::percent_decode_str;

use crate::config::{AuthMode, Config};
use crate::metrics::{Metrics, Operation};
use crate::sigv4::{self, Credentials, SignedRequest};
use crate::store::{Store, StoreError};
use crate::xml;

/// Shared state of the HTTP layer.
pub struct Gateway {
    /// The store.
    pub store: Arc<Store>,
    /// Configuration.
    pub config: Arc<Config>,
    /// Credentials when authentication is enabled.
    pub credentials: Option<Credentials>,
    /// Metrics.
    pub metrics: Arc<Metrics>,
    requests: AtomicU64,
}

impl Gateway {
    /// The HTTP layer over a store.
    #[must_use]
    pub fn new(
        store: Arc<Store>,
        config: Arc<Config>,
        credentials: Option<Credentials>,
        metrics: Arc<Metrics>,
    ) -> Self {
        Self { store, config, credentials, metrics, requests: AtomicU64::new(0) }
    }
}

/// An S3 error response.
#[derive(Debug)]
pub struct S3Error {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl S3Error {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self { status, code, message: message.into() }
    }
}

impl From<StoreError> for S3Error {
    fn from(e: StoreError) -> Self {
        match e {
            StoreError::NoSuchBucket => {
                Self::new(StatusCode::NOT_FOUND, "NoSuchBucket", "The specified bucket does not exist")
            }
            StoreError::NoSuchKey => Self::new(StatusCode::NOT_FOUND, "NoSuchKey", "The specified key does not exist"),
            StoreError::NoSuchUpload => {
                Self::new(StatusCode::NOT_FOUND, "NoSuchUpload", "The specified upload does not exist")
            }
            StoreError::InvalidBucketName => {
                Self::new(StatusCode::BAD_REQUEST, "InvalidBucketName", "The specified bucket is not valid")
            }
            StoreError::BucketNotEmpty => {
                Self::new(StatusCode::CONFLICT, "BucketNotEmpty", "The bucket you tried to delete is not empty")
            }
            StoreError::InvalidArgument(m) => Self::new(StatusCode::BAD_REQUEST, "InvalidArgument", m),
            StoreError::InvalidPart(m) => Self::new(StatusCode::BAD_REQUEST, "InvalidPart", m),
            e => {
                tracing::error!(error = %e, "store failure");
                Self::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "InternalError",
                    "We encountered an internal error. Please try again.",
                )
            }
        }
    }
}

type Reply = Response<Full<Bytes>>;

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

fn http_date(t: i64) -> String {
    httpdate::fmt_http_date(UNIX_EPOCH + std::time::Duration::from_secs(t.max(0) as u64))
}

fn xml_reply(status: StatusCode, body: String) -> Reply {
    let mut r = Response::new(Full::new(Bytes::from(body)));
    *r.status_mut() = status;
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/xml"));
    r
}

fn empty(status: StatusCode) -> Reply {
    let mut r = Response::new(Full::new(Bytes::new()));
    *r.status_mut() = status;
    r
}

/// What a request addresses.
enum Target {
    Service,
    Bucket(String),
    Object(String, String),
}

fn target(path: &str) -> Result<Target, S3Error> {
    let trimmed = path.strip_prefix('/').unwrap_or(path);
    if trimmed.is_empty() {
        return Ok(Target::Service);
    }
    let (bucket, key) = trimmed.split_once('/').unwrap_or((trimmed, ""));
    let decode = |s: &str| {
        percent_decode_str(s)
            .decode_utf8()
            .map(|c| c.into_owned())
            .map_err(|_| S3Error::new(StatusCode::BAD_REQUEST, "InvalidURI", "The path is not valid UTF-8"))
    };
    let bucket = decode(bucket)?;
    if key.is_empty() { Ok(Target::Bucket(bucket)) } else { Ok(Target::Object(bucket, decode(key)?)) }
}

fn query_params(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            let d = |s: &str| percent_decode_str(s).decode_utf8_lossy().into_owned();
            (d(k), d(v))
        })
        .collect()
}

fn param<'a>(params: &'a [(String, String)], name: &str) -> Option<&'a str> {
    params.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
}

fn has(params: &[(String, String)], name: &str) -> bool {
    params.iter().any(|(k, _)| k == name)
}

/// Runs blocking store work off the async threads.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, StoreError> + Send + 'static) -> Result<T, S3Error> {
    match tokio::task::spawn_blocking(f).await {
        Ok(r) => r.map_err(S3Error::from),
        Err(e) => {
            tracing::error!(error = %e, "store task failed");
            Err(S3Error::new(StatusCode::INTERNAL_SERVER_ERROR, "InternalError", "internal error"))
        }
    }
}

/// Handles one HTTP request.
pub async fn handle<B>(gw: Arc<Gateway>, req: Request<B>) -> Result<Reply, Infallible>
where
    B: hyper::body::Body + Send + 'static,
    B::Data: Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    let started = Instant::now();
    let id = format!("{:016X}", gw.requests.fetch_add(1, Ordering::Relaxed) ^ 0x5DEE_CE66_D1CE_4E5B);
    let path = req.uri().path().to_owned();
    let method = req.method().clone();
    let (op, result) = route(&gw, req).await;
    let mut reply = match result {
        Ok(r) => r,
        Err(e) => {
            if e.status.is_server_error() {
                tracing::warn!(%method, %path, code = e.code, "request failed");
            }
            let mut r = if method == Method::HEAD {
                empty(e.status)
            } else {
                xml_reply(e.status, xml::error(e.code, &e.message, &path, &id))
            };
            *r.status_mut() = e.status;
            r
        }
    };
    let h = reply.headers_mut();
    h.insert("x-amz-request-id", HeaderValue::from_str(&id).unwrap_or(HeaderValue::from_static("0")));
    h.insert(header::SERVER, HeaderValue::from_static("tomoz"));
    gw.metrics.request(op, reply.status().as_u16(), started.elapsed().as_secs_f64());
    Ok(reply)
}

async fn read_body<B>(body: B, limit: usize) -> Result<Bytes, S3Error>
where
    B: hyper::body::Body,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    match Limited::new(body, limit).collect().await {
        Ok(c) => Ok(c.to_bytes()),
        Err(e) if e.downcast_ref::<http_body_util::LengthLimitError>().is_some() => Err(S3Error::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "EntityTooLarge",
            "Your proposed upload exceeds the maximum allowed size",
        )),
        Err(e) => Err(S3Error::new(StatusCode::BAD_REQUEST, "IncompleteBody", format!("reading the body failed: {e}"))),
    }
}

fn header_list(headers: &HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .map(|(n, v)| (n.as_str().to_ascii_lowercase(), String::from_utf8_lossy(v.as_bytes()).into_owned()))
        .collect()
}

async fn route<B>(gw: &Arc<Gateway>, req: Request<B>) -> (Operation, Result<Reply, S3Error>)
where
    B: hyper::body::Body + Send + 'static,
    B::Data: Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    let (parts, body) = req.into_parts();
    let path = parts.uri.path().to_owned();
    // Internal endpoints: '_' cannot start a bucket name.
    if path == "/_tomoz/health" {
        return (Operation::Other, Ok(Response::new(Full::new(Bytes::from_static(b"ok\n")))));
    }
    if path == "/_tomoz/metrics" {
        let mut r = Response::new(Full::new(Bytes::from(gw.metrics.render(gw.store.cache_bytes()))));
        r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; version=0.0.4"));
        return (Operation::Other, Ok(r));
    }
    let query = parts.uri.query().unwrap_or("").to_owned();
    let params = query_params(&query);
    let target = match target(&path) {
        Ok(t) => t,
        Err(e) => return (Operation::Other, Err(e)),
    };
    let op = operation(&parts.method, &target, &params);
    let headers = header_list(&parts.headers);
    // Room for aws-chunked framing on top of the largest object.
    let limit = usize::try_from(gw.config.limits.max_object_bytes).unwrap_or(usize::MAX).saturating_add(1 << 20);
    let raw = match read_body(body, limit).await {
        Ok(b) => b,
        Err(e) => return (op, Err(e)),
    };
    let data = match authenticate(gw, &parts.method, &path, &query, &headers, &raw) {
        Ok(d) => d,
        Err(e) => {
            gw.metrics.auth_failures.inc();
            return (op, Err(e));
        }
    };
    (op, dispatch(gw, &parts.method, target, &params, &headers, data).await)
}

fn operation(method: &Method, target: &Target, params: &[(String, String)]) -> Operation {
    match (method, target) {
        (_, Target::Object(..)) if has(params, "uploadId") || has(params, "uploads") => Operation::Multipart,
        (&Method::PUT, Target::Object(..)) => Operation::Put,
        (&Method::GET, Target::Object(..)) => Operation::Get,
        (&Method::HEAD, Target::Object(..)) => Operation::Head,
        (&Method::DELETE, Target::Object(..)) => Operation::Delete,
        (&Method::POST, Target::Bucket(_)) if has(params, "delete") => Operation::Delete,
        (&Method::GET, Target::Bucket(_) | Target::Service) => Operation::List,
        (_, Target::Bucket(_)) => Operation::Bucket,
        _ => Operation::Other,
    }
}

fn authenticate(
    gw: &Gateway,
    method: &Method,
    path: &str,
    query: &str,
    headers: &[(String, String)],
    raw: &[u8],
) -> Result<Vec<u8>, S3Error> {
    let bad_digest = |m: &str| S3Error::new(StatusCode::BAD_REQUEST, "BadDigest", m.to_owned());
    let data = match (gw.config.auth.mode, &gw.credentials) {
        (AuthMode::Sigv4, Some(creds)) => {
            let req = SignedRequest { method: method.as_str(), path, query, headers };
            let skew = gw.config.auth.max_clock_skew_seconds as i64;
            let verified = sigv4::verify(&req, creds, &gw.config.region, now(), skew).map_err(|e| match e {
                sigv4::AuthError::Signature => S3Error::new(
                    StatusCode::FORBIDDEN,
                    "SignatureDoesNotMatch",
                    "The request signature we calculated does not match the signature you provided",
                ),
                sigv4::AuthError::Skewed => S3Error::new(
                    StatusCode::FORBIDDEN,
                    "RequestTimeTooSkewed",
                    "The difference between the request time and the server's time is too large",
                ),
                sigv4::AuthError::UnknownKey => S3Error::new(
                    StatusCode::FORBIDDEN,
                    "InvalidAccessKeyId",
                    "The access key ID you provided does not exist",
                ),
                e => S3Error::new(StatusCode::FORBIDDEN, "AccessDenied", e.to_string()),
            })?;
            sigv4::payload(raw, &verified.payload, Some(&verified), headers).map_err(|e| bad_digest(&e.to_string()))?
        }
        (AuthMode::Sigv4, None) => {
            return Err(S3Error::new(StatusCode::FORBIDDEN, "AccessDenied", "no credentials configured"));
        }
        (AuthMode::None, _) => {
            let mode = sigv4::unauthenticated_payload(headers);
            sigv4::payload(raw, &mode, None, headers).map_err(|e| bad_digest(&e.to_string()))?
        }
    };
    if let Some((_, md5)) = headers.iter().find(|(n, _)| n == "content-md5") {
        let actual = base64(&Md5::digest(&data));
        if actual != md5.trim() {
            return Err(bad_digest("Content-MD5 does not match the body"));
        }
    }
    Ok(data)
}

fn base64(data: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in data.chunks(3) {
        let n =
            (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { A[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { A[n as usize & 63] as char } else { '=' });
    }
    out
}

/// A parsed `Range` header for an object of `size` bytes.
fn parse_range(spec: &str, size: u64) -> Result<(u64, u64), ()> {
    let r = spec.trim().strip_prefix("bytes=").ok_or(())?;
    if r.contains(',') {
        return Err(());
    }
    let (a, b) = r.split_once('-').ok_or(())?;
    let (start, end) = match (a.trim(), b.trim()) {
        ("", n) => {
            let n: u64 = n.parse().map_err(|_| ())?;
            if n == 0 {
                return Err(());
            }
            (size.saturating_sub(n), size.saturating_sub(1))
        }
        (s, "") => (s.parse().map_err(|_| ())?, size.saturating_sub(1)),
        (s, e) => {
            let (s, e): (u64, u64) = (s.parse().map_err(|_| ())?, e.parse().map_err(|_| ())?);
            (s, e.min(size.saturating_sub(1)))
        }
    };
    if size == 0 || start > end || start >= size { Err(()) } else { Ok((start, end)) }
}

fn object_headers(r: &mut Reply, meta: &crate::store::ObjectMeta) {
    let h = r.headers_mut();
    if let Ok(v) = HeaderValue::from_str(&meta.etag) {
        h.insert(header::ETAG, v);
    }
    if let Ok(v) = HeaderValue::from_str(&http_date(meta.modified)) {
        h.insert(header::LAST_MODIFIED, v);
    }
    let ct = meta.content_type.as_deref().unwrap_or("application/octet-stream");
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(ct).unwrap_or(HeaderValue::from_static("application/octet-stream")),
    );
    h.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    if meta.archived {
        h.insert("x-tomoz-storage", HeaderValue::from_static("archive"));
    }
}

#[allow(clippy::too_many_lines)]
async fn dispatch(
    gw: &Arc<Gateway>,
    method: &Method,
    target: Target,
    params: &[(String, String)],
    headers: &[(String, String)],
    data: Vec<u8>,
) -> Result<Reply, S3Error> {
    let store = gw.store.clone();
    let header = |name: &str| headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone());
    match (method, target) {
        (&Method::GET, Target::Service) => {
            let buckets = blocking(move || store.buckets()).await?;
            Ok(xml_reply(StatusCode::OK, xml::list_buckets(&buckets)))
        }
        (&Method::PUT, Target::Bucket(b)) => {
            blocking(move || store.create_bucket(&b)).await?;
            Ok(empty(StatusCode::OK))
        }
        (&Method::HEAD, Target::Bucket(b)) => {
            if blocking(move || store.has_bucket(&b)).await? {
                Ok(empty(StatusCode::OK))
            } else {
                Err(StoreError::NoSuchBucket.into())
            }
        }
        (&Method::DELETE, Target::Bucket(b)) => {
            blocking(move || store.delete_bucket(&b)).await?;
            Ok(empty(StatusCode::NO_CONTENT))
        }
        (&Method::GET, Target::Bucket(b)) if has(params, "location") => {
            let region = gw.config.region.clone();
            if !blocking(move || store.has_bucket(&b)).await? {
                return Err(StoreError::NoSuchBucket.into());
            }
            Ok(xml_reply(
                StatusCode::OK,
                format!(
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<LocationConstraint xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">{}</LocationConstraint>",
                    xml::escape(&region)
                ),
            ))
        }
        (&Method::GET, Target::Bucket(b)) => {
            let v1 = param(params, "list-type") != Some("2");
            let prefix = param(params, "prefix").unwrap_or("").to_owned();
            let delimiter = param(params, "delimiter").filter(|d| !d.is_empty()).map(str::to_owned);
            let max = param(params, "max-keys")
                .and_then(|m| m.parse::<usize>().ok())
                .unwrap_or(1000)
                .min(gw.config.limits.max_keys as usize);
            let continuation = param(params, "continuation-token").map(str::to_owned);
            let start_after = param(params, "start-after").map(str::to_owned);
            let marker = param(params, "marker").map(str::to_owned);
            let after = if v1 {
                marker.clone()
            } else if let Some(t) = &continuation {
                Some(xml::unhex(t).ok_or_else(|| {
                    S3Error::new(StatusCode::BAD_REQUEST, "InvalidArgument", "invalid continuation token")
                })?)
            } else {
                start_after.clone()
            };
            let (bb, pp, dd) = (b.clone(), prefix.clone(), delimiter.clone());
            let listing = blocking(move || store.list(&bb, &pp, dd.as_deref(), after.as_deref(), max)).await?;
            let lp = xml::ListParams {
                bucket: &b,
                prefix: &prefix,
                delimiter: delimiter.as_deref(),
                max_keys: max,
                continuation: continuation.as_deref(),
                start_after: start_after.as_deref(),
                url_encode: param(params, "encoding-type") == Some("url"),
                v1,
                marker: marker.as_deref(),
            };
            Ok(xml_reply(StatusCode::OK, xml::list_objects(&lp, &listing)))
        }
        (&Method::POST, Target::Bucket(b)) if has(params, "delete") => {
            let text = String::from_utf8(data)
                .map_err(|_| S3Error::new(StatusCode::BAD_REQUEST, "MalformedXML", "invalid UTF-8"))?;
            let (keys, quiet) = xml::parse_delete(&text);
            if keys.len() > 1000 {
                return Err(S3Error::new(StatusCode::BAD_REQUEST, "MalformedXML", "at most 1000 keys per request"));
            }
            let (deleted, errors) = blocking(move || {
                let mut deleted = Vec::new();
                let mut errors = Vec::new();
                for k in keys {
                    match store.delete(&b, &k) {
                        Ok(()) => deleted.push(k),
                        Err(StoreError::NoSuchBucket) => return Err(StoreError::NoSuchBucket),
                        Err(e) => errors.push((k, "InternalError".to_owned(), e.to_string())),
                    }
                }
                Ok((deleted, errors))
            })
            .await?;
            Ok(xml_reply(StatusCode::OK, xml::delete_result(&deleted, &errors, quiet)))
        }
        (&Method::POST, Target::Object(b, k)) if has(params, "uploads") => {
            let ct = header("content-type");
            let (bb, kk) = (b.clone(), k.clone());
            let id = blocking(move || store.create_upload(&bb, &kk, ct.as_deref())).await?;
            Ok(xml_reply(StatusCode::OK, xml::initiate_multipart(&b, &k, &id)))
        }
        (&Method::PUT, Target::Object(b, k)) if has(params, "uploadId") => {
            let upload = param(params, "uploadId").unwrap_or("").to_owned();
            let number: u32 = param(params, "partNumber")
                .and_then(|n| n.parse().ok())
                .ok_or_else(|| S3Error::new(StatusCode::BAD_REQUEST, "InvalidArgument", "invalid partNumber"))?;
            let etag = blocking(move || store.upload_part(&b, &k, &upload, number, &data)).await?;
            let mut r = empty(StatusCode::OK);
            if let Ok(v) = HeaderValue::from_str(&etag) {
                r.headers_mut().insert(header::ETAG, v);
            }
            Ok(r)
        }
        (&Method::POST, Target::Object(b, k)) if has(params, "uploadId") => {
            let upload = param(params, "uploadId").unwrap_or("").to_owned();
            let text = String::from_utf8(data)
                .map_err(|_| S3Error::new(StatusCode::BAD_REQUEST, "MalformedXML", "invalid UTF-8"))?;
            let parts =
                xml::parse_complete(&text).map_err(|e| S3Error::new(StatusCode::BAD_REQUEST, "MalformedXML", e))?;
            let (bb, kk) = (b.clone(), k.clone());
            let meta = blocking(move || store.complete_upload(&bb, &kk, &upload, &parts)).await?;
            Ok(xml_reply(StatusCode::OK, xml::complete_multipart(&b, &k, &meta.etag)))
        }
        (&Method::DELETE, Target::Object(b, k)) if has(params, "uploadId") => {
            let upload = param(params, "uploadId").unwrap_or("").to_owned();
            blocking(move || store.abort_upload(&b, &k, &upload)).await?;
            Ok(empty(StatusCode::NO_CONTENT))
        }
        (&Method::PUT, Target::Object(b, k)) => {
            if let Some(source) = header("x-amz-copy-source") {
                let source = percent_decode_str(source.trim_start_matches('/')).decode_utf8_lossy().into_owned();
                let (sb, sk) = source
                    .split_once('/')
                    .ok_or_else(|| S3Error::new(StatusCode::BAD_REQUEST, "InvalidArgument", "invalid copy source"))?;
                let (sb, sk) = (sb.to_owned(), sk.split('?').next().unwrap_or(sk).to_owned());
                let ct = header("content-type");
                let meta = blocking(move || {
                    let (src, body) = store.get(&sb, &sk)?;
                    store.put(&b, &k, &body, ct.as_deref().or(src.content_type.as_deref()))
                })
                .await?;
                return Ok(xml_reply(StatusCode::OK, xml::copy_result(&meta.etag, meta.modified)));
            }
            let ct = header("content-type");
            let meta = blocking(move || store.put(&b, &k, &data, ct.as_deref())).await?;
            let mut r = empty(StatusCode::OK);
            if let Ok(v) = HeaderValue::from_str(&meta.etag) {
                r.headers_mut().insert(header::ETAG, v);
            }
            Ok(r)
        }
        (&Method::GET, Target::Object(b, k)) => {
            let (meta, body) = blocking(move || store.get(&b, &k)).await?;
            let size = body.len() as u64;
            let (status, slice) = match header("range") {
                Some(spec) => match parse_range(&spec, size) {
                    Ok((s, e)) => (StatusCode::PARTIAL_CONTENT, Some((s, e))),
                    Err(()) => {
                        let mut err = S3Error::new(
                            StatusCode::RANGE_NOT_SATISFIABLE,
                            "InvalidRange",
                            "The requested range is not satisfiable",
                        );
                        err.message.push_str(&format!(" (object size {size})"));
                        return Err(err);
                    }
                },
                None => (StatusCode::OK, None),
            };
            let bytes = match slice {
                Some((s, e)) => Bytes::copy_from_slice(&body[s as usize..=e as usize]),
                None => Bytes::from(Arc::try_unwrap(body).unwrap_or_else(|a| (*a).clone())),
            };
            let mut r = Response::new(Full::new(bytes));
            *r.status_mut() = status;
            object_headers(&mut r, &meta);
            if let Some((s, e)) = slice
                && let Ok(v) = HeaderValue::from_str(&format!("bytes {s}-{e}/{size}"))
            {
                r.headers_mut().insert(header::CONTENT_RANGE, v);
            }
            Ok(r)
        }
        (&Method::HEAD, Target::Object(b, k)) => {
            let meta = blocking(move || store.head(&b, &k)).await?;
            let mut r = empty(StatusCode::OK);
            object_headers(&mut r, &meta);
            r.headers_mut().insert(header::CONTENT_LENGTH, HeaderValue::from(meta.size));
            Ok(r)
        }
        (&Method::DELETE, Target::Object(b, k)) => {
            blocking(move || store.delete(&b, &k)).await?;
            Ok(empty(StatusCode::NO_CONTENT))
        }
        _ => Err(S3Error::new(
            StatusCode::NOT_IMPLEMENTED,
            "NotImplemented",
            "This operation is not implemented by the Tomoz gateway",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges() {
        assert_eq!(parse_range("bytes=0-9", 100), Ok((0, 9)));
        assert_eq!(parse_range("bytes=90-", 100), Ok((90, 99)));
        assert_eq!(parse_range("bytes=-10", 100), Ok((90, 99)));
        assert_eq!(parse_range("bytes=50-1000", 100), Ok((50, 99)));
        assert!(parse_range("bytes=100-", 100).is_err());
        assert!(parse_range("bytes=5-1", 100).is_err());
        assert!(parse_range("bytes=0-1,5-6", 100).is_err());
        assert!(parse_range("items=0-1", 100).is_err());
    }

    #[test]
    fn targets() {
        assert!(matches!(target("/").unwrap(), Target::Service));
        assert!(matches!(target("/b").unwrap(), Target::Bucket(b) if b == "b"));
        assert!(matches!(target("/b/").unwrap(), Target::Bucket(b) if b == "b"));
        assert!(matches!(target("/b/a%20b/c").unwrap(), Target::Object(b, k) if b == "b" && k == "a b/c"));
        assert_eq!(base64(b"hello"), "aGVsbG8=");
    }
}
