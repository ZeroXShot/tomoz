//! The gateway's parsers of request data: aws-chunked bodies, S3 XML
//! documents, continuation tokens and credential files.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tomoz_gateway::sigv4::{Credentials, Payload, payload};
use tomoz_gateway::xml;

#[derive(Arbitrary, Debug)]
struct Input<'a> {
    headers: Vec<(String, String)>,
    body: &'a [u8],
    text: &'a str,
}

fuzz_target!(|input: Input<'_>| {
    let headers: Vec<(String, String)> = input.headers.into_iter().map(|(n, v)| (n.to_ascii_lowercase(), v)).collect();
    let _ = payload(input.body, &Payload::UnsignedChunks, None, &headers);
    let _ = payload(input.body, &Payload::Hashed(input.text.to_owned()), None, &headers);
    let _ = xml::parse_complete(input.text);
    let _ = xml::parse_delete(input.text);
    if let Some(key) = xml::unhex(input.text) {
        assert_eq!(xml::hex(key.as_bytes()), input.text.to_ascii_lowercase());
    }
    let _ = Credentials::parse(input.text);
    let escaped = xml::escape(input.text);
    assert!(!escaped.contains('<') && !escaped.contains('>'));
});
