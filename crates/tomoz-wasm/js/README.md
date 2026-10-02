# @zeroxshot/tomoz (JavaScript / WebAssembly)

Decode and encode [Tomoz](https://github.com/ZeroXShot/tomoz) containers in
browsers and Node.js. The codec is the Rust implementation compiled to
WebAssembly (with 128-bit SIMD); it computes exactly the same integers as the
native library, so a container decodes to the same samples — and a volume
encodes to the same bytes — everywhere. No dependencies, no network access.

```sh
npm install @zeroxshot/tomoz
```

```js
import { load } from "@zeroxshot/tomoz";

const tomoz = await load(); // fetches tomoz_wasm.wasm next to the module

const info = tomoz.info(bytes); // { depth, height, width, bits, signed, tiles, sha256, … }

// Whole volume (checks every tile checksum and the SHA-256 of the samples):
const { depth, height, width, data } = tomoz.decode(bytes); // data: Int16Array | Uint16Array

// Or only slices [40, 42): decodes just the tiles that hold them.
const part = tomoz.decode(bytes, { slices: [40, 42] });

// Encoding (tiles of 16 slices × 512 rows here):
const container = tomoz.encode({ depth, height, width, bits: 12, signed: true, data }, { slab: 16 });
```

`load()` also accepts a URL, a `Response`, bytes or a compiled
`WebAssembly.Module`, for bundlers and Content-Security-Policy setups.
Errors from the codec are thrown as `TomozError`. One instance is not
thread-safe; use one per worker (the demo viewer runs the codec in a Web
Worker).

## Build and test

```sh
npm run build   # cargo build --release --target wasm32-unknown-unknown -p tomoz-wasm
npm test        # conformance: the native container bytes, bit for bit
npm run demo    # the viewer on http://127.0.0.1:8765/demo/
```

The demo opens `.tmz` containers or `.npy` arrays (which it compresses in the
page), shows slices with window/level, decodes slabs on demand and verifies
the decoded samples against the container's SHA-256 with WebCrypto.

## C ABI

The module exports a small C ABI (`tomoz_alloc`, `tomoz_free`,
`tomoz_decode`, `tomoz_encode`, `tomoz_info`, `tomoz_result_len`,
`tomoz_error`, `tomoz_error_len`), documented in
[`src/lib.rs`](../src/lib.rs), for hosts other than JavaScript.
