// Tomoz for JavaScript: decode and encode Tomoz containers in browsers and
// Node.js. The codec runs as WebAssembly (crates/tomoz-wasm) and computes
// exactly the same integers as the native library, so a volume decodes to the
// same samples, and encodes to the same bytes, on every platform.

const DEFAULT_MODULE = new URL("./tomoz_wasm.wasm", import.meta.url);
const HEADER_BYTES = 20;

/** An error reported by the codec (corrupt data, invalid volume, ...). */
export class TomozError extends Error {
  constructor(message) {
    super(message);
    this.name = "TomozError";
  }
}

async function readModule(source) {
  if (source instanceof WebAssembly.Module) return source;
  if (source instanceof ArrayBuffer || ArrayBuffer.isView(source)) return WebAssembly.compile(source);
  if (typeof Response !== "undefined" && source instanceof Response) {
    if (typeof WebAssembly.compileStreaming === "function" && source.headers.get("content-type") === "application/wasm") {
      return WebAssembly.compileStreaming(source);
    }
    return WebAssembly.compile(await source.arrayBuffer());
  }
  const url = source instanceof URL ? source : new URL(String(source), import.meta.url);
  if (url.protocol === "file:") {
    const { readFile } = await import("node:fs/promises");
    return WebAssembly.compile(await readFile(url));
  }
  const response = await fetch(url);
  if (!response.ok) throw new TomozError(`cannot load ${url}: HTTP ${response.status}`);
  return readModule(response);
}

/**
 * Loads the codec.
 *
 * @param {URL | string | Response | BufferSource | WebAssembly.Module} [source]
 *   The WebAssembly module; by default `tomoz_wasm.wasm` next to this file.
 * @returns {Promise<Tomoz>}
 */
export async function load(source = DEFAULT_MODULE) {
  if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
    throw new TomozError("big-endian platforms are not supported");
  }
  const module = await readModule(source);
  const instance = await WebAssembly.instantiate(module, {});
  return new Tomoz(instance);
}

function asBytes(data) {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  throw new TypeError("expected an ArrayBuffer or a typed array");
}

function sampleBytes(data, signed) {
  if ((signed && data instanceof Int16Array) || (!signed && data instanceof Uint16Array)) return asBytes(data);
  const out = new Uint16Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = data[i] & 0xffff;
  return asBytes(out);
}

/** A loaded instance of the codec. Not safe to share between threads. */
export class Tomoz {
  #exports;
  #broken = null;

  /** @param {WebAssembly.Instance} instance */
  constructor(instance) {
    this.#exports = instance.exports;
  }

  #error() {
    const ex = this.#exports;
    const ptr = ex.tomoz_error() >>> 0;
    const len = ex.tomoz_error_len() >>> 0;
    return new TextDecoder().decode(new Uint8Array(ex.memory.buffer, ptr, len));
  }

  // Copies `input` into the module, calls `name(ptr, len, ...args)` and
  // returns a copy of the result buffer.
  #call(name, input, ...args) {
    if (this.#broken) throw new TomozError(`the codec stopped after an internal error (${this.#broken}); load it again`);
    const ex = this.#exports;
    const len = input.byteLength;
    let ptr = 0;
    try {
      ptr = ex.tomoz_alloc(len) >>> 0;
      new Uint8Array(ex.memory.buffer, ptr, len).set(input);
      const out = ex[name](ptr, len, ...args) >>> 0;
      if (out === 0) throw new TomozError(this.#error());
      const outLen = ex.tomoz_result_len() >>> 0;
      const copy = new Uint8Array(ex.memory.buffer, out, outLen).slice();
      ex.tomoz_free(out, outLen);
      return copy;
    } catch (e) {
      // A trap (a bug in the codec) leaves the module in an unknown state.
      if (e instanceof WebAssembly.RuntimeError) this.#broken = e.message;
      throw e;
    } finally {
      if (!this.#broken && ptr !== 0) ex.tomoz_free(ptr, len);
    }
  }

  /**
   * The header of a container: shape, sample format, tiling, models and the
   * SHA-256 of the samples.
   *
   * @param {BufferSource} container
   */
  info(container) {
    return JSON.parse(new TextDecoder().decode(this.#call("tomoz_info", asBytes(container))));
  }

  /**
   * Decodes a container, or only slices `[start, end)` of it (decoding just
   * the tiles that hold them). A full decode verifies the SHA-256 of the
   * samples; every decode verifies the checksums of the tiles it reads.
   *
   * @param {BufferSource} container
   * @param {{ slices?: [number, number] }} [options]
   * @returns {{ depth: number, height: number, width: number, bits: number,
   *   signed: boolean, data: Int16Array | Uint16Array }}
   */
  decode(container, { slices } = {}) {
    let z0 = 0;
    let z1 = 0;
    if (slices !== undefined) {
      [z0, z1] = slices;
      if (!Number.isInteger(z0) || !Number.isInteger(z1) || z0 < 0 || z1 <= z0 || z1 > 0xffffffff) {
        throw new RangeError("slices must be [start, end) with 0 <= start < end");
      }
    }
    const out = this.#call("tomoz_decode", asBytes(container), z0, z1);
    const header = new DataView(out.buffer, 0, HEADER_BYTES);
    const [depth, height, width, bits, signed] = [0, 4, 8, 12, 16].map((o) => header.getUint32(o, true));
    const n = (out.byteLength - HEADER_BYTES) / 2;
    const data = signed ? new Int16Array(out.buffer, HEADER_BYTES, n) : new Uint16Array(out.buffer, HEADER_BYTES, n);
    return { depth, height, width, bits, signed: signed !== 0, data };
  }

  /**
   * Encodes a volume. `data` holds `depth × height × width` samples, slice
   * after slice, row after row; values must fit in `bits` bits (two's
   * complement when `signed`).
   *
   * @param {{ depth: number, height: number, width: number, bits: number,
   *   signed?: boolean, data: ArrayLike<number> }} volume
   * @param {{ slab?: number, stripe?: number, packing?: boolean }} [options]
   *   Tiles of `slab` slices × `stripe` rows (default 32 × 512); `packing`
   *   (default true) codes sparse value sets through a histogram.
   * @returns {Uint8Array}
   */
  encode(volume, { slab = 0, stripe = 0, packing = true } = {}) {
    const { depth, height, width, bits, signed = false, data } = volume;
    for (const [k, v] of Object.entries({ depth, height, width, bits, slab, stripe })) {
      if (!Number.isInteger(v) || v < 0 || v > 0xffffffff) throw new RangeError(`${k} must be a non-negative integer`);
    }
    if (data.length !== depth * height * width) {
      throw new TomozError(`${data.length} samples for ${depth}×${height}×${width}`);
    }
    return this.#call("tomoz_encode", sampleBytes(data, signed), depth, height, width, bits, signed ? 1 : 0, slab, stripe,
      packing ? 1 : 0);
  }
}
