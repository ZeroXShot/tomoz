// Tests of the WebAssembly build: `npm test` after `npm run build`.

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { before, describe, test } from "node:test";

import { load, TomozError } from "../tomoz.mjs";
import { phantom } from "./phantom.mjs";

const casesUrl = new URL("../../../tomoz-codec/tests/conformance/cases.json", import.meta.url);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const bytesOf = (a) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

let tomoz;
before(async () => {
  tomoz = await load();
});

describe("conformance", async () => {
  const doc = JSON.parse(await readFile(casesUrl, "utf8"));
  for (const c of doc.cases) {
    test(c.name, () => {
      const volume = phantom(c);
      assert.equal(sha256(bytesOf(volume.data)), c.samples_sha256, "phantom generator differs from the Rust one");
      const container = tomoz.encode(volume, { slab: c.slab, stripe: c.stripe, packing: c.packing });
      assert.equal(sha256(container), c.container_sha256, "the WebAssembly build encodes differently from native");
      assert.equal(container.byteLength, c.container_bytes);

      const decoded = tomoz.decode(container);
      assert.deepEqual(
        { ...decoded, data: undefined },
        { depth: c.depth, height: c.height, width: c.width, bits: c.bits, signed: c.signed, data: undefined },
      );
      assert.deepEqual(decoded.data, volume.data);

      const start = Math.floor(c.depth / 2);
      const part = tomoz.decode(container, { slices: [start, c.depth] });
      assert.equal(part.depth, c.depth - start);
      assert.deepEqual(part.data, volume.data.subarray(start * c.height * c.width));

      const info = tomoz.info(container);
      assert.equal(info.sha256, c.samples_sha256);
      assert.equal(info.model2d, doc.model_2d);
      assert.equal(info.model3d, doc.model_3d);
    });
  }
});

describe("api", () => {
  test("accepts any array of samples", () => {
    const data = [0, 1, 2, 3, 4, 5, 6, 1023];
    const container = tomoz.encode({ depth: 2, height: 2, width: 2, bits: 10, data });
    assert.deepEqual(Array.from(tomoz.decode(container).data), data);
    const signed = tomoz.encode({ depth: 1, height: 2, width: 2, bits: 16, signed: true, data: [-32768, -1, 0, 32767] });
    assert.deepEqual(Array.from(tomoz.decode(signed.buffer).data), [-32768, -1, 0, 32767]);
  });

  test("reports invalid volumes", () => {
    assert.throws(() => tomoz.encode({ depth: 1, height: 1, width: 2, bits: 8, data: [0, 256] }), TomozError);
    assert.throws(() => tomoz.encode({ depth: 1, height: 1, width: 3, bits: 8, data: [0] }), /1 samples for 1×1×3/);
    assert.throws(() => tomoz.encode({ depth: -1, height: 1, width: 1, bits: 8, data: [] }), RangeError);
  });

  test("reports corrupt containers and keeps working", () => {
    const volume = phantom({ depth: 3, height: 16, width: 16, bits: 12, signed: false, seed: 9, noise: 30, step: 1 });
    const container = tomoz.encode(volume);
    assert.throws(() => tomoz.decode(new Uint8Array([1, 2, 3])), /not a Tomoz/);
    for (const at of [container.byteLength - 1, Math.floor(container.byteLength / 2), 40]) {
      const bad = container.slice();
      bad[at] ^= 0x10;
      assert.throws(() => tomoz.decode(bad), TomozError);
    }
    assert.throws(() => tomoz.decode(container.subarray(0, container.byteLength - 7)), TomozError);
    assert.throws(() => tomoz.decode(container, { slices: [2, 9] }), TomozError);
    assert.throws(() => tomoz.decode(container, { slices: [2, 2] }), RangeError);
    assert.deepEqual(tomoz.decode(container).data, volume.data);
  });

  test("loads from bytes", async () => {
    const bytes = await readFile(new URL("../tomoz_wasm.wasm", import.meta.url));
    const other = await load(bytes);
    const container = other.encode({ depth: 1, height: 1, width: 1, bits: 1, data: [1] });
    assert.deepEqual(Array.from(tomoz.decode(container).data), [1]);
  });
});
