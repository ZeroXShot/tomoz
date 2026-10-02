// The codec side of the demo viewer, off the main thread. Holds one container
// at a time and a small cache of decoded slabs, so scrolling inside a slab
// does not decode again.

import { load } from "../tomoz.mjs";

const tomozReady = load();
const SLAB_CACHE = 6;

let container = null; // Uint8Array
let info = null;
const slabs = new Map(); // slab index -> Int16Array | Uint16Array (insertion order = LRU order)

const now = () => performance.now();

function hex(buffer) {
  return Array.from(new Uint8Array(buffer), (b) => b.toString(16).padStart(2, "0")).join("");
}

function bitsFor(min, max, signed) {
  let bits = 1;
  if (signed) {
    while (bits < 16 && (min < -(2 ** (bits - 1)) || max > 2 ** (bits - 1) - 1)) bits++;
  } else {
    while (bits < 16 && max > 2 ** bits - 1) bits++;
  }
  return bits;
}

// A NumPy .npy file of 8- or 16-bit integers, C order, 2-D or 3-D.
function parseNpy(bytes) {
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  if (!magic.every((b, i) => bytes[i] === b)) throw new Error("not a .npy file");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const major = bytes[6];
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const start = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(bytes.subarray(start, start + headerLen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1];
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1]?.split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  if (!descr || !fortran || !shape) throw new Error("malformed .npy header");
  if (fortran === "True") throw new Error("Fortran-ordered arrays are not supported");
  if (shape.length !== 2 && shape.length !== 3) throw new Error(`expected a 2-D or 3-D array, got ${shape.length}-D`);
  const [depth, height, width] = shape.length === 2 ? [1, ...shape] : shape;
  const n = depth * height * width;
  const body = bytes.subarray(start + headerLen);
  const types = { "|u1": [Uint8Array, false], "|i1": [Int8Array, false], "<u2": [Uint16Array, false], "<i2": [Int16Array, true] };
  const type = types[descr];
  if (!type) throw new Error(`dtype ${descr} is not supported: use 8- or 16-bit little-endian integers`);
  const [Kind] = type;
  if (body.byteLength < n * Kind.BYTES_PER_ELEMENT) throw new Error(".npy data is truncated");
  const raw = new Kind(body.slice(0, n * Kind.BYTES_PER_ELEMENT).buffer);
  const signed = Kind === Int16Array || Kind === Int8Array;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = raw[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const data = signed ? Int16Array.from(raw) : Uint16Array.from(raw);
  return { depth, height, width, bits: bitsFor(min, max, signed), signed, data };
}

// A synthetic chest CT: reconstruction circle with padding outside, noisy
// air, table, subcutaneous fat, muscle, lungs with vessels, heart, aorta,
// vertebra and ribs, changing along the volume. Signed 12-bit samples in
// Hounsfield-like units.
function phantom(depth = 64, height = 256, width = 256) {
  let state = 0x2545f491;
  const rand = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const gauss = () => rand() + rand() + rand() - 1.5; // std 0.5
  const inside = (y, x, cy, cx, ry, rx) => ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1;
  const vessels = Array.from({ length: 70 }, () => ({
    side: rand() < 0.5 ? -1 : 1,
    y: 0.32 + rand() * 0.3,
    x: 0.06 + rand() * 0.17,
    r: 0.004 + rand() * 0.008,
    dy: (rand() - 0.5) * 0.25,
    dx: (rand() - 0.5) * 0.25,
  }));
  const ribs = Array.from({ length: 22 }, (_, i) => ({ angle: (i / 22) * 2 * Math.PI + 0.07, phase: rand() * 6.28 }));
  const data = new Int16Array(depth * height * width);
  let i = 0;
  for (let z = 0; z < depth; z++) {
    const t = z / Math.max(1, depth - 1); // 0 (upper chest) .. 1 (lower)
    const lungRy = 0.17 + 0.05 * Math.sin(Math.PI * (0.15 + 0.8 * t));
    const heartR = 0.06 + 0.07 * t;
    for (let yi = 0; yi < height; yi++) {
      const y = yi / height;
      for (let xi = 0; xi < width; xi++) {
        const x = xi / width;
        let v;
        let sigma = 10;
        if ((y - 0.5) ** 2 + (x - 0.5) ** 2 > 0.249) {
          v = -2048; // outside the reconstruction circle
          sigma = 0;
        } else if (inside(y, x, 0.52, 0.5, 0.33, 0.43)) {
          v = -95; // subcutaneous fat
          if (inside(y, x, 0.52, 0.5, 0.3, 0.395)) {
            v = 45; // muscle and soft tissue
            for (const r of ribs) {
              const ry = 0.52 + 0.28 * Math.sin(r.angle);
              const rx = 0.5 + 0.37 * Math.cos(r.angle);
              if (Math.sin(r.phase + z * 0.35) > -0.2 && (y - ry) ** 2 + (x - rx) ** 2 < 0.00018) v = 760;
            }
            for (const side of [-1, 1]) {
              const cx = 0.5 + side * 0.17;
              if (inside(y, x, 0.47, cx, lungRy, 0.13) && !inside(y, x, 0.42, 0.5 + 0.06, heartR + 0.05, heartR + 0.06)) {
                v = -860;
                sigma = 14;
                for (const k of vessels) {
                  if (k.side !== side) continue;
                  const vy = k.y + k.dy * t;
                  const vx = 0.5 + side * (k.x + k.dx * t);
                  if ((y - vy) ** 2 + (x - vx) ** 2 < k.r * k.r) v = 30;
                }
              }
            }
            if (inside(y, x, 0.43, 0.56, heartR + 0.04, heartR + 0.05)) v = 40; // heart
            if (inside(y, x, 0.6, 0.45, 0.032, 0.032)) v = 170; // aorta (contrast)
            if (inside(y, x, 0.72, 0.5, 0.055, 0.06)) v = inside(y, x, 0.72, 0.5, 0.042, 0.047) ? 260 : 720; // vertebral body
            if (inside(y, x, 0.8, 0.5, 0.028, 0.075) && !inside(y, x, 0.775, 0.5, 0.012, 0.02)) v = 640; // posterior arch
          }
        } else if (y > 0.875 && y < 0.9 && Math.abs(x - 0.5) < 0.4) {
          v = 120; // table
        } else {
          v = -1000; // air
          sigma = 6;
        }
        data[i++] = Math.max(-2048, Math.min(2047, Math.round(v + gauss() * 2 * sigma)));
      }
    }
  }
  return { depth, height, width, bits: 12, signed: true, data };
}

function openContainer(bytes) {
  container = bytes;
  slabs.clear();
  info = null;
  return withTomoz((tomoz) => {
    info = tomoz.info(container);
    return { info, bytes: container.byteLength };
  });
}

async function withTomoz(f) {
  return f(await tomozReady);
}

async function encode(volume, source, options = {}) {
  const tomoz = await tomozReady;
  const t0 = now();
  const bytes = tomoz.encode(volume, options);
  const seconds = (now() - t0) / 1000;
  const opened = await openContainer(bytes);
  return { ...opened, encoded: { source, rawBytes: volume.data.byteLength, seconds } };
}

async function slab(index) {
  if (slabs.has(index)) {
    const data = slabs.get(index);
    slabs.delete(index);
    slabs.set(index, data);
    return { data, ms: 0, cached: true };
  }
  const tomoz = await tomozReady;
  const z0 = index * info.slab;
  const z1 = Math.min(info.depth, z0 + info.slab);
  const t0 = now();
  const { data } = tomoz.decode(container, { slices: [z0, z1] });
  const ms = now() - t0;
  slabs.set(index, data);
  while (slabs.size > SLAB_CACHE) slabs.delete(slabs.keys().next().value);
  return { data, ms, cached: false };
}

async function slice(z) {
  const index = Math.floor(z / info.slab);
  const { data, ms, cached } = await slab(index);
  const plane = info.height * info.width;
  const offset = (z - index * info.slab) * plane;
  const out = data.slice(offset, offset + plane);
  return { z, data: out, ms, cached, slab: index, cachedSlabs: [...slabs.keys()] };
}

async function verify() {
  const tomoz = await tomozReady;
  const t0 = now();
  const { data } = tomoz.decode(container); // also checks the SHA-256 itself
  const decodeMs = now() - t0;
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const digest = hex(await crypto.subtle.digest("SHA-256", bytes));
  return { decodeMs, digest, expected: info.sha256, rawBytes: data.byteLength };
}

const handlers = {
  open: ({ bytes, name }) => {
    const b = new Uint8Array(bytes);
    if (name.toLowerCase().endsWith(".npy")) return encode(parseNpy(b), name);
    return openContainer(b);
  },
  // Small tiles, so that the tile map shows random access at work.
  phantom: () => encode(phantom(), "synthetic chest CT", { slab: 8, stripe: 64 }),
  slice: ({ z }) => slice(z),
  verify: () => verify(),
  download: () => ({ bytes: container.slice() }),
};

self.onmessage = async ({ data: { id, op, ...args } }) => {
  try {
    const result = await handlers[op](args);
    const transfer = result?.data ? [result.data.buffer] : result?.bytes instanceof Uint8Array ? [result.bytes.buffer] : [];
    self.postMessage({ id, result }, transfer);
  } catch (e) {
    self.postMessage({ id, error: e?.message ?? String(e) });
  }
};
