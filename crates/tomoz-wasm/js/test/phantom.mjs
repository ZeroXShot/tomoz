// The synthetic phantom of the conformance cases. Must stay identical to
// `phantom` in crates/tomoz-codec/tests/conformance.rs.

export function valueRange(bits, signed) {
  return signed ? [-(2 ** (bits - 1)), 2 ** (bits - 1) - 1] : [0, 2 ** bits - 1];
}

export function phantom({ depth, height, width, bits, signed, seed, noise, step }) {
  const [lo, hi] = valueRange(bits, signed);
  const range = hi - lo + 1;
  let state = seed >>> 0;
  const rand = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state >>> 16;
  };
  const cy = Math.floor(height / 2);
  const cx = Math.floor(width / 2);
  const r = Math.floor((Math.min(height, width) * 2) / 5);
  const data = signed ? new Int16Array(depth * height * width) : new Uint16Array(depth * height * width);
  let i = 0;
  for (let z = 0; z < depth; z++) {
    const rz = r - (z % 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        let v;
        if (d2 > rz * rz) v = 0;
        else if (d2 > (rz - 2) * (rz - 2)) v = Math.floor((range * 3) / 4);
        else v = Math.floor(range / 4) + ((x * 7 + y * 3 + z * 11) % (Math.floor(range / 8) + 1)) + (rand() % (noise + 1));
        data[i++] = lo + Math.floor(Math.min(v, range - 1) / step) * step;
      }
    }
  }
  return { depth, height, width, bits, signed, data };
}
