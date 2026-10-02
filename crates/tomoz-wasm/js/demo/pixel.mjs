// Pixel art drawn from code: the wordmark, status lamps, the tile map and
// ordered dithering. Every graphic of the demo comes from here.

const GLYPHS = {
  T: ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
  O: [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
  M: ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
  Z: ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
};

// 4×4 Bayer matrix, thresholds in [0, 1).
const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map((v) => (v + 0.5) / 16);

export const dithered = (x, y, level) => level > BAYER[(y & 3) * 4 + (x & 3)];

/** A 4×4 tile of ordered dithering at `level`, as a CSS image. */
export function ditherTile(color, level) {
  const canvas = document.createElement("canvas");
  canvas.width = 4;
  canvas.height = 4;
  const g = canvas.getContext("2d");
  g.fillStyle = color;
  for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) if (dithered(x, y, level)) g.fillRect(x, y, 1, 1);
  return `url(${canvas.toDataURL()})`;
}

export function colors(element = document.documentElement) {
  const s = getComputedStyle(element);
  const get = (name) => s.getPropertyValue(name).trim();
  return { ink: get("--ink"), paper: get("--paper"), signal: get("--signal"), faint: get("--faint") };
}

function setup(canvas, w, h, scale) {
  canvas.width = w * scale;
  canvas.height = h * scale;
  canvas.style.width = `${w * scale}px`;
  canvas.style.height = `${h * scale}px`;
  const g = canvas.getContext("2d");
  g.imageSmoothingEnabled = false;
  return g;
}

/** The TOMOZ wordmark, ink on a solid signal-coloured drop shadow. */
export function wordmark(canvas, scale = 4) {
  const text = "TOMOZ";
  const pitch = 7;
  const w = text.length * pitch;
  const h = 8;
  const g = setup(canvas, w, h, scale);
  const c = colors();
  const glyphPixels = (dx, dy, color) => {
    g.fillStyle = color;
    [...text].forEach((ch, i) => {
      GLYPHS[ch].forEach((row, y) => {
        [...row].forEach((on, x) => {
          if (on === "#") g.fillRect((i * pitch + x + dx) * scale, (y + dy) * scale, scale, scale);
        });
      });
    });
  };
  glyphPixels(1, 1, c.signal);
  glyphPixels(0, 0, c.ink);
}

const LAMP = [
  "..####..",
  ".#....#.",
  "#..##..#",
  "#.#...##",
  "#.#...##",
  "#.....##",
  ".#..###.",
  "..####..",
];

/** An 8×8 lamp: "off", "busy" (half-lit, dithered), "on" or "fail". */
export function lamp(canvas, state, scale = 2) {
  const g = setup(canvas, 8, 8, scale);
  const c = colors();
  g.clearRect(0, 0, canvas.width, canvas.height);
  LAMP.forEach((row, y) => {
    [...row].forEach((p, x) => {
      const rim = p === "#";
      let color = null;
      if (rim) color = c.ink;
      else if (y > 0 && y < 7 && x > 0 && x < 7) {
        const inside = (x - 3.5) ** 2 + (y - 3.5) ** 2 < 9;
        if (!inside) return;
        if (state === "on") color = c.signal;
        else if (state === "busy") color = dithered(x, y, 0.5) ? c.signal : null;
        else if (state === "fail") color = (x + y) % 2 === 0 ? c.ink : null;
        else color = dithered(x, y, 0.15) ? c.faint : null;
      }
      if (color) {
        g.fillStyle = color;
        g.fillRect(x * scale, y * scale, scale, scale);
      }
    });
  });
}

/**
 * The tiles of a container: columns are slabs (slices), rows are stripes
 * (rows of the image). `current` is the slab being shown, `cached` the slabs
 * held decoded.
 */
export function tileMap(canvas, { slabs, stripes, current, cached }, cell = 18) {
  const maxWidth = canvas.parentElement.clientWidth || 600;
  const size = Math.max(3, Math.min(cell, Math.floor((maxWidth - 2) / Math.max(1, slabs)) - 1));
  const w = slabs * (size + 1) + 1;
  const h = stripes * (size + 1) + 1;
  canvas.width = w;
  canvas.height = h;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const g = canvas.getContext("2d");
  const c = colors();
  g.fillStyle = c.paper;
  g.fillRect(0, 0, w, h);
  for (let s = 0; s < slabs; s++) {
    for (let r = 0; r < stripes; r++) {
      const x0 = 1 + s * (size + 1);
      const y0 = 1 + r * (size + 1);
      const state = s === current ? "current" : cached.includes(s) ? "cached" : "idle";
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const edge = x === 0 || y === 0 || x === size - 1 || y === size - 1;
          let color = null;
          if (state === "current") color = c.signal;
          else if (state === "cached") color = edge || dithered(x, y, 0.5) ? c.ink : null;
          else color = edge ? c.faint : null;
          if (color) {
            g.fillStyle = color;
            g.fillRect(x0 + x, y0 + y, 1, 1);
          }
        }
      }
    }
  }
}
