// The demo viewer: opens a Tomoz container (or compresses a .npy volume or a
// synthetic phantom) in the browser, shows slices with random access, and
// verifies the decoded samples against the SHA-256 in the header.

import { colors, ditherTile, lamp, tileMap, wordmark } from "./pixel.mjs";

const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL("./worker.mjs", import.meta.url), { type: "module" });
const pending = new Map();
let nextId = 1;

worker.onmessage = ({ data: { id, result, error } }) => {
  const p = pending.get(id);
  pending.delete(id);
  if (error) p.reject(new Error(error));
  else p.resolve(result);
};

function call(op, args = {}, transfer = []) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, op, ...args }, transfer);
  });
}

const fmt = new Intl.NumberFormat("en-US");
const bytesText = (n) => `${fmt.format(n)} B`;
const ms = (t) => `${t.toFixed(1)} ms`;

function log(text) {
  const line = document.createElement("li");
  const t = new Date();
  const time = document.createElement("time");
  time.textContent = t.toTimeString().slice(0, 8);
  line.append(time, document.createTextNode(text));
  $("log").prepend(line);
  while ($("log").children.length > 40) $("log").lastChild.remove();
}

// Lamp states, redrawn when the theme changes.
const lamps = { "status-lamp": "off", "verify-lamp": "off" };
function setLamp(id, state) {
  lamps[id] = state;
  lamp($(id), state);
}

function setStatus(state, text) {
  setLamp("status-lamp", state);
  $("status").textContent = text;
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

// ---------------------------------------------------------------- state

const view = {
  info: null,
  bytes: 0,
  z: 0,
  shown: -1,
  loading: false,
  pixels: null, // current slice samples
  window: 400,
  level: 40,
  auto: true,
  cached: [],
};

// ---------------------------------------------------------------- header

function row(term, value, wide = false) {
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  dd.textContent = value;
  if (wide) dd.className = "wide";
  return [dt, dd];
}

function showHeader(info, bytes, encoded) {
  const n = info.depth * info.height * info.width;
  const stripes = Math.ceil(info.height / info.stripe);
  const slabs = Math.ceil(info.depth / info.slab);
  const items = [
    row("shape", `${info.depth} × ${info.height} × ${info.width}`),
    row("samples", `${info.bits}-bit ${info.signed ? "signed" : "unsigned"}`),
    row("tiles", `${info.tiles}: ${plural(slabs, "slab")} of ${info.slab} slices × ${plural(stripes, "stripe")} of ${info.stripe} rows`),
    row("packing", info.packed ? "histogram" : "none"),
    row("container", bytesText(bytes)),
    row("rate", `${((8 * bytes) / n).toFixed(3)} bits per voxel`),
    row("ratio", `${((2 * n) / bytes).toFixed(2)} : 1 against 16-bit samples`),
    row("2-D model", info.model2d, true),
    row("3-D model", info.model3d, true),
    row("SHA-256", info.sha256, true),
  ];
  if (encoded) {
    const mbs = encoded.rawBytes / 1e6 / encoded.seconds;
    items.splice(4, 0, row("encoded", `${encoded.source}, ${(encoded.seconds * 1000).toFixed(0)} ms (${mbs.toFixed(1)} MB/s)`));
  }
  $("header").replaceChildren(...items.flat());
}

// ---------------------------------------------------------------- slices

// Window from the 1st to the 99th percentile, ignoring the lowest value
// (padding outside the field of view, or background).
function autoWindow(pixels) {
  const step = Math.max(1, Math.floor(pixels.length / 40000));
  let min = Infinity;
  for (let i = 0; i < pixels.length; i += step) min = Math.min(min, pixels[i]);
  const sample = [];
  for (let i = 0; i < pixels.length; i += step) if (pixels[i] !== min) sample.push(pixels[i]);
  if (sample.length === 0) sample.push(min);
  sample.sort((a, b) => a - b);
  const lo = sample[Math.floor(sample.length * 0.01)];
  const hi = sample[Math.min(sample.length - 1, Math.floor(sample.length * 0.99))];
  view.window = Math.max(1, hi - lo);
  view.level = Math.round((hi + lo) / 2);
}

function render() {
  const { info, pixels } = view;
  if (!info || !pixels) return;
  const canvas = $("view");
  if (canvas.width !== info.width || canvas.height !== info.height) {
    canvas.width = info.width;
    canvas.height = info.height;
  }
  const g = canvas.getContext("2d");
  const image = g.createImageData(info.width, info.height);
  const lo = view.level - view.window / 2;
  const scale = 255 / view.window;
  const out = image.data;
  for (let i = 0, j = 0; i < pixels.length; i++, j += 4) {
    const v = Math.max(0, Math.min(255, (pixels[i] - lo) * scale)) | 0;
    out[j] = v;
    out[j + 1] = v;
    out[j + 2] = v;
    out[j + 3] = 255;
  }
  g.putImageData(image, 0, 0);
  $("wl").textContent = `W ${view.window}  L ${view.level}${view.auto ? "  (auto)" : ""}`;
  fitCanvas();
}

function fitCanvas() {
  const { info } = view;
  if (!info) return;
  const box = $("stage");
  const k = Math.min(box.clientWidth / info.width, box.clientHeight / info.height);
  const scale = k >= 1 ? Math.floor(k) : k;
  $("view").style.width = `${Math.max(1, Math.floor(info.width * scale))}px`;
  $("view").style.height = `${Math.max(1, Math.floor(info.height * scale))}px`;
}

function drawTiles() {
  const { info } = view;
  if (!info) return;
  tileMap($("tiles"), {
    slabs: Math.ceil(info.depth / info.slab),
    stripes: Math.ceil(info.height / info.stripe),
    current: Math.floor(view.z / info.slab),
    cached: view.cached,
  });
}

async function showSlice(z) {
  view.z = z;
  $("slice").value = String(z);
  $("slice-label").textContent = `slice ${z + 1} / ${view.info.depth}`;
  drawTiles();
  if (view.loading) return; // the loop below picks up the latest request
  view.loading = true;
  try {
    while (view.shown !== view.z) {
      const want = view.z;
      const r = await call("slice", { z: want });
      view.pixels = r.data;
      view.shown = want;
      view.cached = r.cachedSlabs;
      if (!r.cached) {
        const z0 = r.slab * view.info.slab;
        const z1 = Math.min(view.info.depth, z0 + view.info.slab);
        const stripes = Math.ceil(view.info.height / view.info.stripe);
        log(`decoded slab ${r.slab} (slices ${z0 + 1}–${z1}, ${plural(stripes, "tile")}) in ${ms(r.ms)}`);
      }
      if (view.auto) autoWindow(view.pixels);
      render();
      drawTiles();
    }
  } catch (e) {
    setStatus("fail", e.message);
    log(`error: ${e.message}`);
  } finally {
    view.loading = false;
  }
}

// ---------------------------------------------------------------- opening

async function opened(result, label) {
  view.info = result.info;
  view.bytes = result.bytes;
  view.shown = -1;
  view.cached = [];
  view.auto = true;
  showHeader(result.info, result.bytes, result.encoded);
  $("slice").max = String(result.info.depth - 1);
  $("slice").disabled = result.info.depth < 2;
  $("verify").disabled = false;
  $("save").hidden = !result.encoded;
  setLamp("verify-lamp", "off");
  $("verify-result").textContent = "";
  document.body.classList.add("loaded");
  setStatus("on", label);
  log(result.encoded ? `compressed ${label}: ${bytesText(result.encoded.rawBytes)} → ${bytesText(result.bytes)}` : `opened ${label}`);
  await showSlice(Math.floor(result.info.depth / 2));
}

async function openFile(file) {
  setStatus("busy", `reading ${file.name}`);
  try {
    const buffer = await file.arrayBuffer();
    if (file.name.toLowerCase().endsWith(".npy")) setStatus("busy", `compressing ${file.name}`);
    await opened(await call("open", { bytes: buffer, name: file.name }, [buffer]), file.name);
  } catch (e) {
    setStatus("fail", e.message);
    log(`error: ${e.message}`);
  }
}

async function openUrl(url) {
  setStatus("busy", `fetching ${url}`);
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const buffer = await response.arrayBuffer();
    const name = new URL(url, location.href).pathname.split("/").pop() || "container";
    await opened(await call("open", { bytes: buffer, name }, [buffer]), name);
  } catch (e) {
    setStatus("fail", `${url}: ${e.message}`);
  }
}

async function verify() {
  $("verify").disabled = true;
  setLamp("verify-lamp", "busy");
  $("verify-result").textContent = "decoding every tile";
  try {
    const r = await call("verify");
    const ok = r.digest === r.expected;
    setLamp("verify-lamp", ok ? "on" : "fail");
    $("verify-result").textContent = ok
      ? `bit-exact: WebCrypto SHA-256 of the ${bytesText(r.rawBytes)} decoded equals the header (${ms(r.decodeMs)}, ${(r.rawBytes / 1e3 / r.decodeMs).toFixed(1)} MB/s)`
      : `MISMATCH: decoded ${r.digest}`;
    log(ok ? `verified full decode in ${ms(r.decodeMs)}` : "verification failed");
  } catch (e) {
    setLamp("verify-lamp", "fail");
    $("verify-result").textContent = e.message;
  } finally {
    $("verify").disabled = false;
  }
}

async function save() {
  const { bytes } = await call("download");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" }));
  a.download = "volume.tmz";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

// ---------------------------------------------------------------- theme

function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  for (const b of document.querySelectorAll("[data-set-theme]")) {
    const active = (theme ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "carbon" : "paper")) === b.dataset.setTheme;
    b.setAttribute("aria-pressed", String(active));
  }
  document.documentElement.style.setProperty("--dither", ditherTile(colors().faint, 0.07));
  wordmark($("wordmark"));
  for (const [id, state] of Object.entries(lamps)) lamp($(id), state);
  drawTiles();
}

function storedTheme() {
  try {
    return localStorage.getItem("tomoz-theme");
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------- wiring

$("file").addEventListener("change", (e) => e.target.files[0] && openFile(e.target.files[0]));
$("phantom").addEventListener("click", async () => {
  setStatus("busy", "generating and compressing a 64 × 256 × 256 synthetic chest CT");
  try {
    await opened(await call("phantom"), "synthetic chest CT");
  } catch (e) {
    setStatus("fail", e.message);
  }
});
$("verify").addEventListener("click", verify);
$("save").addEventListener("click", save);
$("slice").addEventListener("input", (e) => showSlice(Number(e.target.value)));
$("auto").addEventListener("click", () => {
  view.auto = true;
  if (view.pixels) autoWindow(view.pixels);
  render();
});

const drop = document.body;
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  document.body.classList.add("dragging");
});
drop.addEventListener("dragleave", (e) => {
  if (e.target === document.body || !document.body.contains(e.relatedTarget)) document.body.classList.remove("dragging");
});
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  document.body.classList.remove("dragging");
  const file = e.dataTransfer.files[0];
  if (file) openFile(file);
});

const stage = $("view");
let dragFrom = null;
stage.addEventListener("pointerdown", (e) => {
  dragFrom = { x: e.clientX, y: e.clientY, window: view.window, level: view.level };
  stage.setPointerCapture(e.pointerId);
});
stage.addEventListener("pointermove", (e) => {
  if (!dragFrom || !view.info) return;
  const range = 2 ** view.info.bits;
  const k = range / 1500;
  view.auto = false;
  view.window = Math.max(1, Math.round(dragFrom.window + (e.clientX - dragFrom.x) * k));
  view.level = Math.round(dragFrom.level - (e.clientY - dragFrom.y) * k);
  render();
});
stage.addEventListener("pointerup", () => {
  dragFrom = null;
});
stage.addEventListener(
  "wheel",
  (e) => {
    if (!view.info) return;
    e.preventDefault();
    const z = Math.max(0, Math.min(view.info.depth - 1, view.z + Math.sign(e.deltaY)));
    if (z !== view.z) showSlice(z);
  },
  { passive: false },
);
document.addEventListener("keydown", (e) => {
  if (!view.info || e.target instanceof HTMLInputElement && e.target.type !== "range") return;
  const step = { ArrowUp: -1, ArrowDown: 1, PageUp: -10, PageDown: 10 }[e.key];
  if (step === undefined) return;
  e.preventDefault();
  showSlice(Math.max(0, Math.min(view.info.depth - 1, view.z + step)));
});
for (const b of document.querySelectorAll("[data-set-theme]")) {
  b.addEventListener("click", () => {
    try {
      localStorage.setItem("tomoz-theme", b.dataset.setTheme);
    } catch {
      // Private mode: the choice lasts for this page only.
    }
    applyTheme(b.dataset.setTheme);
  });
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(storedTheme()));
new ResizeObserver(() => {
  fitCanvas();
  drawTiles();
}).observe($("stage"));

applyTheme(storedTheme());
setStatus("off", "no volume: open a .tmz container or a .npy array, or compress the phantom");
const src = new URLSearchParams(location.search).get("src");
if (src) openUrl(src);
