// Builds the WebAssembly module (release, SIMD; see .cargo/config.toml) and
// copies it next to tomoz.mjs.

import { execFileSync } from "node:child_process";
import { copyFileSync } from "node:fs";

const root = new URL("../../../", import.meta.url);
execFileSync("cargo", ["build", "--release", "--target", "wasm32-unknown-unknown", "-p", "tomoz-wasm"], {
  cwd: root,
  stdio: "inherit",
});
copyFileSync(new URL("target/wasm32-unknown-unknown/release/tomoz_wasm.wasm", root), new URL("tomoz_wasm.wasm", import.meta.url));
