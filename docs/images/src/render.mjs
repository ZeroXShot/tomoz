import { chromium } from "playwright-core";
import { writeFileSync } from "node:fs";
const [page, out, png] = process.argv.slice(2);
// CHROMIUM: path to a Chromium or headless-shell binary (default: the one
// installed by `npx playwright install chromium`).
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM || undefined,
  args: ["--allow-file-access-from-files"],
});
const p = await browser.newPage({ viewport: { width: 1800, height: 1400 } });
p.on("console", (m) => m.type() === "error" && console.log("console:", m.text().slice(0, 300)));
await p.goto(`file://${page}`);
await p.waitForFunction(() => document.getElementById("out").value !== "loading", null, { timeout: 120000 });
const svg = await p.inputValue("#out");
if (svg.startsWith("error")) { console.log(svg); process.exit(1); }
writeFileSync(out, svg);
await p.waitForTimeout(1500);
const el = await p.$("svg");
await el.screenshot({ path: png });
console.log("svg bytes", svg.length);
await browser.close();
