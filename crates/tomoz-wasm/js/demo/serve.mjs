// A minimal static server for trying the demo locally:
//   node demo/serve.mjs [port]    (then open http://127.0.0.1:<port>/demo/)
// It serves the package directory on the loopback interface only.

import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, normalize, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const port = Number(process.argv[2] ?? process.env.PORT ?? 8765);
const types = {
  ".html": "text/html; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".tmz": "application/octet-stream",
  ".npy": "application/octet-stream",
};

createServer(async (req, res) => {
  try {
    const path = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    let file = normalize(root + path);
    if (!file.startsWith(root) || file.split(sep).includes("node_modules")) throw new Error("forbidden");
    if ((await stat(file)).isDirectory()) file += `${sep}index.html`;
    await stat(file);
    res.writeHead(200, {
      "content-type": types[extname(file)] ?? "application/octet-stream",
      "cache-control": "no-store",
      "x-content-type-options": "nosniff",
    });
    createReadStream(file).pipe(res);
  } catch {
    res.writeHead(404, { "content-type": "text/plain" }).end("not found\n");
  }
}).listen(port, "127.0.0.1", () => {
  console.log(`Tomoz demo on http://127.0.0.1:${port}/demo/`);
});
