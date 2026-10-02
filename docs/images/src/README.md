# Diagram sources

`architecture.svg` is generated, not drawn by hand, so that it can be kept in
step with the code:

```sh
python3 architecture.py            # writes skeleton.js (Excalidraw skeleton elements)
npm install playwright-core        # once; then a Chromium build:
npx playwright install chromium    # or set CHROMIUM=/path/to/chromium
node render.mjs "$PWD/render.html" ../architecture.svg architecture.png
```

`render.html` loads Excalidraw 0.18 from esm.sh, converts the skeleton and
exports an SVG with the hand-drawn font embedded. Style: dark background,
roughness 2, Virgil, neon strokes; the only graphic is a pixel-art sprite.
