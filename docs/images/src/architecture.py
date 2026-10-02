"""Excalidraw skeleton of docs/images/architecture.svg (see README.md here)."""

import itertools
import json
from pathlib import Path

BG = "#0d1017"
TEXT = "#e6edf3"
CYAN, MAGENTA, LIME, AMBER, VIOLET = "#22e3ff", "#ff3cac", "#b6ff3b", "#ffb627", "#a98bff"
seed = itertools.count(1000)
els = []


def box(id, x, y, w, h, color, text, size=17, dashed=False, fill=None):
    els.append(
        {
            "type": "rectangle",
            "id": id,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "strokeColor": color,
            "backgroundColor": fill or "transparent",
            "fillStyle": "hachure" if fill else "solid",
            "strokeWidth": 2,
            "strokeStyle": "dashed" if dashed else "solid",
            "roughness": 2,
            "seed": next(seed),
            "label": {"text": text, "fontSize": size, "fontFamily": 1, "strokeColor": TEXT},
        }
    )


def text(x, y, s, size=18, color=TEXT):
    els.append(
        {
            "type": "text",
            "x": x,
            "y": y,
            "text": s,
            "fontSize": size,
            "fontFamily": 1,
            "strokeColor": color,
            "roughness": 2,
            "seed": next(seed),
        }
    )


def arrow(a, b, x, y, dx, dy, color, label=None):
    e = {
        "type": "arrow",
        "x": x,
        "y": y,
        "points": [[0, 0], [dx, dy]],
        "strokeColor": color,
        "strokeWidth": 2,
        "roughness": 2,
        "seed": next(seed),
        "endArrowhead": "arrow",
        "start": {"id": a},
        "end": {"id": b},
    }
    if label:
        e["label"] = {"text": label, "fontSize": 15, "fontFamily": 1, "strokeColor": color}
    els.append(e)


def sprite(x, y, rows, palette, px=6):
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch in palette:
                els.append(
                    {
                        "type": "rectangle",
                        "x": x + c * px,
                        "y": y + r * px,
                        "width": px,
                        "height": px,
                        "strokeColor": "transparent",
                        "backgroundColor": palette[ch],
                        "fillStyle": "solid",
                        "strokeWidth": 1,
                        "roughness": 0,
                        "seed": next(seed),
                    }
                )


# Title with a pixel-art slice stack.
sprite(
    40,
    26,
    [
        "....cccccccc",
        "...c......cc",
        "..cccccccc.c",
        ".c......cc.c",
        "mmmmmmmm.c.c",
        "m......m.cc.",
        "m..ll..m.c..",
        "m.llll.mc...",
        "m..ll..m....",
        "mmmmmmmm....",
    ],
    {"c": CYAN, "m": MAGENTA, "l": LIME},
)
text(130, 30, "TOMOZ  —  architecture and data flow", 34)
text(132, 78, "every box is a crate or a step that exists in this repository", 16, "#8b98a9")

# Clients
text(40, 128, "clients", 20, CYAN)
box("pacs", 40, 160, 290, 82, CYAN, "PACS\nOrthanc · dcm4chee (S3 storage)")
box("sdk", 40, 262, 290, 82, CYAN, "S3 SDKs\nboto3 · aws-cli · rclone")
box("py", 40, 404, 290, 82, CYAN, "Python · CLI\nNumPy · .npy · NIfTI · DICOM")
box("web", 40, 506, 290, 82, CYAN, "browser · Node.js\nWebAssembly + SIMD128")

# Gateway
text(410, 128, "tomoz-gateway  (S3 API, tokio + hyper)", 20, MAGENTA)
box("auth", 410, 160, 560, 64, MAGENTA, "SigV4 · aws-chunked · multipart · limits")
box(
    "put",
    410,
    244,
    560,
    70,
    MAGENTA,
    "PUT → raw file (tmp, fsync, rename)\n→ SQLite index commit (WAL, single source of truth)",
)
box(
    "compact",
    410,
    334,
    560,
    92,
    MAGENTA,
    "compactor: series quiet ≥ quiet_seconds\npack → restore every object + SHA-256 → switch\nUPDATE … WHERE blob = old  (writers always win)",
)
box(
    "get",
    410,
    446,
    560,
    70,
    MAGENTA,
    "GET → raw file (SHA-256) or one decoded slab\nLRU cache in bytes · re-lookup if a writer raced",
)
box("ops", 410, 536, 560, 52, MAGENTA, "/_tomoz/metrics (Prometheus) · /_tomoz/health")

arrow("pacs", "auth", 330, 200, 80, -8, CYAN, "S3")
arrow("sdk", "auth", 330, 300, 80, -90, CYAN)

# Codec stack
text(1040, 128, "codec stack", 20, LIME)
box(
    "archive",
    1040,
    160,
    262,
    112,
    LIME,
    "tomoz-archive · .tmzd\nstacks by series,\ngeometry, position\nSHA-256 per file",
    16,
)
box("dicom", 1318, 160, 122, 112, LIME, "tomoz-dicom\nPart 10\nexact byte\noffsets", 16)
box(
    "codec",
    1040,
    300,
    400,
    92,
    LIME,
    "tomoz-codec · .tmz\ntiles: 32 slices × 512 rows\nCRC-32C per tile · SHA-256 per volume",
)
box("nn", 1040, 430, 192, 112, LIME, "tomoz-nn\ninteger MLP\n30-48-48-6\nNEON·dot·AVX2·WASM", 16)
box("ent", 1248, 430, 192, 112, LIME, "tomoz-entropy\nrange coder\nadaptive models\nno_std", 16)

arrow("compact", "archive", 970, 380, 70, -150, MAGENTA)
arrow("get", "archive", 970, 470, 70, -212, MAGENTA)
text(978, 486, "pack /\nrestore", 15, MAGENTA)
arrow("archive", "dicom", 1302, 216, 16, 0, LIME)
arrow("archive", "codec", 1171, 272, 0, 28, LIME)
arrow("codec", "nn", 1136, 392, 0, 38, LIME)
arrow("codec", "ent", 1344, 392, 0, 38, LIME)

# Embedded use: one lane under the gateway into the codec.
els.append(
    {
        "type": "arrow",
        "x": 330,
        "y": 445,
        "points": [[0, 0], [40, 0], [40, 180], [675, 180], [675, -64], [710, -64]],
        "strokeColor": CYAN,
        "strokeWidth": 2,
        "roughness": 2,
        "seed": next(seed),
        "endArrowhead": "arrow",
    }
)
els.append(
    {
        "type": "arrow",
        "x": 330,
        "y": 547,
        "points": [[0, 0], [40, 0]],
        "strokeColor": CYAN,
        "strokeWidth": 2,
        "roughness": 2,
        "seed": next(seed),
        "endArrowhead": None,
    }
)
text(420, 600, "embedded use, no gateway: encode · decode · pack · unpack", 15, CYAN)

# Constraints
text(1500, 128, "design constraints", 20, AMBER)
box("c1", 1500, 160, 290, 96, AMBER, "DETERMINISTIC\nsame bytes on x86-64,\nAArch64 and WebAssembly", 16, dashed=True)
box("c2", 1500, 272, 290, 80, AMBER, "LOSSLESS, VERIFIED\nchecksums on every decode", 16, dashed=True)
box("c3", 1500, 368, 290, 80, AMBER, "RANDOM ACCESS\none slab per slice read", 16, dashed=True)
box(
    "c4",
    1500,
    464,
    290,
    96,
    AMBER,
    "UNTRUSTED INPUT\nfuzzed parsers, no allocation\nfrom unvalidated sizes",
    16,
    dashed=True,
)

# Lab pipeline
text(40, 668, "lab  (Python)  —  how the built-in models are made and measured", 20, VIOLET)
box("tcia", 40, 704, 230, 84, VIOLET, "TCIA public data\nCC BY 3.0 / 4.0", 16)
box("manifest", 310, 704, 260, 84, VIOLET, "manifest tz1.json\nsites disjoint train / test\nSHA-256 pinned", 16)
box("train", 610, 704, 260, 84, VIOLET, "train (PyTorch, CPU)\nfloat → QAT, 2^k scales", 16)
box("models", 910, 704, 260, 84, VIOLET, "tz1-2d / tz1-3d .tzm\n4.4 KB + 5.3 KB, embedded", 16)
box("golden", 1210, 704, 270, 84, VIOLET, "golden vectors\nRust ≡ Python, per sample", 16)
box("eval", 1520, 704, 270, 84, VIOLET, "eval vs JPEG-LS, JPEG 2000,\nHTJ2K, JPEG-XL, zstd", 16)
arrow("tcia", "manifest", 270, 746, 40, 0, VIOLET)
arrow("manifest", "train", 570, 746, 40, 0, VIOLET)
arrow("train", "models", 870, 746, 40, 0, VIOLET)
arrow("models", "golden", 1170, 746, 40, 0, VIOLET)
arrow("golden", "eval", 1480, 746, 40, 0, VIOLET)
arrow("models", "nn", 1100, 704, 0, -162, VIOLET)
text(1110, 600, "weights + priors,\nbuilt in", 15, VIOLET)

skeleton = json.dumps({"background": BG, "elements": els})
Path(__file__).with_name("skeleton.js").write_text(f"window.SKELETON = {skeleton};\n")
print(len(els), "elements")
