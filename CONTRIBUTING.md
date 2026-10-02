# Contributing

Thanks for your interest in Tomoz. Issues and pull requests are welcome;
for larger changes, please open an issue first to agree on the approach.

## Building and testing

The Rust toolchain is pinned in `rust-toolchain.toml` (rustup installs it on
first use). Everything below runs offline once dependencies are fetched.

```sh
cargo test --workspace --exclude tomoz-python   # all Rust tests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check

# WebAssembly build and its JavaScript tests (Node.js 18+)
cd crates/tomoz-wasm/js && npm run build && npm test

# Python bindings
cd crates/tomoz-python && maturin develop --release && pytest -q tests

# Lab (datasets, training, evaluation)
cd lab && uv sync --group dev && uv run pytest -q && uv run ruff check src tests
```

## Rules that keep the format stable

Tomoz containers must decode identically everywhere, forever. Therefore:

- **Integer arithmetic only** in anything that influences predictions or
  coding (see [ADR 0001](docs/decisions/0001-integer-network.md)).
- **Conformance hashes** (`crates/tomoz-codec/tests/conformance/cases.json`)
  pin the exact bytes produced for fixed inputs. If your change alters them,
  it changes the format: that needs a new format version, a migration story
  and an explicit discussion. Refresh them only for intended changes:
  `TOMOZ_BLESS=1 cargo test -p tomoz-codec --test conformance`.
- **Golden vectors** tie the Rust codec to the Python reference. Changes to
  features, the network or the head must be made in both, followed by
  `tomoz-lab golden ../crates/tomoz-codec/tests/golden/predictions.json`.
- **New SIMD kernels** must be tested against the scalar kernel
  (`crates/tomoz-nn/tests/kernels.rs`), and the conformance test must pass
  with them (`TOMOZ_KERNEL=<name> cargo test -p tomoz-codec --test conformance`).
- **Parsers of untrusted input** never panic, never allocate from an
  unvalidated size, and have a fuzz target (`fuzz/`, run with
  `cargo +nightly fuzz run <target>`).

## Style

- Code follows `rustfmt.toml` and the lints in `Cargo.toml`; Python follows
  `ruff` (see `lab/pyproject.toml`).
- Comments explain why, not what. Public items are documented.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat(codec): …`, `fix(gateway): …`) and stay small and coherent.

## Data

Never commit medical images, even public ones: datasets are downloaded by
`tomoz-lab fetch` and pinned by hash. Test fixtures are synthetic.

## Licence

By contributing you agree that your contributions are licensed under the
Apache License 2.0, like the rest of the project.
