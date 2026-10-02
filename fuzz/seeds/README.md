# Seed corpus

Small synthetic inputs that let the fuzzers start from valid data instead of
an empty corpus: Tomoz containers and an archive (prefixed with the flag
byte `0x01`, which makes the targets point them at the built-in models),
tiny generated DICOM files (joined with `\xffSPLIT\xff` for `archive_pack`),
and the built-in model files. None of it is patient data.

```sh
mkdir -p corpus/codec_decode && cp seeds/codec_decode/* corpus/codec_decode/
cargo +nightly fuzz run codec_decode
```
