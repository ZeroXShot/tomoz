//! `tomoz`: compress, decompress and inspect medical image volumes.

mod io;

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use tomoz_codec::{DecodeOptions, EncodeOptions, Model, ModelSet, Volume};

#[derive(Parser)]
#[command(name = "tomoz", version, about = "Learned lossless compression for medical image volumes")]
struct Cli {
    /// Model files (TZM1) to use instead of, and in addition to, the built-in models.
    #[arg(long, global = true, value_name = "FILE")]
    model: Vec<PathBuf>,
    /// Log level (error, warn, info, debug, trace); `serve` defaults to info.
    #[arg(long, global = true)]
    log: Option<String>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compress a volume (.npy, .nii, .nii.gz).
    Encode {
        input: PathBuf,
        /// Output file (default: input with .tmz appended).
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Slices per tile.
        #[arg(long, default_value_t = 32)]
        slab: u16,
        /// Rows per tile.
        #[arg(long, default_value_t = 512)]
        stripe: u16,
        /// Disable histogram packing.
        #[arg(long)]
        no_packing: bool,
    },
    /// Decompress a .tmz file to .npy, or to .nii when it came from NIfTI.
    Decode {
        input: PathBuf,
        /// Output file; the extension selects the format (.npy, .nii, .nii.gz).
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Show the header of a .tmz file.
    Info {
        input: PathBuf,
        /// Print JSON.
        #[arg(long)]
        json: bool,
    },
    /// Decode a .tmz file and check every checksum.
    Verify { input: PathBuf },
    /// Byte-exact DICOM archives.
    #[command(subcommand)]
    Dicom(DicomCommand),
    /// Run the S3-compatible storage gateway.
    Serve {
        /// Configuration file (TOML); `TOMOZ__SECTION__KEY` variables override it.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Print the effective configuration and exit.
        #[arg(long)]
        print_config: bool,
    },
    /// Measure compression ratio and speed on a volume.
    Bench {
        input: PathBuf,
        /// Repetitions of encode and decode.
        #[arg(long, default_value_t = 3)]
        repeat: usize,
        /// Worker threads (default: one per core).
        #[arg(long)]
        threads: Option<usize>,
        /// Print JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum DicomCommand {
    /// Pack DICOM files (or directories of them) into one archive.
    Pack {
        /// Files and directories (searched recursively).
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Output archive (.tmzd).
        #[arg(short, long)]
        output: PathBuf,
        /// zstd level for headers and stored files.
        #[arg(long, default_value_t = 19)]
        zstd_level: i32,
    },
    /// Restore every file of an archive into a directory.
    Unpack {
        archive: PathBuf,
        /// Output directory.
        #[arg(short, long)]
        output: PathBuf,
    },
    /// List the instances of an archive.
    Ls { archive: PathBuf },
    /// Restore every instance in memory and check its SHA-256.
    Verify { archive: PathBuf },
}

fn main() {
    let cli = Cli::parse();
    let default = if matches!(cli.command, Command::Serve { .. }) { "info" } else { "warn" };
    let level = cli.log.clone().unwrap_or_else(|| default.to_owned());
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_new(&level).unwrap_or_else(|_| default.into()))
        .with_writer(std::io::stderr)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .init();
    #[allow(clippy::print_stderr)] // Errors are reported on stderr, not through the logger.
    if let Err(e) = run(cli) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

/// Models used for encoding and the registry used for decoding.
fn models(paths: &[PathBuf]) -> Result<(ModelSet, Vec<Model>)> {
    let builtin = ModelSet::builtin();
    let mut registry = vec![builtin.two_d.clone(), builtin.three_d.clone()];
    let mut set = builtin;
    for p in paths {
        let m = Model::from_bytes(&fs::read(p).with_context(|| format!("reading {}", p.display()))?)
            .with_context(|| format!("loading model {}", p.display()))?;
        match m.kind() {
            tomoz_codec::ModelKind::TwoD => set.two_d = m.clone(),
            tomoz_codec::ModelKind::ThreeD => set.three_d = m.clone(),
        }
        registry.push(m);
    }
    Ok((set, registry))
}

/// Reads a volume and the metadata that restores its original file.
fn read_volume(path: &Path) -> Result<(Volume, Vec<u8>)> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let name = path.to_string_lossy().to_lowercase();
    if name.ends_with(".npy") {
        Ok((io::npy::read(&bytes)?, Vec::new()))
    } else if name.ends_with(".nii") || name.ends_with(".nii.gz") {
        let n = io::nifti::read(&io::nifti::maybe_gunzip(bytes)?)?;
        let meta = n.metadata();
        Ok((n.volume, meta))
    } else {
        bail!("unknown input format for {} (expected .npy, .nii or .nii.gz)", path.display())
    }
}

fn write_volume(path: &Path, volume: &Volume, metadata: &[u8]) -> Result<()> {
    let name = path.to_string_lossy().to_lowercase();
    let bytes = if name.ends_with(".npy") {
        io::npy::write(volume)
    } else if name.ends_with(".nii") || name.ends_with(".nii.gz") {
        let raw = io::nifti::Nifti::restore(metadata, volume).context("the container does not hold a NIfTI header")?;
        if name.ends_with(".gz") {
            use std::io::Write;
            let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            gz.write_all(&raw)?;
            gz.finish()?
        } else {
            raw
        }
    } else {
        bail!("unknown output format for {} (expected .npy, .nii or .nii.gz)", path.display())
    };
    fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

#[allow(clippy::print_stdout)]
fn run(cli: Cli) -> Result<()> {
    let (set, registry) = models(&cli.model)?;
    match cli.command {
        Command::Encode { input, output, slab, stripe, no_packing } => {
            let (volume, metadata) = read_volume(&input)?;
            let options = EncodeOptions { slab, stripe, packing: !no_packing, ..EncodeOptions::with_models(set) };
            let t = Instant::now();
            let bytes = tomoz_codec::encode_with_metadata(&volume, &metadata, &options)?;
            let secs = t.elapsed().as_secs_f64();
            let output = output.unwrap_or_else(|| PathBuf::from(format!("{}.tmz", input.display())));
            fs::write(&output, &bytes).with_context(|| format!("writing {}", output.display()))?;
            let samples = volume.samples().len() as f64;
            println!(
                "{} → {}: {:.3} bits/sample, ratio {:.2} (vs 16-bit), {:.1} MB/s",
                input.display(),
                output.display(),
                bytes.len() as f64 * 8.0 / samples,
                volume.raw_bytes() as f64 / bytes.len() as f64,
                volume.raw_bytes() as f64 / 1e6 / secs
            );
        }
        Command::Decode { input, output } => {
            let bytes = fs::read(&input).with_context(|| format!("reading {}", input.display()))?;
            let volume = tomoz_codec::decode(&bytes, &DecodeOptions::with_registry(&registry))?;
            let header = tomoz_codec::inspect(&bytes)?;
            write_volume(&output, &volume, &header.metadata)?;
        }
        Command::Info { input, json } => {
            let bytes = fs::read(&input).with_context(|| format!("reading {}", input.display()))?;
            let h = tomoz_codec::inspect(&bytes)?;
            let samples = u64::from(h.depth) * u64::from(h.height) * u64::from(h.width);
            let info = serde_json::json!({
                "version": h.version,
                "shape": [h.depth, h.height, h.width],
                "bits": h.bits,
                "signed": h.signed,
                "value_offset": h.offset,
                "max_mapped": h.max_mapped,
                "histogram_packing": h.packed(),
                "tiles": {"slab": h.slab, "stripe": h.stripe, "count": h.tile_count()},
                "models": {"2d": h.model_2d.to_string(), "3d": h.model_3d.to_string()},
                "sha256": h.sha256.iter().map(|b| format!("{b:02x}")).collect::<String>(),
                "metadata_bytes": h.metadata.len(),
                "compressed_bytes": bytes.len(),
                "bits_per_sample": bytes.len() as f64 * 8.0 / samples as f64,
            });
            if json {
                println!("{}", serde_json::to_string_pretty(&info)?);
            } else {
                for (k, v) in info.as_object().into_iter().flatten() {
                    println!("{k:>18}: {v}");
                }
            }
        }
        Command::Verify { input } => {
            let bytes = fs::read(&input).with_context(|| format!("reading {}", input.display()))?;
            let t = Instant::now();
            let volume = tomoz_codec::decode(&bytes, &DecodeOptions::with_registry(&registry))?;
            println!(
                "{}: ok ({} samples, decoded at {:.1} MB/s)",
                input.display(),
                volume.samples().len(),
                volume.raw_bytes() as f64 / 1e6 / t.elapsed().as_secs_f64()
            );
        }
        Command::Dicom(cmd) => dicom(cmd, set, &registry)?,
        Command::Serve { config, print_config } => {
            let config = tomoz_gateway::Config::load(config.as_deref(), std::env::vars())?;
            if print_config {
                print!("{}", config.to_toml());
                return Ok(());
            }
            let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
            runtime.block_on(async {
                let shutdown = async {
                    let ctrl_c = tokio::signal::ctrl_c();
                    #[cfg(unix)]
                    {
                        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok();
                        let terminate = async {
                            match term.as_mut() {
                                Some(t) => {
                                    t.recv().await;
                                }
                                None => std::future::pending::<()>().await,
                            }
                        };
                        tokio::select! { _ = ctrl_c => {}, () = terminate => {} }
                    }
                    #[cfg(not(unix))]
                    {
                        let _ = ctrl_c.await;
                    }
                };
                tomoz_gateway::run(config, set, shutdown).await
            })?;
        }
        Command::Bench { input, repeat, threads, json } => {
            let (volume, _) = read_volume(&input)?;
            let kernel = set.three_d.network().kernel().name();
            let options = EncodeOptions { threads, ..EncodeOptions::with_models(set) };
            let decode_options = DecodeOptions { threads, ..DecodeOptions::with_registry(&registry) };
            let mut best_enc = f64::MAX;
            let mut best_dec = f64::MAX;
            let mut bytes = Vec::new();
            for _ in 0..repeat.max(1) {
                let t = Instant::now();
                bytes = tomoz_codec::encode(&volume, &options)?;
                best_enc = best_enc.min(t.elapsed().as_secs_f64());
                let t = Instant::now();
                let back = tomoz_codec::decode(&bytes, &decode_options)?;
                best_dec = best_dec.min(t.elapsed().as_secs_f64());
                if back != volume {
                    bail!("round trip mismatch");
                }
            }
            let mb = volume.raw_bytes() as f64 / 1e6;
            let bps = bytes.len() as f64 * 8.0 / volume.samples().len() as f64;
            if json {
                let out = serde_json::json!({
                    "input": input.display().to_string(),
                    "shape": [volume.depth(), volume.height(), volume.width()],
                    "bytes": bytes.len(),
                    "bits_per_sample": bps,
                    "encode_seconds": best_enc,
                    "decode_seconds": best_dec,
                    "encode_mb_per_s": mb / best_enc,
                    "decode_mb_per_s": mb / best_dec,
                    "kernel": kernel,
                    "threads": threads,
                });
                println!("{out}");
            } else {
                println!(
                    "{}: {bps:.3} bits/sample, encode {:.1} MB/s, decode {:.1} MB/s ({kernel} kernel)",
                    input.display(),
                    mb / best_enc,
                    mb / best_dec
                );
            }
        }
    }
    Ok(())
}

/// Files under `inputs`, directories searched recursively, with names
/// relative to the input they were found under.
fn collect_files(inputs: &[PathBuf]) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let mut stack = vec![input.clone()];
            while let Some(dir) = stack.pop() {
                let mut entries: Vec<_> = fs::read_dir(&dir)?.collect::<std::io::Result<_>>()?;
                entries.sort_by_key(fs::DirEntry::path);
                for e in entries {
                    let path = e.path();
                    if e.file_type()?.is_dir() {
                        stack.push(path);
                    } else {
                        let rel = path.strip_prefix(input).unwrap_or(&path).to_string_lossy().replace('\\', "/");
                        out.push((rel, path));
                    }
                }
            }
        } else {
            let name = input.file_name().map_or_else(|| "file".into(), |n| n.to_string_lossy().into_owned());
            out.push((name, input.clone()));
        }
    }
    Ok(out)
}

/// A relative path that stays inside the output directory, or `None`.
fn safe_relative(name: &str) -> Option<PathBuf> {
    let mut p = PathBuf::new();
    for part in name.split(['/', '\\']) {
        match part {
            "" | "." => {}
            ".." => return None,
            s if s.contains(':') => return None,
            s => p.push(s),
        }
    }
    (!p.as_os_str().is_empty()).then_some(p)
}

#[allow(clippy::print_stdout)]
fn dicom(cmd: DicomCommand, set: ModelSet, registry: &[Model]) -> Result<()> {
    let registry: Vec<Model> = registry.to_vec();
    match cmd {
        DicomCommand::Pack { inputs, output, zstd_level } => {
            let files = collect_files(&inputs)?;
            let data: Vec<(String, Vec<u8>)> = files
                .iter()
                .map(|(n, p)| Ok((n.clone(), fs::read(p).with_context(|| format!("reading {}", p.display()))?)))
                .collect::<Result<_>>()?;
            let refs: Vec<(String, &[u8])> = data.iter().map(|(n, b)| (n.clone(), b.as_slice())).collect();
            let options = tomoz_archive::PackOptions {
                zstd_level,
                ..tomoz_archive::PackOptions::new(EncodeOptions::with_models(set))
            };
            let t = Instant::now();
            let (bytes, report) = tomoz_archive::pack(&refs, &options)?;
            let secs = t.elapsed().as_secs_f64();
            // Check before writing: an archive is only useful if it restores.
            let archive = tomoz_archive::Archive::open(&bytes)?;
            archive.restore_all(&registry).context("verification of the new archive failed")?;
            fs::write(&output, &bytes).with_context(|| format!("writing {}", output.display()))?;
            println!(
                "{} files, {:.1} MB → {:.1} MB (ratio {:.2}) in {:.1} s: {} coded in {} stacks, {} stored{}",
                report.instances,
                report.input_bytes as f64 / 1e6,
                report.output_bytes as f64 / 1e6,
                report.input_bytes as f64 / report.output_bytes.max(1) as f64,
                secs,
                report.coded,
                report.stacks,
                report.instances - report.coded,
                if report.stored.is_empty() { String::new() } else { format!(" ({:?})", report.stored) }
            );
        }
        DicomCommand::Unpack { archive, output } => {
            let bytes = fs::read(&archive).with_context(|| format!("reading {}", archive.display()))?;
            let a = tomoz_archive::Archive::open(&bytes)?;
            let files = a.restore_all(&registry)?;
            for (i, (entry, data)) in a.instances().iter().zip(files).enumerate() {
                let rel = safe_relative(&entry.name).unwrap_or_else(|| PathBuf::from(format!("instance-{i:06}.dcm")));
                let path = output.join(rel);
                if let Some(dir) = path.parent() {
                    fs::create_dir_all(dir)?;
                }
                fs::write(&path, data).with_context(|| format!("writing {}", path.display()))?;
            }
            println!("restored {} files into {}", a.instances().len(), output.display());
        }
        DicomCommand::Ls { archive } => {
            let bytes = fs::read(&archive).with_context(|| format!("reading {}", archive.display()))?;
            let a = tomoz_archive::Archive::open(&bytes)?;
            for e in a.instances() {
                let how = match &e.kind {
                    tomoz_archive::InstanceKind::Stored { .. } => "stored".to_owned(),
                    tomoz_archive::InstanceKind::Coded { stack, first_slice, layout, .. } => {
                        format!(
                            "stack {stack} slice {first_slice} ({}x{}x{})",
                            layout.frames, layout.rows, layout.columns
                        )
                    }
                };
                println!("{:>12}  {}  {}", e.size, how, e.name);
            }
        }
        DicomCommand::Verify { archive } => {
            let bytes = fs::read(&archive).with_context(|| format!("reading {}", archive.display()))?;
            let a = tomoz_archive::Archive::open(&bytes)?;
            let t = Instant::now();
            let files = a.restore_all(&registry)?;
            let total: usize = files.iter().map(Vec::len).sum();
            println!(
                "{}: {} files ok ({:.1} MB restored in {:.2} s)",
                archive.display(),
                files.len(),
                total as f64 / 1e6,
                t.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::safe_relative;
    use std::path::PathBuf;

    #[test]
    fn archive_names_cannot_escape() {
        assert_eq!(safe_relative("a/b.dcm"), Some(PathBuf::from("a/b.dcm")));
        assert_eq!(safe_relative("./a//b.dcm"), Some(PathBuf::from("a/b.dcm")));
        assert_eq!(safe_relative("/etc/passwd"), Some(PathBuf::from("etc/passwd")));
        assert_eq!(safe_relative("../x"), None);
        assert_eq!(safe_relative("a/../../x"), None);
        assert_eq!(safe_relative("C:\\x"), None);
        assert_eq!(safe_relative(""), None);
    }
}
