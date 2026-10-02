/** An error reported by the codec (corrupt data, invalid volume, ...). */
export class TomozError extends Error {}

/** The header of a container, as returned by {@link Tomoz.info}. */
export interface Info {
  depth: number;
  height: number;
  width: number;
  bits: number;
  signed: boolean;
  /** Slices per tile. */
  slab: number;
  /** Rows per tile. */
  stripe: number;
  tiles: number;
  /** Whether the samples are coded through a histogram of the values used. */
  packed: boolean;
  /** SHA-256 of the samples as 16-bit little-endian integers (hex). */
  sha256: string;
  /** Identifiers of the models the container was coded with. */
  model2d: string;
  model3d: string;
}

/** A volume of samples, slice after slice, row after row. */
export interface Volume {
  depth: number;
  height: number;
  width: number;
  /** Significant bits per sample (1 to 16). */
  bits: number;
  signed: boolean;
  data: Int16Array | Uint16Array;
}

export interface EncodeOptions {
  /** Slices per tile (default 32). */
  slab?: number;
  /** Rows per tile (default 512). */
  stripe?: number;
  /** Code sparse value sets through a histogram (default true). */
  packing?: boolean;
}

export interface DecodeOptions {
  /** Decode only slices `[start, end)`. */
  slices?: [number, number];
}

/** A loaded instance of the codec. Not safe to share between threads. */
export class Tomoz {
  private constructor(instance: WebAssembly.Instance);
  info(container: BufferSource): Info;
  decode(container: BufferSource, options?: DecodeOptions): Volume;
  encode(
    volume: Omit<Volume, "data" | "signed"> & { signed?: boolean; data: ArrayLike<number> },
    options?: EncodeOptions,
  ): Uint8Array;
}

/**
 * Loads the codec; by default the `tomoz_wasm.wasm` file next to the module.
 */
export function load(source?: URL | string | Response | BufferSource | WebAssembly.Module): Promise<Tomoz>;
