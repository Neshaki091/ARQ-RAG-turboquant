/**
 * server/services/turboquant.js  (Node.js — server side)
 * TurboQuant compression stats for 1536-dimensional embeddings
 * Matches VECTOR(1536) in supabase/002_migrate_1536.sql
 */

function makeRng(seed) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Pre-build a 64×1536 random rotation flat matrix (data-oblivious, seed=42)
// Dùng Float32Array phẳng 1 chiều (Flat Array) để tối ưu hóa CPU Cache và V8 Engine
const ROTATION_ROWS = 64;
const DIM = 1536;
const _rng = makeRng(42);
const ROTATION_FLAT = new Float32Array(ROTATION_ROWS * DIM);

let _idx = 0;
for (let i = 0; i < ROTATION_ROWS; i++) {
  for (let j = 0; j < DIM; j++) {
    const u1 = _rng() + 1e-10;
    const u2 = _rng();
    ROTATION_FLAT[_idx++] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

/**
 * Compute TurboQuant compression statistics for a 1536-dim embedding.
 * @param {number[]} embedding  float32 array, length 1536
 * @param {number}   bits       quantisation bits per dimension (default 8)
 * @returns {object}
 */
export function computeStats(embedding, bits = 8) {
  const dim = embedding.length;           // 1536
  const originalBytes = dim * 4;          // float32 = 4 bytes/dim  → 6144 B

  // PolarQuant: project onto 64 rotated axes
  // Vòng lặp For cực nhanh (JIT-optimized) thay vì .map() và .reduce()
  const rotated = new Float32Array(ROTATION_ROWS);
  let rIdx = 0;
  for (let i = 0; i < ROTATION_ROWS; i++) {
    let sum = 0;
    for (let j = 0; j < dim; j++) {
      sum += ROTATION_FLAT[rIdx++] * (embedding[j] ?? 0);
    }
    rotated[i] = sum;
  }

  // Scalar quantise each rotated coordinate
  const levels = 2 ** bits;
  const quantised = new Uint8Array(ROTATION_ROWS);
  for (let i = 0; i < ROTATION_ROWS; i++) {
    const c = Math.max(-1, Math.min(1, rotated[i]));
    quantised[i] = Math.round(((c + 1) / 2) * (levels - 1));
  }

  // QJL correction: 32 binary bits for residual error
  const qjlBits = 32;
  const quantBytes = Math.ceil((quantised.length * bits) / 8);  // 64 B (8-bit)
  const qjlBytes   = Math.ceil(qjlBits / 8);                    // 4 B
  const compressedBytes = quantBytes + qjlBytes;                 // 68 B

  return {
    originalDim:      dim,
    compressedDim:    quantised.length + qjlBits,
    originalBytes,
    compressedBytes,
    compressionRatio: Number((originalBytes / compressedBytes).toFixed(2)),
    bitsPerDim:       Number(((compressedBytes * 8) / dim).toFixed(3)),
    quantisationBits: bits,
  };
}
