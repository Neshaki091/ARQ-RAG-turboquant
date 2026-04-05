/**
 * server/services/productquant.js
 * Product Quantization (PQ) — simulated for 1536-dim embeddings
 * Used for 3-way comparison: Vanilla RAG vs ARQ+PQ vs ARQ+TurboQuant
 *
 * PQ splits a d-dim vector into M sub-vectors of d/M dims each,
 * then quantizes each sub-vector to one of K centroids (codebook).
 * Training phase (offline) uses k-means — simulated here.
 */

// PQ config: 1536-dim / 48 subspaces = 32-dim each, 256 centroids
const M       = 48;   // number of sub-spaces
const K       = 256;  // centroids per sub-space (8-bit → 1 byte each)
const SUB_DIM = 32;   // 1536 / 48 = 32 dims per subspace

// Seeded RNG for reproducible codebook "training" simulation
function makeRng(seed) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Pre-built simulated codebook — shape [M][K][SUB_DIM]
// (represents k-means centroids after "offline training")
let _codebook = null;
function getCodebook() {
  if (_codebook) return _codebook;
  const rng = makeRng(7777);
  _codebook = Array.from({ length: M }, () =>
    Array.from({ length: K }, () =>
      Array.from({ length: SUB_DIM }, () => {
        const u1 = rng() + 1e-10;
        const u2 = rng();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * 0.4;
      })
    )
  );
  return _codebook;
}

// Initialize codebook eagerly
getCodebook();

/**
 * L2 distance squared between two vectors
 */
function l2sq(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - (b[i] ?? 0);
    s += d * d;
  }
  return s;
}

/**
 * Find nearest centroid index for a sub-vector
 */
function nearestCentroid(subvec, centroids) {
  let best = 0, bestDist = Infinity;
  for (let k = 0; k < centroids.length; k++) {
    const d = l2sq(subvec, centroids[k]);
    if (d < bestDist) { bestDist = d; best = k; }
  }
  return { code: best, dist: bestDist };
}

/**
 * Compute Product Quantization compression stats for a 1536-dim embedding.
 * Simulates the FULL PQ pipeline: split → assign → reconstruct → residual.
 *
 * @param {number[]} embedding  float32 array, length 1536
 * @returns {object}
 */
export function computePQStats(embedding) {
  const codebook = getCodebook();
  const dim = embedding.length;  // 1536

  // ── Phase 1: Split into M sub-vectors ──────────────────────────────────
  const subvectors = Array.from({ length: M }, (_, m) =>
    embedding.slice(m * SUB_DIM, (m + 1) * SUB_DIM)
  );

  // ── Phase 2: Encode — find nearest centroid per sub-space ───────────────
  const t0 = Date.now();
  const codes     = [];
  let   totalDist = 0;
  for (let m = 0; m < M; m++) {
    const { code, dist } = nearestCentroid(subvectors[m], codebook[m]);
    codes.push(code);
    totalDist += dist;
  }
  const encodeMs = Date.now() - t0;

  // ── Phase 3: Reconstruct from codes (for residual analysis) ────────────
  const reconstructed = [];
  for (let m = 0; m < M; m++) {
    const centroid = codebook[m][codes[m]];
    for (let i = 0; i < SUB_DIM; i++) {
      reconstructed.push(centroid[i]);
    }
  }

  // ── Residual L2 error ───────────────────────────────────────────────────
  let residualNorm = 0;
  for (let i = 0; i < dim; i++) {
    const diff = embedding[i] - (reconstructed[i] ?? 0);
    residualNorm += diff * diff;
  }
  residualNorm = Math.sqrt(residualNorm);

  // ── Storage calculation ─────────────────────────────────────────────────
  // PQ stores M codes (1 byte each) + codebook (pre-stored, not per-vector)
  const originalBytes   = dim * 4;        // float32 = 4 bytes × 1536 = 6144 B
  const compressedBytes = M;              // M bytes (8-bit codes) = 48 B
  // Note: codebook itself = M × K × SUB_DIM × 4B = 48 × 256 × 32 × 4 = 1.57 MB (amortized)
  const codebookOverheadBytes = M * K * SUB_DIM * 4;  // total, amortized over N docs

  return {
    algorithm:          'Product Quantization',
    originalDim:        dim,
    subspaces:          M,
    centroids:          K,
    subDim:             SUB_DIM,
    codes:              codes.slice(0, 16),  // first 16 for display
    originalBytes,
    compressedBytes,
    compressionRatio:   Number((originalBytes / compressedBytes).toFixed(2)),
    bitsPerDim:         Number(((compressedBytes * 8) / dim).toFixed(3)),
    residualNorm:       Number(residualNorm.toFixed(4)),
    avgQuantError:      Number((totalDist / M).toFixed(6)),
    encodeMs,
    codebookOverheadMB: Number((codebookOverheadBytes / 1024 / 1024).toFixed(2)),
    trainingRequired:   true,
    estimatedTrainTimeMin: 'O(N·M·K·iters) — typically 15–120 min for large datasets',
  };
}

/**
 * Simulate the "warmup cost" of PQ (k-means training phase).
 * Returns timing metadata representing the offline training overhead.
 */
export function getPQTrainingMeta() {
  return {
    phase:          'offline_training',
    description:    'k-means clustering on full document corpus',
    subspaces:      M,
    centroids:      K,
    iterations:     100,
    estimatedMs:    null,   // Would be real minutes on actual dataset
    simulatedMs:    0,      // "pre-trained" in this demo
    note:           'In production, PQ requires retraining when documents change.',
  };
}
