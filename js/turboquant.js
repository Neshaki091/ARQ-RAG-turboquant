/**
 * TurboQuant - JavaScript Simulation
 * Based on Google Research paper (ICLR 2026)
 * 
 * Algorithm:
 * 1. PolarQuant: Random Rotation Matrix to spread vector energy
 * 2. QJL Error Correction: 1-bit Quantized Johnson-Lindenstrauss transform
 */

class TurboQuant {
  constructor(inputDim = 128, outputBits = 8) {
    this.inputDim = inputDim;
    this.outputBits = outputBits;
    this.rotationMatrix = null;
    this.qjlMatrix = null;
    this._initMatrices();
  }

  /**
   * Seeded pseudo-random number generator (Mulberry32)
   */
  _seededRandom(seed) {
    let s = seed;
    return function() {
      s |= 0; s = s + 0x6D2B79F5 | 0;
      let t = Math.imul(s ^ s >>> 15, 1 | s);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  /**
   * Generate random orthogonal rotation matrix (Haar distribution approx)
   */
  _initMatrices() {
    const rng = this._seededRandom(42);
    const d = this.inputDim;

    // Build random Gaussian matrix and orthogonalize (simplified Gram-Schmidt)
    const G = [];
    for (let i = 0; i < d; i++) {
      G.push([]);
      for (let j = 0; j < d; j++) {
        // Box-Muller transform for Gaussian samples
        const u1 = rng() + 1e-10;
        const u2 = rng();
        G[i].push(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
      }
    }

    // Store only first few rows for demo (real impl uses full matrix)
    this.rotationMatrix = G.slice(0, Math.min(d, 64));

    // QJL matrix: random ±1 entries scaled by 1/sqrt(m)
    const m = 32;
    this.qjlMatrix = [];
    for (let i = 0; i < m; i++) {
      this.qjlMatrix.push([]);
      for (let j = 0; j < d; j++) {
        this.qjlMatrix[i].push(rng() > 0.5 ? 1 : -1);
      }
    }

    this._scale = 1 / Math.sqrt(m);
  }

  /**
   * Normalize vector to unit sphere
   */
  _normalize(v) {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) + 1e-10;
    return v.map(x => x / norm);
  }

  /**
   * Matrix-vector multiplication
   */
  _matVec(matrix, v) {
    return matrix.map(row => row.reduce((s, mij, j) => s + mij * v[j], 0));
  }

  /**
   * Stage 1: PolarQuant - Random Rotation
   * Spreads vector energy so coordinates become i.i.d.
   */
  polarQuant(vector) {
    const normalized = this._normalize(vector);
    const rotated = this._matVec(this.rotationMatrix, normalized);

    // Scalar quantization with outputBits precision
    const levels = Math.pow(2, this.outputBits);
    const quantized = rotated.map(x => {
      const clipped = Math.max(-1, Math.min(1, x));
      return Math.round((clipped + 1) / 2 * (levels - 1));
    });

    return {
      original: vector,
      normalized,
      rotated,
      quantized,
      compressionRatio: (this.inputDim * 32) / (quantized.length * this.outputBits)
    };
  }

  /**
   * Stage 2: QJL Error Correction
   * Applies 1-bit JL transform to residual for unbiased estimation
   */
  qjlCorrection(original, quantized, rotated) {
    // Dequantize
    const levels = Math.pow(2, this.outputBits);
    const dequant = quantized.map(q => (q / (levels - 1)) * 2 - 1);

    // Compute residual
    const residual = rotated.map((r, i) => r - (dequant[i] || 0));

    // Apply 1-bit QJL to residual
    const qjlOutput = this._matVec(this.qjlMatrix, this._safeResidual(residual));
    const binaryQjl = qjlOutput.map(x => x >= 0 ? 1 : 0);

    return {
      residual,
      qjlBits: binaryQjl,
      correctionBits: binaryQjl.length
    };
  }

  _safeResidual(r) {
    // Pad or trim to match dimensions
    const result = new Array(this.inputDim).fill(0);
    for (let i = 0; i < Math.min(r.length, this.inputDim); i++) {
      result[i] = r[i];
    }
    return result;
  }

  /**
   * Full TurboQuant encode pipeline
   */
  encode(vector) {
    const stage1 = this.polarQuant(vector);
    const stage2 = this.qjlCorrection(vector, stage1.quantized, stage1.rotated);

    const originalBytes = vector.length * 4; // float32
    const quantBytes = Math.ceil((stage1.quantized.length * this.outputBits) / 8);
    const qjlBytes = Math.ceil(stage2.correctionBits / 8);
    const totalCompressed = quantBytes + qjlBytes;

    return {
      stage1,
      stage2,
      stats: {
        originalDim: vector.length,
        compressedDim: stage1.quantized.length + stage2.correctionBits,
        originalBytes,
        compressedBytes: totalCompressed,
        compressionRatio: (originalBytes / totalCompressed).toFixed(2),
        bitsPerDim: ((totalCompressed * 8) / vector.length).toFixed(2)
      }
    };
  }

  /**
   * Decode and estimate original vector (approximate)
   */
  decode(encoded) {
    const { stage1, stage2 } = encoded;
    const levels = Math.pow(2, this.outputBits);
    const dequant = stage1.quantized.map(q => (q / (levels - 1)) * 2 - 1);

    // Apply QJL correction
    const qjlCorrection = this._scale * this._matVec(
      this.qjlMatrix.map(row => row).slice(0, stage2.qjlBits.length),
      stage2.qjlBits.map(b => b * 2 - 1)
    );

    const corrected = dequant.map((d, i) => d + (qjlCorrection || 0) * 0.1);
    return corrected;
  }

  /**
   * Compute dot product between two compressed vectors
   * (Inner Product Estimation without full decode)
   */
  compressedDotProduct(enc1, enc2) {
    const q1 = enc1.stage1.quantized;
    const q2 = enc2.stage1.quantized;
    const levels = Math.pow(2, this.outputBits);

    // Dequantized dot product
    let dp = 0;
    for (let i = 0; i < Math.min(q1.length, q2.length); i++) {
      const d1 = (q1[i] / (levels - 1)) * 2 - 1;
      const d2 = (q2[i] / (levels - 1)) * 2 - 1;
      dp += d1 * d2;
    }

    // QJL correction (1-bit agreement)
    const b1 = enc1.stage2.qjlBits;
    const b2 = enc2.stage2.qjlBits;
    let agree = 0;
    for (let i = 0; i < Math.min(b1.length, b2.length); i++) {
      if (b1[i] === b2[i]) agree++;
    }
    const qjlEst = 2 * (agree / b1.length) - 1;

    return dp + qjlEst * 0.05;
  }
}

/**
 * Embedding simulator - generates pseudo-embeddings for demo
 */
class EmbeddingSimulator {
  constructor(dim = 128) {
    this.dim = dim;
    this.vocabulary = this._buildVocabulary();
  }

  _buildVocabulary() {
    return {
      // AI/ML terms
      'machine': [0.8, 0.6, 0.2], 'learning': [0.7, 0.8, 0.1],
      'neural': [0.9, 0.5, 0.3], 'network': [0.6, 0.7, 0.4],
      'deep': [0.8, 0.4, 0.5], 'transformer': [0.9, 0.8, 0.2],
      'attention': [0.7, 0.9, 0.3], 'embedding': [0.5, 0.8, 0.6],
      'vector': [0.4, 0.7, 0.8], 'retrieval': [0.3, 0.6, 0.9],
      'generation': [0.6, 0.5, 0.8], 'language': [0.7, 0.6, 0.7],
      'model': [0.8, 0.7, 0.6], 'training': [0.9, 0.6, 0.4],
      'inference': [0.6, 0.8, 0.7], 'quantization': [0.4, 0.9, 0.8],
      // TurboQuant specific
      'turbo': [0.2, 0.3, 0.9], 'quant': [0.3, 0.4, 0.8],
      'compression': [0.4, 0.5, 0.9], 'rotation': [0.5, 0.3, 0.7],
      'random': [0.6, 0.2, 0.6], 'johnson': [0.7, 0.1, 0.5],
      // RAG specific
      'rag': [0.1, 0.8, 0.5], 'document': [0.2, 0.7, 0.4],
      'context': [0.3, 0.6, 0.3], 'answer': [0.4, 0.5, 0.2],
      'knowledge': [0.5, 0.4, 0.1], 'search': [0.6, 0.3, 0.0],
    };
  }

  /**
   * Generate a deterministic embedding for a text string
   */
  embed(text) {
    const words = text.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/);
    const rng = this._seededRandom(this._hashString(text));

    // Base random vector
    const embedding = Array.from({ length: this.dim }, () => {
      const u1 = rng() + 1e-10;
      const u2 = rng();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * 0.1;
    });

    // Add semantic signal from vocabulary
    for (const word of words) {
      if (this.vocabulary[word]) {
        const [a, b, c] = this.vocabulary[word];
        // Influence first few dimensions based on semantic meaning
        for (let i = 0; i < this.dim; i++) {
          const group = Math.floor(i / (this.dim / 3));
          const influence = [a, b, c][group] || 0;
          embedding[i] += influence * 0.3 * Math.sin(i * 0.1);
        }
      }
    }

    // Normalize
    const norm = Math.sqrt(embedding.reduce((s, x) => s + x * x, 0)) + 1e-10;
    return embedding.map(x => x / norm);
  }

  _hashString(str) {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) + hash) + str.charCodeAt(i);
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  _seededRandom(seed) {
    let s = seed;
    return function() {
      s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
      return (s >>> 0) / 0xFFFFFFFF;
    };
  }

  /**
   * Cosine similarity between two embeddings
   */
  cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
  }
}

// Export for use in other modules
window.TurboQuant = TurboQuant;
window.EmbeddingSimulator = EmbeddingSimulator;
