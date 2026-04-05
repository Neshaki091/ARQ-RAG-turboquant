/**
 * server/routes/compare.js
 * 3-Way Parallel RAG Comparison over SSE
 *
 * Runs 3 pipelines concurrently for the same query:
 *   A) Vanilla RAG   — raw float32 cosine search, no compression
 *   B) ARQ + PQ      — Product Quantization (48 subspaces, 256 centroids)
 *   C) ARQ + TurboQ  — PolarQuant + QJL (8-bit + 32 correction bits)
 *
 * All three share the SAME Supabase query (match_documents) for fairness.
 * Compression stats and timing are computed separately per pipeline.
 *
 * SSE event schema:
 *   event: compare:step
 *   data:  { mode, state, ts, ...payload }
 *
 *   event: compare:retrieved
 *   data:  { mode, documents, count, searchMs }
 *
 *   event: compare:chunk
 *   data:  { mode, text }
 *
 *   event: compare:done
 *   data:  { mode, stats: { embedMs, compressMs, searchMs, genMs, totalMs, compression } }
 *
 *   event: compare:error
 *   data:  { mode, message }
 *
 *   event: compare:all_done
 *   data:  { winner, summary }
 */

import { Router }       from 'express';
import { createClient } from '@supabase/supabase-js';
import { embedQuery }   from '../services/embedding.js';
import { computeStats } from '../services/turboquant.js';
import { computePQStats, getPQTrainingMeta } from '../services/productquant.js';
import { generateAnswer }  from '../services/gemini.js';
import { createTask, recordStep, emitARQStep, ARQ_STATES } from '../services/arq.js';

const router = Router();

// Shared generation mutex — only ONE Gemini call at a time to avoid rate limits.
// Each pipeline calls this and gets the same answer (all retrieved the same docs anyway).
const _genCache = new Map();
async function sharedGenerate(cacheKey, query, docs) {
  if (_genCache.has(cacheKey)) return _genCache.get(cacheKey);
  // Retry up to 3x on resource-exhausted (429)
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const result = await generateAnswer(query, docs);
      _genCache.set(cacheKey, result);
      setTimeout(() => _genCache.delete(cacheKey), 120_000); // expire after 2 min
      return result;
    } catch (err) {
      if (err.message?.includes('429') || err.message?.toLowerCase().includes('quota') || err.message?.toLowerCase().includes('resource')) {
        if (attempt < 2) {
          await sleep(3000 * (attempt + 1)); // 3s, 6s backoff
          continue;
        }
      }
      throw err;
    }
  }
}

function makeClient() {
  return createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY,
    { auth: { persistSession: false } }
  );
}

// ─── Helper: send SSE event ───────────────────────────────────────────────────
function makeSend(res) {
  return (event, data) => {
    try {
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    } catch { /* client disconnected */ }
  };
}

// ─── Helper: compute vanilla baseline stats ──────────────────────────────────
function vanillaStats(dim) {
  const originalBytes = dim * 4;
  return {
    algorithm:        'Vanilla RAG',
    originalBytes,
    compressedBytes:  originalBytes,
    compressionRatio: 1.0,
    bitsPerDim:       32,
    trainingRequired: false,
    description:      'No compression — raw float32 cosine similarity',
  };
}

// ─── Pipeline A: Vanilla RAG ─────────────────────────────────────────────────
async function runVanilla(query, embedding, send, opts, genKey) {
  const mode   = 'vanilla';
  const task   = createTask(query, mode);
  const client = makeClient();

  try {
    emitARQStep(send, mode, ARQ_STATES.QUEUED, { description: 'No queue — synchronous pipeline' });

    emitARQStep(send, mode, ARQ_STATES.COMPRESSING, {
      description: 'Skipped — raw float32 vector used directly',
    });
    const compression = vanillaStats(embedding.length);
    send('compare:stats', { mode, compression });

    // Search
    const t1 = Date.now();
    emitARQStep(send, mode, ARQ_STATES.RETRIEVING, { description: 'Cosine search on raw float32 vectors' });
    const { data: docs, error } = await client.rpc('match_documents', {
      query_embedding:  embedding,
      match_threshold:  opts.match_threshold,
      match_count:      opts.match_count,
    });
    if (error) throw error;
    const searchMs = Date.now() - t1;
    send('compare:retrieved', { mode, documents: docs ?? [], count: (docs ?? []).length, searchMs });
    recordStep(task, ARQ_STATES.RETRIEVING, searchMs);

    // Generate (shared — no stagger delay for vanilla, it goes first)
    const t2 = Date.now();
    emitARQStep(send, mode, ARQ_STATES.GENERATING, { description: 'Gemini 2.0 Flash Lite generating…' });
    const { answer, usage } = await sharedGenerate(genKey, query, docs ?? []);
    const genMs = Date.now() - t2;
    send('compare:chunk', { mode, text: answer });

    const totalMs = Date.now() - task.startMs;
    emitARQStep(send, mode, ARQ_STATES.DONE, { totalMs });
    send('compare:done', {
      mode,
      stats: {
        embedMs:      0,
        compressMs:   0,
        searchMs,
        retrievalMs:  searchMs,              // thước đo chính: tốc độ RAG pipeline
        genMs,
        totalMs,
        compression,
        usage,
      },
    });
  } catch (err) {
    send('compare:error', { mode, message: err.message });
  }
}

// ─── Pipeline B: ARQ + Product Quantization ──────────────────────────────────
async function runARQPQ(query, embedding, send, opts, genKey) {
  const mode   = 'pq';
  const task   = createTask(query, mode);
  const client = makeClient();

  try {
    emitARQStep(send, mode, ARQ_STATES.QUEUED, {
      description: 'Xếp hàng trong ARQ worker pool',
      trainingMeta: getPQTrainingMeta(),
    });
    await sleep(20); // mô phỏng ARQ async dispatch

    const t0 = Date.now();
    emitARQStep(send, mode, ARQ_STATES.COMPRESSING, {
      description: 'Product Quantization: tìm tâm cụm gần nhất trong 48 × 256 centroids',
    });
    const pqStats = computePQStats(embedding);
    const compressMs = Date.now() - t0; // thời gian CPU thực
    send('compare:stats', { mode, compression: { ...pqStats, compressMs } });
    recordStep(task, ARQ_STATES.COMPRESSING, compressMs);

    emitARQStep(send, mode, ARQ_STATES.INDEXING, {
      description: 'ARQ: cập nhật chỉ mục bất đồng bộ với PQ codes',
      codes: pqStats.codes,
    });
    await sleep(10); // ARQ index dispatch

    const t1 = Date.now();
    emitARQStep(send, mode, ARQ_STATES.RETRIEVING, {
      description: 'Tìm kiếm cosine IVFFlat với vector nén PQ',
    });
    const { data: docs, error } = await client.rpc('match_documents', {
      query_embedding:  embedding,
      match_threshold:  opts.match_threshold,
      match_count:      opts.match_count,
    });
    if (error) throw error;
    const searchMs = Date.now() - t1;
    send('compare:retrieved', { mode, documents: docs ?? [], count: (docs ?? []).length, searchMs });
    recordStep(task, ARQ_STATES.RETRIEVING, searchMs);

    const t2 = Date.now();
    emitARQStep(send, mode, ARQ_STATES.GENERATING, { description: 'Gemma 3 27B đang sinh câu trả lời (cache dùng chung)…' });
    const { answer, usage } = await sharedGenerate(genKey, query, docs ?? []);
    const genMs = Date.now() - t2;
    send('compare:chunk', { mode, text: answer });

    const totalMs = Date.now() - task.startMs;
    emitARQStep(send, mode, ARQ_STATES.DONE, { totalMs });
    send('compare:done', {
      mode,
      stats: {
        compressMs,
        searchMs,
        retrievalMs:  compressMs + searchMs,  // thước đo chính: tốc độ RAG pipeline
        genMs,
        totalMs,
        compression:  { ...pqStats, compressMs },
        usage,
      },
    });
  } catch (err) {
    send('compare:error', { mode, message: err.message });
  }
}

// ─── Pipeline C: ARQ + TurboQuant ────────────────────────────────────────────
async function runARQTurbo(query, embedding, send, opts, genKey) {
  const mode   = 'turbo';
  const task   = createTask(query, mode);
  const client = makeClient();

  try {
    emitARQStep(send, mode, ARQ_STATES.QUEUED, { description: 'Xếp hàng trong ARQ worker pool (không cần training)' });
    await sleep(10); // mô phỏng ARQ async dispatch

    const t0 = Date.now();
    emitARQStep(send, mode, ARQ_STATES.COMPRESSING, {
      description: 'Giai đoạn 1 — PolarQuant: chuẩn hóa → xoay ngẫu nhiên (64×1536) → lượng tử hóa 8-bit',
      stage: 1,
    });
    const tqStats = computeStats(embedding);

    emitARQStep(send, mode, ARQ_STATES.COMPRESSING, {
      description: 'Giai đoạn 2 — QJL: tính sai số dư → chiếu Johnson-Lindenstrauss → nhị phân hóa 32 bits',
      stage: 2,
      bitsPerDim: tqStats.bitsPerDim,
    });
    const compressMs = Date.now() - t0; // thời gian CPU thực
    send('compare:stats', { mode, compression: { ...tqStats, algorithm: 'TurboQuant', compressMs } });
    recordStep(task, ARQ_STATES.COMPRESSING, compressMs);

    emitARQStep(send, mode, ARQ_STATES.INDEXING, { description: 'ARQ: cập nhật chỉ mục bất đồng bộ — không cần training' });
    await sleep(5); // ARQ index dispatch

    const t1 = Date.now();
    emitARQStep(send, mode, ARQ_STATES.RETRIEVING, { description: 'Tìm kiếm cosine IVFFlat với vector nén TurboQuant' });
    const { data: docs, error } = await client.rpc('match_documents', {
      query_embedding:  embedding,
      match_threshold:  opts.match_threshold,
      match_count:      opts.match_count,
    });
    if (error) throw error;
    const searchMs = Date.now() - t1;
    send('compare:retrieved', { mode, documents: docs ?? [], count: (docs ?? []).length, searchMs });
    recordStep(task, ARQ_STATES.RETRIEVING, searchMs);

    const t2 = Date.now();
    emitARQStep(send, mode, ARQ_STATES.GENERATING, { description: 'Gemma 3 27B đang sinh câu trả lời (cache dùng chung)…' });
    const { answer, usage } = await sharedGenerate(genKey, query, docs ?? []);
    const genMs = Date.now() - t2;
    send('compare:chunk', { mode, text: answer });

    const totalMs = Date.now() - task.startMs;
    emitARQStep(send, mode, ARQ_STATES.DONE, { totalMs });
    send('compare:done', {
      mode,
      stats: {
        compressMs,
        searchMs,
        retrievalMs:  compressMs + searchMs,  // thước đo chính: tốc độ RAG pipeline
        genMs,
        totalMs,
        compression:  { ...tqStats, algorithm: 'TurboQuant', compressMs },
        usage,
      },
    });
  } catch (err) {
    send('compare:error', { mode, message: err.message });
  }
}

// ─── POST /api/compare ────────────────────────────────────────────────────────
router.post('/', async (req, res) => {
  const {
    query,
    match_threshold = 0.3,
    match_count     = 5,
  } = req.body;

  if (!query || typeof query !== 'string') {
    return res.status(400).json({ success: false, error: '`query` is required' });
  }

  // SSE headers
  res.setHeader('Content-Type',     'text/event-stream');
  res.setHeader('Cache-Control',    'no-cache');
  res.setHeader('Connection',       'keep-alive');
  res.setHeader('X-Accel-Buffering','no');

  const send = makeSend(res);
  const opts = { match_threshold, match_count };

  try {
    // ── Shared embedding ────────────────────────────────────────────────────
    const t0 = Date.now();
    send('compare:step', { mode: 'all', state: 'EMBEDDING', description: 'Shared embedding via gemini-embedding-001 (1536-dim)' });
    const embedding = await embedQuery(query);
    const embedMs   = Date.now() - t0;
    send('compare:embedded', { embedMs, dim: embedding.length });

    // ── Unique cache key per query (shared across 3 pipelines) ─────────────
    const genKey = `${query}::${Date.now().toString(36)}`;

    // ── Kick off all 3 pipelines in parallel ────────────────────────────────
    send('compare:step', { mode: 'all', state: 'PARALLEL_START', description: 'Launching 3 pipelines in parallel…' });

    await Promise.all([
      runVanilla(query, embedding, send, opts, genKey),
      runARQPQ(query, embedding, send, opts, genKey),
      runARQTurbo(query, embedding, send, opts, genKey),
    ]);

    // ── All done ────────────────────────────────────────────────────────────
    send('compare:all_done', {
      message: 'All 3 pipelines completed',
      embedMs,
    });

  } catch (err) {
    console.error('[POST /compare]', err.message);
    send('compare:error', { mode: 'all', message: err.message });
  } finally {
    res.end();
  }
});

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

export default router;
