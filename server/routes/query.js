/**
 * server/routes/query.js
 * RAG query pipeline: embed → vector search (Supabase) → Gemini generate
 * Supports both one-shot JSON and SSE streaming responses.
 */
import { Router } from 'express';
import { createClient } from '@supabase/supabase-js';
import { embedQuery } from '../services/embedding.js';
import { computeStats } from '../services/turboquant.js';
import { generateAnswer, streamAnswer } from '../services/gemini.js';

const router = Router();

function supabase() {
  return createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY,
    { auth: { persistSession: false } }
  );
}

// ─── POST /api/query ──────────────────────────────────────────────────────────
// Body: { query, match_threshold?, match_count? }
// Returns full JSON response (non-streaming)
router.post('/', async (req, res) => {
  const {
    query,
    match_threshold = 0.3,
    match_count     = 5,
  } = req.body;

  if (!query || typeof query !== 'string') {
    return res.status(400).json({ success: false, error: '`query` is required' });
  }

  try {
    const t0 = Date.now();

    // 1. Embed the query (gemini-embedding-001, 1536-dim)
    const queryEmbedding = await embedQuery(query);
    const embedMs = Date.now() - t0;

    // 2. TurboQuant stats for the query vector
    const tqStats = computeStats(queryEmbedding);

    // 3. Semantic search via Supabase match_documents RPC
    const t1 = Date.now();
    const { data: retrieved, error } = await supabase().rpc('match_documents', {
      query_embedding: queryEmbedding,
      match_threshold,
      match_count,
    });
    if (error) throw error;
    const searchMs = Date.now() - t1;

    // 4. Generate answer with Gemini 2.0 Flash Lite
    const t2 = Date.now();
    const { answer, usage } = await generateAnswer(query, retrieved ?? []);
    const genMs = Date.now() - t2;

    const totalMs = Date.now() - t0;

    res.json({
      success: true,
      query,
      retrieved:  retrieved ?? [],
      answer,
      stats: {
        embedMs,
        searchMs,
        genMs,
        totalMs,
        docsFound:  (retrieved ?? []).length,
        turboquant: tqStats,
        usage,
      },
    });
  } catch (err) {
    console.error('[POST /query]', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

// ─── POST /api/query/stream ───────────────────────────────────────────────────
// Server-Sent Events (SSE) streaming version.
// Events: start | retrieved | chunk | done | error
router.post('/stream', async (req, res) => {
  const {
    query,
    match_threshold = 0.3,
    match_count     = 5,
  } = req.body;

  if (!query || typeof query !== 'string') {
    return res.status(400).json({ success: false, error: '`query` is required' });
  }

  // Set SSE headers early
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');

  const send = (event, data) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  try {
    const t0 = Date.now();

    // 1. Embed query
    send('start', { status: 'embedding' });
    const queryEmbedding = await embedQuery(query);
    const tqStats = computeStats(queryEmbedding);
    send('embedded', { embedMs: Date.now() - t0, turboquant: tqStats });

    // 2. Vector search
    send('searching', { status: 'searching' });
    const { data: retrieved, error } = await supabase().rpc('match_documents', {
      query_embedding: queryEmbedding,
      match_threshold,
      match_count,
    });
    if (error) throw error;

    send('retrieved', { documents: retrieved ?? [], count: (retrieved ?? []).length });

    // 3. Stream Gemini generation
    await streamAnswer(query, retrieved ?? [], res);

    // streamAnswer() closes res itself; done event emitted inside it

  } catch (err) {
    console.error('[POST /query/stream]', err.message);
    res.write(`event: error\ndata: ${JSON.stringify({ message: err.message })}\n\n`);
    res.end();
  }
});

export default router;
