/**
 * server/routes/documents.js
 * CRUD for knowledge-base documents.
 * Schema: id, chunk_id, content, heading, source, chunk_index, embedding, created_at
 */
import { Router } from 'express';
import { createClient } from '@supabase/supabase-js';
import { embedText, generateChunkId } from '../services/embedding.js';
import { computeStats } from '../services/turboquant.js';

const router = Router();

function supabase() {
  return createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY,
    { auth: { persistSession: false } }
  );
}

// ─── GET /api/documents ──────────────────────────────────────────────────────
// Return all docs (no embeddings to keep payload small)
router.get('/', async (_req, res) => {
  try {
    const { data, error } = await supabase()
      .from('documents')
      .select('id, chunk_id, content, heading, source, chunk_index, created_at')
      .order('chunk_index', { ascending: true });

    if (error) throw error;
    res.json({ success: true, documents: data, count: data.length });
  } catch (err) {
    console.error('[GET /documents]', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

// ─── POST /api/documents ─────────────────────────────────────────────────────
// Ingest one chunk: embed → upsert into Supabase
// Body: { content, heading?, source?, chunk_index? }
router.post('/', async (req, res) => {
  const {
    content,
    heading    = null,
    source     = 'manual',
    chunk_index = 0,
  } = req.body;

  if (!content) {
    return res.status(400).json({ success: false, error: '`content` is required' });
  }

  try {
    const t0 = Date.now();

    // 1. Dedup key
    const chunk_id = generateChunkId(content);

    // 2. Embed with gemini-embedding-001 (1536-dim)
    const embedding = await embedText(content);
    const embedMs   = Date.now() - t0;

    // 3. TurboQuant stats (metadata only — DB stores raw float32)
    const tqStats = computeStats(embedding);

    // 4. Upsert (chunk_id UNIQUE → update on conflict)
    const { data, error } = await supabase()
      .from('documents')
      .upsert(
        { chunk_id, content, heading, source, chunk_index, embedding },
        { onConflict: 'chunk_id' }
      )
      .select('id, chunk_id, heading, source, chunk_index, created_at')
      .single();

    if (error) throw error;

    const totalMs = Date.now() - t0;
    console.log(`[INGEST] chunk_id=${chunk_id.slice(0, 8)}… | embed: ${embedMs}ms | total: ${totalMs}ms`);

    res.status(201).json({
      success: true,
      document: data,
      stats: { embedMs, totalMs, turboquant: tqStats },
    });
  } catch (err) {
    console.error('[POST /documents]', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

// ─── DELETE /api/documents/:id ────────────────────────────────────────────────
router.delete('/:id', async (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (isNaN(id)) return res.status(400).json({ success: false, error: 'Invalid id' });

  try {
    const { error } = await supabase().from('documents').delete().eq('id', id);
    if (error) throw error;
    res.json({ success: true, message: `Document ${id} deleted` });
  } catch (err) {
    console.error('[DELETE /documents/:id]', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

export default router;
