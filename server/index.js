/**
 * server/index.js
 * Express app entry point — ARQ-RAG × TurboQuant backend
 */
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

import documentsRouter from './routes/documents.js';
import queryRouter     from './routes/query.js';
import compareRouter   from './routes/compare.js';
import ingestRouter    from './routes/ingest.js';

// ─── Config ───────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
const __dirname = dirname(fileURLToPath(import.meta.url));

// Validate required env vars on startup
const REQUIRED_ENV = ['GOOGLE_API_KEY', 'SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY'];
for (const key of REQUIRED_ENV) {
  if (!process.env[key]) {
    console.error(`❌  Missing environment variable: ${key}`);
    process.exit(1);
  }
}

// ─── App ──────────────────────────────────────────────────────────────────────
const app = express();

// Middleware
app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '2mb' }));

// Serve static frontend from project root
const ROOT = join(__dirname, '..');
app.use(express.static(ROOT, {
  // Don't serve server/ directory
  index: 'index.html',
}));

// ─── API Routes ───────────────────────────────────────────────────────────────
app.use('/api/documents', documentsRouter);
app.use('/api/documents', ingestRouter);
app.use('/api/query',     queryRouter);
app.use('/api/compare',   compareRouter);

// Health check
app.get('/api/health', (_req, res) => {
  res.json({
    status:  'ok',
    model:   'gemini-embedding-001 + gemma-3-27b-it (fallback: gemini-1.5-flash)',
    dim:     1536,
    version: '1.0.0',
  });
});

// ─── Fallback SPA ─────────────────────────────────────────────────────────────
app.get('*', (_req, res) => {
  res.sendFile(join(ROOT, 'index.html'));
});

// ─── Start ────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🚀  ARQ-RAG × TurboQuant server`);
  console.log(`   http://localhost:${PORT}\n`);
  console.log(`   Embedding : gemini-embedding-001 (1536-dim)`);
  console.log(`   LLM       : gemma-3-27b-it`);
  console.log(`   Supabase  : ${process.env.SUPABASE_URL}`);
  console.log();
});

export default app;
