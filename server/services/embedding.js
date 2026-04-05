/**
 * server/services/embedding.js
 * Google gemini-embedding-001 — 1536-dim (Matryoshka)
 * Matches VECTOR(1536) in supabase/002_migrate_1536.sql
 */
import { GoogleGenerativeAI } from '@google/generative-ai';
import { createHash } from 'crypto';

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const OUTPUT_DIM = 1536;
const embeddingModel = genAI.getGenerativeModel({ model: 'gemini-embedding-001' });

/**
 * Generate a stable chunk_id (MD5) for upsert dedup.
 * Matches the chunk_id UNIQUE constraint in the documents table.
 */
export function generateChunkId(text) {
  return createHash('md5').update(text).digest('hex');
}

/**
 * Generate a 1536-dimensional embedding for a document.
 * @param {string} text
 * @returns {Promise<number[]>} float32 array, length 1536
 */
export async function embedText(text) {
  if (!text || typeof text !== 'string') {
    throw new Error('embedText: text must be a non-empty string');
  }

  const result = await embeddingModel.embedContent({
    content: { parts: [{ text }], role: 'user' },
    taskType: 'RETRIEVAL_DOCUMENT',
    outputDimensionality: OUTPUT_DIM,
  });

  return result.embedding.values; // float32 array, length 1536
}

/**
 * Generate a 1536-dimensional embedding for a query.
 * @param {string} query
 * @returns {Promise<number[]>} float32 array, length 1536
 */
export async function embedQuery(query) {
  if (!query || typeof query !== 'string') {
    throw new Error('embedQuery: query must be a non-empty string');
  }

  const result = await embeddingModel.embedContent({
    content: { parts: [{ text: query }], role: 'user' },
    taskType: 'RETRIEVAL_QUERY',
    outputDimensionality: OUTPUT_DIM,
  });

  return result.embedding.values;
}

/**
 * Batch embed multiple texts (sequential to avoid rate limits).
 * @param {string[]} texts
 * @returns {Promise<number[][]>}
 */
export async function embedBatch(texts) {
  const embeddings = [];
  for (const text of texts) {
    const emb = await embedText(text);
    embeddings.push(emb);
    // Small delay to respect rate limits
    await new Promise(r => setTimeout(r, 100));
  }
  return embeddings;
}
