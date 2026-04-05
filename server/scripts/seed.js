/**
 * server/scripts/seed.js
 * Seeds the Supabase knowledge base with 8 sample documents.
 * Uses the 002_migrate_1536.sql schema (chunk_id, content, heading, source, chunk_index)
 *
 * Run with:  npm run seed
 */
import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import { embedText, generateChunkId } from '../services/embedding.js';
import { computeStats } from '../services/turboquant.js';

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY,
  { auth: { persistSession: false } }
);

const DOCUMENTS = [
  {
    heading: 'TurboQuant: Efficient Vector Quantization for RAG',
    content:
      'TurboQuant is a novel vector quantization algorithm developed by Google Research (ICLR 2026). It uses a two-stage approach: Stage 1 (PolarQuant) applies a random rotation matrix to the embedding vector, spreading its energy so coordinates become nearly independent. Stage 2 applies a 1-bit Quantized Johnson-Lindenstrauss (QJL) transform to the residual error, providing an unbiased inner-product estimate. TurboQuant achieves near-zero indexing time without dataset-specific training, making it ideal for dynamic RAG systems.',
    source: 'arxiv:2504.19874',
  },
  {
    heading: 'Retrieval-Augmented Generation (RAG) Architecture',
    content:
      'RAG combines a retrieval mechanism with a large language model generator. When a user query is received it is embedded into a dense vector and compared against a knowledge base using maximum inner-product search (MIPS). The most relevant document chunks are returned and injected as context into the LLM prompt. This grounds the model\'s output in factual documents and drastically reduces hallucination. Key pipeline stages: embed → compress → search → retrieve → augment → generate.',
    source: 'rag-overview',
  },
  {
    heading: 'Vector Databases and Similarity Search',
    content:
      'Vector databases store high-dimensional embedding vectors and enable efficient similarity search. Common indexing algorithms include IVFFlat (inverted file with flat quantization, fast approximate search), HNSW (hierarchical navigable small world graphs, excellent recall), and ScaNN (Google\'s SOAR algorithm). Supabase uses pgvector which supports both IVFFlat and HNSW indexes natively. Cosine similarity (1 - cosine distance) is the standard metric for semantic search.',
    source: 'vector-db-overview',
  },
  {
    heading: 'Product Quantization vs TurboQuant',
    content:
      'Product Quantization (PQ) has been the standard for approximate nearest-neighbour search since 2011. It requires expensive k-means training on the full dataset (offline phase of hours) and must be retrained when documents are added or changed. TurboQuant is data-oblivious: no training is required, each vector is compressed independently in microseconds. Recall@10 for TurboQuant (8-bit) is comparable to PQ while eliminating the operational burden of offline re-indexing.',
    source: 'turboquant-vs-pq',
  },
  {
    heading: 'LLM Context Windows and KV-Cache Compression',
    content:
      'Large Language Models process tokens within a fixed context window. Long-context RAG (many retrieved chunks) can exceed this limit. TurboQuant addresses this by compressing the KV-cache activations, allowing the model to fit more document context on the same hardware. Combined with RAG, this enables processing of longer and richer context with reduced GPU memory footprint, effectively extending the usable context window without additional hardware.',
    source: 'llm-optimization',
  },
  {
    heading: 'ARQ: Asynchronous Request Queue in RAG Systems',
    content:
      'ARQ (Asynchronous Request Queue) decouples document ingestion from query serving in production RAG deployments. Document embedding jobs are queued asynchronously so that large batch ingestion does not block real-time query handling. Combined with TurboQuant compression, new documents can be embedded and indexed in near-real-time. This architecture supports high-throughput ingestion pipelines and low-latency query serving simultaneously using a shared Supabase vector store.',
    source: 'arq-architecture',
  },
  {
    heading: 'Gemini Embedding Models: From 768 to 1536 Dimensions',
    content:
      'Google\'s gemini-embedding-001 model supports Matryoshka Representation Learning, allowing output dimensions from 1 up to 3072. Setting outputDimensionality=1536 produces compact embeddings that balance storage efficiency and semantic richness. These embeddings are language-agnostic and support over 100 languages. For RAG in Supabase, a VECTOR(1536) column with an IVFFlat or HNSW index provides fast cosine similarity search across millions of document chunks.',
    source: 'gemini-embedding',
  },
  {
    heading: 'Johnson-Lindenstrauss Lemma and Random Projections',
    content:
      'The Johnson-Lindenstrauss (JL) lemma states that a set of n points in high-dimensional Euclidean space can be projected to O(log n / ε²) dimensions while approximately preserving all pairwise distances within a factor of (1 ± ε). Random projection matrices with Rademacher (±1) entries satisfy this property efficiently. TurboQuant\'s QJL stage exploits this theorem: a 1-bit projection of the quantization residual provides an unbiased estimator of the residual\'s inner product contribution, correcting the bias introduced by scalar quantization.',
    source: 'math-foundations',
  },
];

async function seed() {
  console.log('🌱  Seeding ARQ-RAG knowledge base...\n');
  let inserted = 0, skipped = 0;

  for (let i = 0; i < DOCUMENTS.length; i++) {
    const { heading, content, source } = DOCUMENTS[i];
    const chunk_id    = generateChunkId(content);
    const chunk_index = i;

    process.stdout.write(`  [${i + 1}/${DOCUMENTS.length}] ${heading.slice(0, 50)}… `);

    try {
      const embedding = await embedText(content);
      const tqStats   = computeStats(embedding);

      const { error } = await supabase
        .from('documents')
        .upsert(
          { chunk_id, content, heading, source, chunk_index, embedding },
          { onConflict: 'chunk_id' }
        );

      if (error) throw error;

      console.log(`✅  (${tqStats.compressionRatio}× compressed)`);
      inserted++;
    } catch (err) {
      console.log(`❌  ${err.message}`);
      skipped++;
    }

    // Respect rate limit
    if (i < DOCUMENTS.length - 1) await new Promise(r => setTimeout(r, 300));
  }

  console.log(`\n✨  Done. Inserted/updated: ${inserted} | Errors: ${skipped}`);
  process.exit(skipped > 0 ? 1 : 0);
}

seed();
