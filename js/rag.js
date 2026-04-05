/**
 * js/rag.js  — Frontend RAG client
 * Calls the real Express backend at /api/*
 * Replaces the previous mock EmbeddingSimulator / KnowledgeBase
 */

const API_BASE = window.location.origin; // same-origin when served by Express

class RAGClient {
  constructor() {
    this._cache = null;
  }

  /**
   * Fetch all documents from Supabase knowledge base.
   * @returns {Promise<Array>}
   */
  async listDocuments() {
    const res = await fetch(`${API_BASE}/api/documents`);
    const body = await res.json();
    if (!body.success) throw new Error(body.error);
    return body.documents;
  }

  /**
   * Ingest a single document chunk.
   * @param {{ content: string, heading?: string, source?: string, chunk_index?: number }} doc
   */
  async ingestDocument(doc) {
    const res = await fetch(`${API_BASE}/api/documents`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(doc),
    });
    const body = await res.json();
    if (!body.success) throw new Error(body.error);
    return body;
  }

  /**
   * Run the full RAG pipeline (non-streaming).
   * @param {string} query
   * @param {{ match_threshold?: number, match_count?: number }} opts
   * @returns {Promise<{ query, retrieved, answer, stats }>}
   */
  async query(query, opts = {}) {
    const res = await fetch(`${API_BASE}/api/query`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query, ...opts }),
    });
    const body = await res.json();
    if (!body.success) throw new Error(body.error);
    return body;
  }

  /**
   * Run RAG pipeline with SSE streaming.
   * @param {string} query
   * @param {object} opts
   * @param {{
   *   onStart?:     (data) => void,
   *   onEmbedded?:  (data) => void,
   *   onSearching?: (data) => void,
   *   onRetrieved?: (data) => void,
   *   onChunk?:     (text: string) => void,
   *   onDone?:      (data) => void,
   *   onError?:     (msg: string) => void,
   * }} callbacks
   */
  async queryStream(query, opts = {}, callbacks = {}) {
    const res = await fetch(`${API_BASE}/api/query/stream`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query, ...opts }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      callbacks.onError?.(`${res.status} ${err.error || res.statusText}`);
      return;
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buf     = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buf += decoder.decode(value, { stream: true });

      // Parse complete SSE messages from buffer
      const messages = buf.split('\n\n');
      buf = messages.pop(); // keep incomplete tail

      for (const msg of messages) {
        const lines = msg.split('\n');
        let event = 'message', data = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) event = line.slice(7).trim();
          if (line.startsWith('data: '))  data  = line.slice(6).trim();
        }

        if (!data) continue;
        let parsed;
        try { parsed = JSON.parse(data); } catch { continue; }

        switch (event) {
          case 'start':      callbacks.onStart?.(parsed);          break;
          case 'embedded':   callbacks.onEmbedded?.(parsed);       break;
          case 'searching':  callbacks.onSearching?.(parsed);      break;
          case 'retrieved':  callbacks.onRetrieved?.(parsed);      break;
          case 'chunk':      callbacks.onChunk?.(parsed.text);     break;
          case 'done':       callbacks.onDone?.(parsed);           break;
          case 'error':      callbacks.onError?.(parsed.message);  break;
        }
      }
    }
  }

  /**
   * Health check
   */
  async health() {
    const res  = await fetch(`${API_BASE}/api/health`);
    return res.json();
  }
}

// Export singleton
window.ragClient = new RAGClient();
