/**
 * js/demo.js — Main UI Controller (3-way comparison version)
 * Connects frontend to /api/compare endpoint.
 * Delegates visualization to window.arqVisualizer (arq-visualizer.js)
 */

class DemoController {
  constructor() {
    this.isProcessing = false;
    this._bindEvents();
    this._initOnLoad();
  }

  // ─── Init ─────────────────────────────────────────────────────────────────
  async _initOnLoad() {
    this._initPipelineViz();
    this._animateHeroStats();

    try {
      const [health, documents] = await Promise.all([
        ragClient.health(),
        ragClient.listDocuments(),
      ]);
      this._renderKnowledgeBase(documents);
      this._updateSupabaseBadge(true);
      console.log('[Demo] Health:', health);
    } catch (err) {
      console.warn('[Demo] Backend not reachable:', err.message);
      this._showBackendWarning();
      this._updateSupabaseBadge(false);
    }
  }

  _showBackendWarning() {
    const grid = document.getElementById('docs-grid');
    if (grid) {
      grid.innerHTML = `
        <div style="grid-column:1/-1; padding:32px; text-align:center; color:var(--accent-amber);">
          ⚠️ Server backend chưa chạy.<br>
          <small style="color:var(--text-muted)">Khởi động bằng <code>npm run dev</code> rồi làm mới trang.</small>
        </div>`;
    }
  }

  _updateSupabaseBadge(ok) {
    // Optional: add a status indicator if present
    const el = document.getElementById('supabase-status');
    if (el) {
      el.className = ok ? 'status-dot status-ok' : 'status-dot status-err';
      el.title     = ok ? 'Supabase connected' : 'Supabase unreachable';
    }
  }

  // ─── Event bindings ───────────────────────────────────────────────────────
  _bindEvents() {
    document.getElementById('query-btn')
      ?.addEventListener('click', () => this.runCompare());

    document.getElementById('query-input')
      ?.addEventListener('keydown', e => { if (e.key === 'Enter') this.runCompare(); });

    document.querySelectorAll('.query-chip').forEach(chip =>
      chip.addEventListener('click', () => {
        const q = chip.dataset.query;
        const inp = document.getElementById('query-input');
        if (inp) inp.value = q;
        this.runCompare(q);
      })
    );

    document.querySelectorAll('.algo-tab').forEach(tab =>
      tab.addEventListener('click', () => {
        document.querySelectorAll('.algo-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.algo-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tab.dataset.target)?.classList.add('active');
      })
    );

    // Bắt sự kiện cho phần Nhúng (Ingestion)
    const btnIngest = document.getElementById('btn-ingest');
    const modalIngest = document.getElementById('ingest-modal');
    const closeIngest = document.getElementById('btn-close-ingest');

    if (btnIngest && modalIngest) {
      btnIngest.addEventListener('click', () => {
         modalIngest.classList.remove('hidden');
         this.runIngest();
      });
      closeIngest.addEventListener('click', () => {
         if (!this.isIngesting) {
            modalIngest.classList.add('hidden');
         }
      });
    }
  }

  // ─── Ingestion ────────────────────────────────────────────────────────────
  async runIngest() {
    if (this.isIngesting) return;
    this.isIngesting = true;
    const btn = document.getElementById('btn-ingest');
    const logs = document.getElementById('ingest-logs');
    const statsLabel = document.getElementById('ingest-stats-label');
    const truncate = document.getElementById('truncate-db')?.checked ?? true;
    
    if (btn) {
       btn.disabled = true;
       btn.style.opacity = '0.5';
       btn.innerHTML = '<span class="spinner"></span> Đang xử lý...';
    }
    if (logs) logs.innerHTML = '';
    const addLog = (msg, isErr = false) => {
        if (!logs) return;
        const div = document.createElement('div');
        if (isErr) div.style.color = '#ef4444';
        div.textContent = `> ${msg}`;
        logs.appendChild(div);
        logs.scrollTop = logs.scrollHeight;
    };

    const API = window.location.origin;
    const url = `${API}/api/documents/ingest-crawled?truncate=${truncate}`;
    
    try {
      const source = new EventSource(url);
      
      source.addEventListener('log', (e) => addLog(JSON.parse(e.data).message));
      
      source.addEventListener('stats', (e) => {
          const data = JSON.parse(e.data);
          if (statsLabel) statsLabel.textContent = `${data.current} / ${data.total} Bài Báo`;
      });
      
      source.addEventListener('error', (e) => addLog(JSON.parse(e.data).message, true));
      
      source.addEventListener('done', (e) => {
          addLog('🎉 HOÀN TẤT NHÚNG DỮ LIỆU CÀO!');
          source.close();
          this.isIngesting = false;
          if (btn) {
              btn.disabled = false;
              btn.style.opacity = '1';
              btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg> Nhúng Thư Viện Cào (PDF)`;
          }
          this._initOnLoad(); // Refresh document count
      });
      
      source.onerror = (e) => {
          // EventSource error could mean complete if server closed early without 'done' event, or error.
          source.close();
          if (this.isIngesting) {
              addLog('⚠ Kết nối stream bị đóng. Tiến trình có thể đã hoàn tất hoặc bị lỗi mạng.', true);
              this.isIngesting = false;
              if (btn) {
                  btn.disabled = false;
                  btn.style.opacity = '1';
                  btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg> Nhúng Thư Viện Cào (PDF)`;
              }
              this._initOnLoad();
          }
      };

    } catch (err) {
      this.isIngesting = false;
      addLog('Lỗi gọi API: ' + err.message, true);
      if (btn) {
         btn.disabled = false;
         btn.style.opacity = '1';
         btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg> Nhúng Thư Viện Cào (PDF)`;
      }
    }
  }

  // ─── Run 3-way comparison ─────────────────────────────────────────────────
  async runCompare(override = null) {
    if (this.isProcessing) return;
    const input = document.getElementById('query-input');
    const query = override ?? input?.value?.trim();
    if (!query) return;
    if (input) input.value = query;

    this.isProcessing = true;
    this._setButtonLoading(true);

    // Reset visualizer
    window.arqVisualizer?.reset();
    this._highlightPipelineStep(-1);

    try {
      await this._streamCompare(query);
    } catch (err) {
      console.error('[Demo] compare error:', err.message);
    } finally {
      this.isProcessing = false;
      this._setButtonLoading(false);
    }
  }

  async _streamCompare(query) {
    const API = window.location.origin;
    const res = await fetch(`${API}/api/compare`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query, match_threshold: 0.3, match_count: 5 }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || res.statusText);
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buf     = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buf += decoder.decode(value, { stream: true });

      const messages = buf.split('\n\n');
      buf = messages.pop();

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

        this._handleSSEEvent(event, parsed);
      }
    }
  }

  _handleSSEEvent(event, data) {
    const viz = window.arqVisualizer;
    if (!viz) return;

    switch (event) {
      case 'compare:embedded':
        this._highlightPipelineStep(1);
        viz.onSharedEmbedding(data);
        break;

      case 'compare:step': {
        const { mode, state } = data;
        if (mode === 'all') {
          // Global step
          if (state === 'EMBEDDING')      this._highlightPipelineStep(1);
          if (state === 'PARALLEL_START') this._highlightPipelineStep(2);
        } else {
          viz.onStep(data);
          if (state === 'RETRIEVING') this._highlightPipelineStep(3);
          if (state === 'GENERATING') this._highlightPipelineStep(5);
        }
        break;
      }

      case 'compare:stats':
        viz.onStats(data);
        break;

      case 'compare:retrieved':
        this._highlightPipelineStep(4);
        viz.onRetrieved(data);
        break;

      case 'compare:chunk':
        viz.onChunk(data);
        break;

      case 'compare:done':
        viz.onDone(data);
        break;

      case 'compare:error':
        viz.onError(data);
        break;

      case 'compare:all_done':
        this._highlightPipelineStep(5);
        console.log('[Demo] All pipelines done:', data);
        break;
    }
  }

  // ─── Knowledge Base render ────────────────────────────────────────────────
  _renderKnowledgeBase(docs) {
    const grid = document.getElementById('docs-grid');
    if (!grid) return;

    grid.innerHTML = '';
    docs.forEach((doc, i) => {
      const card = document.createElement('div');
      card.className = 'doc-card';
      card.style.animationDelay = `${i * 0.07}s`;
      const preview = (doc.content || '').substring(0, 100);
      card.innerHTML = `
        <div class="doc-card-header">
          <span class="doc-emoji">📄</span>
          <span class="doc-category">${doc.source || 'chung'}</span>
        </div>
        <h4 class="doc-title">${doc.heading || '(không có tiêu đề)'}</h4>
        <p class="doc-snippet">${preview}…</p>
        <div class="doc-stats">
          <span>📐 1536 chiều</span>
          <span>🔢 đoạn ${doc.chunk_index ?? 0}</span>
          <span>⚡ ~90× nén</span>
        </div>`;
      grid.appendChild(card);
    });

    const statDocs = document.getElementById('stat-docs');
    if (statDocs) this._countUp(statDocs, 0, docs.length, 800, ' docs');
  }

  // ─── Pipeline viz (section overview) ─────────────────────────────────────
  _initPipelineViz() {
    document.querySelectorAll('.pipeline-step').forEach((step, i) => {
      setTimeout(() => step.classList.add('visible'), i * 180 + 400);
    });
    document.querySelectorAll('.pipeline-connector').forEach((conn, i) => {
      setTimeout(() => conn.classList.add('visible'), i * 180 + 600);
    });
  }

  _highlightPipelineStep(idx) {
    document.querySelectorAll('.pipeline-step').forEach((step, i) => {
      step.classList.toggle('active',    i === idx);
      step.classList.toggle('completed', i < idx);
    });
    document.querySelectorAll('.pipeline-connector').forEach((conn, i) => {
      conn.classList.toggle('active', i < idx);
    });
  }

  // ─── Button state ─────────────────────────────────────────────────────────
  _setButtonLoading(loading) {
    const btn = document.getElementById('query-btn');
    if (!btn) return;
    btn.disabled = loading;
    btn.innerHTML = loading
      ? '<span class="btn-spinner"></span> Đang chạy…'
      : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg> Chạy Cả 3';
  }

  // ─── Hero counter animation ───────────────────────────────────────────────
  _animateHeroStats() {
    [
      { id: 'stat-dim',   end: 1536, suffix: 'd' },
      { id: 'stat-ratio', end: 90.4, suffix: '×', dec: 1 },
      { id: 'stat-speed', end: 97,   suffix: '%' },
    ].forEach(({ id, end, suffix, dec = 0 }) => {
      const el = document.getElementById(id);
      if (el) this._countUp(el, 0, end, 1200, suffix, dec);
    });
  }

  _countUp(el, from, to, ms, suffix = '', dec = 0) {
    const start = performance.now();
    const step = (now) => {
      const t = Math.min((now - start) / ms, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = (from + (to - from) * eased).toFixed(dec) + suffix;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.demoController = new DemoController();
});
