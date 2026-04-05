/**
 * js/arq-visualizer.js
 * ARQ + TurboQuant / PQ  Step-by-step Visualizer
 *
 * Manages the 3-column parallel comparison UI:
 *   col[vanilla] | col[pq] | col[turbo]
 *
 * Each column has:
 *  - ARQ State Machine animation
 *  - Compression stats panel
 *  - Retrieved docs + similarity scores
 *  - Generated answer (streaming)
 *  - Timing breakdown
 */

// ─── ARQ State Config ────────────────────────────────────────────────────────
const ARQ_STATE_ORDER = [
  'QUEUED', 'EMBEDDING', 'COMPRESSING', 'INDEXING', 'RETRIEVING', 'GENERATING', 'DONE'
];

const STATE_META = {
  IDLE:        { icon: '⏸',  label: 'Chờ',          color: '#475569' },
  QUEUED:      { icon: '📥', label: 'Xếp hàng',      color: '#f59e0b' },
  EMBEDDING:   { icon: '🔢', label: 'Nhúng vector',  color: '#8b5cf6' },
  COMPRESSING: { icon: '⚡', label: 'Nén vector',    color: '#06b6d4' },
  INDEXING:    { icon: '🗂', label: 'Lập chỉ mục',  color: '#3b82f6' },
  RETRIEVING:  { icon: '🔍', label: 'Tìm kiếm',      color: '#10b981' },
  GENERATING:  { icon: '🤖', label: 'Sinh câu trả lời', color: '#ec4899' },
  DONE:        { icon: '✅', label: 'Hoàn thành',    color: '#22c55e' },
  ERROR:       { icon: '❌', label: 'Lỗi',            color: '#ef4444' },
};

const MODE_META = {
  vanilla: { label: 'Vanilla RAG',   color: '#ef4444', icon: '🔴' },
  pq:      { label: 'ARQ + PQ',      color: '#f59e0b', icon: '🟡' },
  turbo:   { label: 'ARQ + TurboQ',  color: '#10b981', icon: '🟢' },
};

// ─────────────────────────────────────────────────────────────────────────────
class ARQVisualizer {
  constructor() {
    this._states   = { vanilla: 'IDLE', pq: 'IDLE', turbo: 'IDLE' };
    this._timers   = { vanilla: null, pq: null, turbo: null };
    this._startTs  = {};
    this._stats    = {};
    this._answers  = { vanilla: '', pq: '', turbo: '' };
    this._stepLogs = { vanilla: [], pq: [], turbo: [] };
    this._doneCount = 0;
    this._results  = {};
  }

  // ─── Public API ───────────────────────────────────────────────────────────

  reset() {
    this._states    = { vanilla: 'IDLE', pq: 'IDLE', turbo: 'IDLE' };
    this._answers   = { vanilla: '', pq: '', turbo: '' };
    this._stepLogs  = { vanilla: [], pq: [], turbo: [] };
    this._stats     = {};
    this._results   = {};
    this._doneCount = 0;

    ['vanilla', 'pq', 'turbo'].forEach(m => {
      this._clearColumn(m);
      this._setColumnState(m, 'IDLE');
    });

    // Clear timing table
    const tbody = document.getElementById('compare-timing-body');
    if (tbody) tbody.innerHTML = '';
  }

  onSharedEmbedding(data) {
    const el = document.getElementById('compare-embed-info');
    if (el) {
      el.innerHTML = `<span class="log-ts">${this._now()}</span> ⚡ Embedding complete — <strong>${data.dim}</strong>-dim vector in <strong>${data.embedMs}ms</strong>`;
      el.classList.add('visible');
    }
    // Mark all columns as past embedding
    ['vanilla', 'pq', 'turbo'].forEach(m => this._setState(m, 'QUEUED'));
  }

  onStep(data) {
    const { mode, state, description } = data;
    if (!MODE_META[mode]) return;
    this._setState(mode, state);
    if (description) this._appendLog(mode, state, description);
  }

  onStats(data) {
    const { mode, compression } = data;
    this._stats[mode] = compression;
    this._renderCompressionPanel(mode, compression);
  }

  onRetrieved(data) {
    const { mode, documents, count, searchMs } = data;
    this._appendLog(mode, 'RETRIEVING', `🔍 Trụy xuất ${count} tài liệu trong ${searchMs}ms`);
    this._renderRetrievedDocs(mode, documents);
  }

  onChunk(data) {
    const { mode, text } = data;
    this._answers[mode] += text;
    this._setState(mode, 'GENERATING');
    this._renderAnswer(mode, this._answers[mode]);
  }

  onDone(data) {
    const { mode, stats } = data;
    this._results[mode] = stats;
    this._setState(mode, 'DONE');
    const retMs = stats.retrievalMs ?? stats.searchMs;
    this._appendLog(mode, 'DONE',
      `✅ Retrieval: ${retMs}ms (nén ${stats.compressMs ?? 0}ms + tìm kiếm ${stats.searchMs}ms)`);
    this._renderTiming(mode, stats);
    this._doneCount++;
    if (this._doneCount >= 3) this._renderFinalComparison();
  }

  onError(data) {
    const { mode, message } = data;
    this._setState(mode, 'ERROR');
    this._appendLog(mode, 'ERROR', message);
  }

  // ─── Internal render helpers ─────────────────────────────────────────────

  _now() {
    return new Date().toISOString().split('T')[1].slice(0, 12);
  }

  _setState(mode, state) {
    if (!ARQ_STATE_ORDER.includes(state) && state !== 'IDLE' && state !== 'ERROR') return;
    this._states[mode] = state;
    this._setColumnState(mode, state);
  }

  _setColumnState(mode, state) {
    const col   = document.getElementById(`col-${mode}`);
    if (!col) return;

    const meta  = STATE_META[state] || STATE_META.IDLE;
    const badge  = col.querySelector('.arq-state-badge');
    if (badge) {
      badge.textContent = `${meta.icon} ${meta.label}`;
      badge.style.setProperty('--badge-color', meta.color);
    }

    // Highlight state nodes
    col.querySelectorAll('.arq-node').forEach(node => {
      const ns = node.dataset.state;
      node.classList.remove('active', 'done', 'error');
      if (ns === state) node.classList.add('active');
      else if (ARQ_STATE_ORDER.indexOf(ns) < ARQ_STATE_ORDER.indexOf(state)) node.classList.add('done');
    });
  }

  _appendLog(mode, state, desc) {
    const log = document.getElementById(`log-${mode}`);
    if (!log) return;

    const meta = STATE_META[state] || { icon: '→', color: '#64748b' };
    const entry = document.createElement('div');
    entry.className = 'arq-log-entry';
    entry.innerHTML = `
      <span class="arq-log-ts">${this._now()}</span>
      <span class="arq-log-icon" style="color:${meta.color}">${meta.icon}</span>
      <span class="arq-log-text">${desc}</span>`;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
  }

  _clearColumn(mode) {
    const log = document.getElementById(`log-${mode}`);
    if (log) log.innerHTML = '';

    const ans = document.getElementById(`answer-${mode}`);
    if (ans) ans.textContent = '';

    const comp = document.getElementById(`compression-${mode}`);
    if (comp) comp.innerHTML = '';

    const docs = document.getElementById(`docs-${mode}`);
    if (docs) docs.innerHTML = '';
  }

  _renderCompressionPanel(mode, c) {
    const el = document.getElementById(`compression-${mode}`);
    if (!el) return;

    const isVanilla = mode === 'vanilla';
    const ratio = c.compressionRatio ?? 1;
    const pct   = isVanilla ? 100 : Math.min(100, (1 / ratio) * 100).toFixed(1);
    const barColor = MODE_META[mode]?.color ?? '#10b981';

    el.innerHTML = `
      <div class="comp-header">
        <span class="comp-algo">${c.algorithm ?? 'Unknown'}</span>
        <span class="comp-ratio ${ratio > 50 ? 'comp-ratio--best' : ''}">${ratio}×</span>
      </div>
      <div class="comp-bar-wrap">
        <div class="comp-bar" style="width:${pct}%; background:${barColor}"></div>
        <span class="comp-pct">${pct}%</span>
      </div>
      <div class="comp-details">
        <div class="comp-row"><span>Original</span><strong>${c.originalBytes}B</strong></div>
        <div class="comp-row"><span>Compressed</span><strong>${c.compressedBytes}B</strong></div>
        <div class="comp-row"><span>Bits/dim</span><strong>${c.bitsPerDim}</strong></div>
        ${c.training_required || c.trainingRequired
          ? `<div class="comp-row comp-warn"><span>⚠ Training</span><strong>Required</strong></div>`
          : `<div class="comp-row comp-ok"><span>✓ Training</span><strong>None</strong></div>`}
        ${c.compressMs != null
          ? `<div class="comp-row"><span>Encode time</span><strong>${c.compressMs}ms</strong></div>`
          : ''}
        ${c.residualNorm != null
          ? `<div class="comp-row"><span>Residual L2</span><strong>${c.residualNorm}</strong></div>`
          : ''}
      </div>`;
  }

  _renderRetrievedDocs(mode, docs) {
    const el = document.getElementById(`docs-${mode}`);
    if (!el) return;

    if (!docs || docs.length === 0) {
      el.innerHTML = '<p class="no-docs">No docs above threshold</p>';
      return;
    }

    el.innerHTML = docs.slice(0, 3).map((d, i) => {
      const score = ((d.similarity ?? 0) * 100).toFixed(1);
      return `
        <div class="mini-doc">
          <div class="mini-doc-header">
            <span class="mini-rank">#${i + 1}</span>
            <span class="mini-title">${(d.heading || d.source || 'Document').substring(0, 36)}</span>
            <span class="mini-score">${score}%</span>
          </div>
          <div class="mini-score-bar">
            <div class="mini-score-fill" style="width:${score}%;background:${MODE_META[mode].color}"></div>
          </div>
        </div>`;
    }).join('');
  }

  _renderAnswer(mode, text) {
    const el = document.getElementById(`answer-${mode}`);
    if (el) el.textContent = text;
  }

  _renderTiming(mode, stats) {
    const tbody = document.getElementById('compare-timing-body');
    if (!tbody) return;

    const existing = document.getElementById(`timing-row-${mode}`);
    if (existing) existing.remove();

    const meta     = MODE_META[mode];
    const retMs    = stats.retrievalMs ?? stats.searchMs;
    const isFastest = () => {
      const others = ['vanilla', 'pq', 'turbo']
        .filter(m => m !== mode && this._results[m])
        .map(m => this._results[m].retrievalMs ?? this._results[m].searchMs ?? Infinity);
      return others.length > 0 && others.every(t => retMs <= t);
    };

    const row = document.createElement('tr');
    row.id    = `timing-row-${mode}`;
    row.innerHTML = `
      <td><span style="color:${meta.color}">${meta.icon} ${meta.label}</span></td>
      <td>${stats.compressMs ?? 0}ms</td>
      <td>${stats.searchMs ?? '—'}ms</td>
      <td style="color:${meta.color};font-weight:700">
        ${retMs}ms
        ${isFastest() ? '<span style="font-size:10px;margin-left:4px">🏆</span>' : ''}
      </td>
      <td style="color:var(--text-muted);font-size:12px">${stats.genMs}ms <em>(dùng chung)</em></td>
      <td>${stats.compression?.compressionRatio ?? 1}×</td>
      <td>${stats.compression?.trainingRequired ? '⚠ Cần' : '✓ Không'}</td>`;
    tbody.appendChild(row);
  }

  _renderFinalComparison() {
    const wrap = document.getElementById('compare-winner');
    if (!wrap) return;

    const modes  = ['vanilla', 'pq', 'turbo'];
    const ranked = modes
      .filter(m => this._results[m])
      .map(m => {
        const s      = this._results[m];
        const retMs  = s.retrievalMs ?? s.searchMs ?? 9999;
        const ratio  = s.compression?.compressionRatio ?? 1;
        return { mode: m, retMs, ratio, stats: s };
      })
      .sort((a, b) => a.retMs - b.retMs); // nướt nhất = nhanh nhất

    if (!ranked.length) return;
    const winner = ranked[0];
    const meta   = MODE_META[winner.mode];

    // Build retrievalMs map for chart
    const retrievalTimings = Object.fromEntries(
      ranked.map(({ mode, retMs }) => [mode, retMs])
    );

    wrap.innerHTML = `
      <div class="winner-badge">
        🏆 Nhanh nhất: <strong style="color:${meta.color}">${meta.label}</strong>
        <span class="winner-reason">
          ${winner.retMs}ms retrieval · ${winner.ratio}× nén
          ${winner.mode !== 'vanilla' ? '· Không cần training lại khi cập nhật' : ''}
        </span>
      </div>`;
    wrap.classList.add('visible');

    // Cập nhật benchmark chart với thời gian retrieval thực tế
    if (typeof window.updateBenchmarkChart === 'function') {
      window.updateBenchmarkChart(retrievalTimings);
    }
  }
}

// Export singleton
window.arqVisualizer = new ARQVisualizer();
