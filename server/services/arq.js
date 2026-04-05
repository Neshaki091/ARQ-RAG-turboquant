/**
 * server/services/arq.js
 * ARQ — Asynchronous RAG Queue simulation
 * Models the state machine for async document ingestion + query serving.
 *
 * States: IDLE → QUEUED → EMBEDDING → COMPRESSING → INDEXING → READY → RETRIEVING → GENERATING → DONE
 */

export const ARQ_STATES = {
  IDLE:        'IDLE',
  QUEUED:      'QUEUED',
  EMBEDDING:   'EMBEDDING',
  COMPRESSING: 'COMPRESSING',
  INDEXING:    'INDEXING',
  READY:       'READY',
  RETRIEVING:  'RETRIEVING',
  GENERATING:  'GENERATING',
  DONE:        'DONE',
  ERROR:       'ERROR',
};

/**
 * Emit a structured ARQ step event via SSE.
 * @param {Function} send  - (event, data) => void
 * @param {string} mode    - 'turbo' | 'pq' | 'vanilla'
 * @param {string} state   - ARQ_STATES value
 * @param {object} payload - additional data
 */
export function emitARQStep(send, mode, state, payload = {}) {
  send(`arq:step`, {
    mode,
    state,
    ts: Date.now(),
    ...payload,
  });
}

/**
 * Simulate the ARQ queue enqueue phase (near-instant).
 * Returns a task descriptor.
 */
export function createTask(query, mode) {
  return {
    id:      `${mode}-${Date.now().toString(36)}`,
    query,
    mode,
    state:   ARQ_STATES.QUEUED,
    startMs: Date.now(),
    steps:   [],
  };
}

/**
 * Record a timing step into the task log.
 */
export function recordStep(task, state, ms) {
  task.steps.push({ state, ms, ts: Date.now() });
  task.state = state;
  return ms;
}
