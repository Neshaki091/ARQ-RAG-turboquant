/**
 * server/services/gemini.js
 * Gemma 3 27B via Google Generative AI SDK.
 *
 * NOTE: Gemma models do NOT support the `systemInstruction` parameter.
 * The system prompt is embedded directly inside buildRagPrompt().
 */
import { GoogleGenerativeAI } from '@google/generative-ai';

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);

const MODEL = 'gemma-3-27b-it';

const GENERATION_CONFIG = {
  temperature: 0.7,
  topK: 40,
  topP: 0.95,
  maxOutputTokens: 1024,
};

// System instruction embedded in prompt (Gemma doesn't support systemInstruction param)
const SYSTEM_INSTRUCTION = `Bạn là trợ lý AI chính xác và am hiểu cho hệ thống ARQ-RAG được cung cấp bởi TurboQuant vector compression.

Khi trả lời câu hỏi:
1. LUÔN dựa trên các tài liệu ngữ cảnh được cung cấp
2. Ngắn gọn nhưng đầy đủ — khoảng 3-5 câu
3. Nếu ngữ cảnh không trả lời đầy đủ câu hỏi, hãy nói rõ điều đó
4. Nếu được hỏi về TurboQuant, giải thích 2 giai đoạn: PolarQuant (xoay ngẫu nhiên) và QJL (hiệu chỉnh sai số)
5. Dùng thuật ngữ kỹ thuật chính xác nhưng dễ hiểu

Định dạng: Đoạn văn xuôi. KHÔNG dùng tiêu đề markdown hay gạch đầu dòng.`;

// ─── Mock answer fallback ──────────────────────────────────────────────────────
const MOCK_ANSWERS = {
  default: (query, docs) => {
    const topDoc  = docs[0];
    const heading = topDoc?.heading || topDoc?.source || 'cơ sở tri thức';
    return `[Chế độ Demo — Đã hết quota Gemma] Dựa trên "${heading}" và ${docs.length} tài liệu được truy xuất, ` +
      `pipeline ARQ-RAG đã tìm thấy ngữ cảnh liên quan cho câu hỏi: "${query}". ` +
      `Nén TurboQuant (PolarQuant + QJL) giảm kích thước vector ~90× mà không cần training, ` +
      `giúp truy xuất nhanh hơn Vanilla RAG. Product Quantization đạt ~128× nén ` +
      `nhưng cần chạy lại k-means khi thêm tài liệu mới. ` +
      `Toàn bộ thống kê nén và thời gian đo là thực — chỉ phần trả lời này là mô phỏng do hết quota API.`;
  },
};

function mockGenerate(query, documents) {
  return {
    answer: MOCK_ANSWERS.default(query, documents),
    usage:  { promptTokenCount: 0, candidatesTokenCount: 0, mock: true },
  };
}

// ─── Quota error detector ──────────────────────────────────────────────────────
function isQuotaError(err) {
  const msg = err?.message?.toLowerCase() ?? '';
  return msg.includes('429') || msg.includes('quota') || msg.includes('resource_exhausted');
}

// ─── Build RAG prompt ─────────────────────────────────────────────────────────
// System instruction is prepended here since Gemma ignores the systemInstruction param.
function buildRagPrompt(query, documents) {
  const contextBlocks = documents
    .map((doc, i) =>
      `[Document ${i + 1}] "${doc.heading || doc.source || 'Document'}" (similarity: ${((doc.similarity ?? 0) * 100).toFixed(1)}%)\n${doc.content}`
    )
    .join('\n\n');

  return `${SYSTEM_INSTRUCTION}\n\n---\n\nContext Documents:\n${contextBlocks}\n\n---\nUser Question: ${query}\n\nAnswer based on the above context:`;
}

// ─── Generate (non-streaming) ─────────────────────────────────────────────────
/**
 * @param {string} query
 * @param {Array}  documents
 * @returns {Promise<{answer: string, usage: object}>}
 */
export async function generateAnswer(query, documents) {
  const prompt = buildRagPrompt(query, documents);

  try {
    const model = genAI.getGenerativeModel({
      model: MODEL,
      generationConfig: GENERATION_CONFIG,
    });

    console.log(`[Gemma] Calling ${MODEL}…`);
    const result   = await model.generateContent(prompt);
    const response = result.response;

    console.log(`[Gemma] ✓ Done`);
    return {
      answer: response.text(),
      usage:  { ...response.usageMetadata, model: MODEL },
    };
  } catch (err) {
    if (isQuotaError(err)) {
      console.warn(`[Gemma] ⚠ Quota hit — using mock answer for demo.`);
      return mockGenerate(query, documents);
    }
    throw err;
  }
}

// ─── Stream answer via SSE ────────────────────────────────────────────────────
/**
 * @param {string} query
 * @param {Array}  documents
 * @param {object} res  - Express response in SSE mode
 */
export async function streamAnswer(query, documents, res) {
  res.setHeader('Content-Type',      'text/event-stream');
  res.setHeader('Cache-Control',     'no-cache');
  res.setHeader('Connection',        'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');

  const sendEvent = (event, data) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  const prompt = buildRagPrompt(query, documents);

  try {
    const model = genAI.getGenerativeModel({
      model: MODEL,
      generationConfig: GENERATION_CONFIG,
    });

    const streamResult = await model.generateContentStream(prompt);
    sendEvent('start', { status: 'generating', model: MODEL });

    for await (const chunk of streamResult.stream) {
      const text = chunk.text();
      if (text) sendEvent('chunk', { text });
    }

    const finalResponse = await streamResult.response;
    sendEvent('done', { status: 'complete', usage: finalResponse.usageMetadata || {} });
  } catch (err) {
    if (isQuotaError(err)) {
      console.warn(`[Gemma/stream] Quota hit — streaming mock.`);
      const mock = mockGenerate(query, documents);
      sendEvent('start', { status: 'generating', model: 'mock' });
      sendEvent('chunk', { text: mock.answer });
      sendEvent('done',  { status: 'complete', usage: mock.usage });
    } else {
      sendEvent('error', { message: err.message });
    }
  } finally {
    res.end();
  }
}
