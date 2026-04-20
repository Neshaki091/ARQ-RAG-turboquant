"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import {
  Send,
  Cpu,
  Zap,
  Database,
  BarChart2,
  ChevronDown,
  CheckCircle2,
  AlertCircle,
  Loader2,
  BookOpen,
  Clock,
  Hash,
  Layers,
} from "lucide-react";

// ── Types ───────────────────────────────────────────────────────────────

interface Metrics {
  total_latency_ms?: number;
  embed_latency_ms?: number;
  adc_table_ms?: number;
  retrieve_latency_ms?: number;
  rerank_latency_ms?: number;
  llm_latency_ms?: number;
  analyze_latency_ms?: number;
  retrieval_count?: number;
  rerank_count?: number;
  query_complexity?: string;
  effective_limit?: number;
  effective_top_k?: number;
  pq_config?: { M: number; K: number; compression_ratio: string };
  rerank_method?: string;
  note?: string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  model?: string;
  metrics?: Metrics;
  sources?: string[];
  streaming?: boolean;
}

// ── Model Config ────────────────────────────────────────────────────────

const MODELS = [
  {
    id: "pq",
    label: "RAG-PQ",
    description: "Product Quantization — ADC reranking, nén 384x",
    color: "#8b5cf6",
    icon: "⚡",
  },
  {
    id: "raw",
    label: "RAG-Raw",
    description: "Float32 baseline — không nén, không reranking",
    color: "#6b7280",
    icon: "📊",
  },
  {
    id: "sq8",
    label: "RAG-SQ8",
    description: "Scalar Quantization 8-bit — nén 4x",
    color: "#3b82f6",
    icon: "🔢",
  },
  {
    id: "arq",
    label: "ARQ-RAG",
    description: "TurboQuant — ADC + QJL combined reranking",
    color: "#06b6d4",
    icon: "🚀",
  },
  {
    id: "adaptive",
    label: "Adaptive",
    description: "Dynamic top-k — tự điều chỉnh theo query complexity",
    color: "#10b981",
    icon: "🧠",
  },
] as const;

type ModelId = (typeof MODELS)[number]["id"];

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Helper: Format metric value ─────────────────────────────────────────
function fmtMs(val?: number): string {
  if (val == null) return "—";
  return val < 1000 ? `${val.toFixed(0)}ms` : `${(val / 1000).toFixed(2)}s`;
}

// ── MetricsPanel ────────────────────────────────────────────────────────
function MetricsPanel({ metrics, model }: { metrics: Metrics; model: string }) {
  const modelConfig = MODELS.find((m) => m.id === model);
  const color = modelConfig?.color || "#8b5cf6";

  const rows = [
    { label: "Total", value: fmtMs(metrics.total_latency_ms), icon: <Clock size={12} /> },
    { label: "Embed", value: fmtMs(metrics.embed_latency_ms), icon: <Cpu size={12} /> },
    { label: "ADC Table", value: fmtMs(metrics.adc_table_ms), icon: <Layers size={12} /> },
    { label: "Retrieve", value: fmtMs(metrics.retrieve_latency_ms), icon: <Database size={12} /> },
    { label: "Rerank", value: fmtMs(metrics.rerank_latency_ms), icon: <BarChart2 size={12} /> },
    { label: "LLM", value: fmtMs(metrics.llm_latency_ms), icon: <Zap size={12} /> },
  ].filter((r) => r.value !== "—");

  return (
    <div
      style={{
        background: "rgba(13,13,30,0.8)",
        border: `1px solid ${color}30`,
        borderRadius: "10px",
        padding: "12px 14px",
        marginTop: "10px",
        fontSize: "0.78rem",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "6px",
          marginBottom: "8px",
          color: color,
          fontWeight: 600,
        }}
      >
        <BarChart2 size={13} />
        Metrics
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "4px 12px",
        }}
      >
        {rows.map((row) => (
          <div
            key={row.label}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              color: "var(--text-secondary)",
            }}
          >
            <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              {row.icon} {row.label}
            </span>
            <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
              {row.value}
            </span>
          </div>
        ))}
      </div>

      {/* PQ specific info */}
      {metrics.pq_config && (
        <div
          style={{
            marginTop: "8px",
            paddingTop: "8px",
            borderTop: `1px solid ${color}20`,
            color: "var(--text-secondary)",
            fontSize: "0.72rem",
          }}
        >
          PQ: M={metrics.pq_config.M}, K={metrics.pq_config.K},{" "}
          nén {metrics.pq_config.compression_ratio}
        </div>
      )}

      {/* Retrieval stats */}
      {(metrics.retrieval_count != null || metrics.rerank_count != null) && (
        <div
          style={{
            marginTop: "8px",
            paddingTop: "8px",
            borderTop: `1px solid ${color}20`,
            display: "flex",
            gap: "12px",
            color: "var(--text-secondary)",
          }}
        >
          {metrics.retrieval_count != null && (
            <span>
              <Hash size={10} style={{ display: "inline", marginRight: 3 }} />
              Candidates: {metrics.retrieval_count}
            </span>
          )}
          {metrics.rerank_count != null && metrics.rerank_count > 0 && (
            <span>→ Top-K: {metrics.rerank_count}</span>
          )}
          {metrics.query_complexity && (
            <span style={{ color: metrics.query_complexity === "COMPLEX" ? "#f59e0b" : "#10b981" }}>
              {metrics.query_complexity}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// ── ModelSelector ───────────────────────────────────────────────────────
function ModelSelector({
  selected,
  onChange,
}: {
  selected: ModelId;
  onChange: (m: ModelId) => void;
}) {
  const [open, setOpen] = useState(false);
  const current = MODELS.find((m) => m.id === selected)!;

  return (
    <div style={{ position: "relative" }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          background: "var(--bg-card)",
          border: `1px solid ${current.color}40`,
          borderRadius: "10px",
          padding: "8px 14px",
          color: "var(--text-primary)",
          cursor: "pointer",
          fontSize: "0.875rem",
          fontWeight: 500,
          transition: "all 0.2s",
          minWidth: "180px",
        }}
        onMouseEnter={(e) => {
          (e.target as HTMLElement).style.borderColor = current.color;
        }}
        onMouseLeave={(e) => {
          (e.target as HTMLElement).style.borderColor = `${current.color}40`;
        }}
      >
        <span>{current.icon}</span>
        <span style={{ color: current.color }}>{current.label}</span>
        <ChevronDown
          size={14}
          style={{
            marginLeft: "auto",
            transform: open ? "rotate(180deg)" : "none",
            transition: "transform 0.2s",
            color: "var(--text-muted)",
          }}
        />
      </button>

      {open && (
        <div
          style={{
            position: "absolute",
            top: "calc(100% + 6px)",
            left: 0,
            right: 0,
            background: "var(--bg-card)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "10px",
            overflow: "hidden",
            zIndex: 50,
            boxShadow: "0 20px 40px rgba(0,0,0,0.5)",
            animation: "fadeIn 0.15s ease",
          }}
        >
          {MODELS.map((m) => (
            <button
              key={m.id}
              onClick={() => {
                onChange(m.id);
                setOpen(false);
              }}
              style={{
                display: "block",
                width: "100%",
                textAlign: "left",
                padding: "10px 14px",
                background: selected === m.id ? `${m.color}15` : "transparent",
                border: "none",
                cursor: "pointer",
                transition: "background 0.15s",
                borderLeft: selected === m.id ? `2px solid ${m.color}` : "2px solid transparent",
              }}
              onMouseEnter={(e) => {
                if (selected !== m.id)
                  (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.04)";
              }}
              onMouseLeave={(e) => {
                if (selected !== m.id)
                  (e.currentTarget as HTMLElement).style.background = "transparent";
              }}
            >
              <div
                style={{
                  fontSize: "0.875rem",
                  fontWeight: 500,
                  color: m.color,
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                {m.icon} {m.label}
              </div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "2px" }}>
                {m.description}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ── ChatMessage ──────────────────────────────────────────────────────────
function ChatMessage({ msg }: { msg: Message }) {
  const modelCfg = MODELS.find((m) => m.id === msg.model);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: msg.role === "user" ? "row-reverse" : "row",
        gap: "10px",
        alignItems: "flex-start",
        animation: "fadeIn 0.25s ease",
      }}
    >
      {/* Avatar */}
      <div
        style={{
          width: "32px",
          height: "32px",
          borderRadius: "50%",
          flexShrink: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "14px",
          background:
            msg.role === "user"
              ? "linear-gradient(135deg, #3b82f6, #8b5cf6)"
              : `${modelCfg?.color || "#8b5cf6"}22`,
          border:
            msg.role !== "user" ? `1px solid ${modelCfg?.color || "#8b5cf6"}44` : "none",
        }}
      >
        {msg.role === "user" ? "U" : modelCfg?.icon || "🤖"}
      </div>

      <div style={{ maxWidth: "82%", minWidth: "150px" }}>
        {/* Bubble */}
        <div
          style={{
            background:
              msg.role === "user"
                ? "linear-gradient(135deg, #3730a3, #4c1d95)"
                : "var(--bg-card)",
            border:
              msg.role === "user"
                ? "1px solid rgba(139,92,246,0.3)"
                : "1px solid var(--border-subtle)",
            borderRadius:
              msg.role === "user" ? "16px 4px 16px 16px" : "4px 16px 16px 16px",
            padding: "12px 16px",
          }}
        >
          {msg.streaming && msg.content === "" ? (
            <span style={{ color: "var(--text-muted)", display: "flex", gap: 4 }}>
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: "var(--accent-purple)",
                    display: "inline-block",
                    animation: `typing 1.2s ${i * 0.2}s infinite`,
                  }}
                />
              ))}
            </span>
          ) : msg.role === "user" ? (
            <p style={{ color: "#e2e8f0", fontSize: "0.9rem", lineHeight: 1.6 }}>
              {msg.content}
            </p>
          ) : (
            <div className="markdown-body">
              <ReactMarkdown
                remarkPlugins={[remarkMath]}
                rehypePlugins={[rehypeKatex]}
              >
                {msg.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* Metrics */}
        {msg.metrics && !msg.streaming && (
          <MetricsPanel metrics={msg.metrics} model={msg.model || "pq"} />
        )}

        {/* Sources */}
        {msg.sources && msg.sources.length > 0 && (
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "4px",
              marginTop: "6px",
            }}
          >
            {[...new Set(msg.sources)].slice(0, 4).map((src) => (
              <span
                key={src}
                style={{
                  fontSize: "0.7rem",
                  padding: "2px 8px",
                  background: "rgba(139,92,246,0.1)",
                  border: "1px solid rgba(139,92,246,0.2)",
                  borderRadius: "20px",
                  color: "var(--text-secondary)",
                  display: "flex",
                  alignItems: "center",
                  gap: "4px",
                }}
              >
                <BookOpen size={9} />
                {src.length > 25 ? src.slice(0, 25) + "…" : src}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────
export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [selectedModel, setSelectedModel] = useState<ModelId>("pq");
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const chatBottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Scroll to bottom khi có message mới
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(async () => {
    const query = input.trim();
    if (!query || loading) return;

    const userMsg: Message = {
      id: `u_${Date.now()}`,
      role: "user",
      content: query,
    };

    const assistantId = `a_${Date.now()}`;
    const assistantMsg: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      model: selectedModel,
      streaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setLoading(true);
    setStreaming(true);

    try {
      // Dùng streaming endpoint
      const resp = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, model: selectedModel }),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let fullContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (data === "[DONE]" || data.startsWith("[ERROR]")) break;
          fullContent += data;

          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: fullContent }
                : m
            )
          );
        }
      }

      // Fetch metrics từ non-streaming endpoint
      const metaResp = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, model: selectedModel }),
      });

      if (metaResp.ok) {
        const result = await metaResp.json();
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: result.answer, // Dùng answer đầy đủ
                  streaming: false,
                  metrics: result.metrics,
                  sources: result.sources,
                }
              : m
          )
        );
      } else {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, streaming: false } : m
          )
        );
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : "Lỗi không xác định";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: `❌ Lỗi kết nối đến backend: ${errMsg}`,
                streaming: false,
              }
            : m
        )
      );
    } finally {
      setLoading(false);
      setStreaming(false);
      inputRef.current?.focus();
    }
  }, [input, loading, selectedModel]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const currentModel = MODELS.find((m) => m.id === selectedModel)!;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        background: "var(--bg-primary)",
        overflow: "hidden",
      }}
    >
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "14px 24px",
          background: "var(--bg-secondary)",
          borderBottom: "1px solid var(--border-subtle)",
          flexShrink: 0,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div
            style={{
              width: "36px",
              height: "36px",
              background: "linear-gradient(135deg, #8b5cf6, #06b6d4)",
              borderRadius: "10px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "18px",
              boxShadow: "0 0 20px rgba(139,92,246,0.4)",
            }}
          >
            ⚡
          </div>
          <div>
            <h1
              style={{
                fontSize: "1rem",
                fontWeight: 700,
                background: "linear-gradient(90deg, #8b5cf6, #06b6d4)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              ARQ-RAG TurboQuant
            </h1>
            <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: "1px" }}>
              Vector Quantization Research Dashboard
            </p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <ModelSelector selected={selectedModel} onChange={setSelectedModel} />

          {/* Status dot */}
          <div
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              background: loading ? "#f59e0b" : "#10b981",
              boxShadow: `0 0 6px ${loading ? "#f59e0b" : "#10b981"}`,
              animation: loading ? "pulse-glow 1.5s infinite" : "none",
            }}
          />
        </div>
      </header>

      {/* ── Chat Area ───────────────────────────────────────────────── */}
      <main
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "24px",
          display: "flex",
          flexDirection: "column",
          gap: "20px",
        }}
      >
        {messages.length === 0 ? (
          /* Empty state */
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              gap: "16px",
              animation: "slideUp 0.5s ease",
            }}
          >
            <div
              style={{
                width: "72px",
                height: "72px",
                background: "linear-gradient(135deg, #8b5cf622, #06b6d422)",
                border: "1px solid rgba(139,92,246,0.3)",
                borderRadius: "20px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "32px",
              }}
            >
              ⚡
            </div>
            <div style={{ textAlign: "center" }}>
              <h2 style={{ fontSize: "1.2rem", fontWeight: 600, color: "var(--text-primary)" }}>
                ARQ-RAG Research Dashboard
              </h2>
              <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", marginTop: "6px" }}>
                Hiện đang dùng{" "}
                <span style={{ color: currentModel.color, fontWeight: 500 }}>
                  {currentModel.icon} {currentModel.label}
                </span>
              </p>
              <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "4px" }}>
                {currentModel.description}
              </p>
            </div>

            {/* Model comparison grid */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(5, 1fr)",
                gap: "8px",
                marginTop: "12px",
                width: "100%",
                maxWidth: "600px",
              }}
            >
              {MODELS.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setSelectedModel(m.id)}
                  style={{
                    background:
                      selectedModel === m.id ? `${m.color}20` : "var(--bg-card)",
                    border: `1px solid ${selectedModel === m.id ? m.color : "var(--border-subtle)"}`,
                    borderRadius: "10px",
                    padding: "10px 8px",
                    cursor: "pointer",
                    textAlign: "center",
                    transition: "all 0.2s",
                  }}
                >
                  <div style={{ fontSize: "18px" }}>{m.icon}</div>
                  <div
                    style={{
                      fontSize: "0.7rem",
                      color: m.color,
                      fontWeight: 600,
                      marginTop: "4px",
                    }}
                  >
                    {m.label}
                  </div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => <ChatMessage key={msg.id} msg={msg} />)
        )}
        <div ref={chatBottomRef} />
      </main>

      {/* ── Input Area ──────────────────────────────────────────────── */}
      <footer
        style={{
          padding: "16px 24px",
          background: "var(--bg-secondary)",
          borderTop: "1px solid var(--border-subtle)",
          flexShrink: 0,
        }}
      >
        <div
          style={{
            display: "flex",
            gap: "10px",
            alignItems: "flex-end",
            maxWidth: "900px",
            margin: "0 auto",
          }}
        >
          <div
            style={{
              flex: 1,
              background: "var(--bg-input)",
              border: `1px solid ${
                input ? "var(--border-active)" : "var(--border-subtle)"
              }`,
              borderRadius: "14px",
              padding: "12px 16px",
              transition: "border-color 0.2s",
            }}
          >
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Hỏi về Vector Quantization, RAG, thuật toán... (${currentModel.label})`}
              disabled={loading}
              rows={1}
              style={{
                width: "100%",
                background: "transparent",
                border: "none",
                outline: "none",
                color: "var(--text-primary)",
                fontSize: "0.9rem",
                lineHeight: 1.5,
                resize: "none",
                fontFamily: "var(--font-sans)",
                maxHeight: "120px",
                overflowY: "auto",
              }}
              onInput={(e) => {
                const el = e.currentTarget;
                el.style.height = "auto";
                el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
              }}
            />
          </div>

          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            style={{
              width: "44px",
              height: "44px",
              borderRadius: "12px",
              background:
                loading || !input.trim()
                  ? "var(--bg-card)"
                  : `linear-gradient(135deg, ${currentModel.color}, ${currentModel.color}cc)`,
              border: "1px solid var(--border-subtle)",
              cursor: loading || !input.trim() ? "not-allowed" : "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: loading || !input.trim() ? "var(--text-muted)" : "white",
              transition: "all 0.2s",
              flexShrink: 0,
              boxShadow:
                input.trim() && !loading
                  ? `0 0 20px ${currentModel.color}40`
                  : "none",
            }}
          >
            {loading ? (
              <Loader2 size={18} style={{ animation: "spin 1s linear infinite" }} />
            ) : (
              <Send size={18} />
            )}
          </button>
        </div>

        <p
          style={{
            textAlign: "center",
            fontSize: "0.7rem",
            color: "var(--text-muted)",
            marginTop: "8px",
          }}
        >
          Enter để gửi • Shift+Enter để xuống dòng • Hỗ trợ Markdown + LaTeX
        </p>
      </footer>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
