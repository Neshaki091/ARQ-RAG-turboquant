"use client";

import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Activity,
  Database,
  Cpu,
  Download,
  Play,
  Layers,
  AlertCircle,
  FileText,
  CheckCircle2,
  Loader2,
  Trash2,
  Terminal,
  Square
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area
} from "recharts";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const API_BASE = "http://localhost:8000";

export default function Dashboard() {
  const [status, setStatus] = useState<any>({ status: "IDLE", progress: 0 });
  const [pdfs, setPdfs] = useState<string[]>([]);
  const [numFiles, setNumFiles] = useState(5);
  const [metricsHistory, setMetricsHistory] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [benchmarkBatchSize, setBenchmarkBatchSize] = useState(20);

  // Chat State
  const [messages, setMessages] = useState<any[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [chatConfig, setChatConfig] = useState({
    model: "groq",
    collection: "vector_arq"
  });

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || isTyping) return;

    const userMsg = { role: "user", content: inputMessage };
    setMessages(prev => [...prev, userMsg]);
    const currentQuery = inputMessage;
    setInputMessage("");
    setIsTyping(true);

    // Thêm tin nhắn assistant tạm thời để hiển thị status
    const assistantMsgId = Date.now();
    setMessages(prev => [...prev, {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      status: "⏳ Đang kết nối..."
    }]);

    try {
      const response = await fetch(`${API_BASE}/chat-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: currentQuery,
          model: chatConfig.model,
          collection: chatConfig.collection
        })
      });

      if (!response.body) throw new Error("Không có phản hồi từ server");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            if (data.type === "status") {
              setMessages(prev => prev.map(m =>
                m.id === assistantMsgId ? { ...m, status: data.message } : m
              ));
            } else if (data.type === "final") {
              if (data.update_only) {
                // Chỉ cập nhật điểm số cho tin nhắn hiện tại
                setMessages(prev => prev.map(m =>
                  m.id === assistantMsgId ? {
                    ...m,
                    scores: data.scores,
                    status: null // Tắt trạng thái đang chấm điểm
                  } : m
                ));
              } else {
                // Hiển thị câu trả lời chính ngay lập tức
                setMessages(prev => prev.map(m =>
                  m.id === assistantMsgId ? {
                    ...m,
                    content: data.answer,
                    status: null,
                    scores: data.scores,
                    latency: data.latency,
                    sources: data.sources
                  } : m
                ));
              }
            }
          } catch (e) {
            console.error("Lỗi parse stream line:", line);
          }
        }
      }
    } catch (err: any) {
      setError("Chat failed: " + err.message);
      setMessages(prev => prev.filter(m => m.id !== assistantMsgId));
    } finally {
      setIsTyping(false);
    }
  };

  // Poll status
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE}/status`);
        setStatus(res.data);

        // Luôn cập nhật lịch sử tài nguyên nếu có dữ liệu RAM từ backend
        if (res.data.ram_usage) {
          setMetricsHistory(prev => [
            ...prev.slice(-19),
            {
              time: new Date().toLocaleTimeString(),
              ram: res.data.sys_ram_usage || res.data.ram_usage, // Dùng RAM hệ thống nếu có
              latency: res.data.last_latency || 0
            }
          ]);
        }
      } catch (err) {
        console.error("Failed to fetch status");
      }
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    axios.get(`${API_BASE}/pdfs`).then(res => setPdfs(res.data.files)).catch(() => { });
  }, []);

  const handlePurge = async () => {
    const secret = prompt("VUI LÒNG NHẬP SECRET KEY ĐỂ XÓA SẠCH DỮ LIỆU:");
    if (!secret) return;

    if (confirm("Hành động này sẽ xóa toàn bộ Qdrant collections và các tệp nén cục bộ. Bạn có chắc chắn không?")) {
      setError(null);
      try {
        const res = await axios.post(`${API_BASE}/purge-data`, { secret_key: secret });
        alert(res.data.message);
        window.location.reload(); // Refresh để cập nhật lại trạng thái sạch
      } catch (err: any) {
        setError(err.response?.data?.detail || "Purge failed");
      }
    }
  };

  const runAction = async (endpoint: string, data?: any) => {
    setError(null);
    try {
      await axios.post(`${API_BASE}/${endpoint}`, data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Action failed");
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0c] text-slate-200 p-8 font-sans selection:bg-indigo-500/30">
      {/* Background Glow */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden -z-10">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-500/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-500/10 blur-[120px] rounded-full" />
      </div>

      <header className="max-w-7xl mx-auto mb-12 flex justify-between items-end">
        <div>
          <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400 mb-2">
            ARQ-RAG Research Dashboard
          </h1>
          <p className="text-slate-400 flex items-center gap-2">
            <Layers size={18} className="text-indigo-400" />
            Empirical Benchmarking of Vector Quantization Models
          </p>
        </div>
        <div className="flex gap-4">
          <div className={`px-4 py-2 rounded-full border border-slate-800 bg-slate-900/50 flex items-center gap-2 text-sm`}>
            <div className={`w-2 h-2 rounded-full ${["IDLE", "COMPLETED"].includes(status.status) ? "bg-green-500" : "bg-yellow-500 animate-pulse"}`} />
            System {status.status}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-12 gap-6">

        {/* Control Panel */}
        <section className="col-span-12 lg:col-span-4 space-y-6">
          <div className="bg-slate-900/40 border border-slate-800 p-6 rounded-2xl backdrop-blur-xl">
            <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
              <Activity size={20} className="text-indigo-400" />
              Pipeline Control
            </h3>

            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
                <label className="text-xs uppercase tracking-wider text-slate-500 block mb-3">Auto Embedding Config</label>
                <div className="flex gap-3 items-center">
                  <input
                    type="number"
                    value={numFiles}
                    onChange={(e) => setNumFiles(parseInt(e.target.value))}
                    className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 w-20 text-center focus:ring-1 ring-indigo-500 outline-none"
                  />
                  <span className="text-slate-400 text-sm">PDF Files</span>
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={() => runAction("run-crawl")}
                  disabled={!["IDLE", "COMPLETED"].includes(status.status)}
                  className="flex-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 transition-colors rounded-lg py-3 flex items-center justify-center gap-2 font-medium border border-slate-700"
                >
                  <Database size={18} /> Run Crawl data
                </button>
                {!["IDLE", "COMPLETED"].includes(status.status) && (
                  <button
                    onClick={() => runAction("stop-crawl")}
                    className="px-4 bg-red-500/10 hover:bg-red-500/20 text-red-500 transition-colors rounded-lg py-3 flex items-center justify-center gap-2 font-medium border border-red-500/20 shadow-lg shadow-red-500/5"
                    title="Stop Current Task"
                  >
                    <Square size={16} fill="currentColor" /> Stop
                  </button>
                )}
              </div>

              <button
                onClick={() => runAction("run-auto-pipeline", { num_files: numFiles })}
                disabled={!["IDLE", "COMPLETED"].includes(status.status)}
                className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:opacity-50 transition-all rounded-lg py-3 flex items-center justify-center gap-2 font-semibold border border-indigo-500/30 shadow-lg shadow-indigo-500/10"
              >
                <Activity size={18} /> Auto Embedding
              </button>

              <div className="flex gap-2">
                <select
                  value={benchmarkBatchSize}
                  onChange={(e) => setBenchmarkBatchSize(parseInt(e.target.value))}
                  className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-4 text-sm font-medium focus:ring-1 ring-indigo-500 outline-none w-24"
                >
                  <option value={20}>20 Q</option>
                  <option value={30}>30 Q</option>
                  <option value={40}>40 Q</option>
                </select>
                <button
                  onClick={() => runAction("run-benchmark", { batch_size: benchmarkBatchSize })}
                  disabled={!["IDLE", "COMPLETED"].includes(status.status)}
                  className="flex-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 transition-colors rounded-lg py-4 flex items-center justify-center gap-2 font-bold text-lg border border-slate-700 shadow-xl"
                >
                  <Play size={20} fill="currentColor" /> Research
                </button>
              </div>

              <button
                onClick={() => runAction("run-generate-testset")}
                disabled={!["IDLE", "COMPLETED"].includes(status.status)}
                className="w-full bg-emerald-600/10 hover:bg-emerald-600/20 text-emerald-500 transition-all rounded-lg py-3 flex items-center justify-center gap-2 font-semibold border border-emerald-500/30"
              >
                <CheckCircle2 size={18} /> Generate Ground Truth
              </button>
            </div>

            {error && (
              <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex items-center gap-2">
                <AlertCircle size={14} /> {error}
              </div>
            )}
          </div>

          <div className="bg-slate-900/40 border border-slate-800 p-6 rounded-2xl backdrop-blur-xl">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <FileText size={20} className="text-indigo-400" />
              Dataset Files ({pdfs.length})
            </h3>
            <div className="max-h-[300px] overflow-y-auto pr-2 space-y-2 custom-scrollbar">
              {pdfs.map(file => (
                <div key={file} className="p-3 rounded-lg bg-slate-800/20 border border-slate-700/30 text-sm text-slate-400 truncate flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                  {file}
                </div>
              ))}
            </div>

            <div className="mt-6 pt-4 border-t border-slate-800/60">
              <label className="text-[10px] uppercase tracking-[0.2em] text-red-500/60 block mb-3 font-bold">Danger Zone</label>
              <button
                onClick={handlePurge}
                className="w-full bg-red-500/10 hover:bg-red-500/20 text-red-500 transition-colors rounded-lg py-3 flex items-center justify-center gap-2 text-sm font-medium border border-red-500/20"
              >
                <Trash2 size={16} /> Purge All Data
              </button>
            </div>
          </div>
        </section>

        {/* Monitoring & Stats */}
        <section className="col-span-12 lg:col-span-8 space-y-6">
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-slate-900/40 border border-slate-800 p-6 rounded-2xl backdrop-blur-xl">
              <div className="justify-between items-start mb-4 flex">
                <span className="text-slate-400 text-sm">System Progress</span>
                {["INGESTING", "EMBEDDING", "INDEXING", "BENCHMARKING"].includes(status.status) && <Loader2 size={18} className="text-indigo-400 animate-spin" />}
              </div>
              <div className="text-3xl font-bold text-white mb-4">{status.progress}%</div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${status.progress}%` }}
                  className="bg-gradient-to-r from-indigo-500 to-purple-500 h-full"
                />
              </div>
            </div>

            <div className="bg-slate-900/40 border border-slate-800 p-6 rounded-2xl backdrop-blur-xl">
              <div className="flex justify-between items-start mb-4">
                <span className="text-slate-400 text-sm">Benchmark Result</span>
                {status.excel_url && <CheckCircle2 size={18} className="text-green-400" />}
              </div>
              {status.excel_url ? (
                <a
                  href={status.excel_url}
                  target="_blank"
                  className="inline-flex items-center gap-2 bg-green-600 hover:bg-green-500 text-white px-4 py-2 rounded-lg transition-colors font-medium mt-1"
                >
                  <Download size={18} /> Download Excel
                </a>
              ) : (
                <div className="text-slate-500 italic mt-2">Báo cáo sẽ xuất hiện sau khi hoàn tất</div>
              )}
            </div>
          </div>

          <div className="bg-slate-900/40 border border-slate-800 p-6 rounded-2xl backdrop-blur-xl h-[500px] flex flex-col">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Activity size={20} className="text-indigo-400" />
              Real-time Chat Demonstration
            </h3>

            <div className="flex gap-2 mb-4">
              <select
                className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1 text-xs outline-none"
                value={chatConfig.model}
                onChange={(e) => setChatConfig({ ...chatConfig, model: e.target.value })}
              >
                <option value="groq">Groq Cloud (GPT-OSS 20B)</option>
              </select>
              <select
                className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1 text-xs outline-none"
                value={chatConfig.collection}
                onChange={(e) => setChatConfig({ ...chatConfig, collection: e.target.value })}
              >
                <option value="vector_raw">Standard RAG</option>
                <option value="vector_adaptive">Adaptive RAG</option>
                <option value="vector_pq">PQ RAG</option>
                <option value="vector_sq8">Scalar RAG</option>
                <option value="vector_arq">ARQ-RAG (TurboQuant)</option>
              </select>
            </div>

            <div className="flex-1 overflow-y-auto mb-4 space-y-4 pr-2 custom-scrollbar">
              {messages.map((m, i) => (
                <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] p-3 rounded-2xl ${m.role === 'user' ? 'bg-indigo-600' : 'bg-slate-800/80 border border-slate-700'}`}>
                    {m.status && (
                      <div className="flex items-center gap-2 mb-3 p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-[11px] text-indigo-400 font-medium italic animate-pulse">
                        <Loader2 size={12} className="animate-spin" />
                        {m.status}
                      </div>
                    )}
                    <div className="text-sm markdown-container leading-relaxed">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                        components={{
                          table: ({ node, ...props }) => <div className="overflow-x-auto my-4 border border-slate-700/50 rounded-xl"><table className="min-w-full divide-y divide-slate-700/50" {...props} /></div>,
                          th: ({ node, ...props }) => <th className="px-4 py-3 bg-slate-800/50 text-left text-xs font-bold text-slate-300 uppercase tracking-wider" {...props} />,
                          td: ({ node, ...props }) => <td className="px-4 py-3 text-xs border-t border-slate-700/50 text-slate-400" {...props} />,
                          ul: ({ node, ...props }) => <ul className="list-disc ml-6 space-y-2 my-3" {...props} />,
                          ol: ({ node, ...props }) => <ol className="list-decimal ml-6 space-y-2 my-3" {...props} />,
                          li: ({ node, ...props }) => <li className="text-sm" {...props} />,
                          h1: ({ node, ...props }) => <h1 className="text-xl font-bold mt-6 mb-3 text-white border-b border-slate-700/50 pb-2" {...props} />,
                          h2: ({ node, ...props }) => <h2 className="text-lg font-bold mt-5 mb-2 text-white" {...props} />,
                          h3: ({ node, ...props }) => <h3 className="text-base font-bold mt-4 mb-2 text-indigo-400" {...props} />,
                          p: ({ node, ...props }) => <p className="mb-4 last:mb-0" {...props} />,
                          code: ({ node, ...props }) => <code className="bg-slate-900/80 px-1.5 py-0.5 rounded text-indigo-300 font-mono text-[13px] border border-slate-700/50" {...props} />,
                          a: ({ node, ...props }) => <a className="text-indigo-400 hover:text-indigo-300 underline underline-offset-4" {...props} />
                        }}
                      >
                        {m.content}
                      </ReactMarkdown>
                    </div>
                    {m.latency ? (
                      <div className="mt-3 pt-3 border-t border-slate-700/50 space-y-2">
                        <div className="flex justify-between text-[10px] uppercase tracking-tighter text-slate-400 font-semibold items-center">
                          <span>{m.scores ? "Quality Metrics (RAGAS)" : "System Metrics"}</span>
                          <span className="bg-slate-900/80 px-2 py-1 rounded text-indigo-300">⏳ {Math.round(m.latency * 1000)} ms</span>
                        </div>
                        {m.scores && (
                          <div className="space-y-1 mt-2">
                            <MetricBar label="Faithfulness" value={m.scores.faithfulness} color="bg-emerald-500" />
                            <MetricBar label="Relevance" value={m.scores.answer_relevancy} color="bg-blue-500" />
                          </div>
                        )}
                      </div>
                    ) : null}
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="flex justify-start">
                  <div className="bg-slate-800/80 p-3 rounded-2xl border border-slate-700">
                    <Loader2 size={16} className="animate-spin text-slate-400" />
                  </div>
                </div>
              )}
            </div>

            <form onSubmit={handleChatSubmit} className="relative">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Hỏi bất cứ điều gì về tài liệu..."
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 pr-12 text-sm focus:ring-1 ring-indigo-500 outline-none"
              />
              <button
                type="submit"
                className="absolute right-2 top-2 p-2 bg-indigo-600 rounded-lg hover:bg-indigo-500 transition-colors"
                disabled={isTyping}
              >
                <Play size={14} fill="currentColor" />
              </button>
            </form>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-900/40 border border-slate-800 p-6 rounded-2xl backdrop-blur-xl h-[400px]">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Cpu size={20} className="text-indigo-400" />
                Real-time Resource Monitor
              </h3>
              <div className="w-full h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={metricsHistory}>
                    <defs>
                      <linearGradient id="colorRam" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="time" hide />
                    <YAxis stroke="#475569" fontSize={12} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                    />
                    <Area
                      type="monotone"
                      dataKey="ram"
                      stroke="#818cf8"
                      fillOpacity={1}
                      fill="url(#colorRam)"
                      name="RAM (MB)"
                    />
                    <Line
                      type="monotone"
                      dataKey="latency"
                      stroke="#f472b6"
                      dot={false}
                      name="Latency (ms)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-[#0b0c10] border border-slate-800 p-6 rounded-2xl backdrop-blur-xl h-[400px] flex flex-col font-mono text-[10px] md:text-xs shadow-inner">
              <h3 className="text-sm font-semibold mb-4 flex items-center gap-2 text-slate-400 font-sans tracking-wide">
                <Terminal size={16} className="text-emerald-500" />
                Backend Live Logs
              </h3>
              <div className="flex-1 overflow-y-auto space-y-1 custom-scrollbar pr-2 flex flex-col-reverse">
                {[...(status?.logs || [])].reverse().map((log: string, i: number) => {
                  const parts = log.split(" | ");
                  const time = parts[0];
                  const module = parts[1];
                  const level = parts[2];
                  const message = parts.slice(3).join(" | ");

                  const getLevelStyle = (lvl: string) => {
                    switch (lvl?.trim()) {
                      case "ERROR": return "text-red-400 font-bold bg-red-400/10 px-1 rounded";
                      case "WARNING": return "text-yellow-400 font-bold bg-yellow-400/10 px-1 rounded";
                      case "INFO": return "text-emerald-400/80";
                      default: return "text-slate-500";
                    }
                  };

                  return (
                    <div key={i} className="py-1 text-[11px] border-b border-slate-800/30 font-mono flex gap-2 items-start leading-relaxed">
                      <span className="text-slate-600 shrink-0">[{time}]</span>
                      <span className="text-indigo-400/70 shrink-0 font-semibold w-20 truncate">[{module}]</span>
                      <span className={`${getLevelStyle(level)} shrink-0 w-16 text-center`}>{level}</span>
                      <span className="text-slate-300 break-words">{message}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </section>

      </main>

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #1e293b00;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #334155;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #475569;
        }
      `}</style>
    </div>
  );
}

function MetricBar({ label, value, color }: { label: string, value: number, color: string }) {
  const percentage = (value * 100).toFixed(0);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[9px] text-slate-500 uppercase">
        <span>{label}</span>
        <span>{percentage}%</span>
      </div>
      <div className="w-full bg-slate-900 h-1 rounded-full overflow-hidden">
        <div
          className={`${color} h-full transition-all duration-1000`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
