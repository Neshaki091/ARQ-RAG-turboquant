"use client";

import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Activity, Database, Cpu, Download, Play, Layers, AlertCircle, FileText,
  CheckCircle2, Loader2, Trash2, Terminal, Clock, Zap, RefreshCw, Activity as ActivityIcon
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Line, LineChart
} from "recharts";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";

const API_BASE = "https://apiarqrag.evbtranding.site";

export default function Dashboard() {
  const [status, setStatus] = useState<any>({ status: "IDLE", progress: 0 });
  const [pdfs, setPdfs] = useState<string[]>([]);
  const [numFiles, setNumFiles] = useState(5);
  const [metricsHistory, setMetricsHistory] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [benchmarkBatchSize, setBenchmarkBatchSize] = useState(20);
  const [activeTab, setActiveTab] = useState("dashboard");
  const [benchmarkHistory, setBenchmarkHistory] = useState<any[]>([]);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Polling status
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE}/status`);
        setStatus(res.data);
        setMetricsHistory(prev => [...prev.slice(-29), {
          time: new Date().toLocaleTimeString(),
          ram: res.data.ram_usage,
          latency: res.data.last_latency
        }]);
      } catch (err) {
        console.error("Failed to fetch status");
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Fetch PDFs
  useEffect(() => {
    const fetchPdfs = async () => {
      try {
        const res = await axios.get(`${API_BASE}/pdfs`);
        setPdfs(res.data.files);
      } catch (err) {
        console.error("Failed to fetch PDFs");
      }
    };
    fetchPdfs();
    const interval = setInterval(fetchPdfs, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchBenchmarkHistory = async (model = "all") => {
    try {
      const res = await axios.get(`${API_BASE}/api/benchmark/history?model=${model}`);
      setBenchmarkHistory(res.data.results);
    } catch (err) {
      console.error("Failed to fetch history");
    }
  };

  useEffect(() => {
    if (activeTab === "research") {
      fetchBenchmarkHistory();
      const interval = setInterval(() => fetchBenchmarkHistory(), 5000);
      return () => clearInterval(interval);
    }
  }, [activeTab]);

  // Chat State
  const [messages, setMessages] = useState<any[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [chatConfig, setChatConfig] = useState({ model: "groq", collection: "vector_arq" });

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || isTyping) return;
    const userMsg = { role: "user", content: inputMessage };
    setMessages(prev => [...prev, userMsg]);
    const currentQuery = inputMessage;
    setInputMessage("");
    setIsTyping(true);
    try {
      const response = await fetch(`${API_BASE}/chat-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: currentQuery, ...chatConfig })
      });
      if (!response.body) throw new Error("No response body");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMsg = { role: "assistant", content: "" };
      setMessages(prev => [...prev, assistantMsg]);
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");
        lines.forEach(line => {
          if (!line.trim()) return;
          try {
            const data = JSON.parse(line);
            if (data.type === "final") {
              // Backend trả về câu trả lời đầy đủ
              assistantMsg.content = data.answer || "";
              setMessages(prev => [...prev.slice(0, -1), { ...assistantMsg }]);
            } else if (data.type === "text") {
              // Fallback: streaming token-by-token
              assistantMsg.content += data.content;
              setMessages(prev => [...prev.slice(0, -1), { ...assistantMsg }]);
            } else if (data.type === "error") {
              assistantMsg.content = `❌ ${data.message}`;
              setMessages(prev => [...prev.slice(0, -1), { ...assistantMsg }]);
            }
          } catch (e) {}
        });
      }
    } catch (err) {
      setError("Chat failed");
    } finally {
      setIsTyping(false);
    }
  };

  const handleRunBenchmarkTest = async (model: string) => {
    setError(null);
    try {
      await axios.post(`${API_BASE}/api/benchmark/run-test`, { model, batch_size: benchmarkBatchSize });
      alert(`Bắt đầu thực nghiệm: ${model}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Run test failed");
    }
  };

  const handleClearHistory = async (model = "all") => {
    if (!confirm("Xóa lịch sử thực nghiệm?")) return;
    try {
      await axios.delete(`${API_BASE}/api/benchmark/clear?model=${model}`);
      fetchBenchmarkHistory();
    } catch (err: any) {
      setError("Xóa thất bại");
    }
  };

  const handlePurge = async (target: string) => {
    const secret = prompt("Nhập Secret Key để xác nhận xóa:");
    if (!secret) return;
    try {
      const res = await axios.post(`${API_BASE}/purge-data`, { secret_key: secret, target });
      alert(res.data.message);
    } catch (err: any) {
      alert(err.response?.data?.detail || "Purge failed");
    }
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 p-8 font-sans selection:bg-blue-500/30">
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden -z-10">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-500/5 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-teal-500/5 blur-[120px] rounded-full" />
      </div>

      <header className="max-w-7xl mx-auto mb-8 flex justify-between items-center">
        <div className="flex items-center gap-4">
          <img src="/logo.png" alt="Logo" className="w-12 h-12 object-contain" />
          <div>
            <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-700 to-slate-800 mb-1">TurboQuant Dashboard</h1>
            <p className="text-slate-500 text-sm font-medium">Adaptive Retrieval-Augmented Generation Research Platform</p>
          </div>
        </div>
        <div className="flex gap-4">
          <div className="px-4 py-2 rounded-xl border border-slate-200 bg-white shadow-sm flex items-center gap-2 text-sm font-semibold text-slate-600">
            <div className={`w-2 h-2 rounded-full ${["IDLE", "COMPLETED"].includes(status.status) ? "bg-green-500" : "bg-blue-500 animate-pulse"}`} />
            System {status.status}
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto mb-8 flex gap-4">
        <button onClick={() => setActiveTab("dashboard")} className={`px-6 py-2 rounded-xl font-bold transition-all flex items-center gap-2 ${activeTab === "dashboard" ? "bg-blue-600 text-white shadow-lg" : "bg-white text-slate-500 border border-slate-200"}`}>
          <ActivityIcon size={18} /> Dashboard
        </button>
        <button onClick={() => setActiveTab("research")} className={`px-6 py-2 rounded-xl font-bold transition-all flex items-center gap-2 ${activeTab === "research" ? "bg-indigo-600 text-white shadow-lg" : "bg-white text-slate-500 border border-slate-200"}`}>
          <Zap size={18} /> Model Research
        </button>
      </div>

      <main className="max-w-7xl mx-auto">
        <AnimatePresence mode="wait">
          {activeTab === "dashboard" ? (
            <motion.div key="dashboard" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="grid grid-cols-12 gap-6">
              <section className="col-span-12 lg:col-span-4 space-y-6">
                <div className="bg-white border border-slate-200 p-6 rounded-2xl shadow-sm">
                  <h3 className="text-lg font-bold mb-6 flex items-center gap-2 text-slate-800"><ActivityIcon size={20} className="text-blue-600" /> Pipeline Control</h3>
                  <div className="space-y-4">
                    <button onClick={() => axios.post(`${API_BASE}/run-ingest`, { num_files: numFiles })} className="w-full bg-slate-900 hover:bg-slate-800 text-white py-3 rounded-xl flex items-center justify-center gap-3 transition-all font-bold group">
                      <Layers size={18} className="group-hover:rotate-12 transition-transform" /> Ingest & Chunk ({numFiles} files)
                    </button>
                    <button onClick={() => axios.post(`${API_BASE}/run-embed`)} className="w-full border-2 border-slate-200 hover:border-blue-500 hover:text-blue-600 py-3 rounded-xl flex items-center justify-center gap-3 transition-all font-bold text-slate-600">
                      <Cpu size={18} /> Create Embeddings
                    </button>
                    <button onClick={() => axios.post(`${API_BASE}/run-generate-testset`)} className="w-full bg-blue-50 text-blue-700 hover:bg-blue-100 py-3 rounded-xl flex items-center justify-center gap-3 transition-all font-bold border border-blue-100">
                      <FileText size={18} /> Generate GT Testset
                    </button>
                  </div>
                </div>
                <div className="bg-white border border-slate-200 p-6 rounded-2xl shadow-sm">
                  <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-slate-800"><FileText size={20} className="text-blue-600" /> Dataset Files ({pdfs.length})</h3>
                  <div className="max-h-[200px] overflow-y-auto pr-2 space-y-2 custom-scrollbar">
                    {pdfs.map(file => (
                      <div key={file} className="p-3 rounded-lg bg-slate-50 border border-slate-100 text-sm text-slate-600 truncate flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-500" /> {file}
                      </div>
                    ))}
                  </div>
                  <div className="mt-6 pt-4 border-t border-slate-100">
                    <p className="text-[10px] uppercase font-bold text-red-400 mb-2">Admin Actions</p>
                    <div className="grid grid-cols-2 gap-2">
                      <button onClick={() => handlePurge("vector")} className="p-2 border border-orange-200 text-orange-600 rounded-lg text-xs font-bold bg-orange-50">Vector DB</button>
                      <button onClick={() => handlePurge("pdf")} className="p-2 border border-red-200 text-red-600 rounded-lg text-xs font-bold bg-red-50">PDF & Data</button>
                    </div>
                  </div>
                </div>
              </section>

              <section className="col-span-12 lg:col-span-8 space-y-6">
                <div className="grid grid-cols-2 gap-6">
                  <div className="bg-white border border-slate-200 p-6 rounded-2xl shadow-sm">
                    <div className="flex justify-between mb-2"><span className="text-slate-500 text-sm font-bold">Progress</span><span className="text-blue-600 font-bold">{status.progress}%</span></div>
                    <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden"><div className="bg-blue-600 h-full transition-all" style={{ width: `${status.progress}%` }} /></div>
                  </div>
                  <div className="bg-white border border-slate-200 p-6 rounded-2xl shadow-sm">
                    <span className="text-slate-500 text-sm font-bold block mb-2">Resources</span>
                    <div className="flex justify-between items-end"><span className="text-2xl font-bold text-slate-800">{status.ram_usage} <span className="text-xs text-slate-400">MB</span></span><span className="text-xs text-emerald-500 font-bold">● Live</span></div>
                  </div>
                </div>
                <div className="bg-white border border-slate-200 rounded-2xl shadow-sm h-[500px] flex flex-col p-6">
                  <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-slate-800"><ActivityIcon size={20} className="text-blue-600" /> Interaction Console</h3>
                  <div className="flex-1 overflow-y-auto mb-4 space-y-4 pr-2 custom-scrollbar">
                    {messages.length === 0 ? <div className="h-full flex flex-col items-center justify-center opacity-30 gap-4"><ActivityIcon size={60} /><p className="font-bold">Hệ thống sẵn sàng xử lý truy vấn</p></div> : messages.map((m, i) => (
                      <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}><div className={`max-w-[85%] p-4 rounded-2xl shadow-sm ${m.role === "user" ? "bg-blue-600 text-white" : "bg-slate-50 border border-slate-200"}`}><div className="text-sm prose prose-slate max-w-none"><ReactMarkdown>{m.content}</ReactMarkdown></div></div></div>
                    ))}
                  </div>
                  <form onSubmit={handleChatSubmit} className="relative flex items-center gap-3 border-t border-slate-100 pt-4">
                    <select value={chatConfig.collection} onChange={(e) => setChatConfig(prev=>({...prev, collection: e.target.value}))} className="p-3 bg-slate-100 rounded-xl text-xs font-bold outline-none border-none">
                      <option value="vector_raw">RAW</option><option value="vector_pq">PQ</option><option value="vector_sq8">SQ8</option><option value="vector_arq">ARQ</option><option value="vector_adaptive">ADAPTIVE</option>
                    </select>
                    <input value={inputMessage} onChange={(e) => setInputMessage(e.target.value)} placeholder="Nhập câu hỏi nghiên cứu..." className="flex-1 p-3 bg-white border border-slate-200 rounded-xl outline-none focus:ring-2 focus:ring-blue-500/20" />
                    <button type="submit" disabled={isTyping} className="p-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-all"><Play size={20} /></button>
                  </form>
                </div>
              </section>
            </motion.div>
          ) : (
            <motion.div key="research" initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.98 }} className="space-y-6">
              <div className="grid grid-cols-12 gap-6">
                <div className="col-span-12 lg:col-span-4 space-y-6">
                  <div className="bg-white border border-slate-200 p-6 rounded-2xl shadow-sm">
                    <h3 className="text-lg font-bold mb-6 flex items-center gap-2 text-indigo-800"><Zap size={20} className="text-indigo-600" /> Research Control</h3>
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-2">
                        {["vector_raw", "vector_pq", "vector_sq8", "vector_arq", "vector_adaptive"].map(m => (
                          <button key={m} onClick={() => handleRunBenchmarkTest(m)} disabled={status.benchmark_running} className={`p-3 rounded-lg border text-[10px] font-bold transition-all ${status.benchmark_model === m ? "bg-indigo-600 text-white" : "bg-slate-50 border-slate-200"}`}>{m.replace("vector_", "").toUpperCase()}</button>
                        ))}
                      </div>
                      <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
                        <label className="text-xs font-bold text-slate-500 mb-2 block">Batch Samples</label>
                        <input type="number" value={benchmarkBatchSize} onChange={(e) => setBenchmarkBatchSize(parseInt(e.target.value))} className="w-full p-2 bg-white border border-slate-200 rounded text-sm outline-none" />
                      </div>
                      <button onClick={() => handleClearHistory()} className="w-full py-2 bg-red-50 text-red-600 rounded-lg text-xs font-bold border border-red-100">Dọn dẹp lịch sử</button>
                    </div>
                  </div>
                </div>
                <div className="col-span-12 lg:col-span-8">
                  <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
                    <div className="p-4 bg-slate-50 border-b border-slate-200 flex justify-between items-center"><h3 className="font-bold text-slate-800 flex items-center gap-2"><Clock size={18} className="text-indigo-600" /> Research Results</h3><button onClick={() => fetchBenchmarkHistory()}><RefreshCw size={16} className={status.benchmark_running ? "animate-spin" : ""} /></button></div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left">
                        <thead className="text-[10px] font-bold text-slate-400 bg-slate-50/50 uppercase tracking-tighter">
                          <tr><th className="p-4">Model</th><th className="p-4">Question</th><th className="p-4">Latency</th><th className="p-4">RAM</th></tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {benchmarkHistory.length === 0 ? <tr><td colSpan={4} className="p-12 text-center text-slate-400 italic">Chưa có kết quả thực nghiệm.</td></tr> : benchmarkHistory.map((res: any, i: number) => (
                            <tr key={i} className="text-xs hover:bg-slate-50/50 transition-colors">
                              <td className="p-4 font-bold text-indigo-600">{res.model_name?.replace("vector_", "").toUpperCase()}</td>
                              <td className="p-4 max-w-sm truncate">{res.question}</td>
                              <td className="p-4 font-mono font-bold text-emerald-600">{res.latency_ms?.toFixed(0)}ms</td>
                              <td className="p-4 font-mono font-bold text-amber-600">{res.peak_ram_mb?.toFixed(1)}MB</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: #f1f5f9; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
      `}</style>
    </div>
  );
}
