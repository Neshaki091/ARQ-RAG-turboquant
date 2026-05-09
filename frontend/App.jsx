import React, { useState, useEffect } from 'react';
import { Send, Upload, FileText, Cpu, Search, Trash2, Book, Loader2, Sparkles, MessageSquare, Zap, Activity, Shield } from 'lucide-react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';

const API_BASE = "http://localhost:8000";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [mode, setMode] = useState("fast");
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' or 'simulate'
  const [simQueries, setSimQueries] = useState([]);
  const [simCount, setSimCount] = useState(8);
  const [simResults, setSimResults] = useState({});
  const [isSimulating, setIsSimulating] = useState(false);

  // Load benchmark queries từ CSV (Mô phỏng bằng cách fetch text)
  const loadBenchmark = async () => {
    try {
      const defaultQueries = [
        "RAG là gì?", "Vector Quantization hoạt động như thế nào?", 
        "Giải thích về phương pháp Q-MLLM", "So sánh ARES và RAGAS",
        "Lattice Vector Quantization là gì?", "RobustRAG bảo vệ hệ thống như thế nào?",
        "Lợi ích của việc nén 2-bit", "TurboQuant nhanh hơn bao nhiêu lần?",
        "Làm thế nào để tối ưu hóa RAG?", "Cơ chế của IVF trong TurboQuant",
        "Tại sao dùng SQ+QJL?", "Độ trễ của 2-bit và 4-bit khác nhau thế nào?"
      ];
      
      let randomSet = [];
      for(let i=0; i<simCount; i++) {
        randomSet.push(defaultQueries[Math.floor(Math.random() * defaultQueries.length)]);
      }
      setSimQueries(randomSet);
    } catch (e) { console.error(e); }
  };

  const streamSingleQuery = async (query, idx) => {
    const startTime = Date.now();
    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: query, mode: mode })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let isMetaEnd = false;
      let fullContent = "";
      let ttftRecorded = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });

        if (!isMetaEnd) {
          if (chunk.includes("--META_END--")) {
            const parts = chunk.split("--META_END--");
            const meta = JSON.parse(parts[0]);
            setSimResults(prev => ({
              ...prev,
              [idx]: { ...prev[idx], ...meta, retrieval: meta.latency }
            }));
            isMetaEnd = true;
            if (parts[1]) {
              if (!ttftRecorded) {
                setSimResults(prev => ({ ...prev, [idx]: { ...prev[idx], ttft: `${Date.now() - startTime}ms` } }));
                ttftRecorded = true;
              }
              fullContent += parts[1];
              setSimResults(prev => ({ ...prev, [idx]: { ...prev[idx], content: fullContent } }));
            }
          }
        } else {
          if (!ttftRecorded && chunk.trim()) {
            setSimResults(prev => ({ ...prev, [idx]: { ...prev[idx], ttft: `${Date.now() - startTime}ms` } }));
            ttftRecorded = true;
          }
          fullContent += chunk;
          setSimResults(prev => ({ ...prev, [idx]: { ...prev[idx], content: fullContent } }));
        }
      }
    } catch (err) {
      setSimResults(prev => ({ ...prev, [idx]: { ...prev[idx], content: "Lỗi kết nối!" } }));
    }
  };

  const handleRunSimulation = async () => {
    setIsSimulating(true);
    setSimResults({});
    
    // Bắn loạt song song
    const tasks = simQueries.map((query, idx) => streamSingleQuery(query, idx));
    await Promise.all(tasks);
    
    setIsSimulating(false);
  };

  const addSimQuery = () => {
    setSimQueries([...simQueries, "Câu hỏi mới..."]);
  };

  const updateSimQuery = (index, value) => {
    const newQueries = [...simQueries];
    newQueries[index] = value;
    setSimQueries(newQueries);
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await axios.get(`${API_BASE}/documents`);
      setDocuments(res.data.documents || []);
    } catch (err) {
      console.error("Failed to fetch documents", err);
    }
  };

  const handleSend = async () => {
    if (!input || loading) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    // Tạo một message trống cho bot để fill dần
    const botMsgId = Date.now();
    setMessages(prev => [...prev, {
      id: botMsgId,
      role: 'bot',
      content: "",
      loading: true
    }]);

    try {
      const sendTime = Date.now();
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input, mode: mode })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let isMetaEnd = false;
      let fullContent = "";
      let ttftRecorded = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        if (!isMetaEnd) {
          if (chunk.includes("--META_END--")) {
            const parts = chunk.split("--META_END--");
            const metaStr = parts[0];
            try {
              const meta = JSON.parse(metaStr);
              setMessages(prev => prev.map(m =>
                m.id === botMsgId ? { ...m, ...meta, loading: false } : m
              ));
            } catch (e) { console.error("Meta parse error", e); }

            isMetaEnd = true;
            if (parts[1]) {
              // Nhận được token đầu tiên ngay sau META_END
              if (!ttftRecorded) {
                const ttft = Date.now() - sendTime;
                setMessages(prev => prev.map(m =>
                  m.id === botMsgId ? { ...m, ttft: `${ttft}ms` } : m
                ));
                ttftRecorded = true;
              }
              fullContent += parts[1];
              setMessages(prev => prev.map(m =>
                m.id === botMsgId ? { ...m, content: fullContent } : m
              ));
            }
          }
        } else {
          // Nhận các token tiếp theo
          if (!ttftRecorded && chunk.trim()) {
            const ttft = Date.now() - sendTime;
            setMessages(prev => prev.map(m =>
              m.id === botMsgId ? { ...m, ttft: `${ttft}ms` } : m
            ));
            ttftRecorded = true;
          }
          fullContent += chunk;
          setMessages(prev => {
            const newMsgs = [...prev];
            const botIdx = newMsgs.findIndex(m => m.id === botMsgId);
            if (botIdx !== -1) {
              newMsgs[botIdx] = { ...newMsgs[botIdx], content: fullContent };
            }
            return newMsgs;
          });
        }
      }
    } catch (err) {
      console.error(err);
      setMessages(prev => prev.map(m =>
        m.id === botMsgId ? { ...m, content: "Có lỗi xảy ra!", loading: false } : m
      ));
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      await axios.post(`${API_BASE}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      fetchDocuments();
    } catch (err) {
      alert("Lỗi khi tải lên!");
    } finally {
      setUploading(false);
      e.target.value = null;
    }
  };

  const handleDelete = async (filename) => {
    if (!confirm(`Bạn có chắc chắn muốn xóa tài liệu "${filename}"?`)) return;
    try {
      await axios.delete(`${API_BASE}/documents/${filename}`);
      fetchDocuments();
    } catch (err) {
      alert("Lỗi khi xóa tài liệu!");
    }
  };

  const handleImport = async () => {
    if (!confirm("Bạn có muốn nạp bộ dữ liệu pre-computed (1114 papers) không? Việc này sẽ ghi đè dữ liệu hiện tại.")) return;
    setLoading(true);
    try {
      await axios.post(`${API_BASE}/import-precomputed`);
      alert("Nạp dữ liệu thành công! Cả 2-bit và 4-bit Engine đã được sẵn sàng.");
      fetchDocuments();
    } catch (err) {
      alert("Lỗi khi nạp dữ liệu!");
    } finally {
      setLoading(false);
    }
  };

  const handleCleanup = async () => {
    if (!confirm("Bạn có muốn dọn dẹp các phiên bản Index cũ để tiết kiệm dung lượng không?")) return;
    setLoading(true);
    try {
      await axios.post(`${API_BASE}/cleanup`);
      alert("Đã dọn dẹp thành công!");
    } catch (err) {
      alert("Lỗi khi dọn dẹp!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <div className="sidebar">
        <div className="brand">
          <div className="logo-container">
            <Cpu className="logo-icon" />
          </div>
          <div className="brand-text">
            <h1>ARQ-RAG</h1>
            <span>TurboQuant Engine</span>
          </div>
        </div>

        <div className="mode-selector">
          <div className="section-header">
            <Zap size={14} />
            <h3>CHẾ ĐỘ TRUY VẤN</h3>
          </div>
          <div className="mode-tabs">
            <div className={`mode-tab ${mode === 'raw' ? 'active' : ''}`} onClick={() => setMode('raw')}>
              <Shield size={14} />
              <span>Raw</span>
            </div>
            <div className={`mode-tab ${mode === 'fast' ? 'active' : ''}`} onClick={() => setMode('fast')}>
              <Zap size={14} />
              <span>Fast</span>
            </div>
            <div className={`mode-tab ${mode === 'ultra' ? 'active' : ''}`} onClick={() => setMode('ultra')}>
              <Activity size={14} />
              <span>Ultra</span>
            </div>
          </div>
          <p className="mode-desc">
            {mode === 'raw' && "Tìm kiếm vector gốc (Chính xác cao, Chậm)"}
            {mode === 'fast' && "TurboQuant 4-bit (Cân bằng, Nhanh)"}
            {mode === 'ultra' && "TurboQuant 2-bit (Tối ưu, Cực nhanh)"}
          </p>
        </div>

        <div className="upload-section">
          <label className={`upload-card ${uploading ? 'disabled' : ''}`}>
            <div className="upload-icon-box">
              {uploading ? <Loader2 className="spin" /> : <Upload />}
            </div>
            <div className="upload-info">
              <span className="upload-title">{uploading ? "Đang xử lý..." : "Tải tài liệu mới"}</span>
              <span className="upload-subtitle">Tự động nén 2/4-bit</span>
            </div>
            <input type="file" hidden onChange={handleUpload} accept=".pdf" disabled={uploading} />
          </label>
        </div>

        <button className="import-btn" onClick={handleImport} disabled={loading || uploading}>
          <Sparkles size={16} />
          Nạp Papers (Hybrid-Bit)
        </button>

        <button className="cleanup-btn" onClick={handleCleanup} disabled={loading || uploading}>
          <Trash2 size={16} />
          Dọn dẹp phiên bản cũ
        </button>

        <div className="doc-section">
          <div className="section-header">
            <Book size={16} />
            <h3>KHO TRI THỨC</h3>
            <span className="count">{documents.length}</span>
          </div>
          <div className="doc-list">
            {documents.length === 0 ? (
              <div className="empty-docs">Chưa có tài liệu nào</div>
            ) : (
              documents.map((doc, idx) => (
                <div key={idx} className="doc-item">
                  <div className="doc-icon"><FileText size={14} /></div>
                  <div className="doc-name">{doc}</div>
                  <button className="delete-btn" onClick={() => handleDelete(doc)}>
                    <Trash2 size={14} />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="main-content">
        <header className="main-header">
          <div className="header-info">
            <MessageSquare size={20} className="text-primary" />
            <div className="tabs-nav">
              <button className={`tab-link ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
                <MessageSquare size={16} /> Chat
              </button>
              <button className={`tab-link ${activeTab === 'simulate' ? 'active' : ''}`} onClick={() => setActiveTab('simulate')}>
                <Activity size={16} /> Simulate Load
              </button>
            </div>
          </div>
          <div className="header-actions">
            <div className="engine-badge">
              <Sparkles size={14} />
              <span>{mode.toUpperCase()} MODE ACTIVE</span>
            </div>
          </div>
        </header>

        {activeTab === 'chat' ? (
          <div className="chat-container">
            <div className="messages-area">
              {messages.length === 0 && (
                <div className="welcome-screen">
                  <div className="welcome-art"><Cpu size={64} className="art-icon" /></div>
                  <h3>Khám phá kiến thức của bạn</h3>
                  <p>Hệ thống TurboQuant đang sẵn sàng xử lý 1080 bài báo khoa học.</p>
                </div>
              )}
              {messages.map((m, i) => (
                <div key={i} className={`message-wrapper ${m.role}`}>
                  <div className="message-avatar">{m.role === 'user' ? 'U' : 'AI'}</div>
                  <div className="message-content">
                    <div className="metrics-row">
                      {m.complexity && <div className={`complexity-tag ${m.complexity.toLowerCase()}`}>🧠 {m.complexity}</div>}
                      {m.latency && <div className="latency-tag">🔍 Retrieval: {m.latency}</div>}
                      {m.ttft && <div className="ttft-tag">🚀 TTFT: {m.ttft}</div>}
                    </div>
                    <div className="text-body"><ReactMarkdown>{m.content}</ReactMarkdown></div>
                    {m.sources && (
                      <div className="message-sources">
                        {m.sources.map((s, si) => <span key={si} className="source-tag">{s}</span>)}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {loading && <div className="typing-indicator"><span></span><span></span><span></span></div>}
            </div>

            <div className="input-container">
              <div className="input-wrapper">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                  placeholder="Hỏi bất cứ điều gì..."
                  disabled={loading}
                />
                <button className="send-btn" onClick={handleSend} disabled={!input || loading}>
                  <Send size={20} />
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="simulate-container">
            <div className="sim-dashboard">
              <div className="sim-actions">
                <div className="sim-count-control">
                  <label>Số lượng:</label>
                  <input 
                    type="number" 
                    value={simCount} 
                    onChange={(e) => setSimCount(parseInt(e.target.value) || 1)} 
                    min="1" max="20"
                  />
                </div>
                <button onClick={loadBenchmark} className="btn-secondary">Nạp mẫu ({simCount} câu)</button>
                <button 
                  onClick={handleRunSimulation} 
                  className={`btn-fire ${isSimulating ? 'loading' : ''}`}
                  disabled={isSimulating || simQueries.length === 0}
                >
                  <Zap size={18} /> {isSimulating ? 'Đang mô phỏng...' : 'BẮN LOẠT (FIRE)'}
                </button>
              </div>

              {Object.keys(simResults).length > 0 && (
                <div className="sim-stats">
                  <div className="stat-box highlight">
                    <span className="label">Đang xử lý</span>
                    <span className="value">{Object.keys(simResults).length}/{simQueries.length}</span>
                  </div>
                  <div className="stat-box">
                    <span className="label">Chế độ</span>
                    <span className="value">{mode.toUpperCase()}</span>
                  </div>
                </div>
              )}

              <div className="sim-table-header">
                <div className="h-col num">#</div>
                <div className="h-col query">CÂU HỎI (QUERY)</div>
                <div className="h-col answer">AI STREAMING & METRICS</div>
              </div>
              <div className="sim-query-list">
                {simQueries.map((q, idx) => (
                  <div key={idx} className="sim-row">
                    <span className="row-num">{idx + 1}</span>
                    <div className="sim-content-grid">
                      <div className="sim-query-col">
                        <div className="q-text">{q}</div>
                      </div>
                      <div className="sim-answer-col">
                        {simResults[idx] ? (
                          <div className="answer-wrapper">
                            <div className="mini-metrics">
                              {simResults[idx].retrieval && <span className="m-tag r">R: {simResults[idx].retrieval}</span>}
                              {simResults[idx].ttft && <span className="m-tag t">TTFT: {simResults[idx].ttft}</span>}
                            </div>
                            <div className="answer-text">
                              {simResults[idx].content || "..."}
                            </div>
                          </div>
                        ) : (
                          <div className="answer-placeholder">Sẵn sàng...</div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                {simQueries.length === 0 && <div className="empty-sim">Nhấn "Nạp mẫu" để chuẩn bị.</div>}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
