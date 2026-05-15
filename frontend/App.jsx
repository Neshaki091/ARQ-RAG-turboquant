import React, { useState, useEffect } from 'react';
import { Send, Upload, FileText, Cpu, Search, Trash2, Book, Loader2, Sparkles, MessageSquare, Zap, Activity, Shield, LogOut, User, Lock, Layers, Play, Menu, ChevronLeft } from 'lucide-react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';

const API_BASE = import.meta.env.VITE_API_BASE || "https://neshaki-arq-rag-turboquant.hf.space";

export default function App() {
  const [token, setToken] = useState(localStorage.getItem("token") || "");
  const [user, setUser] = useState(null);
  const [isRegistering, setIsRegistering] = useState(false);
  const [authForm, setAuthForm] = useState({ username: "", password: "" });
  
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [mode, setMode] = useState("balance");
  const [scope, setScope] = useState("both"); // 'user', 'system', 'both'
  const [activeTab, setActiveTab] = useState('chat');
  const [simQueries, setSimQueries] = useState([]);
  const [simResults, setSimResults] = useState({});
  const [isSimulating, setIsSimulating] = useState(false);
  const [simCount, setSimCount] = useState(32); // Mặc định 32, tối đa 50
  
  // New States for Sessions
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState("default");
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  
  const [selectedDocChunks, setSelectedDocChunks] = useState(null);
  const [viewingDocName, setViewingDocName] = useState("");
  const [adminUsers, setAdminUsers] = useState([]);
  
  const [systemChunks, setSystemChunks] = useState([]);
  const [systemTotal, setSystemTotal] = useState(0);
  const [systemOffset, setSystemOffset] = useState(0);
  const [adminSubTab, setAdminSubTab] = useState('users'); // 'users', 'system'
  
  const [telemetry, setTelemetry] = useState({ cpu: 0, ram: 0, uptime: 0 });

  useEffect(() => {
    if (token) {
      fetchUser();
      fetchSessions();
    }
  }, [token]);

  useEffect(() => {
    if (token && activeSessionId) {
      fetchMessages(activeSessionId);
      fetchDocuments(activeSessionId);
    }
  }, [activeSessionId, token]);

  useEffect(() => {
    if (user?.role === 'admin') {
      fetchAdminUsers();
    }
  }, [user]);

  useEffect(() => {
    const timer = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE}/system/stats`);
        setTelemetry({
          cpu: res.data.cpu_percent,
          ram: res.data.memory_mb,
          uptime: res.data.uptime
        });
      } catch (e) {}
    }, 5000);
    return () => clearInterval(timer);
  }, []);

  const fetchAdminUsers = async () => {
    try {
      const res = await axios.get(`${API_BASE}/admin/users`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setAdminUsers(res.data.users || []);
    } catch (err) {
      console.error("Failed to fetch admin users", err);
    }
  };

  const fetchSystemChunks = async (offset = 0) => {
    try {
      const res = await axios.get(`${API_BASE}/admin/system/chunks?offset=${offset}&limit=100`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setSystemChunks(res.data.chunks || []);
      setSystemTotal(res.data.total || 0);
      setSystemOffset(res.data.offset);
    } catch (err) {
      console.error("Failed to fetch system chunks", err);
    }
  };

  useEffect(() => {
    if (activeTab === 'admin' && adminSubTab === 'system') {
      fetchSystemChunks(0);
    }
  }, [activeTab, adminSubTab]);

  const fetchUser = async () => {
    try {
      const res = await axios.get(`${API_BASE}/me`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setUser(res.data);
    } catch (err) {
      handleLogout();
    }
  };

  const fetchDocuments = async (sid) => {
    try {
      const url = sid ? `${API_BASE}/documents?session_id=${sid}` : `${API_BASE}/documents`;
      const res = await axios.get(url, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setDocuments(res.data.documents || []);
    } catch (err) {
      console.error("Failed to fetch documents", err);
    }
  };

  const fetchSessions = async () => {
    try {
      const res = await axios.get(`${API_BASE}/sessions`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setSessions(res.data.sessions || []);
    } catch (err) {
      console.error("Failed to fetch sessions", err);
    }
  };

  const fetchMessages = async (sid) => {
    if (!sid) return;
    setIsHistoryLoading(true);
    try {
      const res = await axios.get(`${API_BASE}/sessions/${sid}/messages`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      // Convert backend roles to frontend roles
      const formatted = res.data.messages.map(m => ({
        role: m.role === 'assistant' ? 'bot' : 'user',
        content: m.content,
        timestamp: m.created_at
      }));
      setMessages(formatted);
    } catch (err) {
      console.error("Failed to fetch messages", err);
    } finally {
      setIsHistoryLoading(false);
    }
  };

  const handleLoadBenchmark = async () => {
    try {
      const res = await axios.get(`${API_BASE}/benchmark/queries`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const all = Array.isArray(res.data) ? res.data : [];
      if (all.length === 0) {
        alert("Bộ câu hỏi rỗng!");
        return;
      }
      // Lấy ngẫu nhiên theo số lượng simCount
      const shuffled = [...all].sort(() => 0.5 - Math.random());
      const selected = shuffled.slice(0, Math.min(simCount, 50)).map((q, idx) => {
        let qText = "";
        if (typeof q === 'string') qText = q;
        else qText = q.question || q.text || q.query || "Câu hỏi không xác định";
        
        return {
          id: idx,
          question: qText,
          status: 'idle',
          answer: null,
          latency: 0,
          embed_latency: 0,
          search_latency: 0,
          chunks_count: 0,
          chunks: [],
          complexity: ""
        };
      });
      setSimQueries(selected);
      setSimResults({});
    } catch (err) {
      console.error(err);
      alert("Lỗi tải bộ câu hỏi benchmark!");
    }
  };

  const runSimulation = async () => {
    if (simQueries.length === 0) return;
    setIsSimulating(true);
    setSimResults(null);
    
    const startTime = Date.now();
    let completedCount = 0;
    let totalEmbed = 0;
    let totalSearch = 0;

    // Cập nhật trạng thái tất cả sang pending
    setSimQueries(prev => prev.map(q => ({ ...q, status: 'pending', answer: null })));

    // Gửi MỘT request duy nhất chứa toàn bộ batch để kích hoạt cơ chế Batch Processing (không gọi LLM)
    try {
      const res = await axios.post(`${API_BASE}/chat`, {
        messages_batch: simQueries.map(q => q.question),
        user_id: user.id,
        session_id: "simulation",
        mode: mode,
        scope: scope
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });

      const batchResults = res.data.batch_results || [];
      
      // Cập nhật kết quả cho toàn bộ các câu hỏi cùng lúc
      setSimQueries(prev => prev.map((q, idx) => {
        const bRes = batchResults[idx] || {};
        return {
          ...q,
          status: 'success',
          chunks_count: (bRes.hydrated_results || []).length,
          chunks: bRes.hydrated_results || [],
          complexity: bRes.complexity || "Average",
          embed_latency: bRes.embed_latency || 0,
          search_latency: bRes.search_latency || 0,
          latency: (Date.now() - startTime) / simQueries.length // Độ trễ trung bình cho mỗi câu
        };
      }));

      const totalDuration = Date.now() - startTime;
      const totalEmbed = batchResults.reduce((acc, r) => acc + (r.embed_latency || 0), 0);
      const totalSearch = batchResults.reduce((acc, r) => acc + (r.search_latency || 0), 0);

      setSimResults({
        totalTime: totalDuration,
        avgLatency: totalDuration / simQueries.length,
        avgEmbed: totalEmbed / simQueries.length,
        avgSearch: totalSearch / simQueries.length,
        throughput: (simQueries.length / (totalDuration / 1000)).toFixed(2)
      });

    } catch (err) {
      console.error(`Simulation failed:`, err);
      setSimQueries(prev => prev.map(q => ({ ...q, status: 'error' })));
    }
    
    setIsSimulating(false);
  };

  const handleCreateSession = async () => {
    const newId = `session_${Date.now()}`;
    const title = prompt("Nhập tiêu đề cuộc trò chuyện mới:", "New Chat") || "New Chat";
    try {
      await axios.post(`${API_BASE}/sessions`, 
        { session_id: newId, title: title },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      fetchSessions();
      setActiveSessionId(newId);
    } catch (err) {
      alert("Lỗi khi tạo phiên mới!");
    }
  };

  const handleDeleteSession = async (e, sid) => {
    e.stopPropagation();
    if (!confirm("Xóa cuộc trò chuyện này?")) return;
    try {
      await axios.delete(`${API_BASE}/sessions/${sid}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      fetchSessions();
      if (activeSessionId === sid) setActiveSessionId("default");
    } catch (err) {
      alert("Lỗi khi xóa phiên!");
    }
  };

  const handleAuth = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      if (isRegistering) {
        await axios.post(`${API_BASE}/register`, authForm);
        alert("Đăng ký thành công! Hãy đăng nhập.");
        setIsRegistering(false);
      } else {
        const res = await axios.post(`${API_BASE}/login`, authForm);
        const newToken = res.data.access_token;
        setToken(newToken);
        localStorage.setItem("token", newToken);
      }
    } catch (err) {
      alert(err.response?.data?.detail || "Lỗi xác thực!");
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    setToken("");
    setUser(null);
    localStorage.removeItem("token");
    setMessages([]);
  };

  const handleSend = async () => {
    if (!input || loading) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

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
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ 
          message: input, 
          mode: mode, 
          scope: scope,
          session_id: activeSessionId,
          stream: true
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let isMetaEnd = false;
      let fullContent = "";
      let ttftRecorded = false;
      let buffer = ""; // Bộ đệm để tích lũy dữ liệu meta

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        if (!isMetaEnd) {
          buffer += chunk;
          if (buffer.includes("--META_END--")) {
            const parts = buffer.split("--META_END--");
            const metaStr = parts[0].trim(); // Thêm trim() để loại bỏ khoảng trắng mồi
            try {
              const meta = JSON.parse(metaStr);
              setMessages(prev => prev.map(m =>
                m.id === botMsgId ? { ...m, ...meta, loading: false } : m
              ));
            } catch (e) { console.error("Meta parse error", e); }

            isMetaEnd = true;
            if (parts[1]) {
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
        headers: { 
          "Content-Type": "multipart/form-data",
          "Authorization": `Bearer ${token}`
        }
      });
      fetchDocuments(activeSessionId);
    } catch (err) {
      alert("Lỗi khi tải lên!");
    } finally {
      setUploading(false);
      e.target.value = null;
    }
  };

  const handleViewDoc = async (filename) => {
    try {
      setViewingDocName(filename);
      const res = await axios.get(`${API_BASE}/documents/${filename}/chunks`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setSelectedDocChunks(res.data.chunks || []);
    } catch (err) {
      alert("Lỗi khi tải nội dung tài liệu!");
    }
  };

  const handleDelete = async (filename) => {
    if (!confirm(`Bạn có chắc chắn muốn xóa tài liệu "${filename}"?`)) return;
    try {
      await axios.delete(`${API_BASE}/documents/${filename}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      fetchDocuments(activeSessionId);
    } catch (err) {
      alert("Lỗi khi xóa tài liệu!");
    }
  };

  const [showLeftSidebar, setShowLeftSidebar] = useState(false);
  const [showRightSidebar, setShowRightSidebar] = useState(false);

  if (!token) {
    return (
      <div className="auth-screen">
        <div className="auth-card">
          <div className="brand-vertical">
            <div className="logo-container large">
              <Cpu className="logo-icon" />
            </div>
            <h1>ARQ-RAG</h1>
            <p>Adaptive Routing Quantization</p>
          </div>
          
          <form onSubmit={handleAuth}>
            <div className="input-group">
              <User size={18} />
              <input 
                type="text" 
                placeholder="Tên đăng nhập" 
                required 
                value={authForm.username}
                onChange={e => setAuthForm({...authForm, username: e.target.value})}
              />
            </div>
            <div className="input-group">
              <Lock size={18} />
              <input 
                type="password" 
                placeholder="Mật khẩu" 
                required 
                value={authForm.password}
                onChange={e => setAuthForm({...authForm, password: e.target.value})}
              />
            </div>
            <button className="auth-submit" type="submit" disabled={loading}>
              {loading ? <Loader2 className="spin" /> : (isRegistering ? "Đăng ký" : "Đăng nhập")}
            </button>
          </form>
          
          <div className="auth-toggle">
            {isRegistering ? "Đã có tài khoản?" : "Chưa có tài khoản?"}
            <button onClick={() => setIsRegistering(!isRegistering)}>
              {isRegistering ? "Đăng nhập ngay" : "Đăng ký ngay"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className={`sidebar ${showLeftSidebar ? 'open' : ''}`}>
        <div className="brand">
          <div className="logo-container">
            <Cpu className="logo-icon" />
          </div>
          <div className="brand-text">
            <h2>TurboQuant</h2>
            <span>{user?.username}</span>
          </div>
          <button className="mobile-close" onClick={() => setShowLeftSidebar(false)}>
            <ChevronLeft size={20} />
          </button>
          <button className="logout-btn" onClick={handleLogout} title="Đăng xuất">
            <LogOut size={16} />
          </button>
        </div>

        <div className="mode-selector">
          <div className="section-header">
            <Zap size={14} />
            <h3>CHẾ ĐỘ TRUY VẤN</h3>
          </div>
          <div className="mode-grid">
            {['ultrafast', 'fast', 'balance', 'accuracy', 'adaptive'].map(m => (
              <div 
                key={m} 
                className={`mode-tab ${mode === m ? 'active' : ''}`} 
                onClick={() => setMode(m)}
              >
                <span>{m.charAt(0).toUpperCase() + m.slice(1)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="scope-selector">
          <div className="section-header">
            <Layers size={14} />
            <h3>PHẠM VI BỘ NHỚ</h3>
          </div>
          <div className="scope-tabs">
            <div className={`scope-tab ${scope === 'user' ? 'active' : ''}`} onClick={() => setScope('user')}>
              <span>User</span>
            </div>
            <div className={`scope-tab ${scope === 'system' ? 'active' : ''}`} onClick={() => setScope('system')}>
              <span>System</span>
            </div>
            <div className={`scope-tab ${scope === 'both' ? 'active' : ''}`} onClick={() => setScope('both')}>
              <span>Both</span>
            </div>
          </div>
        </div>

        <div className="upload-section">
          <label className={`upload-card ${uploading ? 'disabled' : ''}`}>
            <div className="upload-icon-box">
              {uploading ? <Loader2 className="spin" /> : <Upload />}
            </div>
            <div className="upload-info">
              <span className="upload-title">Tải tài liệu mới</span>
              <span className="upload-subtitle">Session: {activeSessionId}</span>
            </div>
            <input type="file" hidden onChange={handleUpload} accept=".pdf" disabled={uploading} />
          </label>
        </div>

        <div className="session-section">
          <div className="section-header">
            <MessageSquare size={16} />
            <h3>HỘI THOẠI</h3>
            <button className="new-chat-btn" onClick={handleCreateSession} title="Tạo phiên mới">+</button>
          </div>
          <div className="session-list">
            <div 
              className={`session-item ${activeSessionId === 'default' ? 'active' : ''}`}
              onClick={() => { setActiveSessionId('default'); setShowLeftSidebar(false); }}
            >
              <div className="session-name">Mặc định (General)</div>
            </div>
            {sessions.map(s => (
              <div 
                key={s.id} 
                className={`session-item ${activeSessionId === s.id ? 'active' : ''}`}
                onClick={() => { setActiveSessionId(s.id); setShowLeftSidebar(false); }}
              >
                <div className="session-name" title={s.title}>{s.title}</div>
                <button className="sess-del" onClick={(e) => handleDeleteSession(e, s.id)}><Trash2 size={12} /></button>
              </div>
            ))}
          </div>
        </div>

        <div className="doc-section">
          <div className="section-header">
            <Book size={16} />
            <h3>TÀI LIỆU CỦA BẠN</h3>
            <span className="count">{documents.length}</span>
          </div>
          <div className="doc-list">
            {documents.length === 0 ? (
              <div className="empty-docs">Chưa có tài liệu nào</div>
            ) : (
              documents.map((doc, idx) => (
                <div key={idx} className="doc-item">
                  <div className="doc-icon"><FileText size={14} /></div>
                  <div className="doc-name" title={doc}>{doc}</div>
                  <div className="doc-actions">
                    <button className="view-btn" onClick={() => handleViewDoc(doc)} title="Xem chi tiết">
                      <Search size={14} />
                    </button>
                    <button className="delete-btn" onClick={() => handleDelete(doc)} title="Xóa">
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="main-content">
        <header className="main-header">
          <div className="header-left">
            <button className="mobile-toggle" onClick={() => setShowLeftSidebar(true)}>
              <Menu size={20} />
            </button>
            <div className="header-info">
              <MessageSquare size={20} className="text-primary hide-mobile" />
              <div className="tabs-nav">
                <button className={`tab-link ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
                  Chat
                </button>
                <button className={`tab-link ${activeTab === 'simulate' ? 'active' : ''}`} onClick={() => setActiveTab('simulate')}>
                  Simulate
                </button>
                {user?.role === 'admin' && (
                  <button className={`tab-link ${activeTab === 'admin' ? 'active' : ''}`} onClick={() => setActiveTab('admin')}>
                    Admin
                  </button>
                )}
              </div>
            </div>
          </div>
          <div className="header-actions">
            <div className="engine-badge hide-mobile">
              <Sparkles size={14} />
              <span>{mode.toUpperCase()} + {scope.toUpperCase()}</span>
            </div>
            <button className="mobile-toggle" onClick={() => setShowRightSidebar(!showRightSidebar)}>
              <Activity size={20} />
            </button>
          </div>
        </header>

        <div className="content-layout">
          <div className="center-view">
            {activeTab === 'chat' ? (
              <div className="chat-container">
                <div className="messages-area">
                  {messages.length === 0 && (
                    <div className="welcome-screen">
                      <div className="welcome-art"><Cpu size={64} className="art-icon" /></div>
                      <h3>Chào mừng, {user?.username} {user?.role === 'admin' && <span className="admin-badge">Admin</span>}</h3>
                      <p>Hệ thống ARQ-RAG đang sẵn sàng với bộ nhớ {scope}.</p>
                    </div>
                  )}
                  {messages.map((m, i) => (
                    <div key={i} className={`message-wrapper ${m.role}`}>
                      <div className="message-avatar">{m.role === 'user' ? 'U' : 'AI'}</div>
                      <div className="message-content">
                        <div className="metrics-row">
                          {m.complexity && <div className={`complexity-tag ${m.complexity.toLowerCase()}`}>🧠 {m.complexity}</div>}
                          {m.latency && <div className="latency-tag">🔍 {m.latency}</div>}
                          {m.ttft && <div className="ttft-tag">🚀 {m.ttft}</div>}
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
            ) : activeTab === 'simulate' ? (
              <div className="simulate-container">
                <div className="sim-header-row">
                   <div className="sim-title">
                      <Zap size={24} className="text-primary" />
                      <h2>Simulation Mode</h2>
                      <span className="badge">Dynamic Batching (500ms / 32 Max)</span>
                   </div>
                   <div className="sim-header-actions">
                      <div className="sim-count-input">
                        <span>Số câu:</span>
                        <input 
                          type="number" 
                          min="1" 
                          max="50" 
                          value={simCount} 
                          onChange={(e) => setSimCount(Math.min(50, parseInt(e.target.value) || 1))}
                          disabled={isSimulating}
                        />
                      </div>
                      <button className="sim-btn secondary" onClick={handleLoadBenchmark} disabled={isSimulating}>
                        <Search size={16} />
                        Load {simCount} Queries
                      </button>
                      <button 
                        className={`sim-btn primary ${isSimulating ? 'loading' : ''}`}
                        onClick={runSimulation}
                        disabled={isSimulating || simQueries.length === 0}
                      >
                        {isSimulating ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
                        Start Simulation
                      </button>
                   </div>
                </div>

                {simResults && simResults.totalTime && (
                  <div className="sim-metrics-grid">
                    <div className="sim-metric-card wait">
                      <Zap size={16} className="icon-gold" />
                      <span className="label">Total Wait</span>
                      <span className="value">{(Number(simResults.totalTime) / 1000).toFixed(2)}s</span>
                    </div>
                    <div className="sim-metric-card embed">
                      <Cpu size={16} className="icon-purple" />
                      <span className="label">Avg Embed</span>
                      <span className="value">{Math.round(simResults.avgEmbed)}ms</span>
                    </div>
                    <div className="sim-metric-card search">
                      <Search size={16} className="icon-cyan" />
                      <span className="label">Avg TQ Search</span>
                      <span className="value">{simResults.avgSearch?.toFixed(2)}ms</span>
                    </div>
                    <div className="sim-metric-card highlight">
                      <Activity size={16} className="icon-primary" />
                      <span className="label">Throughput</span>
                      <span className="value">{simResults.throughput || 0} Q/s</span>
                    </div>
                  </div>
                )}

                <div className="sim-grid">
                  {simQueries.length === 0 ? (
                    <div className="sim-empty">
                       <Activity size={48} className="text-dim" />
                       <p>Nhấn "Load 32 Queries" để bắt đầu mô phỏng kịch bản tải thực tế.</p>
                    </div>
                  ) : (
                    simQueries.map((q) => (
                      <div key={q.id} className={`sim-box ${q.status}`}>
                          <div className="sim-box-header">
                             <div className="sim-box-header-left">
                               <span className="idx">#{q.id + 1}</span>
                               {q.status === 'success' && (
                                 <span className={`complexity-badge ${q.complexity?.toLowerCase()}`}>
                                    {q.complexity}
                                 </span>
                               )}
                             </div>
                             {q.status === 'pending' && <Loader2 size={12} className="spin" />}
                          </div>
                         <div className="sim-question" title={q.question}>{q.question}</div>
                         {q.status === 'success' && (
                           <>
                             <div className="sim-chunks-list">
                                {q.chunks && q.chunks.map((c, ci) => (
                                   <div key={ci} className="sim-chunk-item">
                                      {c.text}
                                   </div>
                                ))}
                             </div>
                              <div className="sim-result-footer">
                                <div className="sim-latency-box wait" title="Tổng thời gian chờ (User Experience)">
                                   <Zap size={12} /> {q.latency}ms
                                </div>
                                <div className="sim-latency-box embed" title="Thời gian tạo Vector">
                                   <Cpu size={12} /> {q.embed_latency}ms
                                </div>
                                <div className="sim-latency-box search" title="Thời gian TurboQuant Search">
                                   <Search size={12} /> {q.search_latency}ms
                                </div>
                                <div className="sim-count-box">
                                   {q.chunks_count} Chks
                                </div>
                             </div>
                           </>
                         )}
                      </div>
                    ))
                  )}
                </div>
              </div>
            ) : (
              <div className="admin-container">
                <div className="admin-header">
                  <div className="admin-nav">
                    <button className={`admin-nav-btn ${adminSubTab === 'users' ? 'active' : ''}`} onClick={() => setAdminSubTab('users')}>Người dùng</button>
                    <button className={`admin-nav-btn ${adminSubTab === 'system' ? 'active' : ''}`} onClick={() => setAdminSubTab('system')}>Bộ nhớ Hệ thống (5M)</button>
                  </div>
                  <button className="refresh-btn" onClick={() => adminSubTab === 'users' ? fetchAdminUsers() : fetchSystemChunks(systemOffset)}>
                    <Activity size={16} /> Làm mới
                  </button>
                </div>

                {adminSubTab === 'users' ? (
                  <div className="user-table-wrapper">
                    <table className="user-table">
                      <thead>
                        <tr>
                          <th>ID</th>
                          <th>Username</th>
                          <th>Role</th>
                          <th>Ngày tạo</th>
                        </tr>
                      </thead>
                      <tbody>
                        {adminUsers.map(u => (
                          <tr key={u.id}>
                            <td>{u.id}</td>
                            <td>{u.username}</td>
                            <td><span className={`role-badge ${u.role}`}>{u.role}</span></td>
                            <td>{new Date(u.created_at).toLocaleDateString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="system-browser">
                    <div className="pagination-info">
                      <span>Hiển thị {systemOffset + 1} - {systemOffset + systemChunks.length} trong tổng số {systemTotal.toLocaleString()} vectors</span>
                      <div className="pagination-actions">
                        <button disabled={systemOffset === 0} onClick={() => fetchSystemChunks(Math.max(0, systemOffset - 100))}>Trước</button>
                        <button disabled={systemOffset + 100 >= systemTotal} onClick={() => fetchSystemChunks(systemOffset + 100)}>Tiếp</button>
                      </div>
                    </div>
                    <div className="system-table-wrapper scrollable">
                      <table className="system-table">
                        <thead>
                          <tr>
                            <th>ID</th>
                            <th style={{ width: '60%' }}>Nội dung (Preview)</th>
                            <th>Nguồn</th>
                          </tr>
                        </thead>
                        <tbody>
                          {systemChunks.map(c => (
                            <tr key={c.id}>
                              <td className="text-dim">#{c.id}</td>
                              <td className="chunk-preview">{c.text.substring(0, 200)}...</td>
                              <td className="source-tag-small">{c.source}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
          
          {(activeTab === 'chat' || activeTab === 'simulate') && (
            <div className={`resource-sidebar ${showRightSidebar ? 'open' : ''}`}>
              <div className="sidebar-header">
                <Activity size={14} />
                <h3>TELEMETRY</h3>
              </div>
              
              <div className="telemetry-card">
                <div className="t-label">CPU USAGE</div>
                <div className="t-value">{telemetry.cpu}%</div>
                <div className="t-bar-bg">
                  <div className="t-bar-fill cpu" style={{ width: `${telemetry.cpu}%` }}></div>
                </div>
              </div>

              <div className="telemetry-card">
                <div className="t-label">WORKING SET (RAM)</div>
                <div className="t-value">{telemetry.ram} MB</div>
                <div className="t-bar-bg">
                  <div className="t-bar-fill ram" style={{ width: `${Math.min(100, (telemetry.ram / 8192) * 100)}%` }}></div>
                </div>
              </div>

              <div className="telemetry-card">
                <div className="t-label">UPTIME</div>
                <div className="t-value">{Math.floor(telemetry.uptime / 60)}m {Math.round(telemetry.uptime % 60)}s</div>
              </div>

              <div className="telemetry-info">
                <Shield size={12} />
                <span>TurboQuant v4.0 Active</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Modal xem chi tiết tài liệu */}
      {selectedDocChunks && (
        <div className="modal-overlay" onClick={() => setSelectedDocChunks(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Chi tiết: {viewingDocName}</h3>
              <button className="close-btn" onClick={() => setSelectedDocChunks(null)}>&times;</button>
            </div>
            <div className="modal-body">
              <div className="chunks-list">
                {selectedDocChunks.map((c, ci) => (
                  <div key={ci} className="chunk-card">
                    <div className="chunk-meta">Đoạn {ci + 1} - Trang {c.page || '?' } (ID: {c.id})</div>
                    <div className="chunk-text">{c.text}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
