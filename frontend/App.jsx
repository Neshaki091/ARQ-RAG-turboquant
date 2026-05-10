import React, { useState, useEffect } from 'react';
import { Send, Upload, FileText, Cpu, Search, Trash2, Book, Loader2, Sparkles, MessageSquare, Zap, Activity, Shield, LogOut, User, Lock, Layers } from 'lucide-react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';

const API_BASE = "http://localhost:8000";

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
  const [simCount, setSimCount] = useState(8);
  const [simResults, setSimResults] = useState({});
  const [isSimulating, setIsSimulating] = useState(false);
  
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
      fetchDocuments();
    }
  }, [token]);

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

  const fetchDocuments = async () => {
    try {
      const res = await axios.get(`${API_BASE}/documents`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setDocuments(res.data.documents || []);
    } catch (err) {
      console.error("Failed to fetch documents", err);
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
        body: JSON.stringify({ message: input, mode: mode, scope: scope })
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
      fetchDocuments();
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
      fetchDocuments();
    } catch (err) {
      alert("Lỗi khi xóa tài liệu!");
    }
  };

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
      <div className="sidebar">
        <div className="brand">
          <div className="logo-container">
            <Cpu className="logo-icon" />
          </div>
          <div className="brand-text">
            <h1>ARQ-RAG</h1>
            <span>{user?.username}</span>
          </div>
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
              <span className="upload-subtitle">Gắn với ID: {user?.id}</span>
            </div>
            <input type="file" hidden onChange={handleUpload} accept=".pdf" disabled={uploading} />
          </label>
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
          <div className="header-info">
            <MessageSquare size={20} className="text-primary" />
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
          <div className="header-actions">
            <div className="engine-badge">
              <Sparkles size={14} />
              <span>{mode.toUpperCase()} + {scope.toUpperCase()}</span>
            </div>
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
                <div className="sim-dashboard">
                   <div className="welcome-screen">
                      <Zap size={48} className="art-icon" />
                      <h3>Chế độ Mô phỏng</h3>
                      <p>Chạy các kịch bản kiểm thử hiệu năng và độ chính xác của ARQ-RAG.</p>
                   </div>
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
            <div className="resource-sidebar">
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
