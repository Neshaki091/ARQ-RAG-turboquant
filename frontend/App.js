import React, { useState } from 'react';
import { Send, Upload, FileText, Cpu, Search } from 'lucide-react';
import axios from 'axios';

const API_BASE = "http://localhost:8000";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  const handleSend = async () => {
    if (!input) return;
    
    const newMsg = { role: 'user', content: input };
    setMessages([...messages, newMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post(`${API_BASE}/chat`, { message: input });
      setMessages(prev => [...prev, { 
        role: 'bot', 
        content: res.data.answer,
        strategy: res.data.strategy,
        reason: res.data.reason,
        sources: res.data.sources
      }]);
    } catch (err) {
      console.error(err);
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
      await axios.post(`${API_BASE}/upload`, formData);
      alert("Tải lên và xử lý thành công!");
    } catch (err) {
      alert("Lỗi khi tải lên!");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="dashboard">
      <div className="sidebar">
        <h2 style={{display:'flex', alignItems:'center', gap:'10px'}}>
          <Cpu color="#6366f1" /> ARQ-RAG
        </h2>
        <p style={{color: '#94a3b8', fontSize: '0.9rem'}}>TurboQuant Powered Vector Search</p>
        
        <div style={{marginTop: '3rem'}}>
          <label className="upload-btn" style={{
            display: 'flex', 
            alignItems: 'center', 
            gap: '10px',
            padding: '1rem',
            background: 'rgba(255,255,255,0.05)',
            borderRadius: '0.5rem',
            cursor: 'pointer'
          }}>
            <Upload size={20} />
            {uploading ? "Đang xử lý..." : "Tải lên PDF"}
            <input type="file" hidden onChange={handleUpload} accept=".pdf" />
          </label>
        </div>
      </div>

      <div className="main-content">
        <div className="chat-container">
          {messages.length === 0 && (
            <div style={{textAlign:'center', marginTop:'5rem', color:'#94a3b8'}}>
              <FileText size={48} style={{marginBottom:'1rem'}} />
              <h3>Chào mừng bạn đến với Demo ARQ-RAG</h3>
              <p>Hãy tải lên tài liệu và đặt câu hỏi để bắt đầu.</p>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}`}>
              {m.strategy && (
                <div className="strategy-badge">
                  <Search size={10} style={{marginRight:'5px'}} />
                  Strategy: {m.strategy.toUpperCase()}
                </div>
              )}
              <div>{m.content}</div>
              {m.sources && (
                <div style={{fontSize:'0.7rem', marginTop:'10px', opacity:0.6}}>
                  Nguồn: {m.sources.join(", ")}
                </div>
              )}
            </div>
          ))}
          {loading && <div className="message bot">Đang suy nghĩ...</div>}
        </div>

        <div className="input-area">
          <input 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Đặt câu hỏi về tài liệu của bạn..." 
          />
          <button onClick={handleSend}>
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
}
