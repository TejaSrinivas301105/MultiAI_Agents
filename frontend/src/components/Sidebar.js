import React, { useRef, useState } from "react";
import { uploadFiles } from "../api";

const AGENT_LIST = [
  { name: "Summarizer", key: "summarizer", color: "#10b981", icon: "📝" },
  { name: "MCQ Generator", key: "mcq_generator", color: "#f59e0b", icon: "❓" },
  { name: "Notes Maker", key: "notes_maker", color: "#3b82f6", icon: "📒" },
  { name: "Exam Prep", key: "exam_prep_agent", color: "#ef4444", icon: "🎯" },
  { name: "Concept Explainer", key: "concept_explainer", color: "#8b5cf6", icon: "💡" },
  { name: "Search Agent", key: "search_agent", color: "#ec4899", icon: "🔍" },
  { name: "Chat Agent", key: "chat_agent", color: "#06b6d4", icon: "💬" },
];

export default function Sidebar({
  models,
  selectedModel,
  onModelChange,
  uploadedFiles,
  setUploadedFiles,
  activeAgent,
  activeSubAgent,
  onToast,
}) {
  const fileInputRef = useRef(null);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files).filter((f) =>
      f.name.toLowerCase().endsWith(".pdf")
    );
    setPendingFiles((prev) => [...prev, ...files]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files).filter((f) =>
      f.name.toLowerCase().endsWith(".pdf")
    );
    setPendingFiles((prev) => [...prev, ...files]);
  };

  const removePending = (idx) => {
    setPendingFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleUpload = async () => {
    if (pendingFiles.length === 0) return;
    setUploading(true);
    try {
      const result = await uploadFiles(pendingFiles);
      setUploadedFiles((prev) => [...prev, ...result.files]);
      setPendingFiles([]);
      onToast("success", `✅ ${result.message}`);
    } catch (err) {
      onToast("error", `❌ ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="logo-icon">🧠</div>
          <div>
            <h1>Multi-Agentic RAG</h1>
            <p>College Study Assistant</p>
          </div>
        </div>
      </div>

      <div className="sidebar-content">
        {/* Model Selector */}
        <div>
          <div className="section-label">Language Model</div>
          <select
            className="model-select"
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
          >
            {models.map((m) => (
              <option key={m.key} value={m.key}>
                {m.display_name}
              </option>
            ))}
          </select>
        </div>

        {/* Upload Area */}
        <div>
          <div className="section-label">Upload Study Material</div>
          <div
            className={`upload-area ${dragging ? "dragging" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <span className="upload-icon">📄</span>
            <p>Drop PDFs here or click to browse</p>
            <span className="upload-hint">Supports multiple PDF files</span>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              multiple
              onChange={handleFileSelect}
            />
          </div>

          {/* Pending files */}
          {pendingFiles.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <div className="file-list">
                {pendingFiles.map((f, i) => (
                  <div className="file-item" key={i}>
                    <span className="file-icon">📕</span>
                    <span className="file-name">{f.name}</span>
                    <button className="file-remove" onClick={() => removePending(i)}>
                      ✕
                    </button>
                  </div>
                ))}
              </div>
              <button
                className="upload-btn"
                style={{ marginTop: 10 }}
                onClick={handleUpload}
                disabled={uploading}
              >
                {uploading ? "Uploading..." : `Upload ${pendingFiles.length} file(s)`}
              </button>
            </div>
          )}

          {/* Uploaded files */}
          {uploadedFiles.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <div className="section-label">Uploaded Files</div>
              <div className="file-list">
                {uploadedFiles.map((name, i) => (
                  <div className="file-item" key={i}>
                    <span className="file-icon">📗</span>
                    <span className="file-name">{name}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Agent Info */}
        <div className="agent-info-card">
          <h3>🤖 Available Agents</h3>
          <div className="agent-list">
            {AGENT_LIST.map((a) => (
              <div
                className={`agent-dot ${
                  activeAgent === a.key || activeSubAgent === a.key ? "active" : ""
                }`}
                key={a.key}
              >
                <span
                  className="dot"
                  style={{
                    backgroundColor: a.color,
                    color: a.color,
                  }}
                ></span>
                <span>
                  {a.icon} {a.name}
                  {activeAgent === a.key && (
                    <span
                      style={{
                        fontSize: "0.6rem",
                        marginLeft: 6,
                        color: a.color,
                        fontWeight: 700,
                      }}
                    >
                      ACTIVE
                    </span>
                  )}
                  {activeSubAgent === a.key && activeSubAgent !== "none" && (
                    <span
                      style={{
                        fontSize: "0.6rem",
                        marginLeft: 6,
                        color: a.color,
                        fontWeight: 700,
                      }}
                    >
                      SUB-AGENT
                    </span>
                  )}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
