import React, { useState, useEffect, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import MessageInput from "./components/MessageInput";
import { fetchModels, sendQuery, sendChat } from "./api";

function Toast({ toast }) {
  if (!toast) return null;
  return <div className={`status-toast ${toast.type}`}>{toast.message}</div>;
}

export default function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("gpt-oss-120b");
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeAgent, setActiveAgent] = useState("");
  const [activeSubAgent, setActiveSubAgent] = useState("none");
  const [toast, setToast] = useState(null);

  const showToast = useCallback((type, message) => {
    setToast({ type, message });
    setTimeout(() => setToast(null), 3500);
  }, []);

  // Fetch available models on mount
  useEffect(() => {
    fetchModels()
      .then((data) => {
        setModels(data.models || []);
        if (data.models && data.models.length > 0) {
          setSelectedModel(data.models[0].key);
        }
      })
      .catch(() => {
        // Fallback models if backend is not ready
        setModels([
          { key: "gpt-oss-120b", display_name: "GPT-OSS 120B" },
          { key: "gpt-oss-20b", display_name: "GPT-OSS 20B" },
          { key: "kimi-k2", display_name: "Kimi K2" },
        ]);
      });
  }, []);

  const handleSend = useCallback(
    async (text) => {
      // Add user message
      const userMsg = { role: "user", text };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);
      setActiveAgent("");
      setActiveSubAgent("none");

      try {
        let result;
        const hasFiles = uploadedFiles.length > 0;

        if (hasFiles) {
          // Use the RAG pipeline
          result = await sendQuery(text, selectedModel);
        } else {
          // No files uploaded — use chat agent
          result = await sendChat(text, selectedModel);
        }

        const assistantMsg = {
          role: "assistant",
          text: result.response || "No response generated.",
          agent: result.agent || "unknown",
          sub_agent: result.sub_agent || "none",
          sources: result.sources || [],
        };

        setMessages((prev) => [...prev, assistantMsg]);
        setActiveAgent(result.agent || "");
        setActiveSubAgent(result.sub_agent || "none");
      } catch (err) {
        const errMsg = {
          role: "assistant",
          text: `⚠️ **Error:** ${err.message}`,
          agent: "error",
          sub_agent: "none",
          sources: [],
        };
        setMessages((prev) => [...prev, errMsg]);
        showToast("error", err.message);
      } finally {
        setIsLoading(false);
      }
    },
    [selectedModel, uploadedFiles, showToast]
  );

  return (
    <div className="app-container">
      <Toast toast={toast} />

      <Sidebar
        models={models}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
        uploadedFiles={uploadedFiles}
        setUploadedFiles={setUploadedFiles}
        activeAgent={activeAgent}
        activeSubAgent={activeSubAgent}
        onToast={showToast}
      />

      <div className="main-content">
        <div className="chat-header">
          <h2>💬 Study Chat</h2>
          <div className="agent-badge-group">
            {activeAgent && activeAgent !== "error" && (
              <span className={`agent-badge ${activeAgent}`}>
                🤖 {activeAgent.replace(/_/g, " ")}
              </span>
            )}
            {activeSubAgent && activeSubAgent !== "none" && (
              <span className={`agent-badge ${activeSubAgent}`}>
                ⚡ Sub: {activeSubAgent.replace(/_/g, " ")}
              </span>
            )}
            {!activeAgent && (
              <span style={{ fontSize: "0.78rem", color: "#64748b" }}>
                {uploadedFiles.length > 0
                  ? "Ready — ask a question"
                  : "Upload PDFs or chat freely"}
              </span>
            )}
          </div>
        </div>

        <ChatArea messages={messages} isLoading={isLoading} />

        <MessageInput onSend={handleSend} disabled={isLoading} />
      </div>
    </div>
  );
}
