import { useCallback, useEffect, useState } from "react";
import { Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { fetchModels, sendChat, sendQuery } from "./api";
import ChatArea from "./Components/ChartArea";
import Dashboard from "./Components/Dashboard";
import MessageInput from "./Components/MessageInput";
import Sidebar from "./Components/Sidebar";
import './App.css';

function Toast({ toast }) {
  if (!toast) return null;
  return <div className={`status-toast ${toast.type}`}>{toast.message}</div>;
}

function MainApp() {
  const location = useLocation();
  const initialAgent = location.state?.agentKey || "";

  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("gpt-oss-120b");
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeAgent, setActiveAgent] = useState(initialAgent);
  const [activeSubAgent, setActiveSubAgent] = useState("none");
  const [toast, setToast] = useState(null);

  const showToast = useCallback((type, message) => {
    setToast({ type, message });
    setTimeout(() => setToast(null), 3500);
  }, []);

  useEffect(() => {
    fetchModels()
      .then((data) => {
        setModels(data.models || []);
        if (data.models && data.models.length > 0) {
          setSelectedModel(data.models[0].key);
        }
      })
      .catch(() => {
        setModels([
          { key: "gpt-oss-120b", display_name: "GPT-OSS 120B" },
          { key: "gpt-oss-20b", display_name: "GPT-OSS 20B" },
          { key: "kimi-k2", display_name: "Kimi K2" },
        ]);
      });
  }, []);

  useEffect(() => {
    if (initialAgent && messages.length === 0) {
      setIsLoading(true);
      setTimeout(() => {
        let welcomeText = "";
        switch (initialAgent) {
          case "summarizer":
            welcomeText = "Hello! I am the **Summarizer** agent. Please upload your documents or type your text below, and I will condense it into clear, bite-sized summaries. What would you like to summarize today?";
            break;
          case "mcq_generator":
            welcomeText = "Welcome! I can generate custom multiple-choice questions from your study materials to help test your knowledge. How many questions would you like to practice?";
            break;
          case "concept_explainer":
            welcomeText = "Hi there! I specialize in breaking down complex topics into simple, intuitive explanations. What concept are you struggling with or curious about?";
            break;
          case "exam_prep_agent":
            welcomeText = "Ready to ace your exams? I am the **Exam Prep** agent. Share your syllabus topics or materials, and let's start a targeted practice session. What should we cover first?";
            break;
          default:
            welcomeText = "Hello! I am ready to help you study. How can I assist you today?";
            break;
        }

        setMessages([
          {
            role: "assistant",
            text: welcomeText,
            agent: initialAgent,
            sub_agent: "none",
            sources: [],
          },
        ]);
        setIsLoading(false);
      }, 600);
    }
  }, [initialAgent, messages.length]);

  const handleSend = useCallback(
    async (text) => {
      const userMsg = { role: "user", text };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);

      try {
        let result;
        const hasFiles = uploadedFiles.length > 0;

        // Add agent hint to query if an agent was pre-selected
        let queryText = text;
        if (initialAgent && initialAgent !== "chat_agent") {
          const agentHints = {
            "summarizer": "Summarize: ",
            "mcq_generator": "Generate MCQs: ",
            "notes_maker": "Make notes: ",
            "exam_prep_agent": "Prepare exam questions: ",
            "concept_explainer": "Explain: ",
          };
          const hint = agentHints[initialAgent];
          if (hint && !text.toLowerCase().startsWith(hint.toLowerCase())) {
            queryText = hint + text;
          }
        }

        if (hasFiles) {
          result = await sendQuery(queryText, selectedModel);
        } else {
          result = await sendChat(queryText, selectedModel);
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
    [selectedModel, uploadedFiles, showToast, initialAgent]
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

export default function App() {
  const navigate = useNavigate();

  const handleAgentStart = (agentKey) => {
    navigate("/chat", { state: { agentKey } });
  };

  return (
    <Routes>
      <Route path="/" element={<Dashboard onAgentSelect={handleAgentStart} />} />
      <Route path="/chat" element={<MainApp />} />
    </Routes>
  );
}
