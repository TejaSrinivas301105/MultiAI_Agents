import React, { useRef, useEffect, useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { fetchTTSAudio } from "../api";

const AGENT_LABELS = {
  summarizer: "Summarizer",
  mcq_generator: "MCQ Generator",
  notes_maker: "Notes Maker",
  exam_prep_agent: "Exam Prep",
  concept_explainer: "Concept Explainer",
  search_agent: "Search Agent",
  chat_agent: "Chat Agent",
};

function SourceLinks({ sources }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="sources-section">
      <h4>🔗 Sources</h4>
      {sources.map((src, i) => (
        <a
          key={i}
          className="source-link"
          href={src.link}
          target="_blank"
          rel="noopener noreferrer"
        >
          <span className="link-icon">🌐</span>
          <span>{src.title || src.link}</span>
        </a>
      ))}
    </div>
  );
}

function AudioButton({ text }) {
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const audioRef = useRef(null);

  const handlePlay = useCallback(async () => {
    if (playing && audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setPlaying(false);
      return;
    }

    setLoading(true);
    try {
      const audioUrl = await fetchTTSAudio(text.substring(0, 5000));
      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      audio.onended = () => setPlaying(false);
      audio.play();
      setPlaying(true);
    } catch (err) {
      // Fallback to Web Speech API
      if ("speechSynthesis" in window) {
        const utterance = new SpeechSynthesisUtterance(text.substring(0, 3000));
        utterance.rate = 1;
        utterance.onend = () => setPlaying(false);
        window.speechSynthesis.speak(utterance);
        setPlaying(true);
      }
    } finally {
      setLoading(false);
    }
  }, [text, playing]);

  return (
    <button
      className={`audio-btn ${playing ? "playing" : ""}`}
      onClick={handlePlay}
      disabled={loading}
    >
      {loading ? "⏳" : playing ? "⏹️" : "🔊"}{" "}
      {loading ? "Loading..." : playing ? "Stop Audio" : "Play Audio"}
    </button>
  );
}

function MessageContent({ msg }) {
  const agentKey = msg.agent || "";
  const subAgentKey = msg.sub_agent && msg.sub_agent !== "none" ? msg.sub_agent : null;

  return (
    <div>
      {msg.role === "assistant" && agentKey && (
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
          <span className={`agent-badge ${agentKey}`}>
            🤖 {AGENT_LABELS[agentKey] || agentKey}
          </span>
          {subAgentKey && (
            <span className={`agent-badge ${subAgentKey}`}>
              ⚡ Sub: {AGENT_LABELS[subAgentKey] || subAgentKey}
            </span>
          )}
        </div>
      )}

      <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>

      {msg.role === "assistant" && <SourceLinks sources={msg.sources} />}

      {msg.role === "assistant" && msg.text && (
        <AudioButton text={msg.text} />
      )}
    </div>
  );
}

export default function ChatArea({ messages, isLoading }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  if (messages.length === 0 && !isLoading) {
    return (
      <div className="messages-area">
        <div className="welcome-screen">
          <span className="welcome-icon">🧠</span>
          <h2>Welcome to Multi-Agentic RAG</h2>
          <p>
            Upload your study PDFs and ask anything — summaries, MCQs, notes,
            exam prep, concept explanations, or search the web. The AI
            automatically selects the best agent for your query.
          </p>
          <div className="feature-chips">
            <div className="feature-chip">📝 Summarizer</div>
            <div className="feature-chip">❓ MCQ Generator</div>
            <div className="feature-chip">📒 Notes Maker</div>
            <div className="feature-chip">🎯 Exam Prep</div>
            <div className="feature-chip">💡 Concept Explainer</div>
            <div className="feature-chip">🔍 Web Search</div>
            <div className="feature-chip">💬 Chat</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="messages-area">
      {messages.map((msg, i) => (
        <div key={i} className={`message ${msg.role}`}>
          <div className="message-avatar">
            {msg.role === "user" ? "👤" : "🤖"}
          </div>
          <div className="message-bubble">
            <MessageContent msg={msg} />
          </div>
        </div>
      ))}

      {isLoading && (
        <div className="message assistant">
          <div className="message-avatar">🤖</div>
          <div className="message-bubble">
            <div className="loading-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
