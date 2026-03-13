import React, { useState } from "react";
import "./Dashboard.css";

const AGENTS = [
    {
        name: "Summarizer",
        key: "summarizer",
        color: "#10b981",
        icon: "📝",
        desc: "Instantly condense long documents into clear, bite-sized summaries.",
    },
    {
        name: "MCQ Generator",
        key: "mcq_generator",
        color: "#f59e0b",
        icon: "❓",
        desc: "Test your knowledge with auto-generated custom quiz questions.",
    },
    {
        name: "Concept Explainer",
        key: "concept_explainer",
        color: "#8b5cf6",
        icon: "💡",
        desc: "Break down complex topics into simple, intuitive explanations.",
    },
    {
        name: "Exam Prep",
        key: "exam_prep_agent",
        color: "#ef4444",
        icon: "🎯",
        desc: "Prepare for your exams with targeted practice and study plans.",
    },
    ];

    export default function Dashboard({ onAgentSelect }) {
    const [step, setStep] = useState(1);
    const [isTransitioning, setIsTransitioning] = useState(false);

    const handleGetStarted = () => {
        setIsTransitioning(true);
        setTimeout(() => {
        setStep(2);
        setIsTransitioning(false);
        }, 400); // Wait for fade out
    };

    const handleAgentClick = (agent) => {
        // Add click animation before navigating
        onAgentSelect(agent.key);
    };

    return (
        <div className="dashboard-container">
        {/* Animated background blobs */}
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
        <div className="blob blob-3"></div>

        <div className={`dashboard-content ${isTransitioning ? "fade-out" : "fade-in"}`}>
            {step === 1 && (
            <div className="step-1">
                <div className="hero-section">
                <div className="hero-icon-container">
                    <span className="hero-icon">🧠</span>
                </div>
                <h1 className="hero-title">
                    Multi-Agentic <span className="text-gradient">RAG</span>
                </h1>
                <p className="hero-subtitle">
                    Your intelligent study companion. Tap into the power of specialized AI agents designed to help you learn faster, smarter, and better.
                </p>
                
                <button className="btn-get-started" onClick={handleGetStarted}>
                    Get Started
                    <span className="btn-arrow">→</span>
                </button>
                </div>
            </div>
            )}

            {step === 2 && (
            <div className="step-2">
                <div className="header-animate">
                <h2 className="step-2-title">Choose Your Agent</h2>
                <p className="step-2-subtitle">Select an expert to begin your study session.</p>
                </div>

                <div className="agents-grid">
                {AGENTS.map((agent, index) => (
                    <div 
                    key={agent.key} 
                    className="agent-card"
                    style={{ animationDelay: `${index * 0.1}s` }}
                    onClick={() => handleAgentClick(agent)}
                    >
                    <div 
                        className="agent-card-icon"
                        style={{ backgroundColor: `${agent.color}15`, color: agent.color, borderColor: `${agent.color}40` }}
                    >
                        {agent.icon}
                    </div>
                    <h3 className="agent-card-title">{agent.name}</h3>
                    <p className="agent-card-desc">{agent.desc}</p>
                    
                    <div className="agent-card-action" style={{ color: agent.color }}>
                        Start Session <span>→</span>
                    </div>
                    </div>
                ))}
                </div>
            </div>
            )}
        </div>
        </div>
    );
}
