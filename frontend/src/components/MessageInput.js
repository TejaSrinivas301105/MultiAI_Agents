import React, { useState, useRef, useEffect } from "react";

export default function MessageInput({ onSend, disabled }) {
  const [text, setText] = useState("");
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 120) + "px";
    }
  }, [text]);

  const handleSubmit = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="input-area">
      <div className="input-wrapper">
        <div className="input-field-wrapper">
          <textarea
            ref={textareaRef}
            className="input-field"
            placeholder="Ask anything — summarize, generate MCQs, make notes, explain a concept..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={disabled}
          />
        </div>
        <button
          className="send-btn"
          onClick={handleSubmit}
          disabled={disabled || !text.trim()}
          title="Send message"
        >
          ➤
        </button>
      </div>
    </div>
  );
}
