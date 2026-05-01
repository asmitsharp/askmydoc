"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { UploadCloud, Plus, Send, FileText, Zap, Clock, AlertCircle, BookOpen, ChevronRight } from "lucide-react";

/* ─────────────────────────────────────────────
   Types
───────────────────────────────────────────── */
type Citation = {
  source: string;
  chunk_index: number;
  score: number;
  content: string;
};

type Message = {
  id: string;
  role: "user" | "bot";
  content: string;
  citations?: Citation[];
  latencyMs?: number;
  isError?: boolean;
};

type UploadState = {
  msg: string;
  type: "idle" | "uploading" | "success" | "error";
  filename?: string;
};

/* ─────────────────────────────────────────────
   Welcome Empty State
───────────────────────────────────────────── */
function EmptyState({ onUpload }: { onUpload: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-10 px-8 animate-fade-in">
      {/* Cinematic gradient blob */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 60% 50% at 50% 30%, rgba(113,76,182,0.07) 0%, transparent 70%)",
        }}
      />

      <div className="relative flex flex-col items-center gap-6 text-center max-w-lg">
        <div
          className="w-14 h-14 rounded-[16px] flex items-center justify-center"
          style={{ background: "rgba(113,76,182,0.10)" }}
        >
          <BookOpen size={26} style={{ color: "#714cb6" }} />
        </div>

        <div className="flex flex-col gap-3">
          <h2
            className="font-semibold"
            style={{
              fontSize: "28px",
              lineHeight: 1.2,
              letterSpacing: "-0.62px",
              color: "#292827",
              fontWeight: 600,
            }}
          >
            Ask your documents anything
          </h2>
          <p style={{ fontSize: "16px", color: "#666666", lineHeight: 1.6, fontWeight: 460 }}>
            Hybrid vector + BM25 retrieval with Cohere reranking. Upload a PDF,
            Markdown, or text file — then ask questions grounded in your content.
          </p>
        </div>

        <div className="flex flex-wrap gap-3 justify-center mt-2">
          {[
            "What are the key findings?",
            "Summarise section 2",
            "What does v(T+1) mean?",
          ].map((q) => (
            <span
              key={q}
              className="px-4 py-2 rounded-full border text-sm cursor-default select-none transition-colors"
              style={{
                borderColor: "#e3e3e2",
                color: "#666666",
                background: "#ffffff",
                fontSize: "14px",
                fontWeight: 460,
              }}
            >
              {q}
            </span>
          ))}
        </div>

        <button
          onClick={onUpload}
          className="mt-2 flex items-center gap-2 px-6 py-3 rounded-[8px] font-medium transition-all duration-200 hover:-translate-y-px"
          style={{
            background: "#292827",
            color: "#ffffff",
            fontSize: "15px",
            fontWeight: 500,
          }}
        >
          <UploadCloud size={16} />
          Upload your first document
        </button>
      </div>

      {/* Stats strip */}
      <div
        className="relative flex items-center gap-8 px-8 py-5 rounded-[16px] border"
        style={{ background: "#ffffff", borderColor: "#e3e3e2" }}
      >
        {[
          { icon: <Zap size={15} />, label: "Hybrid retrieval" },
          { icon: <FileText size={15} />, label: "PDF · MD · TXT" },
          { icon: <Clock size={15} />, label: "~2 s latency" },
        ].map(({ icon, label }, i) => (
          <div key={label} className="flex items-center gap-2">
            {i > 0 && (
              <div
                className="absolute"
                style={{
                  left: `${i * 33.3}%`,
                  top: "50%",
                  transform: "translateY(-50%)",
                  width: "1px",
                  height: "24px",
                  background: "#e3e3e2",
                }}
              />
            )}
            <span style={{ color: "#714cb6" }}>{icon}</span>
            <span style={{ fontSize: "13px", color: "#666666", fontWeight: 460 }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   Typing dots
───────────────────────────────────────────── */
function TypingIndicator() {
  return (
    <div className="flex gap-2 max-w-[720px] w-full mx-auto animate-fade-in px-4 md:px-0">
      <div
        className="flex items-center gap-1.5 px-5 py-4 rounded-[16px] rounded-tl-[4px]"
        style={{ background: "#ffffff", border: "1px solid #e3e3e2" }}
      >
        {[0, 150, 300].map((delay) => (
          <span
            key={delay}
            className="block w-1.5 h-1.5 rounded-full"
            style={{
              background: "#714cb6",
              animation: `bounce-dots 1.2s ease-in-out ${delay}ms infinite`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   Citation card
───────────────────────────────────────────── */
function CitationCard({ c, idx }: { c: Citation; idx: number }) {
  const [open, setOpen] = useState(false);

  return (
    <button
      onClick={() => setOpen((v) => !v)}
      className="w-full text-left rounded-[12px] border transition-all duration-200 hover:border-[#714cb6]/40"
      style={{
        background: open ? "rgba(113,76,182,0.04)" : "#f2f0eb",
        borderColor: open ? "rgba(113,76,182,0.35)" : "#e3e3e2",
        padding: "10px 14px",
      }}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-semibold"
            style={{ background: "rgba(113,76,182,0.12)", color: "#714cb6" }}
          >
            {idx + 1}
          </span>
          <span
            className="truncate font-medium"
            style={{ fontSize: "13px", color: "#292827", fontWeight: 500 }}
          >
            {c.source}
          </span>
          <span
            className="shrink-0 px-2 py-0.5 rounded-full text-[11px]"
            style={{ background: "#e3e3e2", color: "#666666", fontWeight: 460 }}
          >
            {(c.score * 100).toFixed(1)}%
          </span>
        </div>
        <ChevronRight
          size={14}
          style={{
            color: "#666666",
            flexShrink: 0,
            transform: open ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 0.2s ease",
          }}
        />
      </div>
      {open && (
        <p
          className="mt-3 leading-relaxed"
          style={{ fontSize: "13px", color: "#666666", fontWeight: 400, lineHeight: 1.65 }}
        >
          {c.content}
        </p>
      )}
    </button>
  );
}

/* ─────────────────────────────────────────────
   Chat bubble
───────────────────────────────────────────── */
function ChatMessage({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";

  return (
    <div
      className={`flex flex-col gap-3 max-w-[720px] w-full mx-auto animate-fade-slide-up ${
        isUser ? "items-end" : "items-start"
      }`}
    >
      {/* Bubble */}
      <div
        className="leading-relaxed"
        style={{
          background: isUser ? "#292827" : "#ffffff",
          color: isUser ? "#ffffff" : "#292827",
          border: isUser ? "none" : "1px solid #e3e3e2",
          borderRadius: isUser ? "16px 16px 4px 16px" : "4px 16px 16px 16px",
          padding: "14px 20px",
          fontSize: "15px",
          fontWeight: 460,
          lineHeight: 1.65,
          maxWidth: isUser ? "85%" : "100%",
          boxShadow: isUser ? "none" : "0 1px 3px rgba(41,40,39,0.05)",
          wordBreak: "break-word",
        }}
      >
        {msg.isError && (
          <div className="flex items-center gap-2 mb-2" style={{ color: "#b91c1c" }}>
            <AlertCircle size={15} />
            <span className="text-sm font-medium">Error</span>
          </div>
        )}
        <div
          className="prose prose-sm max-w-none"
          style={
            {
              "--tw-prose-body": isUser ? "#e5e5e5" : "#292827",
              "--tw-prose-headings": isUser ? "#ffffff" : "#292827",
              "--tw-prose-bold": isUser ? "#ffffff" : "#292827",
              "--tw-prose-links": "#714cb6",
              "--tw-prose-code": isUser ? "#d4c7ff" : "#4e242c",
              "--tw-prose-pre-bg": "#1c1917",
            } as React.CSSProperties
          }
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
        </div>
      </div>

      {/* Citations */}
      {!isUser && msg.citations && msg.citations.length > 0 && (
        <div className="w-full flex flex-col gap-2">
          <span
            style={{ fontSize: "11px", color: "#666666", fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase" }}
          >
            Sources · {msg.citations.length}
          </span>
          {msg.citations.map((c, idx) => (
            <CitationCard key={idx} c={c} idx={idx} />
          ))}
          {msg.latencyMs != null && (
            <span style={{ fontSize: "11px", color: "#9ca3af", textAlign: "right" }}>
              {msg.latencyMs.toLocaleString()} ms
            </span>
          )}
        </div>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────────
   Main Page
───────────────────────────────────────────── */
export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [upload, setUpload] = useState<UploadState>({ type: "idle", msg: "" });
  const [isDragging, setIsDragging] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  /* ── resize textarea ───────────────────── */
  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, []);

  /* ── new session ──────────────────────── */
  const handleNewSession = () => {
    setMessages([]);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  /* ── upload ───────────────────────────── */
  const doUpload = async (file: File) => {
    setUpload({ type: "uploading", msg: `Uploading ${file.name}…`, filename: file.name });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/ingest", { method: "POST", body: formData });
      const data = await res.json();

      if (res.ok) {
        setUpload({ type: "success", msg: `${file.name} queued`, filename: file.name });
        setTimeout(() => setUpload({ type: "idle", msg: "" }), 6000);
      } else {
        setUpload({ type: "error", msg: data.error ?? "Upload failed" });
      }
    } catch (e: any) {
      setUpload({ type: "error", msg: e.message });
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) doUpload(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) doUpload(file);
  };

  /* ── query ────────────────────────────── */
  const handleSend = async () => {
    const text = input.trim();
    if (!text || isTyping) return;

    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    const userMsg: Message = { id: `u-${Date.now()}`, role: "user", content: text };
    setMessages((p) => [...p, userMsg]);
    setIsTyping(true);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });
      const data = await res.json();

      if (res.ok) {
        setMessages((p) => [
          ...p,
          {
            id: `b-${Date.now()}`,
            role: "bot",
            content: data.answer,
            citations: data.citations,
            latencyMs: data.latency_ms,
          },
        ]);
      } else {
        setMessages((p) => [
          ...p,
          {
            id: `b-${Date.now()}`,
            role: "bot",
            content: `${data.error ?? "Something went wrong"}\n\n*Failure: ${data.failure_reason ?? "unknown"}*`,
            isError: true,
          },
        ]);
      }
    } catch (e: any) {
      setMessages((p) => [
        ...p,
        { id: `b-${Date.now()}`, role: "bot", content: e.message, isError: true },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  /* ─────────────────────────────────────────────
     Render
  ───────────────────────────────────────────── */
  return (
    <div className="flex h-full" style={{ background: "#f2f0eb" }}>
      {/* ── Sidebar ─────────────────────────────── */}
      <aside
        className="flex flex-col shrink-0 w-[260px]"
        style={{
          background: "#ffffff",
          borderRight: "1px solid #e3e3e2",
        }}
      >
        {/* Brand */}
        <div className="px-6 py-5 flex items-center gap-2.5" style={{ borderBottom: "1px solid #e3e3e2" }}>
          <div
            className="w-7 h-7 rounded-[6px] flex items-center justify-center shrink-0"
            style={{ background: "#421d24" }}
          >
            <BookOpen size={13} color="#ffffff" />
          </div>
          <span
            style={{
              fontSize: "16px",
              fontWeight: 600,
              color: "#292827",
              letterSpacing: "-0.31px",
            }}
          >
            AskMyDocs
          </span>
        </div>

        {/* New session */}
        <div className="px-4 py-4">
          <button
            onClick={handleNewSession}
            className="w-full flex items-center gap-2.5 px-4 py-2.5 rounded-[8px] transition-all duration-150 hover:bg-[#f2f0eb] active:scale-[0.98]"
            style={{
              fontSize: "14px",
              fontWeight: 500,
              color: "#292827",
              border: "1px solid #e3e3e2",
              background: "transparent",
            }}
          >
            <Plus size={15} />
            New session
          </button>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Upload zone */}
        <div className="px-4 pb-6 flex flex-col gap-3">
          <span
            style={{
              fontSize: "11px",
              fontWeight: 600,
              color: "#666666",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}
          >
            Knowledge Base
          </span>

          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="relative flex flex-col items-center justify-center gap-2 rounded-[12px] border-2 border-dashed cursor-pointer transition-all duration-200 py-6"
            style={{
              borderColor: isDragging ? "#714cb6" : "#dcd7d3",
              background: isDragging ? "rgba(113,76,182,0.04)" : "#f2f0eb",
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              className="absolute inset-0 opacity-0 cursor-pointer"
              accept=".pdf,.md,.markdown,.txt"
              onChange={handleFileChange}
            />
            <div
              className="w-8 h-8 rounded-full flex items-center justify-center"
              style={{ background: "rgba(113,76,182,0.10)" }}
            >
              <UploadCloud size={16} style={{ color: "#714cb6" }} />
            </div>
            <div className="text-center px-2">
              <p style={{ fontSize: "13px", fontWeight: 500, color: "#292827" }}>
                Drop file or click
              </p>
              <p style={{ fontSize: "12px", color: "#666666", marginTop: "2px" }}>
                PDF · MD · TXT
              </p>
            </div>
          </div>

          {/* Upload feedback */}
          {upload.type !== "idle" && (
            <div
              className="flex items-start gap-2 px-3 py-2.5 rounded-[8px] animate-fade-in"
              style={{
                background:
                  upload.type === "success"
                    ? "rgba(21,128,61,0.07)"
                    : upload.type === "error"
                    ? "rgba(185,28,28,0.07)"
                    : "rgba(113,76,182,0.07)",
                border: `1px solid ${
                  upload.type === "success"
                    ? "rgba(21,128,61,0.20)"
                    : upload.type === "error"
                    ? "rgba(185,28,28,0.20)"
                    : "rgba(113,76,182,0.20)"
                }`,
              }}
            >
              {upload.type === "uploading" && (
                <span
                  className="w-3 h-3 rounded-full shrink-0 mt-0.5"
                  style={{
                    background: "#714cb6",
                    animation: "bounce-dots 1s ease-in-out infinite",
                  }}
                />
              )}
              {upload.type === "success" && (
                <span style={{ color: "#15803d", fontSize: "13px" }}>✓</span>
              )}
              {upload.type === "error" && (
                <AlertCircle size={13} style={{ color: "#b91c1c", flexShrink: 0, marginTop: "2px" }} />
              )}
              <p
                style={{
                  fontSize: "12px",
                  color:
                    upload.type === "success"
                      ? "#15803d"
                      : upload.type === "error"
                      ? "#b91c1c"
                      : "#714cb6",
                  fontWeight: 460,
                  lineHeight: 1.4,
                  wordBreak: "break-all",
                }}
              >
                {upload.msg}
              </p>
            </div>
          )}
        </div>
      </aside>

      {/* ── Main area ───────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0" style={{ background: "#f2f0eb" }}>
        {/* Top bar */}
        <header
          className="flex items-center px-8 py-3.5 shrink-0"
          style={{
            background: "rgba(242,240,235,0.85)",
            backdropFilter: "blur(12px)",
            borderBottom: "1px solid #e3e3e2",
          }}
        >
          <span style={{ fontSize: "14px", color: "#666666", fontWeight: 460 }}>
            Hybrid RAG · vector + BM25 + Cohere rerank
          </span>
          <div className="flex-1" />
          <span
            className="px-3 py-1 rounded-full text-xs font-medium"
            style={{ background: "#f2f0eb", border: "1px solid #e3e3e2", color: "#666666" }}
          >
            Go backend · v0.4.0
          </span>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto py-10 px-6 md:px-16 flex flex-col gap-6 relative">
          {messages.length === 0 ? (
            <EmptyState onUpload={() => fileInputRef.current?.click()} />
          ) : (
            <>
              {messages.map((msg) => (
                <ChatMessage key={msg.id} msg={msg} />
              ))}
              {isTyping && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input bar */}
        <div
          className="shrink-0 px-6 md:px-16 py-5"
          style={{ borderTop: "1px solid #dcd7d3", background: "rgba(242,240,235,0.9)" }}
        >
          <div
            className="flex items-end gap-3 max-w-[720px] mx-auto rounded-[16px] px-5 py-3.5 transition-all duration-200"
            style={{
              background: "#ffffff",
              border: "1px solid #e3e3e2",
              boxShadow: "0 2px 8px rgba(41,40,39,0.06)",
            }}
            onFocus={() => {}}
          >
            <textarea
              ref={textareaRef}
              rows={1}
              className="flex-1 resize-none bg-transparent outline-none leading-relaxed placeholder:text-[#b2b0ae]"
              style={{
                fontSize: "15px",
                fontWeight: 460,
                color: "#292827",
                minHeight: "24px",
                maxHeight: "160px",
                fontFamily: "inherit",
              }}
              placeholder="Ask a question about your documents…"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                resizeTextarea();
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
            />
            <button
              disabled={!input.trim() || isTyping}
              onClick={handleSend}
              className="shrink-0 w-9 h-9 rounded-full flex items-center justify-center transition-all duration-150 active:scale-95"
              style={{
                background: input.trim() && !isTyping ? "#292827" : "#e3e3e2",
                color: input.trim() && !isTyping ? "#ffffff" : "#9ca3af",
                cursor: input.trim() && !isTyping ? "pointer" : "not-allowed",
              }}
            >
              <Send size={15} />
            </button>
          </div>
          <p
            className="text-center mt-2"
            style={{ fontSize: "11px", color: "#b2b0ae", fontWeight: 460 }}
          >
            ↵ to send · Shift ↵ for new line
          </p>
        </div>
      </main>
    </div>
  );
}
