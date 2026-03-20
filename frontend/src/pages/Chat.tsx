import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { chatAPI } from '../lib/api';
import { useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import DocumentUpload from '../components/DocumentUpload';

// ── Types ────────────────────────────────────────────────────────
interface MemoryCitation {
  node_type: string;
  retrieval_score: number;
  hop_distance: number | string;
  snippet: string;
  properties: Record<string, any>;
  source?: 'graph' | 'vector' | 'hybrid';
  score_breakdown?: {
    graph_distance: number;
    recency: number;
    confidence: number;
    reinforcement: number;
  };
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  retrieval_ms?: number;
  llm_generation_ms?: number;
  citations?: MemoryCitation[];
}

// ── Node colours ─────────────────────────────────────────────────
const NODE_META: Record<string, { pill: string; dot: string }> = {
  Fact: { pill: 'rgba(99,102,241,0.18)', dot: '#6366f1' },
  Transaction: { pill: 'rgba(45,212,191,0.15)', dot: '#2dd4bf' },
  Asset: { pill: 'rgba(251,191,36,0.15)', dot: '#fbbf24' },
  Goal: { pill: 'rgba(167,139,250,0.15)', dot: '#a78bfa' },
  Entity: { pill: 'rgba(148,163,184,0.12)', dot: '#94a3b8' },
  Message: { pill: 'rgba(56,189,248,0.13)', dot: '#38bdf8' },
};

// ── Shared logo ───────────────────────────────────────────────────
function Logo({ size = 7 }: { size?: number }) {
  const px = size * 4;
  return (
    <div
      className="animate-pulse-glow flex items-center justify-center rounded-xl flex-shrink-0"
      style={{
        width: px, height: px,
        background: 'linear-gradient(135deg,rgba(139,92,246,0.22),rgba(99,102,241,0.12))',
        border: '1px solid rgba(139,92,246,0.3)',
      }}
    >
      <svg style={{ width: px * 0.46, height: px * 0.46 }}
        fill="none" stroke="rgba(167,139,250,0.85)" strokeWidth="2.2" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="3" fill="rgba(167,139,250,0.18)" />
        <path d="M12 2v3m0 14v3M2 12h3m14 0h3m-3.5-6.5-2 2m-7 7-2 2m11 0-2-2m-7-7-2-2" />
      </svg>
    </div>
  );
}

// ── Wave typing indicator ─────────────────────────────────────────
function WaveDots() {
  return (
    <div className="flex items-center gap-1.5 py-1 px-0.5">
      {[0, 1, 2].map(i => (
        <div
          key={i}
          className="rounded-full"
          style={{
            width: 6, height: 6,
            background: 'linear-gradient(135deg,#8b5cf6,#6366f1)',
            animation: `waveFloat 1.2s ease-in-out infinite`,
            animationDelay: `${i * 0.18}s`,
            opacity: 0.8,
          }}
        />
      ))}
    </div>
  );
}

// ── Score bar ─────────────────────────────────────────────────────
function ScoreBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2 flex-shrink-0">
      <div className="w-16 h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.07)' }}>
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            background: 'linear-gradient(90deg,#8b5cf6,#2dd4bf)',
          }}
        />
      </div>
      <span style={{ fontSize: '0.65rem', color: 'var(--text-ghost)', minWidth: '2rem' }}>{pct}%</span>
    </div>
  );
}

// ── Citation card ─────────────────────────────────────────────────
function CitationCard({ c, i }: { c: MemoryCitation; i: number }) {
  const [open, setOpen] = useState(false);
  const meta = NODE_META[c.node_type] ?? { pill: 'rgba(148,163,184,0.12)', dot: '#94a3b8' };
  const isGraph = c.source === 'graph';
  const isVector = c.source === 'vector';

  return (
    <div
      className="rounded-xl overflow-hidden transition-all duration-200"
      style={{ border: '1px solid rgba(255,255,255,0.05)', background: 'rgba(255,255,255,0.018)' }}
    >
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center gap-2.5 px-3 py-2.5 text-left transition-colors hover:bg-white/[0.025]"
      >
        {/* Index */}
        <span style={{ fontSize: '0.65rem', color: 'var(--text-ghost)', minWidth: '1.2rem', textAlign: 'right' }}>
          {i + 1}
        </span>

        {/* Node type pill */}
        <span
          className="rounded-md px-2 py-0.5 flex-shrink-0"
          style={{ fontSize: '0.65rem', fontWeight: 700, background: meta.pill, color: meta.dot, border: `1px solid ${meta.dot}28` }}
        >
          {c.node_type}
        </span>

        {/* Source badge */}
        <span
          className="rounded-md px-2 py-0.5 flex-shrink-0"
          style={{
            fontSize: '0.6rem', fontWeight: 700,
            background: isGraph ? 'rgba(139,92,246,0.14)' : isVector ? 'rgba(45,212,191,0.13)' : 'rgba(99,102,241,0.13)',
            color: isGraph ? '#a78bfa' : isVector ? '#2dd4bf' : '#818cf8',
            border: `1px solid ${isGraph ? '#8b5cf628' : isVector ? '#2dd4bf28' : '#6366f128'}`,
          }}
        >
          {isGraph ? 'Graph' : isVector ? 'Vector' : 'Hybrid'}
        </span>

        {/* Snippet */}
        <span
          className="flex-1 truncate"
          style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}
        >
          {c.snippet || '—'}
        </span>

        {/* Score bar */}
        <ScoreBar value={c.retrieval_score} />

        {/* Hop */}
        <span style={{ fontSize: '0.6rem', color: 'var(--text-ghost)', flexShrink: 0, marginLeft: '0.25rem' }}>
          {c.hop_distance !== 'N/A' ? `${c.hop_distance}hop` : '—'}
        </span>

        {/* Chevron */}
        <svg
          className="flex-shrink-0 transition-transform duration-200"
          style={{ transform: open ? 'rotate(180deg)' : 'none' }}
          width="12" height="12" fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth="2.5" viewBox="0 0 24 24"
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>

      {open && (
        <div
          className="animate-fade-up px-4 py-3 space-y-3"
          style={{ borderTop: '1px solid rgba(255,255,255,0.04)', background: '#0f0f1c' }}
        >
          {/* Score breakdown */}
          {c.score_breakdown && (
            <div>
              <p style={{ fontSize: '0.6rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-ghost)', marginBottom: '0.6rem' }}>
                Score Breakdown
              </p>
              <div className="grid grid-cols-2 gap-2">
                {([
                  ['Graph distance', c.score_breakdown.graph_distance],
                  ['Recency', c.score_breakdown.recency],
                  ['Confidence', c.score_breakdown.confidence],
                  ['Reinforcement', c.score_breakdown.reinforcement],
                ] as [string, number][]).map(([lbl, val]) => (
                  <div key={lbl} className="flex items-center justify-between gap-2">
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{lbl}</span>
                    <ScoreBar value={val} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Properties */}
          {Object.keys(c.properties).length > 0 && (
            <div>
              <p style={{ fontSize: '0.6rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-ghost)', marginBottom: '0.6rem' }}>
                Properties
              </p>
              <div className="space-y-1">
                {Object.entries(c.properties).map(([k, v]) =>
                  v !== undefined && v !== null && v !== '' ? (
                    <div key={k} className="flex gap-3">
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', minWidth: '5.5rem', flexShrink: 0, textTransform: 'capitalize' }}>
                        {k.replace(/_/g, ' ')}
                      </span>
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', wordBreak: 'break-all' }}>
                        {String(v)}
                      </span>
                    </div>
                  ) : null
                )}
              </div>
            </div>
          )}

          {c.snippet && (
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontStyle: 'italic', borderTop: '1px solid rgba(255,255,255,0.04)', paddingTop: '0.6rem' }}>
              "{c.snippet}"
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ── Sources panel ─────────────────────────────────────────────────
function SourcesPanel({ citations }: { citations: MemoryCitation[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-2 transition-opacity hover:opacity-80"
        style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 500 }}
      >
        <svg width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <ellipse cx="12" cy="5" rx="9" ry="3" />
          <path d="M21 12c0 1.66-4.03 3-9 3S3 13.66 3 12" />
          <path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5" />
        </svg>
        <span>{citations.length} memory source{citations.length !== 1 ? 's' : ''}</span>
        <svg
          className="transition-transform duration-200"
          style={{ transform: open ? 'rotate(180deg)' : 'none' }}
          width="10" height="10" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24"
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>
      {open && (
        <div className="mt-2 space-y-1.5 animate-fade-up">
          {citations.map((c, i) => <CitationCard key={i} c={c} i={i} />)}
        </div>
      )}
    </div>
  );
}

// ── Document upload panel ─────────────────────────────────────────
function DocUploadPanel({
  userId,
  onSuccess,
  onError,
  onStart,
  onClose,
}: {
  userId: string;
  onSuccess: (data: any) => void;
  onError: (e: string) => void;
  onStart: () => void;
  onClose: () => void;
}) {
  return (
    <div
      className="animate-scale-in rounded-2xl p-4 space-y-3"
      style={{ background: 'var(--bg-raised)', border: '1px solid rgba(255,255,255,0.07)' }}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div style={{
            width: 28, height: 28, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'linear-gradient(135deg,rgba(139,92,246,0.22),rgba(99,102,241,0.12))',
            border: '1px solid rgba(139,92,246,0.25)',
          }}>
            <svg width="13" height="13" fill="none" stroke="rgba(167,139,250,0.85)" strokeWidth="2" viewBox="0 0 24 24">
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="12" y1="18" x2="12" y2="12" />
              <line x1="9" y1="15" x2="15" y2="15" />
            </svg>
          </div>
          <div>
            <p style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>Upload Document</p>
            <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>PDF, images — extracted into your memory graph</p>
          </div>
        </div>
        <button onClick={onClose} className="btn-ghost" style={{ padding: '0.3rem', border: 'none', borderRadius: '0.5rem' }}>
          <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      <div style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.75rem' }}>
        <DocumentUpload
          userId={userId}
          onUploadSuccess={onSuccess}
          onUploadError={onError}
          onUploadStart={onStart}
        />
      </div>
    </div>
  );
}

// ── Upload success toast ──────────────────────────────────────────
function UploadSuccess({ data, onDismiss }: { data: any; onDismiss: () => void }) {
  return (
    <div
      className="animate-scale-in rounded-2xl p-3.5 flex items-start gap-3"
      style={{ background: 'rgba(45,212,191,0.07)', border: '1px solid rgba(45,212,191,0.18)' }}
    >
      <div style={{
        width: 28, height: 28, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'rgba(45,212,191,0.15)', flexShrink: 0,
      }}>
        <svg width="14" height="14" fill="none" stroke="#2dd4bf" strokeWidth="2.5" viewBox="0 0 24 24">
          <path d="M20 6L9 17l-5-5" />
        </svg>
      </div>
      <div className="flex-1 min-w-0">
        <p style={{ fontSize: '0.8125rem', fontWeight: 600, color: '#2dd4bf' }}>{data.document_name}</p>
        <div className="flex gap-4 mt-1.5 flex-wrap">
          {[
            ['Facts', data.extraction_stats?.facts_extracted || 0],
            ['Entities', data.extraction_stats?.entities_extracted || 0],
            ['Relations', data.extraction_stats?.relationships_extracted || 0],
          ].map(([lbl, val]) => (
            <span key={lbl as string} style={{ fontSize: '0.7rem', color: 'rgba(45,212,191,0.65)' }}>
              <span style={{ color: '#2dd4bf', fontWeight: 600 }}>{val}</span> {lbl}
            </span>
          ))}
        </div>
      </div>
      <button onClick={onDismiss} style={{ flexShrink: 0, background: 'none', border: 'none', cursor: 'pointer', color: 'rgba(45,212,191,0.4)', padding: 4 }}>
        <svg width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
          <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>
    </div>
  );
}

// ── Empty state prompts ───────────────────────────────────────────
const SAMPLE_PROMPTS = [
  { icon: '💹', text: 'I invested ₹50,000 in HDFC mutual fund' },
  { icon: '🎯', text: 'My retirement goal is ₹2 crore by 2045' },
  { icon: '🔍', text: 'What have I invested so far?' },
];

// ── Main Chat Page ────────────────────────────────────────────────
export default function Chat() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [showDocUpload, setShowDocUpload] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState<any>(null);
  const [isUploading, setIsUploading] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  useEffect(() => { scrollToBottom(); }, [messages]);

  const resizeTextarea = () => {
    const ta = inputRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
  };
  useEffect(resizeTextarea, [input]);

  // Load history
  useEffect(() => {
    const load = async () => {
      if (!user) { setIsLoadingHistory(false); return; }
      setIsLoadingHistory(true);
      try {
        const sessions = await chatAPI.getSessions();
        const session = sessions?.[0];
        if (!session) { return; }
        const history = await chatAPI.getSessionMessages(session.id);
        setMessages((history || []).map((msg: any) => ({
          id: msg.id, role: msg.role, content: msg.content,
          timestamp: msg.created_at ? new Date(msg.created_at) : new Date(),
          retrieval_ms: msg.retrieval_time_ms ?? undefined,
          llm_generation_ms: msg.llm_generation_time_ms ?? undefined,
          citations: Array.isArray(msg.memory_citations) ? msg.memory_citations : undefined,
        })));
      } catch (e) { console.error('Failed to load history', e); }
      finally { setIsLoadingHistory(false); }
    };
    load();
  }, [user]);

  const handleSend = async (text?: string, e?: React.FormEvent) => {
    e?.preventDefault();
    const msg = text ?? input.trim();
    if (!msg || isSending || !user) return;
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: msg, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsSending(true);
    try {
      const res = await chatAPI.sendMessage(msg, user.user_id);
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: res.answer || res.message || 'Processed successfully.',
        timestamp: new Date(),
        retrieval_ms: res.retrieval_metrics?.retrieval_ms,
        llm_generation_ms: res.retrieval_metrics?.llm_generation_ms,
        citations: res.memory_citations ?? undefined,
      }]);
    } catch {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(), role: 'assistant',
        content: 'Something went wrong. Please try again.', timestamp: new Date(),
      }]);
    } finally { setIsSending(false); }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleUploadSuccess = (data: any) => {
    setUploadSuccess({
      document_name: data.document_name || 'Document',
      extraction_stats: {
        facts_extracted: data.extraction_stats?.facts_extracted || data.llm_extraction?.facts?.length || 0,
        entities_extracted: data.extraction_stats?.entities_extracted || data.llm_extraction?.entities?.length || 0,
        relationships_extracted: data.extraction_stats?.relationships_extracted || data.llm_extraction?.relationships?.length || 0,
      },
    });
    setIsUploading(false);
    setTimeout(() => setUploadSuccess(null), 6000);
  };

  const empty = messages.length === 0 && !isLoadingHistory;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--bg-base)' }}>

      {/* ── Header ────────────────────────────────────────────── */}
      <header
        className="flex items-center justify-between flex-shrink-0 z-10"
        style={{
          padding: '0.75rem 1.25rem',
          background: 'rgba(7,7,15,0.85)',
          backdropFilter: 'blur(16px)',
          borderBottom: '1px solid var(--border-subtle)',
        }}
      >
        <div className="flex items-center gap-2.5">
          <Logo size={7} />
          <div>
            <div className="flex items-center gap-1.5">
              <p style={{ fontSize: '0.875rem', fontWeight: 700, color: 'var(--text-primary)' }}>GraphMind</p>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#2dd4bf', boxShadow: '0 0 6px #2dd4bf', display: 'inline-block' }} className="animate-blink" />
            </div>
            <p style={{ fontSize: '0.6875rem', color: 'var(--text-ghost)' }}>Cognitive finance assistant</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="hidden sm:block text-right mr-1">
            <p style={{ fontSize: '0.75rem', fontWeight: 500, color: 'var(--text-secondary)' }}>{user?.full_name}</p>
            <p style={{ fontSize: '0.6875rem', color: 'var(--text-ghost)' }}>{user?.email}</p>
          </div>

          <button
            onClick={() => navigate('/mindmap')}
            className="btn-ghost"
            style={{ borderColor: 'rgba(139,92,246,0.2)', color: 'rgba(167,139,250,0.6)' }}
          >
            <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <circle cx="5" cy="12" r="2" /><circle cx="19" cy="5" r="2" /><circle cx="19" cy="19" r="2" />
              <path d="M7 11l10-4M7 13l10 4" />
            </svg>
            <span className="hidden sm:inline">Knowledge Graph</span>
          </button>

          <button onClick={logout} className="btn-ghost">
            <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4M16 17l5-5-5-5M21 12H9" />
            </svg>
            <span className="hidden sm:inline">Logout</span>
          </button>
        </div>
      </header>

      {/* ── Messages ──────────────────────────────────────────── */}
      <main style={{ flex: 1, overflowY: 'auto', padding: '1.5rem 1rem' }}>
        <div style={{ maxWidth: 760, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

          {/* Loading skeleton */}
          {isLoadingHistory && (
            <div className="flex justify-center py-12">
              <div className="flex items-center gap-2" style={{ fontSize: '0.8125rem', color: 'var(--text-ghost)' }}>
                <svg className="animate-spin" width="16" height="16" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Loading your conversation…
              </div>
            </div>
          )}

          {/* Empty state */}
          {empty && (
            <div className="animate-fade-in flex flex-col items-center justify-center py-20 text-center">
              {/* Background glow */}
              <div style={{
                position: 'absolute', width: 500, height: 500,
                background: 'radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 65%)',
                pointerEvents: 'none', filter: 'blur(40px)',
              }} />
              <Logo size={14} />
              <h2 style={{ fontSize: '1.35rem', fontWeight: 700, color: 'var(--text-primary)', margin: '1.25rem 0 0.5rem' }}>
                Your financial memory awaits
              </h2>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', maxWidth: 360, lineHeight: 1.6 }}>
                Share investments, goals, or ask questions. GraphMind builds your knowledge graph automatically.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 mt-8 w-full" style={{ maxWidth: 600 }}>
                {SAMPLE_PROMPTS.map(({ icon, text }) => (
                  <button
                    key={text}
                    onClick={() => { setInput(text); inputRef.current?.focus(); }}
                    className="text-left rounded-xl transition-all duration-200 hover:-translate-y-0.5"
                    style={{
                      padding: '0.875rem 1rem',
                      background: 'var(--bg-raised)',
                      border: '1px solid var(--border-subtle)',
                      cursor: 'pointer',
                    }}
                    onMouseEnter={e => {
                      (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(139,92,246,0.25)';
                      (e.currentTarget as HTMLButtonElement).style.background = 'var(--bg-hover)';
                    }}
                    onMouseLeave={e => {
                      (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--border-subtle)';
                      (e.currentTarget as HTMLButtonElement).style.background = 'var(--bg-raised)';
                    }}
                  >
                    <span style={{ fontSize: '1.1rem', display: 'block', marginBottom: '0.4rem' }}>{icon}</span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>{text}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((msg, idx) => (
            <div
              key={msg.id}
              className="animate-fade-up flex gap-2.5"
              style={{
                justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                animationDelay: `${Math.min(idx * 0.03, 0.15)}s`,
              }}
            >
              {msg.role === 'assistant' && (
                <div style={{ marginTop: 2, flexShrink: 0 }}>
                  <Logo size={7} />
                </div>
              )}

              <div style={{
                maxWidth: '78%',
                borderRadius: msg.role === 'user' ? '1.25rem 1.25rem 0.4rem 1.25rem' : '1.25rem 1.25rem 1.25rem 0.4rem',
                padding: '0.875rem 1.125rem',
                background: msg.role === 'user'
                  ? 'linear-gradient(135deg,rgba(139,92,246,0.18),rgba(99,102,241,0.12))'
                  : 'var(--bg-card)',
                border: msg.role === 'user'
                  ? '1px solid rgba(139,92,246,0.22)'
                  : '1px solid var(--border-subtle)',
              }}>
                {msg.role === 'assistant' ? (
                  <div
                    className="prose-sm"
                    style={{
                      fontSize: '0.875rem', lineHeight: 1.7, color: 'var(--text-primary)',
                    }}
                  >
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  <p style={{ fontSize: '0.875rem', lineHeight: 1.65, color: 'var(--text-primary)', whiteSpace: 'pre-wrap' }}>
                    {msg.content}
                  </p>
                )}

                {/* Citations */}
                {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
                  <SourcesPanel citations={msg.citations} />
                )}

                {/* Perf metrics */}
                {msg.role === 'assistant' && (msg.retrieval_ms !== undefined || msg.llm_generation_ms !== undefined) && (
                  <div
                    className="flex items-center gap-4 mt-2.5 pt-2.5"
                    style={{ borderTop: '1px solid rgba(255,255,255,0.04)', fontSize: '0.6875rem', color: 'var(--text-ghost)' }}
                  >
                    {msg.retrieval_ms !== undefined && (
                      <span className="flex items-center gap-1">
                        <svg width="11" height="11" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                          <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
                        </svg>
                        {Math.round(msg.retrieval_ms)}ms retrieval
                      </span>
                    )}
                    {msg.llm_generation_ms !== undefined && (
                      <span className="flex items-center gap-1">
                        <svg width="11" height="11" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                        </svg>
                        {Math.round(msg.llm_generation_ms)}ms LLM
                      </span>
                    )}
                  </div>
                )}

                <p style={{ fontSize: '0.625rem', color: 'var(--text-ghost)', marginTop: '0.5rem', textAlign: msg.role === 'user' ? 'right' : 'left' }}>
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>

              {msg.role === 'user' && (
                <div
                  className="flex items-center justify-center flex-shrink-0"
                  style={{
                    width: 28, height: 28, borderRadius: 10, marginTop: 2,
                    background: 'var(--bg-raised)', border: '1px solid var(--border-dim)', flexShrink: 0,
                  }}
                >
                  <svg width="13" height="13" fill="none" stroke="rgba(255,255,255,0.35)" strokeWidth="2" viewBox="0 0 24 24">
                    <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" /><circle cx="12" cy="7" r="4" />
                  </svg>
                </div>
              )}
            </div>
          ))}

          {/* Typing indicator */}
          {isSending && (
            <div className="flex gap-2.5 animate-fade-up">
              <Logo size={7} />
              <div style={{
                borderRadius: '1.25rem 1.25rem 1.25rem 0.4rem', padding: '0.875rem 1.125rem',
                background: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
              }}>
                <WaveDots />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* ── Input bar ─────────────────────────────────────────── */}
      <div
        className="flex-shrink-0"
        style={{ padding: '0.875rem 1rem 1.125rem', background: 'var(--bg-base)', borderTop: '1px solid var(--border-subtle)' }}
      >
        <div style={{ maxWidth: 760, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>

          {/* Doc upload panel */}
          {showDocUpload && (
            <DocUploadPanel
              userId={user?.user_id || ''}
              onSuccess={data => { handleUploadSuccess(data); setShowDocUpload(false); }}
              onError={e => { setUploadError(e); setIsUploading(false); }}
              onStart={() => setIsUploading(true)}
              onClose={() => setShowDocUpload(false)}
            />
          )}

          {/* Upload success toast */}
          {uploadSuccess && (
            <UploadSuccess data={uploadSuccess} onDismiss={() => setUploadSuccess(null)} />
          )}

          {/* Upload error */}
          {uploadError && (
            <div
              className="animate-scale-in rounded-xl px-3.5 py-2.5 flex items-center gap-2"
              style={{ background: 'rgba(239,68,68,0.07)', border: '1px solid rgba(239,68,68,0.18)', fontSize: '0.75rem', color: '#f87171' }}
            >
              <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              {uploadError}
              <button onClick={() => setUploadError(null)} style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'currentColor' }}>
                <svg width="11" height="11" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                  <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
          )}

          {/* Chat input pill */}
          <form
            onSubmit={e => handleSend(undefined, e)}
            className="flex items-end gap-2 rounded-2xl transition-all duration-200"
            style={{
              padding: '0.625rem 0.75rem 0.625rem 1rem',
              background: 'var(--bg-card)',
              border: '1px solid var(--border-dim)',
            }}
            onFocusCapture={e => { (e.currentTarget as HTMLFormElement).style.borderColor = 'rgba(139,92,246,0.35)'; (e.currentTarget as HTMLFormElement).style.boxShadow = '0 0 0 3px rgba(139,92,246,0.08)'; }}
            onBlurCapture={e => { (e.currentTarget as HTMLFormElement).style.borderColor = 'var(--border-dim)'; (e.currentTarget as HTMLFormElement).style.boxShadow = 'none'; }}
          >
            {/* Attach button */}
            <button
              type="button"
              onClick={() => setShowDocUpload(v => !v)}
              title="Upload document"
              className="flex-shrink-0 flex items-center justify-center rounded-lg transition-all duration-150"
              style={{
                width: 32, height: 32, background: 'none', border: 'none', cursor: 'pointer',
                color: showDocUpload ? 'rgba(167,139,250,0.8)' : 'var(--text-ghost)',
              }}
              onMouseEnter={e => (e.currentTarget.style.background = 'rgba(139,92,246,0.1)')}
              onMouseLeave={e => (e.currentTarget.style.background = 'none')}
            >
              <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
              </svg>
            </button>

            {/* Textarea */}
            <textarea
              ref={inputRef}
              rows={1}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question or tell me about an investment… (Enter to send)"
              disabled={isSending}
              style={{
                flex: 1, resize: 'none', background: 'transparent',
                border: 'none', outline: 'none',
                fontSize: '0.875rem', color: 'var(--text-primary)',
                lineHeight: 1.6, minHeight: 24, maxHeight: 160,
                fontFamily: 'inherit',
              }}
            />

            {/* Send button */}
            <button
              type="submit"
              disabled={isSending || !input.trim() || isUploading}
              className="flex-shrink-0 flex items-center justify-center rounded-xl transition-all duration-150"
              style={{
                width: 34, height: 34, border: 'none', cursor: 'pointer',
                background: input.trim() && !isSending
                  ? 'linear-gradient(135deg,#8b5cf6,#6366f1)'
                  : 'rgba(255,255,255,0.06)',
              }}
              onMouseEnter={e => { if (input.trim()) (e.currentTarget.style.transform = 'scale(1.06)'); }}
              onMouseLeave={e => { (e.currentTarget.style.transform = 'scale(1)'); }}
            >
              <svg width="14" height="14" fill="none" stroke="white" strokeWidth="2.5" viewBox="0 0 24 24">
                <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </form>

          <p className="text-center" style={{ fontSize: '0.625rem', color: 'var(--text-ghost)' }}>
            GraphMind may make mistakes. Verify important financial decisions independently.
          </p>
        </div>
      </div>
    </div>
  );
}
