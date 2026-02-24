import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const Logo = () => (
  <div className="w-8 h-8 rounded-lg bg-indigo-950 border border-indigo-800/40 flex items-center justify-center flex-shrink-0">
    <svg className="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" strokeWidth="2.2" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v3m0 14v3M2 12h3m14 0h3m-3.5-6.5-2 2m-7 7-2 2m11 0-2-2m-7-7-2-2" />
    </svg>
  </div>
);

export default function Signup() {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const { signup, isLoading } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (password !== confirmPassword) { setError('Passwords do not match'); return; }
    if (password.length < 6) { setError('Password must be at least 6 characters'); return; }
    try {
      await signup(email, password, fullName);
      navigate('/chat');
    } catch (err: any) {
      setError(err.message || 'Failed to create account');
    }
  };

  const Field = ({ label, type, value, onChange, placeholder }: {
    label: string; type: string; value: string;
    onChange: (v: string) => void; placeholder: string;
  }) => (
    <div className="space-y-1.5">
      <label className="text-[11px] font-semibold text-white/40 uppercase tracking-widest">{label}</label>
      <input
        type={type} value={value} onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder} required disabled={isLoading}
        className="w-full px-4 py-3 rounded-xl bg-white/[0.04] border border-white/[0.08] text-sm text-white placeholder-white/20 focus:outline-none focus:border-white/[0.18] transition-all disabled:opacity-50"
      />
    </div>
  );

  return (
    <div className="min-h-screen flex bg-[#0d0d10]">
      <div className="hidden lg:flex flex-col justify-between w-[44%] p-14 border-r border-white/[0.05] bg-[#0f0f13]">
        <div className="flex items-center gap-2.5">
          <Logo /><span className="text-sm font-semibold text-white/80">GraphMind</span>
        </div>
        <div className="space-y-8">
          <div className="space-y-2">
            <span className="text-xs font-semibold tracking-[0.15em] uppercase text-indigo-400/70">Cognitive Finance AI</span>
            <h2 className="text-[2rem] font-bold text-white leading-tight">Start building your<br />financial memory.</h2>
          </div>
          <p className="text-[15px] text-white/35 leading-relaxed">
            Every message you send is extracted into a structured knowledge graph, all personally yours.
          </p>
          <div className="space-y-3 pt-2">
            {[
              ['Persistent Memory', 'Your data lives between every session'],
              ['Smart Extraction', 'AI structures every message into graph nodes'],
              ['Full Explainability', 'Every answer cites its exact source nodes'],
            ].map(([title, desc]) => (
              <div key={title} className="flex items-start gap-3">
                <div className="w-1.5 h-1.5 rounded-full bg-indigo-400/50 mt-1.5 flex-shrink-0" />
                <div>
                  <p className="text-xs font-medium text-white/60">{title}</p>
                  <p className="text-xs text-white/25">{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
        <p className="text-xs text-white/15">© 2026 GraphMind</p>
      </div>

      <div className="flex-1 flex items-center justify-center p-6">
        <div className="w-full max-w-[400px] space-y-7">
          <div className="flex lg:hidden items-center gap-2.5">
            <Logo /><span className="text-sm font-semibold text-white/80">GraphMind</span>
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Create your account</h1>
            <p className="text-sm text-white/35 mt-1">Start your journey !</p>
          </div>
          {error && (
            <div className="flex items-start gap-2.5 p-3.5 rounded-xl bg-red-500/[0.08] border border-red-500/20 text-sm text-red-400">
              <svg className="w-4 h-4 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              {error}
            </div>
          )}
          <form onSubmit={handleSubmit} className="space-y-4">
            <Field label="Full Name" type="text" value={fullName} onChange={setFullName} placeholder="Full Name" />
            <Field label="Email" type="email" value={email} onChange={setEmail} placeholder="Email Address" />
            <div className="grid grid-cols-2 gap-3">
              <Field label="Password" type="password" value={password} onChange={setPassword} placeholder="••••••••" />
              <Field label="Confirm" type="password" value={confirmPassword} onChange={setConfirmPassword} placeholder="••••••••" />
            </div>
            <button type="submit" disabled={isLoading}
              className="w-full py-3 mt-1 rounded-xl bg-indigo-600 hover:bg-indigo-500 active:scale-[0.99] text-sm font-semibold text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed">
              {isLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  Creating account…
                </span>
              ) : 'Create account'}
            </button>
          </form>
          <p className="text-center text-sm text-white/25">
            Have an account?{' '}<Link to="/login" className="text-indigo-400 hover:text-indigo-300 font-medium transition-colors">Sign in</Link>
          </p>
        </div>
      </div>
    </div>
  );
}
