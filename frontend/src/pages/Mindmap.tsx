import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ReactFlow, Controls, Background, BackgroundVariant,
  useNodesState, useEdgesState, MarkerType, Position,
  type Node, type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useAuth } from '../contexts/AuthContext';
import { memoryAPI } from '../lib/api';

const NODE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  User:        { bg: '#312e81', border: '#6366f1', text: '#c7d2fe' },
  Message:     { bg: '#164e63', border: '#06b6d4', text: '#a5f3fc' },
  Fact:        { bg: '#1e3a5f', border: '#3b82f6', text: '#bfdbfe' },
  Asset:       { bg: '#14532d', border: '#22c55e', text: '#bbf7d0' },
  Goal:        { bg: '#451a03', border: '#f59e0b', text: '#fde68a' },
  Transaction: { bg: '#2e1065', border: '#a855f7', text: '#e9d5ff' },
  RiskProfile: { bg: '#450a0a', border: '#ef4444', text: '#fecaca' },
  Entity:      { bg: '#1c1f26', border: '#6b7280', text: '#d1d5db' },
  Preference:  { bg: '#1a2744', border: '#60a5fa', text: '#bfdbfe' },
};

const Logo = () => (
  <div className="w-7 h-7 rounded-lg bg-indigo-950 border border-indigo-800/40 flex items-center justify-center flex-shrink-0">
    <svg className="w-3.5 h-3.5 text-indigo-400" fill="none" stroke="currentColor" strokeWidth="2.2" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v3m0 14v3M2 12h3m14 0h3m-3.5-6.5-2 2m-7 7-2 2m11 0-2-2m-7-7-2-2" />
    </svg>
  </div>
);

export default function Mindmap() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nodeCount, setNodeCount] = useState(0);
  const [edgeCount, setEdgeCount] = useState(0);

  const hierarchyLayout = (rawNodes: any[], rawEdges: any[]) => {
    const userNode = rawNodes.find(n => n.type === 'User');
    if (!userNode) return rawNodes;
    const adj = new Map<string, string[]>();
    rawEdges.forEach(e => {
      if (!adj.has(e.source)) adj.set(e.source, []);
      adj.get(e.source)!.push(e.target);
    });
    const layers = new Map<string, number>();
    const visited = new Set<string>();
    const queue: { id: string; layer: number }[] = [{ id: userNode.id, layer: 0 }];
    layers.set(userNode.id, 0);
    visited.add(userNode.id);
    while (queue.length) {
      const { id, layer } = queue.shift()!;
      (adj.get(id) || []).forEach(cid => {
        if (!visited.has(cid)) { visited.add(cid); layers.set(cid, layer + 1); queue.push({ id: cid, layer: layer + 1 }); }
      });
    }
    rawNodes.forEach(n => { if (!layers.has(n.id)) layers.set(n.id, Math.max(...Array.from(layers.values())) + 1); });
    const byLayer = new Map<number, any[]>();
    rawNodes.forEach(n => {
      const l = layers.get(n.id) || 0;
      if (!byLayer.has(l)) byLayer.set(l, []);
      byLayer.get(l)!.push(n);
    });
    return rawNodes.map(n => {
      const l = layers.get(n.id) || 0;
      const inLayer = byLayer.get(l) || [];
      const idx = inLayer.indexOf(n);
      const totalW = (inLayer.length - 1) * 210;
      return { ...n, x: -totalW / 2 + 400 + idx * 210, y: 100 + l * 190 };
    });
  };

  const loadMindmap = useCallback(async () => {
    if (!user) return;
    setIsLoading(true); setError(null);
    try {
      const data = await memoryAPI.getMindmap();
      const positioned = hierarchyLayout(data.nodes, data.edges);
      setNodeCount(positioned.length);
      setEdgeCount(data.edges.length);
      setNodes(positioned.map((n: any): Node => {
        const schema = NODE_COLORS[n.type] || NODE_COLORS.Entity;
        return {
          id: n.id, type: 'default',
          position: { x: n.x, y: n.y },
          data: {
            label: (
              <div className="text-center px-1">
                <div className="text-xs font-semibold" style={{ color: schema.text }}>{n.label}</div>
                <div className="text-[10px] mt-0.5 opacity-60" style={{ color: schema.text }}>{n.type}</div>
              </div>
            ),
          },
          style: {
            background: schema.bg, border: `1.5px solid ${schema.border}`,
            borderRadius: '10px', padding: '8px 12px', fontSize: '12px',
            minWidth: '110px', boxShadow: `0 0 12px ${schema.border}22`,
          },
          sourcePosition: Position.Bottom, targetPosition: Position.Top,
        };
      }));
      setEdges(data.edges.map((e: any): Edge => ({
        id: e.id, source: e.source, target: e.target,
        label: e.label, type: 'smoothstep', animated: true,
        style: { stroke: '#4f46e5', strokeWidth: 1.5, strokeDasharray: e.label?.includes('CONTRADICTS') ? '4 3' : undefined },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#4f46e5', width: 14, height: 14 },
        labelStyle: { fill: '#a5b4fc', fontSize: 9, fontWeight: 600 },
        labelBgStyle: { fill: '#0d0d10', fillOpacity: 0.85, rx: 4 },
        labelBgPadding: [4, 3] as [number, number],
      })));
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load knowledge graph');
    } finally { setIsLoading(false); }
  }, [user, setNodes, setEdges]);

  useEffect(() => { loadMindmap(); }, [loadMindmap]);

  return (
    <div className="flex flex-col h-screen bg-[#0d0d10]">
      {/* Header */}
      <header className="flex items-center justify-between px-5 py-3.5 border-b border-white/[0.06] bg-[#0d0d10]/90 backdrop-blur-md z-10 flex-shrink-0">
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/chat')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-white/40 hover:text-white/70 hover:bg-white/[0.05] border border-white/[0.06] transition-all">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
              <path d="M19 12H5M12 5l-7 7 7 7" />
            </svg>
            Back
          </button>
          <div className="flex items-center gap-2">
            <Logo />
            <div>
              <p className="text-sm font-semibold text-white/85">Knowledge Graph</p>
              <p className="text-[11px] text-white/30">Your financial memory visualised</p>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {!isLoading && nodes.length > 0 && (
            <div className="hidden sm:flex items-center gap-4 text-xs text-white/30">
              <span><span className="text-white/60 font-medium">{nodeCount}</span> nodes</span>
              <span><span className="text-white/60 font-medium">{edgeCount}</span> edges</span>
            </div>
          )}
          <button onClick={loadMindmap} disabled={isLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-white/40 hover:text-white/70 hover:bg-white/[0.05] border border-white/[0.06] transition-all disabled:opacity-40">
            <svg className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path d="M23 4v6h-6M1 20v-6h6" /><path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" />
            </svg>
            Refresh
          </button>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 relative overflow-hidden">
        {isLoading && nodes.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center space-y-4">
              <div className="w-12 h-12 rounded-2xl bg-white/[0.03] border border-white/[0.07] flex items-center justify-center mx-auto">
                <svg className="w-5 h-5 text-white/40 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
              </div>
              <p className="text-sm text-white/30">Loading knowledge graph…</p>
            </div>
          </div>
        ) : error ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center space-y-4 max-w-sm">
              <div className="w-12 h-12 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center mx-auto">
                <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-semibold text-white/70 mb-1">Failed to load graph</p>
                <p className="text-xs text-white/30">{error}</p>
              </div>
              <button onClick={loadMindmap}
                className="px-4 py-2 rounded-lg bg-white/[0.06] border border-white/[0.08] text-sm text-white/60 hover:text-white/80 hover:bg-white/[0.08] transition-all">
                Try again
              </button>
            </div>
          </div>
        ) : nodes.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center space-y-4 max-w-sm">
              <div className="w-14 h-14 rounded-2xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mx-auto text-3xl">🌐</div>
              <div>
                <p className="text-base font-semibold text-white/70 mb-1">No graph data yet</p>
                <p className="text-sm text-white/30">Start chatting to build your financial knowledge graph.</p>
              </div>
              <button onClick={() => navigate('/chat')}
                className="px-4 py-2 rounded-lg bg-indigo-600 text-sm font-medium text-white hover:bg-indigo-500 transition-colors">
                Go to Chat
              </button>
            </div>
          </div>
        ) : (
          <ReactFlow
            nodes={nodes} edges={edges}
            onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
            fitView fitViewOptions={{ padding: 0.15 }}
            attributionPosition="bottom-left"
            style={{ background: '#0d0d10' }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="rgba(255,255,255,0.04)" />
            <Controls style={{ background: '#18181d', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px' }} />
          </ReactFlow>
        )}

        {/* Floating legend */}
        {nodes.length > 0 && (
          <div className="absolute bottom-4 left-4 bg-[#18181d]/90 backdrop-blur-md rounded-xl border border-white/[0.06] p-4 shadow-xl">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-white/30 mb-3">Node Types</p>
            <div className="space-y-2">
              {Object.entries(NODE_COLORS).map(([type, cols]) => (
                <div key={type} className="flex items-center gap-2.5 text-xs">
                  <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: cols.border, boxShadow: `0 0 5px ${cols.border}55` }} />
                  <span style={{ color: cols.text }} className="opacity-70">{type}</span>
                </div>
              ))}
            </div>
            <div className="mt-3 pt-3 border-t border-white/[0.05] text-[10px] text-white/20 space-y-0.5">
              <p>Drag to rearrange • Scroll to zoom</p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
