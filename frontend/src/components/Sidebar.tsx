import React, { useState, useEffect } from 'react';
import { Settings, RefreshCw, FileText, Plus, Trash2, MessageSquare } from 'lucide-react';
import { cn } from '../lib/utils';
import axios from 'axios';

interface SidebarProps {
    urls: string[];
    setUrls: React.Dispatch<React.SetStateAction<string[]>>;
    onRebuild: () => Promise<void>;
    isRebuilding: boolean;
    useHyde: boolean;
    setUseHyde: React.Dispatch<React.SetStateAction<boolean>>;
    useRerank: boolean;
    setUseRerank: React.Dispatch<React.SetStateAction<boolean>>;
    topK: number;
    setTopK: React.Dispatch<React.SetStateAction<number>>;
    sessionId: string | null;
    setSessionId: (id: string | null) => void;
    onNewChat: () => void;
}

interface Session {
    id: string;
    title: string;
    updated_at: string;
}

const Sidebar: React.FC<SidebarProps> = ({
    urls,
    setUrls,
    onRebuild,
    isRebuilding,
    useHyde,
    setUseHyde,
    useRerank,
    setUseRerank,
    topK,
    setTopK,
    sessionId,
    setSessionId,
    onNewChat,
}) => {
    const [newItem, setNewItem] = useState('');
    const [sessions, setSessions] = useState<Session[]>([]);

    useEffect(() => {
        fetchSessions();
    }, [sessionId]); // Refresh when session changes (e.g. title update)

    const fetchSessions = async () => {
        try {
            const res = await axios.get('http://localhost:8000/sessions');
            setSessions(res.data);
        } catch (e) {
            console.error("Failed to fetch sessions", e);
        }
    };

    const deleteSession = async (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        if (!confirm("Delete this chat?")) return;
        try {
            await axios.delete(`http://localhost:8000/sessions/${id}`);
            if (sessionId === id) {
                onNewChat();
            }
            fetchSessions();
        } catch (e) {
            console.error(e);
        }
    };

    const handleAddUrl = () => {
        if (newItem.trim()) {
            setUrls([...urls, newItem.trim()]);
            setNewItem('');
        }
    };

    const handleRemoveUrl = (idx: number) => {
        setUrls(urls.filter((_, i) => i !== idx));
    };

    const handleSaveConfig = async () => {
        try {
            await axios.post('http://localhost:8000/config', { urls });
            alert("Configuration saved! Don't forget to rebuild index.");
        } catch (e) {
            console.error(e);
            alert("Failed to save config.");
        }
    };

    return (
        <div className="w-80 h-full bg-slate-50 border-r border-slate-200 flex flex-col shadow-sm">
            <div className="p-4 border-b border-slate-200 bg-white/50 backdrop-blur-sm">
                <button
                    onClick={onNewChat}
                    className="w-full bg-blue-600 text-white rounded-xl py-3 px-4 flex items-center justify-center gap-2 hover:bg-blue-700 transition-all font-medium shadow-sm hover:shadow-md active:scale-[0.98]"
                >
                    <Plus size={18} />
                    New Chat
                </button>
            </div>

            <div className="flex-1 overflow-y-auto">
                <div className="p-4 space-y-6">
                    {/* Sessions List */}
                    <div className="space-y-2">
                        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-2">History</h3>
                        <div className="space-y-1">
                            {sessions.map(session => (
                                <button
                                    key={session.id}
                                    onClick={() => setSessionId(session.id)}
                                    className={cn(
                                        "w-full text-left px-3 py-2.5 rounded-lg text-sm flex items-center gap-3 transition-colors group relative",
                                        sessionId === session.id
                                            ? "bg-white shadow-sm ring-1 ring-slate-200 text-slate-800 font-medium"
                                            : "text-slate-600 hover:bg-slate-100"
                                    )}
                                >
                                    <MessageSquare size={16} className={cn(
                                        sessionId === session.id ? "text-blue-500" : "text-slate-400"
                                    )} />
                                    <span className="truncate flex-1 pr-6">{session.title || "New Chat"}</span>

                                    <div
                                        className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-md hover:bg-red-50 text-slate-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-all"
                                        onClick={(e) => deleteSession(e, session.id)}
                                    >
                                        <Trash2 size={14} />
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="h-px bg-slate-200 my-4" />

                    {/* Settings Section */}
                    <div className="space-y-4">
                        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-2">Settings</h3>

                        <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-4 shadow-sm">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2 text-sm text-slate-700">
                                    <span className="w-2 h-2 rounded-full bg-purple-500"></span>
                                    HyDE
                                </div>
                                <label className="relative inline-flex items-center cursor-pointer">
                                    <input type="checkbox" checked={useHyde} onChange={e => setUseHyde(e.target.checked)} className="sr-only peer" />
                                    <div className="w-9 h-5 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-600"></div>
                                </label>
                            </div>

                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2 text-sm text-slate-700">
                                    <span className="w-2 h-2 rounded-full bg-indigo-500"></span>
                                    Rerank
                                </div>
                                <label className="relative inline-flex items-center cursor-pointer">
                                    <input type="checkbox" checked={useRerank} onChange={e => setUseRerank(e.target.checked)} className="sr-only peer" />
                                    <div className="w-9 h-5 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-600"></div>
                                </label>
                            </div>

                            <div className="space-y-2">
                                <div className="flex justify-between text-sm text-slate-700">
                                    <span>Top-K</span>
                                    <span className="font-mono bg-slate-100 px-1.5 rounded text-xs py-0.5">{topK}</span>
                                </div>
                                <input
                                    type="range" min="1" max="20" value={topK} onChange={e => setTopK(Number(e.target.value))}
                                    className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                                />
                            </div>
                        </div>

                        {/* URL Management */}
                        <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-3 shadow-sm">
                            <label className="text-xs font-semibold text-slate-500 uppercase">Knowledge Base</label>
                            <div className="max-h-32 overflow-y-auto space-y-2 custom-scrollbar pr-1">
                                {urls.map((url, idx) => (
                                    <div key={idx} className="flex items-start gap-2 group text-xs text-slate-600">
                                        <div className="bg-slate-50 p-1.5 rounded-md border border-slate-100 flex-1 break-all">
                                            {url}
                                        </div>
                                        <button onClick={() => handleRemoveUrl(idx)} className="text-slate-300 hover:text-red-500 p-1 transition-colors">
                                            <Trash2 size={12} />
                                        </button>
                                    </div>
                                ))}
                            </div>
                            <div className="flex gap-2">
                                <input
                                    className="flex-1 bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 text-xs focus:ring-2 focus:ring-blue-100 focus:border-blue-400 outline-none transition-all"
                                    placeholder="Add URL..."
                                    value={newItem}
                                    onChange={(e) => setNewItem(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && handleAddUrl()}
                                />
                                <button onClick={handleAddUrl} className="bg-slate-100 hover:bg-slate-200 text-slate-600 p-1.5 rounded-lg transition-colors">
                                    <Plus size={14} />
                                </button>
                            </div>

                            <div className="flex gap-2 pt-2">
                                <button onClick={handleSaveConfig} className="flex-1 bg-slate-800 text-white text-xs py-2 rounded-lg hover:bg-slate-900 transition-colors">
                                    Save Config
                                </button>
                                <button
                                    onClick={onRebuild}
                                    disabled={isRebuilding}
                                    className={cn(
                                        "flex-1 flex items-center justify-center gap-1.5 text-xs py-2 rounded-lg border font-medium transition-all",
                                        isRebuilding
                                            ? "bg-amber-50 text-amber-600 border-amber-200 cursor-not-allowed"
                                            : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50"
                                    )}
                                >
                                    <RefreshCw size={12} className={cn(isRebuilding && "animate-spin")} />
                                    {isRebuilding ? "Rebuilding..." : "Rebuild Index"}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="p-4 border-t border-slate-200 bg-white text-xs text-center text-slate-400">
                Lightweight RAG System v1.1
            </div>
        </div>
    );
};

export default Sidebar;
