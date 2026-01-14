import React from 'react';
import { BookOpen, Quote } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface SourceViewerProps {
    sources: string[];
}

const SourceViewer: React.FC<SourceViewerProps> = ({ sources }) => {
    return (
        <div className="w-80 h-full bg-slate-50/50 border-l border-slate-200 flex flex-col backdrop-blur-sm">
            <div className="p-4 border-b border-slate-200 flex items-center gap-2 bg-white/80 backdrop-blur-md sticky top-0 z-10">
                <BookOpen size={18} className="text-blue-600" />
                <h2 className="font-semibold text-slate-800">Retrieved Context</h2>
                <span className="ml-auto text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-medium">
                    {sources.length}
                </span>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {sources.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-slate-400 space-y-2 p-8 text-center opacity-60">
                        <BookOpen size={48} strokeWidth={1} />
                        <p className="text-sm">Ask a question to see retrieved sources here.</p>
                    </div>
                ) : (
                    <AnimatePresence>
                        {sources.map((source, idx) => (
                            <motion.div
                                key={idx}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.3, delay: idx * 0.05 }}
                                className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow group"
                            >
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">
                                        Source {idx + 1}
                                    </span>
                                    {/* Assuming potential future metadata like URL or Title */}
                                    <Quote size={12} className="text-slate-300 group-hover:text-blue-400 transition-colors" />
                                </div>
                                <p className="text-xs text-slate-600 leading-relaxed font-mono bg-slate-50 p-2 rounded border border-slate-100">
                                    {source}
                                </p>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                )}
            </div>
        </div>
    );
};

export default SourceViewer;
