import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Loader2 } from 'lucide-react';
import { cn } from '../lib/utils';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/atom-one-dark.css';

interface Message {
    role: 'user' | 'assistant';
    content: string;
}

interface ChatInterfaceProps {
    messages: Message[];
    setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
    onSendMessage: (msg: string) => Promise<void>;
    isLoading: boolean;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
    messages,
    setMessages,
    onSendMessage,
    isLoading
}) => {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMsg = input.trim();
        setInput('');
        // Optimistically add user message
        setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        await onSendMessage(userMsg);
    };

    return (
        <div className="flex-1 flex flex-col h-full bg-slate-100 relative">
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {messages.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-slate-400 space-y-4 opacity-50">
                        <div className="w-16 h-16 bg-white rounded-2xl flex items-center justify-center shadow-sm">
                            <Bot size={32} className="text-blue-500" />
                        </div>
                        <p className="font-medium">Start a conversation with your RAG Assistant</p>
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <motion.div
                        key={idx}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={cn(
                            "flex gap-4 max-w-3xl mx-auto",
                            msg.role === 'user' ? "flex-row-reverse" : "flex-row"
                        )}
                    >
                        <div className={cn(
                            "w-8 h-8 rounded-lg flex items-center justify-center shrink-0 shadow-sm",
                            msg.role === 'user' ? "bg-blue-600 text-white" : "bg-white text-blue-600 border border-slate-200"
                        )}>
                            {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                        </div>

                        <div className={cn(
                            "rounded-2xl p-4 shadow-sm text-sm leading-relaxed overflow-hidden max-w-[85%]",
                            msg.role === 'user'
                                ? "bg-blue-600 text-white rounded-tr-sm"
                                : "bg-white text-slate-800 border border-slate-200 rounded-tl-sm"
                        )}>
                            <div className={cn(
                                "prose prose-sm max-w-none",
                                msg.role === 'user' ? "prose-invert" : "prose-slate"
                            )}>
                                <ReactMarkdown
                                    remarkPlugins={[remarkGfm]}
                                    rehypePlugins={[rehypeHighlight]}
                                    components={{
                                        pre: ({ node, ...props }) => (
                                            <div className="overflow-auto w-full my-2 bg-slate-950 p-4 rounded-lg">
                                                <pre {...props} />
                                            </div>
                                        )
                                    }}
                                >
                                    {msg.content}
                                </ReactMarkdown>
                            </div>
                        </div>
                    </motion.div>
                ))}

                {isLoading && messages[messages.length - 1]?.role === 'user' && (
                    <div className="flex gap-4 max-w-3xl mx-auto">
                        <div className="w-8 h-8 rounded-lg bg-white text-blue-600 border border-slate-200 flex items-center justify-center shrink-0 shadow-sm">
                            <Bot size={16} />
                        </div>
                        <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm p-4 shadow-sm flex items-center">
                            <Loader2 size={16} className="animate-spin text-slate-400" />
                            <span className="ml-2 text-xs text-slate-500 font-medium">Thinking...</span>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            <div className="p-4 bg-white/80 backdrop-blur-md border-t border-slate-200">
                <div className="max-w-3xl mx-auto">
                    <form onSubmit={handleSubmit} className="relative flex items-center shadow-lg rounded-2xl overflow-hidden bg-white border border-slate-200 focus-within:ring-2 focus-within:ring-blue-100 transition-all">
                        <input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask anything about your documents..."
                            className="flex-1 py-4 pl-6 pr-12 bg-transparent text-sm focus:outline-none text-slate-700 placeholder:text-slate-400"
                        />
                        <button
                            type="submit"
                            disabled={isLoading || !input.trim()}
                            className="absolute right-2 p-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-colors"
                        >
                            <Send size={18} />
                        </button>
                    </form>
                    <div className="text-center mt-2 text-[10px] text-slate-400">
                        Press Enter to send. Use the sidebar to configure settings.
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChatInterface;
