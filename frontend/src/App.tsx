import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import Sidebar from './components/Sidebar'
import ChatInterface from './components/ChatInterface'
import SourceViewer from './components/SourceViewer'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
}

function App() {
  const [urls, setUrls] = useState<string[]>([])
  const [isRebuilding, setIsRebuilding] = useState(false)
  const [topK, setTopK] = useState(10)
  const [useHyde, setUseHyde] = useState(false)
  const [useRerank, setUseRerank] = useState(true)

  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [sources, setSources] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(false)

  // Ref to track if session switch is internal (auto-created) to avoid clearing state
  const isAutoSwitching = useRef(false)

  // Load config on mount
  useEffect(() => {
    fetchConfig()
  }, [])

  // Load messages when sessionId changes
  useEffect(() => {
    if (sessionId) {
      if (isAutoSwitching.current) {
        // This switch was triggered by handleSendMessage creating a new session.
        // We do NOT want to load messages (it's empty) or clear sources (we might be about to set them).
        isAutoSwitching.current = false;
        return;
      }
      loadSessionMessages(sessionId)
    } else {
      setMessages([])
      setSources([])
    }
  }, [sessionId])

  const loadSessionMessages = async (id: string) => {
    try {
      const res = await axios.get(`http://localhost:8000/sessions/${id}`)
      const loadedMessages: Message[] = res.data
      setMessages(loadedMessages)

      // Attempt to restore sources from the last assistant message
      const lastMsg = loadedMessages[loadedMessages.length - 1]
      if (lastMsg && lastMsg.role === 'assistant' && lastMsg.sources && lastMsg.sources.length > 0) {
        setSources(lastMsg.sources)
      } else {
        setSources([])
      }
    } catch (e) {
      console.error("Failed to load session", e)
    }
  }

  const handleNewChat = async () => {
    try {
      const res = await axios.post('http://localhost:8000/sessions')
      setSessionId(res.data.id)
      setMessages([])
      setSources([])
    } catch (e) {
      console.error(e)
    }
  }

  const fetchConfig = async () => {
    try {
      const res = await axios.get('http://localhost:8000/config')
      setUrls(res.data.urls)
    } catch (e) {
      console.error("Failed to fetch config", e)
    }
  }

  const handleRebuild = async () => {
    setIsRebuilding(true)
    try {
      await axios.post('http://localhost:8000/rebuild')
      setTimeout(() => setIsRebuilding(false), 5000)
    } catch (e) {
      console.error(e)
      setIsRebuilding(false)
    }
  }

  const handleSendMessage = async (msg: string) => {
    // Ensure we have a session ID before sending
    let currentSessionId = sessionId
    if (!currentSessionId) {
      try {
        const res = await axios.post('http://localhost:8000/sessions')
        currentSessionId = res.data.id
        // Mark as auto-switching so useEffect doesn't wipe state
        isAutoSwitching.current = true
        setSessionId(currentSessionId)
      } catch (e) {
        console.error("Failed to create session", e)
        return
      }
    }

    setIsLoading(true)
    setSources([])

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: msg,
          history: messages,
          session_id: currentSessionId, // Send session ID
          use_hyde: useHyde,
          top_k: topK,
          rerank_top_k: 8,
          use_rerank: useRerank,
        }),
      })

      if (!response.body) throw new Error("No response body")

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      let assistantMsg = ""
      setMessages(prev => [...prev, { role: 'assistant', content: '' }])

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (!line.trim()) continue
          try {
            const json = JSON.parse(line)
            if (json.type === 'sources') {
              setSources(json.data)
            } else if (json.type === 'content') {
              assistantMsg += json.data
              setMessages(prev => {
                const newMsgs = [...prev]
                newMsgs[newMsgs.length - 1].content = assistantMsg
                return newMsgs
              })
            }
          } catch (e) {
            console.warn("Failed to parse chunk", line)
          }
        }
      }

    } catch (e) {
      console.error(e)
      setMessages(prev => [...prev, { role: 'assistant', content: "Error: Failed to connect to server." }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex h-screen w-screen bg-slate-100 overflow-hidden font-sans text-slate-800">
      <Sidebar
        urls={urls}
        setUrls={setUrls}
        onRebuild={handleRebuild}
        isRebuilding={isRebuilding}
        useHyde={useHyde}
        setUseHyde={setUseHyde}
        useRerank={useRerank}
        setUseRerank={setUseRerank}
        topK={topK}
        setTopK={setTopK}
        sessionId={sessionId}
        setSessionId={setSessionId}
        onNewChat={handleNewChat}
      />

      <ChatInterface
        messages={messages}
        setMessages={setMessages}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
      />

      <SourceViewer sources={sources} />
    </div>
  )
}

export default App
