import { useEffect, useRef, useState } from 'react'

const SUGGESTED = [
  "What is Gobinath's experience?",
  'What AI projects has he built?',
  'How can I contact him?',
  'What roles is he open to?',
]

function SourceCard({ source }) {
  return (
    <div className="source-card">
      <div className="source-title">{source.title}</div>
      <div className="source-excerpt">{source.excerpt}</div>
      <div className="source-score">
        similarity {typeof source.score === 'number' ? source.score.toFixed(3) : source.score}
      </div>
    </div>
  )
}

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`message-row ${isUser ? 'user' : 'bot'}`}>
      <div className={`bubble ${isUser ? 'user' : 'bot'}`}>
        {!isUser && msg.loading ? (
          <span className="typing">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </span>
        ) : (
          <p>{msg.text}</p>
        )}
        {!isUser && msg.sources && msg.sources.length > 0 && (
          <div className="sources">
            <div className="sources-label">Sources</div>
            {msg.sources.map((s, i) => (
              <SourceCard key={i} source={s} />
            ))}
            {typeof msg.confidence === 'number' && (
              <div className="confidence">
                confidence {msg.confidence.toFixed(3)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'bot',
      text: "Hi! I'm a chatbot about Gobinath Rangaraj. Ask me about his experience, skills, projects, or how to contact him.",
    },
  ])
  const [input, setInput] = useState('')
  const [error, setError] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send(question) {
    const q = question.trim()
    if (!q) return
    setError(null)
    setMessages((prev) => [
      ...prev,
      { role: 'user', text: q },
      { role: 'bot', text: '', loading: true },
    ])
    setInput('')

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      })
      if (!res.ok) {
        throw new Error(`Backend error ${res.status}`)
      }
      const data = await res.json()
      setMessages((prev) => [
        ...prev.slice(0, -1),
        {
          role: 'bot',
          text: data.answer,
          sources: data.sources,
          confidence: data.confidence,
        },
      ])
    } catch (e) {
      setError(
        'Could not reach the chatbot backend. Make sure it is running at http://localhost:8000.'
      )
      setMessages((prev) => prev.slice(0, -1))
    }
  }

  function onSubmit(e) {
    e.preventDefault()
    send(input)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">GR</div>
          <div>
            <h1>Gobinath Rangaraj</h1>
            <p className="subtitle">RAG-powered portfolio chatbot</p>
          </div>
        </div>
      </header>

      <main className="chat">
        {messages.map((m, i) => (
          <Message key={i} msg={m} />
        ))}
        <div ref={bottomRef} />
      </main>

      {error && <div className="error-banner">{error}</div>}

      <footer className="composer-wrap">
        <div className="chips">
          {SUGGESTED.map((q) => (
            <button key={q} className="chip" onClick={() => send(q)}>
              {q}
            </button>
          ))}
        </div>
        <form className="composer" onSubmit={onSubmit}>
          <input
            type="text"
            placeholder="Ask about Gobinath..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button type="submit" disabled={!input.trim()}>
            Send
          </button>
        </form>
        <p className="disclaimer">
          Answers come only from the curated knowledge base — citations included.
        </p>
      </footer>
    </div>
  )
}
