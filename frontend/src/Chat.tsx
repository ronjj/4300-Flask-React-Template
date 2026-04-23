import { useState, useRef, useEffect } from 'react'
import SearchIcon from './assets/mag.png'

interface Message {
  text: string
  isUser: boolean
}

interface ChatProps {
  onSearchTerm: (term: string) => void
}

function Chat({ onSearchTerm }: ChatProps): JSX.Element {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inFlightControllerRef = useRef<AbortController | null>(null)
  const inactivityTimerRef = useRef<number | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const resetInactivityTimer = (timeoutMs: number): void => {
    if (inactivityTimerRef.current) window.clearTimeout(inactivityTimerRef.current)
    inactivityTimerRef.current = window.setTimeout(() => {
      inFlightControllerRef.current?.abort()
    }, timeoutMs)
  }

  const sendMessage = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    setMessages(prev => [...prev, { text, isUser: true }])
    setInput('')
    setLoading(true)
    inFlightControllerRef.current?.abort()

    const controller = new AbortController()
    inFlightControllerRef.current = controller
    resetInactivityTimer(120000)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
        signal: controller.signal,
      })

      if (!response.ok) {
        const data = await response.json()
        setMessages(prev => [...prev, { text: 'Error: ' + (data.error || response.status), isUser: false }])
        return
      }

      let assistantText = ''
      setMessages(prev => [...prev, { text: '', isUser: false }])

      if (!response.body) {
        setMessages(prev => [...prev, { text: 'Error: No response body from server', isUser: false }])
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.search_term !== undefined) {
                onSearchTerm(data.search_term)
              }
              resetInactivityTimer(120000)
              if (data.error) {
                setMessages(prev => [...prev.slice(0, -1), { text: 'Error: ' + data.error, isUser: false }])
                return
              }
              if (data.content !== undefined) {
                assistantText += data.content
                setMessages(prev => [...prev.slice(0, -1), { text: assistantText, isUser: false }])
              }
            } catch {}
          }
        }
      }
    } catch {
      const isAbort = controller.signal.aborted
      const errorText = isAbort ? 'Error: Request timed out.' : 'Something went wrong. Check the console.'
      setMessages(prev => {
        if (prev.length > 0 && !prev[prev.length - 1].isUser) {
          return [...prev.slice(0, -1), { text: errorText, isUser: false }]
        }
        return [...prev, { text: errorText, isUser: false }]
      })
    } finally {
      setLoading(false)
      if (inactivityTimerRef.current) window.clearTimeout(inactivityTimerRef.current)
      inactivityTimerRef.current = null
      if (inFlightControllerRef.current === controller) inFlightControllerRef.current = null
    }
  }

  return (
    <>
      <div id="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.isUser ? 'user' : 'assistant'}`}>
            <p>{msg.text}</p>
          </div>
        ))}
        {loading && (
          <div className="loading-indicator visible">
            <span className="loading-dot" />
            <span className="loading-dot" />
            <span className="loading-dot" />
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-bar">
        <form className="input-row" onSubmit={sendMessage}>
          <img src={SearchIcon} alt="" />
          <input
            type="text"
            placeholder="Ask the AI about Keeping Up with the Kardashians"
            value={input}
            onChange={e => setInput(e.target.value)}
            disabled={loading}
            autoComplete="off"
          />
          <button type="submit" disabled={loading}>Send</button>
        </form>
      </div>
    </>
  )
}

export default Chat
