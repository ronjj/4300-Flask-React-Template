import { useState, useRef, useEffect } from 'react'
import SearchIcon from './assets/mag.png'
import { PlayerChatEvidence, PlayerChatResponse, PlayerChatResult } from './types'

interface Message {
  isUser: boolean
  text: string
  rewrittenQuery?: string | null
  results?: PlayerChatResult[]
  evidence?: PlayerChatEvidence[]
  retrievalConfidence?: number | null
  warnings?: string[]
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
    if (messages.length === 0) return
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [messages, loading])

  const formatKeyStats = (stats?: Record<string, unknown>): string[] => {
    if (!stats) return []
    return Object.entries(stats)
      .filter(([, value]) => value != null)
      .slice(0, 3)
      .map(([key, value]) => `${key.replace(/_/g, ' ')}: ${String(value)}`)
  }

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
      const response = await fetch('/api/player-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
        signal: controller.signal,
      })

      const data: PlayerChatResponse = await response.json()

      if (!response.ok) {
        const errorText = data.answer || data.error || `Error: ${response.status}`
        setMessages(prev => [...prev, { text: String(errorText), isUser: false }])
        return
      }

      const answer = (data.answer || '').trim()
      if (!answer) {
        const fallbackError = data.error || 'Error: Invalid response from server'
        setMessages(prev => [...prev, { text: String(fallbackError), isUser: false }])
        return
      }

      const results = Array.isArray(data.results) ? data.results : []
      const warnings = Array.isArray(data.warnings) ? data.warnings : []
      const evidence = Array.isArray(data.evidence) ? data.evidence : []
      const rewrittenQuery = typeof data.rewritten_query === 'string' ? data.rewritten_query : null
      const retrievalConfidence =
        typeof data.retrieval_confidence === 'number' ? data.retrieval_confidence : null

      setMessages(prev => [
        ...prev,
        {
          text: answer,
          isUser: false,
          rewrittenQuery,
          results,
          evidence,
          retrievalConfidence,
          warnings,
        },
      ])

      if (results.length === 1) {
        const playerName = results[0]?.player_name?.trim()
        if (playerName) onSearchTerm(playerName)
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
            {!msg.isUser && msg.rewrittenQuery && (
              <div className="query-expansion-banner">
                <span className="query-expansion-label">INTERPRETED AS</span>
                <span className="query-expansion-text">{msg.rewrittenQuery}</span>
              </div>
            )}
            <p>{msg.text}</p>
            {!msg.isUser && (
              <>
                {(msg.rewrittenQuery != null || msg.retrievalConfidence != null || (msg.results?.length ?? 0) > 0 || (msg.evidence?.length ?? 0) > 0 || (msg.warnings?.length ?? 0) > 0) && (
                  <div className="chat-meta">
                    {msg.rewrittenQuery && (
                      <div className="chat-meta-block">
                        <p className="chat-meta-title">IR query</p>
                        <p className="chat-meta-line">{msg.rewrittenQuery}</p>
                      </div>
                    )}
                    {msg.retrievalConfidence != null && (
                      <p className="chat-meta-line">
                        Retrieval confidence: {msg.retrievalConfidence.toFixed(3)}
                      </p>
                    )}
                    {(msg.results?.length ?? 0) > 0 && (
                      <div className="chat-meta-block">
                        <p className="chat-meta-title">Top results</p>
                        <ul className="chat-meta-list">
                          {msg.results?.slice(0, 3).map((result, resultIndex) => (
                            <li key={`${result.player_id ?? result.player_name ?? 'result'}-${resultIndex}`}>
                              {[result.player_name, result.team].filter(Boolean).join(' · ') || 'Unnamed player'}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {(msg.evidence?.length ?? 0) > 0 && (
                      <div className="chat-meta-block">
                        <p className="chat-meta-title">Retrieved evidence</p>
                        <ul className="chat-meta-list">
                          {msg.evidence?.slice(0, 3).map((item, evidenceIndex) => {
                            const details = [
                              item.player_name,
                              item.team,
                              item.season_label,
                              typeof item.retrieval_score === 'number'
                                ? `score ${item.retrieval_score.toFixed(3)}`
                                : null,
                            ].filter(Boolean)
                            const stats = formatKeyStats(item.key_stats)
                            return (
                              <li key={`${item.evidence_id ?? item.player_name ?? 'evidence'}-${evidenceIndex}`}>
                                <div>{details.join(' · ')}</div>
                                {stats.length > 0 && <div>{stats.join(' | ')}</div>}
                              </li>
                            )
                          })}
                        </ul>
                      </div>
                    )}
                    {(msg.warnings?.length ?? 0) > 0 && (
                      <div className="chat-meta-block">
                        <p className="chat-meta-title">Warnings</p>
                        <ul className="chat-meta-list">
                          {msg.warnings?.map((warning, warningIndex) => (
                            <li key={`${warning}-${warningIndex}`}>{warning}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
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
            placeholder="Ask about similar players, comparisons, or playing styles..."
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
